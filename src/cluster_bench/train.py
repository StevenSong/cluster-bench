"""Run one benchmark cell and write one results JSON.

This is what `accelerate launch` invokes on every rank. It knows nothing about
the sweep -- the strategy, placement and token count arrive as flags, and the
matrix driver is responsible for producing the right combination.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch.distributed as dist

from . import config, data, metrics, modeling, placement as placement_mod, provenance
from . import strategies as strategies_mod


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def _log(msg: str) -> None:
    if _rank() == 0:
        print(f"[cluster-bench] {msg}", flush=True)


def run(spec: config.RunSpec) -> dict[str, Any]:
    from trl import SFTConfig, SFTTrainer

    strategy = strategies_mod.get(spec.strategy)
    place = placement_mod.get(spec.placement)

    blockers = placement_mod.validate(strategy, place)
    if blockers:
        raise SystemExit(
            "this (strategy, placement) cell is not measurable:\n  - "
            + "\n  - ".join(blockers)
        )

    world_size = _world_size()
    if world_size != place.world_size:
        raise SystemExit(
            f"WORLD_SIZE={world_size} but placement {place.name!r} describes "
            f"{place.world_size} ranks -- the launcher and the spec disagree"
        )

    _log(f"run_id={spec.run_id} world_size={world_size}")
    _log(f"tokens/GPU/step={spec.tokens_per_gpu_step} (the pinned control)")

    model, tokenizer, model_info = modeling.load(spec)
    _log(
        f"model params={model_info['params_total'] / 1e9:.2f}B "
        f"hidden={model_info['hidden_size']} layers={model_info['num_hidden_layers']} "
        f"vocab={model_info['vocab_size']} tied={model_info['tie_word_embeddings']}"
    )

    dataset, is_synthetic = data.build(
        spec, model_info["vocab_size"] or 32000, world_size
    )

    dataset_kwargs: dict[str, Any] = {"add_special_tokens": False}
    # Samples are already exactly seq_len tokens; packing them again would
    # reintroduce the variance the synthetic set exists to remove. TRL rejects
    # padding_free without packing unless max_length is None ("already
    # truncated inputs") -- which is exactly what the synthetic set is, so
    # max_length has nothing left to enforce. tokens/GPU/step is unchanged.
    max_length = spec.seq_len
    # The synthetic set ships its own labels, already masked to the same 25/75
    # prompt/completion split, so completion_only_loss has nothing left to do
    # -- and no completion_mask to do it with.
    completion_only_loss = spec.completion_only_loss
    if is_synthetic:
        dataset_kwargs = {"skip_prepare_dataset": True}
        max_length = None
        completion_only_loss = False

    total_steps = spec.warmup_steps + spec.measure_steps

    args = SFTConfig(
        output_dir=str(spec.out_dir / "hf" / spec.run_id),
        max_steps=total_steps,
        per_device_train_batch_size=spec.micro_batch,
        gradient_accumulation_steps=spec.grad_accum,
        learning_rate=spec.learning_rate,
        lr_scheduler_type="constant",  # no schedule-induced drift across cells
        warmup_steps=0,
        weight_decay=0.0,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        seed=spec.seed,
        data_seed=spec.seed,
        max_length=max_length,
        packing=spec.packing and not is_synthetic,
        packing_strategy=spec.packing_strategy,
        padding_free=spec.padding_free,
        use_liger_kernel=spec.liger,
        gradient_checkpointing=spec.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        completion_only_loss=completion_only_loss,
        average_tokens_across_devices=spec.average_tokens_across_devices,
        dataset_kwargs=dataset_kwargs,
        dataloader_num_workers=spec.dataloader_num_workers,
        dataloader_pin_memory=True,
        dataloader_drop_last=True,  # a short final batch would skew the tail
        ddp_find_unused_parameters=False,
        logging_steps=1,  # the loss curve is the correctness gate
        logging_first_step=True,
        save_strategy="no",  # benchmark cells do not need checkpoints
        report_to="none",  # results go to JSON, not a tracker's step hook
    )

    bench = metrics.BenchCallback(
        warmup_steps=spec.warmup_steps,
        measure_steps=spec.measure_steps,
        tokens_per_gpu_step=spec.tokens_per_gpu_step,
        sync_each_step=spec.sync_each_step,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[bench],
    )

    started = time.time()
    trainer.train()
    wall = time.time() - started

    result: dict[str, Any] = {
        "run_id": spec.run_id,
        "strategy": {
            "name": strategy.name,
            "backend": strategy.backend,
            "purpose": strategy.purpose,
            "param_gather": strategy.param_gather,
            "grad_reduce": strategy.grad_reduce,
            "zero_stage": strategy.zero_stage,
            "hpz": strategy.hpz,
            "autotp": strategy.autotp,
        },
        "placement": {
            "name": place.name,
            "links": place.links,
            "world_size": place.world_size,
            "num_machines": place.num_machines,
            # As launched, per machine_rank -- peers run node0's list reversed.
            "devices": [
                list(place.devices_for(r)) for r in range(place.num_machines)
            ],
        },
        "spec": spec.to_dict(),
        "model": model_info,
        "dataset": {"source": spec.dataset, "synthetic": is_synthetic},
        "wall_time_s": wall,
        **bench.summary(world_size),
    }
    result["provenance"] = provenance.collect()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run one cell of the cluster sharding benchmark."
    )
    config.add_arguments(ap)
    args = ap.parse_args()
    spec = config.from_args(args)

    result = run(spec)

    if _rank() == 0:
        out = spec.out_dir / "runs" / f"{spec.run_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str) + "\n")
        s = result
        print(
            f"[cluster-bench] {spec.run_id}: "
            f"p50={s.get('step_time_p50_s', float('nan')):.4f}s "
            f"p95={s.get('step_time_p95_s', float('nan')):.4f}s "
            f"tok/s/gpu={s.get('tokens_per_s_per_gpu') or 0:.0f} "
            f"peak={s.get('peak_mem_allocated_gb') or 0:.1f}GB",
            flush=True,
        )
        print(f"[cluster-bench] wrote {out}", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
