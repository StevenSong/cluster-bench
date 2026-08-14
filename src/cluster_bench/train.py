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

import torch
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


class _TPBatchBroadcaster:
    """Make every rank of a tensor-parallel group see the same micro-batch.

    AutoTP shards each layer across the TP group, so the group is one logical
    model and must be fed one batch. The HF sampler knows nothing about
    DeepSpeed's TP mesh -- it shards the dataset across all WORLD_SIZE ranks --
    so without this the two ranks of a pair run different data. DeepSpeed
    notices: `_configure_tensor_parallel_states` installs a one-shot forward
    pre-hook that asserts "Data inconsistency within the TP group". That assert
    is the only thing standing between us and a silently wrong run, and it only
    checks the first batch, so the fix has to be real rather than good enough
    to get past step 1.

    Broadcasting the source rank's batch means the partner rank's own samples
    are dropped -- the cell consumes half the sequences per step that its rank
    count suggests, which is exactly why the per-device batch is doubled at the
    call site. Sample *order* still differs from a non-TP cell, so this row's
    absolute loss is not comparable to the correctness gate's reference; see
    report.check_loss_curves.
    """

    def __init__(self) -> None:
        self._group = None
        self._src = 0
        self._shapes_checked = False

    def _resolve(self) -> None:
        # Only valid once the engine has built the mesh, i.e. after
        # deepspeed.initialize -- which is why this is lazy.
        from deepspeed.utils import groups

        self._group = groups.get_tensor_model_parallel_group()
        self._src = groups.get_tensor_model_parallel_src_rank()

    def __call__(self, inputs: dict) -> dict:
        if self._group is None:
            self._resolve()
        # Sorted so every rank broadcasts the same tensor in the same order
        # regardless of how the collator happened to build the dict.
        keys = sorted(k for k, v in inputs.items() if torch.is_tensor(v))
        if not self._shapes_checked:
            self._check_shapes(inputs, keys)
            self._shapes_checked = True
        for k in keys:
            dist.broadcast(inputs[k], src=self._src, group=self._group)
        return inputs

    def _check_shapes(self, inputs: dict, keys: list[str]) -> None:
        """A shape mismatch would hang or silently corrupt; say so instead.

        Once, on the first batch: `wrapped` packing gives every rank chunks of
        exactly seq_len so the shapes do match, but that is the assumption the
        whole run rests on and it costs one object broadcast to confirm.
        """
        mine = {k: tuple(inputs[k].shape) for k in keys}
        payload = [mine if dist.get_rank() == self._src else None]
        dist.broadcast_object_list(
            payload,
            src=self._src,
            group=self._group,
            # Explicit, or torch picks a device for the pickled buffer itself
            # and warns about it on every NCCL rank.
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        theirs = payload[0]
        if theirs != mine:
            raise SystemExit(
                "TP group ranks were handed differently shaped batches "
                f"(source {theirs}, this rank {mine}). A TP group must run one "
                "batch; check packing_strategy is 'wrapped' and that "
                "dataloader_drop_last is on."
            )


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

    # Under TP the group's ranks share one batch, so a per-device batch of
    # micro_batch would put only micro_batch/tp_size sequences' worth of work
    # on each GPU: half the compute of every other cell, at half the global
    # batch, and a step time that looks like a comm win it did not earn.
    # Scaling by tp_size restores both -- each GPU does tp_size x micro_batch
    # sequences against 1/tp_size of the model, i.e. the same FLOPs and the
    # same 8192 tok/GPU/step, and the global batch matches the DDP cell's.
    # spec.tokens_per_gpu_step stays 8192 so report.py still groups this cell
    # with its baseline.
    device_micro_batch = spec.micro_batch * strategy.tp_size

    _log(f"run_id={spec.run_id} world_size={world_size}")
    _log(f"tokens/GPU/step={spec.tokens_per_gpu_step} (the pinned control)")
    if strategy.tp_size > 1:
        _log(
            f"tp_size={strategy.tp_size}: per-device batch "
            f"{spec.micro_batch} -> {device_micro_batch} (a TP group shares one "
            f"batch), dp_world_size={world_size // strategy.tp_size}"
        )

    model, tokenizer, model_info = modeling.load(spec)
    _log(
        f"model params={model_info['params_total'] / 1e9:.2f}B "
        f"hidden={model_info['hidden_size']} layers={model_info['num_hidden_layers']} "
        f"vocab={model_info['vocab_size']} tied={model_info['tie_word_embeddings']}"
    )

    dataset, is_synthetic = data.build(
        spec,
        tokenizer,
        model_info["vocab_size"] or 32000,
        world_size,
        device_micro_batch=device_micro_batch,
    )
    _log(f"dataset={spec.dataset} samples={len(dataset)} synthetic={is_synthetic}")

    # Packed sequences cross document boundaries, and only the flash-attention
    # varlen path honours the boundary -- which is why attn_impl defaults to
    # flash_attention_2. Reaching here means someone opted out of it. FLOPs are
    # identical either way, so step times -- what this repo measures -- are
    # unaffected, and the loss-curve gate still compares configs against each
    # other because every cell is contaminated identically. It does mean the
    # absolute loss is not a training loss. Recorded so a future reader does not
    # mistake it for one.
    packed_attn_leak = (
        spec.packing and not is_synthetic and "flash" not in spec.attn_impl
    )
    if packed_attn_leak:
        _log(
            f"NOTE: packing with attn_impl={spec.attn_impl!r} -- attention crosses "
            "packed document boundaries. Step times are unaffected; absolute loss "
            "is not a training loss. Drop --attn-impl to get the default "
            "flash_attention_2 back."
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
        per_device_train_batch_size=device_micro_batch,
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
        tp_size=strategy.tp_size,
    )
    tp_broadcast = _TPBatchBroadcaster() if strategy.tp_size > 1 else None

    # Tokens are counted from the batch the trainer is about to run, which is
    # the only place they are visible in the main process -- a callback never
    # sees the inputs, and the collator runs in a dataloader worker.
    #
    # Both hooks hang off _prepare_inputs rather than training_step: the TP
    # broadcast needs the batch already on the GPU (NCCL cannot broadcast host
    # tensors), and the count has to happen after it, so that what is tallied
    # is the batch the model actually ran. A benchmark cell never evaluates, so
    # this is called exactly once per micro-batch.
    class _BenchSFTTrainer(SFTTrainer):
        def _prepare_inputs(self, inputs):  # noqa: ANN001
            inputs = super()._prepare_inputs(inputs)
            if tp_broadcast is not None:
                inputs = tp_broadcast(inputs)
            bench.count(inputs)
            return inputs

    trainer = _BenchSFTTrainer(
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
            "tp_size": strategy.tp_size,
            # What the trainer was actually given, which is spec.micro_batch
            # only when TP is off. dp_world_size is the group ZeRO shards over.
            "device_micro_batch": device_micro_batch,
            "dp_world_size": world_size // strategy.tp_size,
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
        "dataset": {
            "source": spec.dataset,
            "synthetic": is_synthetic,
            "num_samples": len(dataset),
            "packing": args.packing,
            "packing_strategy": spec.packing_strategy,
            "packed_attention_crosses_documents": packed_attn_leak,
        },
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
