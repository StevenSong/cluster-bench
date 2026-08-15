"""Build a DeepSpeed config dict for a given strategy.

Replaces the single static `ds_zero3.json`, whose `"auto"` bucket sizes made
4B-vs-31B and even cell-vs-cell comparisons invalid. Everything that affects
message size is pinned; only the fields the strategy is actually varying
differ between cells.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import RunSpec
from .strategies import Strategy


def build(spec: RunSpec, strategy: Strategy) -> dict[str, Any]:
    if not strategy.uses_deepspeed:
        raise ValueError(f"{strategy.name} is not a DeepSpeed strategy")

    # stage 0 is a real cell here (`zero0`), not a degenerate one: it is the
    # candidate like-for-like denominator. DeepSpeed reads `stage: 0` as ZeRO
    # off and ignores the rest of this block, but the block is emitted anyway so
    # every cell's config differs only in the fields the strategy is varying.
    zero: dict[str, Any] = {
        "stage": strategy.zero_stage,
        "overlap_comm": spec.overlap_comm,
        "contiguous_gradients": True,
        # Pinned rather than "auto" -- see config.buckets_for_hidden.
        **spec.buckets,
    }

    if strategy.zero_stage >= 2:
        # Staggers which rank a gradient bucket reduces to, so a single rail
        # isn't the target for every bucket at once.
        zero["round_robin_gradients"] = True

    if strategy.zero_stage == 3:
        zero.update(
            {
                "stage3_max_live_parameters": spec.stage3_max_live_parameters,
                "stage3_max_reuse_distance": spec.stage3_max_reuse_distance,
                "stage3_gather_16bit_weights_on_model_save": True,
                "zero_quantized_weights": spec.zero_quantized_weights,
                "zero_quantized_gradients": spec.zero_quantized_gradients,
                "zero_hpz_partition_size": strategy.hpz,
            }
        )
    else:
        # stage3_* and hpZ are stage-3 concepts; leaving them set at stage 1/2
        # is at best ignored and at worst a config error on some versions.
        if strategy.hpz:
            raise ValueError("zero_hpz_partition_size requires stage 3")

    cfg: dict[str, Any] = {
        "bf16": {"enabled": True},
        "zero_optimization": zero,
        "gradient_clipping": "auto",
        "gradient_accumulation_steps": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "steps_per_print": 10**9,  # stdout noise skews nothing but reads badly
        "wall_clock_breakdown": spec.wall_clock_breakdown,
    }

    if strategy.autotp:
        if strategy.zero_stage >= 3:
            # deepspeed 0.19.2, runtime/engine.py::_configure_tensor_parallel_states:
            #     assert self.zero_optimization_stage() <= 2, "Currently, the
            #     compatibility between 'autotp' and 'zero_stage = 3' has not
            #     been validated"
            # It fires at engine init on every rank, so the cell dies ~2 minutes
            # in with a stack trace instead of a config error. Caught here, on
            # node0, before anything is launched. v0.19.4 removed the assert;
            # taking that upgrade means re-running all of 2A on the new runtime.
            raise ValueError(
                f"{strategy.name}: autotp is incompatible with zero stage "
                f"{strategy.zero_stage} on deepspeed 0.19.2 (max stage 2). "
                "Use stage <= 2, or upgrade to >= 0.19.4 and re-run the whole "
                "matrix so every cell shares a runtime."
            )
        # DeepSpeed AutoTP for training. TP shards each layer inside the NVLink
        # pair; ZeRO-2 then shards grads + optimizer state across the remaining
        # (cross-pair) dimension, a DP group of world_size / autotp.
        cfg["tensor_parallel"] = {"autotp_size": strategy.autotp}

    return cfg


def write(spec: RunSpec, strategy: Strategy, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"ds_{strategy.name}.json"
    path.write_text(json.dumps(build(spec, strategy), indent=2) + "\n")
    return path
