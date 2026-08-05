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
        # DeepSpeed AutoTP for training. TP shards inside the NVLink domain;
        # ZeRO-3 then shards across the remaining (cross-pair) dimension.
        cfg["tensor_parallel"] = {"autotp_size": strategy.autotp}

    return cfg


def write(spec: RunSpec, strategy: Strategy, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"ds_{strategy.name}.json"
    path.write_text(json.dumps(build(spec, strategy), indent=2) + "\n")
    return path
