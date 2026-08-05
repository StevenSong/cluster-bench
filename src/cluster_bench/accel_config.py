"""Build an accelerate launch config for a (strategy, placement) cell.

Replaces the static `accelerate_ds.yaml`, which hardcoded 2 machines / 8
processes / DeepSpeed and so could express exactly one cell of the matrix.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import RunSpec
from .placement import Placement
from .strategies import Strategy


def build(
    spec: RunSpec,
    strategy: Strategy,
    placement: Placement,
    ds_config_path: Path | None = None,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "compute_environment": "LOCAL_MACHINE",
        "num_machines": placement.num_machines,
        "num_processes": placement.world_size,
        "machine_rank": spec.machine_rank,
        "main_process_ip": spec.main_process_ip,
        "main_process_port": spec.main_process_port,
        "rdzv_backend": "c10d",
        "same_network": True,
        "use_cpu": False,
    }

    if strategy.backend == "ddp":
        cfg["distributed_type"] = "MULTI_GPU"
        cfg["mixed_precision"] = "bf16"

    elif strategy.backend == "deepspeed":
        if ds_config_path is None:
            raise ValueError("deepspeed strategies need a ds_config_path")
        cfg["distributed_type"] = "DEEPSPEED"
        # No top-level `mixed_precision` here, deliberately. With a
        # `deepspeed_config_file`, accelerate treats precision set in the
        # accelerate config as a conflicting duplicate: it flags the field in
        # ACCELERATE_CONFIG_DS_FIELDS and DeepSpeedPlugin then refuses to
        # start ("the following accelerate config variables will be
        # ignored"). Spelling it "no" does not help -- any value present
        # trips it; the key has to be absent. bf16 comes from the DeepSpeed
        # config's `bf16.enabled` instead, which is where DeepSpeed reads it
        # from regardless.
        cfg["deepspeed_config"] = {
            "deepspeed_config_file": str(ds_config_path),
            # Stage-3 needs zero.Init at construction or the model is
            # materialized whole on every rank before sharding.
            "zero3_init_flag": strategy.zero_stage == 3,
            "deepspeed_multinode_launcher": "standard",
        }

    elif strategy.backend == "fsdp":
        cfg["distributed_type"] = "FSDP"
        cfg["mixed_precision"] = "bf16"
        fsdp: dict[str, Any] = {
            "fsdp_version": 2,
            "fsdp_reshard_after_forward": strategy.fsdp_reshard_after_forward,
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "fsdp_state_dict_type": "SHARDED_STATE_DICT",
            "fsdp_offload_params": False,
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_activation_checkpointing": False,  # HF trainer owns this
        }
        if strategy.fsdp_hybrid:
            # FSDP2 expresses hybrid sharding as a 2-D device mesh: shard
            # within `fsdp_reshard_after_forward` groups of this size,
            # replicate across them. One group == one node.
            fsdp["fsdp_shard_size"] = placement.procs_per_machine
        cfg["fsdp_config"] = fsdp

    else:
        raise ValueError(f"unhandled backend {strategy.backend!r}")

    return cfg


def write(
    spec: RunSpec,
    strategy: Strategy,
    placement: Placement,
    dest_dir: Path,
    ds_config_path: Path | None = None,
) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / f"accelerate_{strategy.name}_{placement.name}.yaml"
    cfg = build(spec, strategy, placement, ds_config_path)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path
