"""Sharding strategies — Matrix 2A.

Each strategy is a pure description of *how parameters, gradients and optimizer
state are distributed*. Turning one into a launcher config is the job of
`ds_config.py` / `accel_config.py`; nothing here touches the filesystem.

The `param_gather` / `grad_reduce` fields are not used by the runtime — they are
carried into the results JSON so the report table can be generated without
re-deriving what each config actually does over the wire.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Backend = Literal["ddp", "deepspeed", "fsdp"]
Scope = Literal["none", "global", "node", "pair"]


@dataclass(frozen=True)
class Strategy:
    name: str
    backend: Backend
    purpose: str

    # --- what crosses the wire, for the report table ---
    param_gather: Scope = "none"
    grad_reduce: Scope = "global"

    # --- deepspeed ---
    zero_stage: int = 0
    # zero_hpz_partition_size: 0 = off (flat), 4 = node-local, 2 = pair-local.
    hpz: int = 0
    autotp: int = 0

    # --- fsdp2 ---
    # reshard_after_forward=True is FULL_SHARD semantics; hybrid additionally
    # confines the parameter shard group to one node.
    fsdp_reshard_after_forward: bool = True
    fsdp_hybrid: bool = False

    # Strategies that cannot run at every GPU count (e.g. hpZ=4 is meaningless
    # on a 2-GPU placement). Checked by placement.validate().
    min_world_size: int = 1
    world_size_multiple_of: int = 1

    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def uses_deepspeed(self) -> bool:
        return self.backend == "deepspeed"


# Matrix 2A. Rows 1->2->3->4 decompose ZeRO's cost: optimizer, then grads, then
# params. Rows 5/6 vary *only* the param-gather scope against row 4.
STRATEGIES: dict[str, Strategy] = {
    "ddp": Strategy(
        name="ddp",
        backend="ddp",
        purpose="compute ceiling / denominator",
        param_gather="none",
        grad_reduce="global",
        notes=(
            "Fits at 4B (8+8+48 = 64 GB/GPU) but not at 31B. This is the "
            "zero-param-comm baseline every overhead number is reported against.",
        ),
    ),
    "zero1": Strategy(
        name="zero1",
        backend="deepspeed",
        zero_stage=1,
        purpose="optimizer sharding only",
        param_gather="none",
        grad_reduce="global",
    ),
    "zero2": Strategy(
        name="zero2",
        backend="deepspeed",
        zero_stage=2,
        purpose="+ gradient sharding",
        param_gather="none",
        grad_reduce="global",
    ),
    "zero3": Strategy(
        name="zero3",
        backend="deepspeed",
        zero_stage=3,
        hpz=0,
        purpose="current baseline (flat ZeRO-3)",
        param_gather="global",
        grad_reduce="global",
    ),
    "zero3-hpz4": Strategy(
        name="zero3-hpz4",
        backend="deepspeed",
        zero_stage=3,
        hpz=4,
        purpose="node-local param gather, no cross-node gathers",
        param_gather="node",
        grad_reduce="global",
        min_world_size=8,
        world_size_multiple_of=4,
        notes=("hpZ=4 requires >=2 gather groups to differ from flat ZeRO-3.",),
    ),
    "zero3-hpz2": Strategy(
        name="zero3-hpz2",
        backend="deepspeed",
        zero_stage=3,
        hpz=2,
        purpose="pair-local (NVLink) param gather, no PCIe/RoCE gathers",
        param_gather="pair",
        grad_reduce="global",
        min_world_size=4,
        world_size_multiple_of=2,
        notes=(
            "Holds a second, node-local copy of every parameter shard: +4 GB at "
            "4B, +31 GB at 31B. The 4B proxy cannot price this tradeoff.",
        ),
    ),
    "fsdp-full": Strategy(
        name="fsdp-full",
        backend="fsdp",
        fsdp_reshard_after_forward=True,
        fsdp_hybrid=False,
        purpose="cross-check vs zero3",
        param_gather="global",
        grad_reduce="global",
    ),
    "fsdp-hybrid": Strategy(
        name="fsdp-hybrid",
        backend="fsdp",
        fsdp_reshard_after_forward=True,
        fsdp_hybrid=True,
        purpose="cross-check vs zero3-hpz4",
        param_gather="node",
        grad_reduce="global",
        min_world_size=8,
        world_size_multiple_of=4,
    ),
    "tp2-zero3": Strategy(
        name="tp2-zero3",
        backend="deepspeed",
        zero_stage=3,
        autotp=2,
        purpose="only sane TP degree on a 2-GPU NVLink domain",
        param_gather="pair",
        grad_reduce="global",
        min_world_size=4,
        world_size_multiple_of=2,
        notes=(
            "TP all-reduces are synchronous and unhideable, unlike ZeRO's "
            "overlappable gathers. Expect a different shape of overhead curve.",
        ),
    ),
}


def get(name: str) -> Strategy:
    try:
        return STRATEGIES[name]
    except KeyError:
        raise SystemExit(
            f"unknown strategy {name!r}; choose from: {', '.join(STRATEGIES)}"
        ) from None
