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
    # reshard_after_forward=True is FULL_SHARD semantics. There is no hybrid
    # counterpart here any more -- see the note above `fsdp-full` below.
    fsdp_reshard_after_forward: bool = True

    # Strategies that cannot run at every GPU count (e.g. hpZ=4 is meaningless
    # on a 2-GPU placement). Checked by placement.validate().
    min_world_size: int = 1
    world_size_multiple_of: int = 1

    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def uses_deepspeed(self) -> bool:
        return self.backend == "deepspeed"

    @property
    def tp_size(self) -> int:
        """Ranks that are fed the same micro-batch. 1 when TP is off.

        Everything downstream that reasons about *tokens* has to divide by this:
        a TP group holds one copy of the batch between its ranks, so a per-rank
        token tally counts every token tp_size times.
        """
        return self.autotp or 1


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
            "The only bf16-in-place cell in the matrix: it holds bf16 params, "
            "grads and Adam moments and updates in place, while every DeepSpeed "
            "and FSDP2 cell keeps fp32 master weights. So it is both the "
            "denominator and the one cell the correctness gate cannot cover. "
            "`zero0` below exists to test whether that can be fixed.",
        ),
    ),
    "zero0": Strategy(
        name="zero0",
        backend="deepspeed",
        zero_stage=0,
        purpose="like-for-like fp32-master denominator candidate",
        param_gather="none",
        grad_reduce="global",
        notes=(
            "Same wire profile as ddp on paper -- no sharding, bf16 gradient "
            "reduction -- but run through DeepSpeed, so it keeps fp32 master "
            "weights and lands in the same numerical family as every cell it "
            "would be the denominator for. That makes it gate-able, which ddp "
            "is not.",
            "Whether it is actually a zero-param-comm ceiling is the open "
            "question this row exists to answer, and it is not obvious: with "
            "`bf16.enabled` and no ZeRO, deepspeed's engine builds a "
            "BF16_Optimizer, which partitions the fp32 master weights over the "
            "DP group and all-gathers the updated bf16 params at the end of "
            "every step. That is ZeRO-1's comm pattern, not DDP's, and no "
            "config flag disables it.",
            "So read the measurement, do not assume the label. Within a few "
            "percent of ddp: a valid like-for-like denominator, and the gate "
            "hole closes (report --baseline zero0). Level with zero1 instead: "
            "stage 0 is ZeRO-1-shaped, ddp stays the denominator, and this row "
            "is measuring DeepSpeed's fixed per-step runtime floor -- which is "
            "worth having either way, since 2A puts two thirds of flat ZeRO-3's "
            "cost somewhere other than the fabric. "
            "report.check_baseline_candidates decides this automatically.",
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
    # `fsdp-hybrid` (2A row 8, FSDP2 HYBRID_SHARD as the cross-check against
    # zero3-hpz4) was here and was DROPPED on 2026-08-15. It never ran: peak
    # memory came back 12.0 GB, identical to fsdp-full to the decimal, and a
    # node-local parameter replica cannot be free -- the DeepSpeed equivalent
    # costs +1.6 GB at hpz4 and +3.2 GB at hpz2 on the same model. accelerate
    # 1.14 accepted `fsdp_shard_size` under `fsdp_version: 2` and silently did
    # not build the 2-D mesh, so the row was a second measurement of
    # fsdp-full (+17.2% against fsdp-full's +15.4%) wearing a different label.
    #
    # To bring it back: emit the FSDP1 spelling (`fsdp_sharding_strategy:
    # HYBRID_SHARD`) or construct the device mesh directly, and confirm it
    # engaged by peak memory going *up* -- not by the config file containing
    # the key, which is exactly what fooled it the first time.
    #
    # The row is not currently worth much: hpZ measured slower than flat
    # ZeRO-3 in every cell of 2A, 2B and 2C, and fsdp-hybrid's job was to
    # cross-check hpZ. Restore it only if node-local replication starts
    # looking useful again.
    "fsdp-full": Strategy(
        name="fsdp-full",
        backend="fsdp",
        fsdp_reshard_after_forward=True,
        purpose="cross-check vs zero3",
        param_gather="global",
        grad_reduce="global",
    ),
    "tp2-zero2": Strategy(
        name="tp2-zero2",
        backend="deepspeed",
        zero_stage=2,
        autotp=2,
        purpose="only sane TP degree on a 2-GPU NVLink domain",
        param_gather="none",
        grad_reduce="global",
        min_world_size=4,
        world_size_multiple_of=2,
        notes=(
            "Stage 2, not 3: deepspeed 0.19.2 asserts "
            "`zero_optimization_stage() <= 2` when autotp is on "
            "(runtime/engine.py). v0.19.4 lifted it, but taking that upgrade "
            "would change the runtime under every other cell of 2A.",
            "TP keeps parameters permanently sharded inside the pair, so there "
            "is no param all-gather for stage 3 to have optimized -- which is "
            "why the ZeRO-2 substitution costs this row nothing it was going "
            "to measure. Its single-variable baseline is row 3 (plain zero2).",
            "param_gather='none' is honest but incomplete: the comm this row "
            "adds is a per-layer activation all-reduce inside the NVLink pair, "
            "which the Scope fields cannot express. Grad reduce is over the DP "
            "group of world/2, so it still crosses PCIe and RoCE.",
            "TP all-reduces are synchronous and unhideable, unlike ZeRO's "
            "overlappable gathers. Expect a different shape of overhead curve.",
            "Both ranks of a TP group consume the *same* micro-batch, so "
            "train.py doubles the per-device batch to hold 8192 tok/GPU/step "
            "of compute, and metrics divides the token tally by tp_size.",
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
