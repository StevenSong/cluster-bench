"""Aggregate results/runs/*.json into the report table.

Headline metric is comm_overhead = step_time_config / step_time_ddp - 1, so
every number is stated against a *measured* zero-param-comm ceiling rather
than against spec FLOPS. The DDP baseline is matched per (placement, tokens)
group -- comparing an 8-GPU ZeRO-3 cell to a 2-GPU DDP cell would be
meaningless.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path
from typing import Any, Iterable

# hpZ optimizes only the param all-gather; grad reduce-scatter stays global in
# every config. Gathers are ~14 of ~21 GB/step, so hpZ cannot recover more than
# about two thirds of flat ZeRO-3's overhead. A larger measured win means
# something is wrong -- usually a cell that silently degraded to a different
# config, or a token count that isn't actually matched.
HPZ_MAX_PLAUSIBLE_RECOVERY = 2 / 3

# Strategies that can serve as the zero-param-comm denominator, best first.
# `ddp` is the measured compute ceiling; `zero0` is the same thing routed
# through DeepSpeed so that it keeps fp32 master weights and can be gated --
# if it turns out to cost no param comm. See check_baseline_candidates.
BASELINE_STRATEGIES = ("ddp", "zero0")

# How close zero0 has to sit to ddp to be usable as a like-for-like
# denominator. The fp32 master weights alone cost an Adam step over ~64-80 GB
# of traffic instead of ~32 GB -- order 15 ms on a ~1 s step, so ~1.5%. Three
# percent leaves room for that plus run-to-run noise, and nothing like enough
# room for a per-step all-gather of every parameter.
BASELINE_LIKE_FOR_LIKE_TOL = 0.03

# How far median grad norms may spread across cells in the same comparison
# group before it stops being run-to-run variation. Cells differing only in how
# parameters are sharded see the same gradients; a reduction that sums where
# the trainer already took a mean puts a factor of the world size between them.
GRAD_NORM_SPREAD_TOL = 2.0

# Backends whose optimizer keeps fp32 master weights. DeepSpeed's bf16 path
# holds an fp32 partitioned copy of every parameter and applies the Adam update
# in fp32; accelerate's FSDP2 path does the same. Plain DDP does not:
# modeling.load builds the model in bf16, `mixed_precision: bf16` does not
# upcast an already-bf16 model, so parameters, gradients and Adam moments are
# all bf16 and the update is applied in place.
#
# At lr=6e-6 an update is a fraction of a bf16 ulp for a typical weight, so the
# two paths train at measurably different rates -- DDP reaches a higher loss at
# the same step. That is a property of the numerics, not of the sharding the
# cell exists to measure, so cells on different sides of this line cannot be
# gated against each other. See check_loss_curves.
#
# Measured on-cluster at 4B/8 GPUs rather than assumed, because the two
# accelerate backends do *not* agree and the plausible reading was wrong:
#   * peak memory: ddp 38.6 GB (pure bf16, unsharded) vs fsdp-full 12.0 and
#     zero3 13.2 -- a pure-bf16 FSDP2 shard would have been ~8-9.
#   * grad norms: DDP's land exactly on the bf16 grid, the rest do not.
#   * loss curves: fsdp-full sits 0.006-0.015 from every DeepSpeed cell over
#     100 steps and 0.288 from DDP.
_FP32_MASTER_BACKENDS = {"deepspeed", "fsdp"}


def _precision_class(r: dict[str, Any]) -> str:
    backend = r.get("strategy", {}).get("backend", "unknown")
    return "fp32-master" if backend in _FP32_MASTER_BACKENDS else "bf16-in-place"


def _is_bf16_exact(x: float) -> bool:
    """True if x lands exactly on the bfloat16 grid (low 16 bits of fp32 zero).

    A bf16 value is exact in fp32 and in a JSON double, so the round-trip is
    lossless and this is a decision rather than a tolerance.
    """
    return struct.unpack("<I", struct.pack("<f", x))[0] & 0xFFFF == 0


def load(runs_dir: Path) -> list[dict[str, Any]]:
    out = []
    for p in sorted(runs_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            print(f"  (skipping unparseable {p})")
    return out


def _group_key(r: dict[str, Any]) -> tuple:
    """What makes two cells comparable to each other.

    `tag` is part of it, and has to be: a tagged re-run is a *different run
    condition*, not a second sample of the same one. The profiling matrix
    (tag=profile) reruns ddp and zero3 at three placements with
    wall_clock_breakdown on, which instruments the DeepSpeed cells and not the
    DDP ones -- so mixing the two sets means a profiled DDP cell can become the
    denominator for an unprofiled ZeRO-3 one, and the overhead column silently
    compares across run conditions. Without this the baseline is also just
    whichever same-named run sorted last, which is not a decision anyone made.
    """
    return (
        r["placement"]["name"],
        r["tokens_per_gpu_step"],
        r["spec"]["seq_len"],
        r["spec"].get("tag", ""),
    )


def _data_key(r: dict[str, Any]) -> tuple:
    """What the sampler walked. Two cells share a loss curve only if this matches.

    `dataset_num_samples: 0` means "size the corpus from warmup + measure
    steps" (data.py::_n_examples_needed), so a matrix with a different step
    budget gets a different `len(dataset)` -- and HF's seeded RandomSampler
    permutes over the dataset length, so the batch order changes even though
    the text is byte-identical head-of-split in both. The gate matrix's 0+60
    and 2A's 20+80 therefore walk different orders from step 1, which reads as
    a uniform ~0.17 loss deviation across every strategy at once.
    """
    d = r.get("dataset", {})
    return (
        d.get("source"),
        d.get("synthetic"),
        d.get("num_samples"),
        d.get("packing"),
        d.get("packing_strategy"),
    )


def _p50_by_group(runs: list[dict[str, Any]]) -> dict[tuple, dict[str, float]]:
    """{(placement, tokens, seq_len): {strategy_name: p50}}."""
    out: dict[tuple, dict[str, float]] = {}
    for r in runs:
        if "step_time_p50_s" in r:
            out.setdefault(_group_key(r), {})[r["strategy"]["name"]] = r[
                "step_time_p50_s"
            ]
    return out


def annotate(
    runs: list[dict[str, Any]], baseline: str = "ddp"
) -> list[dict[str, Any]]:
    """Attach each run's denominator, per (placement, tokens, seq_len) group.

    The requested baseline is used where the group has one, and the group falls
    back through BASELINE_STRATEGIES otherwise -- 2B and 2C do not necessarily
    run every baseline candidate at every placement, and half a table of dashes
    is worse than a table that says which denominator each row used. Which one
    it was is recorded per run and printed, because an overhead number against
    an unstated denominator is not a number.
    """
    order = [baseline] + [s for s in BASELINE_STRATEGIES if s != baseline]
    by_group = _p50_by_group(runs)
    for r in runs:
        group = by_group.get(_group_key(r), {})
        name = next((s for s in order if s in group), None)
        r["_baseline_name"] = name
        base = group.get(name) if name else None
        r["_baseline_p50"] = base
        r["_comm_overhead"] = (
            r["step_time_p50_s"] / base - 1 if base and "step_time_p50_s" in r else None
        )
    return runs


def check_baseline_candidates(runs: list[dict[str, Any]]) -> list[str]:
    """Decide, from the measurement, whether zero0 is a usable denominator.

    The headline metric divides by a DDP cell that runs different optimizer
    arithmetic from every cell it is the denominator for, and that no reference
    loss curve can be gated against. `zero0` -- DeepSpeed with ZeRO off -- is
    the candidate fix: fp32 master weights, so it is in the same numerical
    family, with no sharding, so it should cost no parameter communication.

    "Should" is the whole problem. With `bf16.enabled` and ZeRO off, DeepSpeed
    builds a BF16_Optimizer, which partitions the fp32 master weights over the
    DP group and all-gathers the updated bf16 parameters every step. That is
    ZeRO-1's wire profile under a stage-0 label, and nothing in the config says
    so. Assuming it away would put an all-gather inside the denominator and
    deflate every overhead number in the study -- the same class of error as
    the `fsdp-hybrid` row that measured FULL_SHARD twice.

    So this reads the answer off the runs instead: zero0 next to ddp means the
    denominator can be repaired, zero0 next to zero1 means it cannot, and the
    row is a measurement of DeepSpeed's fixed per-step cost instead.
    """
    notes = []
    for key, group in sorted(_p50_by_group(runs).items()):
        ddp, zero0 = group.get("ddp"), group.get("zero0")
        if not ddp or not zero0:
            continue
        line = f"{key[0]}/t{key[1]}: zero0 {zero0 / ddp - 1:+.1%} vs ddp"
        zero1 = group.get("zero1")
        if zero1:
            line += f", {zero0 / zero1 - 1:+.1%} vs zero1"

        if zero0 / ddp - 1 <= BASELINE_LIKE_FOR_LIKE_TOL:
            line += (
                " -- stage 0 costs no param comm here, so zero0 is a valid "
                "like-for-like denominator *if it also passes the correctness "
                "gate*: a cell that trains differently is not a denominator, "
                "whatever its step time. Gate it first, then re-run with "
                "--baseline zero0."
            )
        elif zero1 and zero0 / zero1 - 1 > BASELINE_LIKE_FOR_LIKE_TOL:
            line += (
                " -- slower than zero1, which shards strictly more. The volume "
                "is not the difference: check peak memory, and if stage 0 is "
                "holding full unpartitioned fp32 master weights and Adam state "
                "(~16 bytes/param on every rank) then nothing is being "
                "gathered and its gradient all-reduce is DDP's 2P exactly. "
                "What stage 0 lacks is overlap -- DeepSpeed reduces at the end "
                "of backward, where torch DDP fires bucketed all-reduces from "
                "gradient hooks during it, and DeepSpeed's own stages 1-3 "
                "overlap via their reduction hooks. Same bytes as DDP, none of "
                "them hidden. Not a zero-param-comm ceiling in wall-clock "
                "terms; keep --baseline ddp."
            )
        elif zero1 and abs(zero0 - zero1) < abs(zero0 - ddp):
            line += (
                " -- level with zero1, not ddp. DeepSpeed's BF16_Optimizer is "
                "partitioning the fp32 master weights and all-gathering the "
                "updated params each step, so stage 0 is ZeRO-1-shaped and "
                "cannot be the zero-param-comm ceiling. Keep --baseline ddp. "
                "What this row does measure is DeepSpeed's fixed per-step "
                "runtime floor."
            )
        else:
            line += (
                " -- between ddp and zero1 (or zero1 not in this group). Not "
                "safe as a denominator on this evidence; run zero1 in the same "
                "group, or profile the cell, before switching --baseline."
            )
        notes.append(line)
    return notes


def check_hpz_plausibility(runs: list[dict[str, Any]]) -> list[str]:
    """The interpretive guard, applied automatically."""
    warnings = []
    flat = {
        _group_key(r): r
        for r in runs
        if r["strategy"]["name"] == "zero3" and r["strategy"]["hpz"] == 0
    }
    for r in runs:
        if not r["strategy"]["hpz"]:
            continue
        ref = flat.get(_group_key(r))
        if not ref or "step_time_p50_s" not in ref:
            continue
        base = r.get("_baseline_p50")
        if not base:
            continue
        flat_overhead = ref["step_time_p50_s"] - base
        recovered = ref["step_time_p50_s"] - r["step_time_p50_s"]
        if flat_overhead <= 0:
            continue
        frac = recovered / flat_overhead
        if frac > HPZ_MAX_PLAUSIBLE_RECOVERY + 0.05:
            warnings.append(
                f"{r['run_id']}: recovers {frac:.0%} of flat ZeRO-3's overhead, "
                f"above the ~{HPZ_MAX_PLAUSIBLE_RECOVERY:.0%} ceiling implied by "
                "gathers being ~2/3 of ZeRO-3 comm. Verify the cell actually ran "
                "the config it claims (grad reduce-scatter stays global under hpZ)."
            )
    return warnings


def check_token_control(runs: list[dict[str, Any]]) -> list[str]:
    """tokens/GPU/step is the control every comparison in this study rests on.

    On real text it is a measured quantity, not a flag: `wrapped` packing is
    supposed to deliver chunks of exactly seq_len, and if it does not, cells
    are being compared at different token counts and every overhead number is
    contaminated. Cheap to check, so it is checked.
    """
    warnings = []
    for r in runs:
        if r.get("tokens_per_step_control_held") is False:
            warnings.append(
                f"{r['run_id']}: measured "
                f"{r.get('tokens_per_gpu_step_measured', 0):.0f} tokens/GPU/step "
                f"against a nominal {r['tokens_per_gpu_step']} "
                f"({r.get('tokens_per_step_drift', 0):.1%} drift). Packing did not "
                "produce exact chunks -- this cell is not comparable to the rest."
            )

    measured: dict[tuple, list[float]] = {_group_key(r): [] for r in runs}
    for r in runs:
        if r.get("tokens_per_gpu_step_measured") is not None:
            measured[_group_key(r)].append(r["tokens_per_gpu_step_measured"])
    for key, vals in measured.items():
        if len(vals) > 1 and min(vals) and max(vals) / min(vals) > 1.01:
            warnings.append(
                f"{key[0]}/t{key[1]}: cells in this comparison group saw different "
                f"token counts ({min(vals):.0f}..{max(vals):.0f}); their step times "
                "are not directly comparable."
            )
    return warnings


def check_loss_curves(
    runs: list[dict[str, Any]], reference: str, tol: float
) -> list[str]:
    """Correctness gate: catches packing-mask and loss-normalization bugs that
    a fast-but-wrong config would otherwise hide behind a good step time."""
    ref = next((r for r in runs if r["run_id"] == reference), None)
    if ref is None:
        return [f"reference run {reference!r} not found; correctness gate not applied"]
    ref_curve = {p["step"]: p["loss"] for p in ref.get("loss_curve", [])}
    if not ref_curve:
        return [f"reference run {reference!r} has no loss curve"]

    ref_tp = ref["strategy"].get("tp_size", 1)
    ref_precision = _precision_class(ref)

    problems = []
    gated = 0
    for r in runs:
        if r["run_id"] == reference:
            continue
        # A 2-GPU placement or a different tokens/GPU/step is a different global
        # batch, so the curve is legitimately different and nothing about the
        # difference is a correctness signal. Silent here rather than a note:
        # pointed at results/runs this is most of the matrix, and thirty lines
        # saying "2B and 2C exist" would bury the cells that were checked.
        if _group_key(r) != _group_key(ref):
            continue
        # Reported, not silent: unlike a placement or token-count difference,
        # this one looks like every cell failing the gate by the same amount,
        # and the cells are otherwise in the same comparison group.
        if _data_key(r) != _data_key(ref):
            problems.append(
                f"{r['run_id']}: not gated -- walked a different corpus "
                f"({r.get('dataset', {}).get('num_samples')} samples vs the "
                f"reference's {ref.get('dataset', {}).get('num_samples')}). "
                "dataset_num_samples=0 sizes the corpus from warmup+measure "
                "steps, so a different step budget reshuffles the batch order. "
                "Gate against a reference run with the same step budget."
            )
            continue
        # A TP cell cannot be gated against a non-TP reference, and not because
        # anything is wrong with it: its TP group is fed one batch, so the
        # sampler walks the corpus in a different order, and HF's
        # num_items_in_batch counts the duplicated batch, which rescales the
        # logged loss. Both shift the curve without saying anything about
        # correctness. Gate a TP cell against another TP cell.
        if r["strategy"].get("tp_size", 1) != ref_tp:
            problems.append(
                f"{r['run_id']}: not gated -- tp_size "
                f"{r['strategy'].get('tp_size', 1)} vs reference's {ref_tp}. "
                "Sample order and loss normalization both differ under TP; "
                "use a TP run as --reference to gate this row."
            )
            continue
        # Same reasoning, different mechanism: an fp32-master cell and a
        # bf16-in-place cell are running different optimizer arithmetic, so
        # their curves separate within a couple of steps no matter how correct
        # both are. Gating one against the other reports a numerics difference
        # as a sharding bug. Use a reference from the same family.
        if _precision_class(r) != ref_precision:
            problems.append(
                f"{r['run_id']}: not gated -- {_precision_class(r)} optimizer "
                f"(backend {r['strategy']['backend']}) vs reference's "
                f"{ref_precision} (backend {ref['strategy']['backend']}). The "
                "two apply the update in different precision, so the curves "
                "separate on numerics alone. Gate this cell against a reference "
                "from the same family."
            )
            continue
        curve = {p["step"]: p["loss"] for p in r.get("loss_curve", [])}
        shared = sorted(set(curve) & set(ref_curve))
        if not shared:
            problems.append(f"{r['run_id']}: no overlapping logged steps with reference")
            continue
        gated += 1
        diffs = [curve[s] - ref_curve[s] for s in shared]
        worst = max(abs(d) for d in diffs)
        if worst > tol:
            # Signed, not just the magnitude. A curve that sits uniformly above
            # the reference is training more slowly -- an effective learning
            # rate or gradient-scaling difference -- while one that straddles it
            # is walking different data or diverging. The two want completely
            # different investigations, and |max| cannot tell them apart.
            mean = sum(diffs) / len(diffs)
            side = (
                "consistently above"
                if min(diffs) > 0
                else "consistently below"
                if max(diffs) < 0
                else "straddling"
            )
            problems.append(
                f"{r['run_id']}: max loss deviation {worst:.4f} over "
                f"{len(shared)} steps exceeds tol {tol} "
                f"(mean {mean:+.4f}, {side} the reference)"
            )

    # The failure this whole gate exists to prevent is a gate that looks like it
    # ran. Say how many cells it actually compared, so "no problems" cannot mean
    # "nothing was checked".
    in_group = sum(
        1
        for r in runs
        if r["run_id"] != reference and _group_key(r) == _group_key(ref)
    )
    problems.append(
        f"gated {gated} of {in_group} cells in {reference}'s "
        f"{_group_key(ref)[0]}/t{_group_key(ref)[1]} group "
        f"({ref_precision}, tp={ref_tp}) at tol {tol}"
    )
    return problems


def check_precision_class_evidence(runs: list[dict[str, Any]]) -> list[str]:
    """Cross-check the declared optimizer precision against the recorded norms.

    _precision_class reads the backend, which is a claim about how the run was
    configured. The grad norms are evidence: a norm computed over bf16 gradients
    lands exactly on the bf16 grid, one computed in fp32 essentially never does.
    If the two disagree, the backend map is stale -- a precision flag changed, or
    a new backend was added -- and the gate is silently mis-grouping cells.

    Only the unambiguous direction is flagged. An fp32-master run whose every
    norm is bf16-exact cannot be a coincidence, but a bf16-in-place run can
    legitimately report a non-exact norm -- a mixed-precision policy may reduce
    or accumulate the norm in fp32 while the parameters themselves stay bf16,
    which is a real configuration rather than a contradiction.
    """
    notes = []
    for r in runs:
        norms = [
            p["grad_norm"]
            for p in r.get("loss_curve", [])
            if isinstance(p.get("grad_norm"), (int, float)) and p["grad_norm"]
        ]
        if len(norms) < 5:
            continue
        if _precision_class(r) == "fp32-master" and all(
            _is_bf16_exact(v) for v in norms
        ):
            notes.append(
                f"{r['run_id']}: declared fp32-master (backend "
                f"{r['strategy']['backend']}) but all {len(norms)} recorded grad "
                "norms are exactly bf16-representable, which means the optimizer "
                "is not keeping fp32 master weights. report._FP32_MASTER_BACKENDS "
                "no longer describes this run, and the correctness gate is "
                "grouping cells by the wrong families."
            )
    return notes


def check_grad_norm_scale(runs: list[dict[str, Any]]) -> list[str]:
    """Are two cells in the same group seeing gradients of the same size?

    The gate says *that* a loss curve separated; this says whether the
    gradients did, which is the difference between "the update is applied
    differently" and "the gradient reaching the optimizer is a different
    number". The usual cause of the second is a reduction that averages where
    the trainer already averaged, or sums where it expected a mean -- so the
    tell is a ratio near the world size (or its reciprocal), not a small drift.

    Compared within (group, precision class, tp_size): bf16-in-place and
    fp32-master cells compute the norm differently, and a TP group's norm is
    taken over a duplicated batch. Medians, so a couple of early spikes during
    warmup do not decide it.
    """
    buckets: dict[tuple, dict[str, float]] = {}
    for r in runs:
        norms = sorted(
            p["grad_norm"]
            for p in r.get("loss_curve", [])
            if isinstance(p.get("grad_norm"), (int, float)) and p["grad_norm"] > 0
        )
        if len(norms) < 5:
            continue
        key = (_group_key(r), _precision_class(r), r["strategy"].get("tp_size", 1))
        buckets.setdefault(key, {})[r["strategy"]["name"]] = norms[len(norms) // 2]

    notes = []
    for (group, precision, _tp), cells in sorted(buckets.items()):
        if len(cells) < 2:
            continue
        lo_name, lo = min(cells.items(), key=lambda kv: kv[1])
        hi_name, hi = max(cells.items(), key=lambda kv: kv[1])
        if hi / lo <= GRAD_NORM_SPREAD_TOL:
            continue
        world = next(
            (r["placement"]["world_size"] for r in runs if _group_key(r) == group),
            None,
        )
        line = (
            f"{group[0]}/t{group[1]} ({precision}): median grad norm spans "
            f"{hi / lo:.1f}x across cells that should agree -- "
            f"{hi_name} {hi:.3g} vs {lo_name} {lo:.3g}"
        )
        if world and abs(hi / lo - world) / world < 0.2:
            line += (
                f". That ratio is the world size ({world}), which is what a "
                "gradient reduction that sums where the trainer already took a "
                "mean looks like -- not a numerics difference. Any loss-curve "
                "deviation in this group is downstream of it, so fix this "
                "first and re-gate."
            )
        else:
            line += (
                ". Gradients of different sizes reach the optimizer in these "
                "cells, so a loss-curve difference between them is not "
                "evidence about sharding."
            )
        notes.append(line)
    return notes


def _fmt(v: Any, spec: str, dash: str = "-") -> str:
    return dash if v is None else format(v, spec)


def table(runs: Iterable[dict[str, Any]]) -> str:
    rows = sorted(
        runs,
        key=lambda r: (
            r["placement"]["name"],
            r["tokens_per_gpu_step"],
            r["spec"].get("tag", ""),
            r.get("_comm_overhead") if r.get("_comm_overhead") is not None else -1,
        ),
    )
    head = (
        f"{'strategy':<14} {'placement':<14} {'tag':<8} {'tok/gpu':>8} "
        f"{'p50 s':>8} {'p95 s':>8} {'p95/p50':>8} {'tok/s/gpu':>10} "
        f"{'peak GB':>8} {'overhead':>9} {'vs':>7}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['strategy']['name']:<14} {r['placement']['name']:<14} "
            # Without this two rows of the same (strategy, placement) are
            # indistinguishable, and a tagged re-run reads as a duplicate.
            f"{r['spec'].get('tag', '') or '-':<8} "
            f"{r['tokens_per_gpu_step']:>8} "
            f"{_fmt(r.get('step_time_p50_s'), '.4f'):>8} "
            f"{_fmt(r.get('step_time_p95_s'), '.4f'):>8} "
            f"{_fmt(r.get('straggler_ratio'), '.2f'):>8} "
            f"{_fmt(r.get('tokens_per_s_per_gpu'), '.0f'):>10} "
            f"{_fmt(r.get('peak_mem_allocated_gb'), '.1f'):>8} "
            f"{_fmt(r.get('_comm_overhead'), '+.1%'):>9} "
            # Which denominator this row's overhead is against. A group without
            # the requested baseline falls back rather than blanking, so this
            # column is the only thing that says a row changed reference.
            f"{r.get('_baseline_name') or '-':>7}"
        )
    return "\n".join(lines)


def provenance_spread(runs: list[dict[str, Any]]) -> list[str]:
    """A matrix compared across two NCCL builds is not a matrix."""
    keys = ("nccl", "driver", "torch")
    notes = []
    for k in keys:
        seen = {str(r.get("provenance", {}).get(k)) for r in runs}
        if len(seen) > 1:
            notes.append(f"mixed {k} across runs: {sorted(seen)}")

    # Everything else is pinned in requirements.txt; flash-attn is installed out
    # of band per node, so it is the one that can drift. Two cells on different
    # attention kernels are not comparable.
    fa = {
        str(r.get("provenance", {}).get("packages", {}).get("flash-attn"))
        for r in runs
    }
    if len(fa) > 1:
        notes.append(f"mixed flash-attn across runs: {sorted(fa)}")
    shas = {
        r.get("provenance", {}).get("git", {}).get("sha") for r in runs
    }
    if len(shas) > 1:
        notes.append(f"mixed git SHAs across runs: {sorted(str(s)[:8] for s in shas)}")
    if any(r.get("provenance", {}).get("git", {}).get("dirty") for r in runs):
        notes.append("at least one run was produced from a dirty working tree")
    return notes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, default=Path("results/runs"))
    ap.add_argument("--reference", default=None,
                    help="run_id whose loss curve is the correctness gate. Only "
                         "cells with the same optimizer precision and tp_size "
                         "are gated against it; the rest are reported as not "
                         "gated, so a full sweep needs one pass per family")
    ap.add_argument("--loss-tol", type=float, default=0.02)
    ap.add_argument("--baseline", default="ddp", choices=BASELINE_STRATEGIES,
                    help="strategy whose step time is the denominator of "
                         "comm_overhead. Groups without it fall back to the "
                         "next candidate; the `vs` column says which each row "
                         "used. Only switch to zero0 once "
                         "check_baseline_candidates says it costs no param comm")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    runs = load(args.runs_dir)
    if not runs:
        raise SystemExit(f"no runs found in {args.runs_dir}")
    runs = annotate(runs, args.baseline)

    print(table(runs))

    for label, notes in (
        ("provenance", provenance_spread(runs)),
        ("token control", check_token_control(runs)),
        ("baseline", check_baseline_candidates(runs)),
        ("plausibility", check_hpz_plausibility(runs)),
        ("optimizer precision", check_precision_class_evidence(runs)),
        ("grad norm scale", check_grad_norm_scale(runs)),
        (
            "correctness gate",
            check_loss_curves(runs, args.reference, args.loss_tol)
            if args.reference
            else ["not run (pass --reference <run_id>)"],
        ),
    ):
        if notes:
            print(f"\n{label}:")
            for n in notes:
                print(f"  ! {n}")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                [
                    {
                        "run_id": r["run_id"],
                        "strategy": r["strategy"]["name"],
                        "placement": r["placement"]["name"],
                        "links": r["placement"]["links"],
                        "tokens_per_gpu_step": r["tokens_per_gpu_step"],
                        "tokens_per_gpu_step_measured": r.get(
                            "tokens_per_gpu_step_measured"
                        ),
                        "step_time_p50_s": r.get("step_time_p50_s"),
                        "step_time_p95_s": r.get("step_time_p95_s"),
                        "tokens_per_s_per_gpu": r.get("tokens_per_s_per_gpu"),
                        "peak_mem_allocated_gb": r.get("peak_mem_allocated_gb"),
                        "comm_overhead": r.get("_comm_overhead"),
                    }
                    for r in runs
                ],
                indent=2,
            )
        )
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
