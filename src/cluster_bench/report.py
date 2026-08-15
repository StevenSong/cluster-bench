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
    return (r["placement"]["name"], r["tokens_per_gpu_step"], r["spec"]["seq_len"])


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


def annotate(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {
        _group_key(r): r["step_time_p50_s"]
        for r in runs
        if r["strategy"]["name"] == "ddp" and "step_time_p50_s" in r
    }
    for r in runs:
        base = baselines.get(_group_key(r))
        r["_baseline_p50"] = base
        r["_comm_overhead"] = (
            r["step_time_p50_s"] / base - 1 if base and "step_time_p50_s" in r else None
        )
    return runs


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

    measured = {
        (r["placement"]["name"], r["tokens_per_gpu_step"], r["spec"]["seq_len"]): []
        for r in runs
    }
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
        worst = max(abs(curve[s] - ref_curve[s]) for s in shared)
        if worst > tol:
            problems.append(
                f"{r['run_id']}: max loss deviation {worst:.4f} over {len(shared)} "
                f"steps exceeds tol {tol}"
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


def _fmt(v: Any, spec: str, dash: str = "-") -> str:
    return dash if v is None else format(v, spec)


def table(runs: Iterable[dict[str, Any]]) -> str:
    rows = sorted(
        runs,
        key=lambda r: (
            r["placement"]["name"],
            r["tokens_per_gpu_step"],
            r.get("_comm_overhead") if r.get("_comm_overhead") is not None else -1,
        ),
    )
    head = (
        f"{'strategy':<14} {'placement':<14} {'tok/gpu':>8} {'p50 s':>8} "
        f"{'p95 s':>8} {'p95/p50':>8} {'tok/s/gpu':>10} {'peak GB':>8} {'overhead':>9}"
    )
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['strategy']['name']:<14} {r['placement']['name']:<14} "
            f"{r['tokens_per_gpu_step']:>8} "
            f"{_fmt(r.get('step_time_p50_s'), '.4f'):>8} "
            f"{_fmt(r.get('step_time_p95_s'), '.4f'):>8} "
            f"{_fmt(r.get('straggler_ratio'), '.2f'):>8} "
            f"{_fmt(r.get('tokens_per_s_per_gpu'), '.0f'):>10} "
            f"{_fmt(r.get('peak_mem_allocated_gb'), '.1f'):>8} "
            f"{_fmt(r.get('_comm_overhead'), '+.1%'):>9}"
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
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    runs = load(args.runs_dir)
    if not runs:
        raise SystemExit(f"no runs found in {args.runs_dir}")
    runs = annotate(runs)

    print(table(runs))

    for label, notes in (
        ("provenance", provenance_spread(runs)),
        ("token control", check_token_control(runs)),
        ("plausibility", check_hpz_plausibility(runs)),
        ("optimizer precision", check_precision_class_evidence(runs)),
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
