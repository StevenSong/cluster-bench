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
from pathlib import Path
from typing import Any, Iterable

# hpZ optimizes only the param all-gather; grad reduce-scatter stays global in
# every config. Gathers are ~14 of ~21 GB/step, so hpZ cannot recover more than
# about two thirds of flat ZeRO-3's overhead. A larger measured win means
# something is wrong -- usually a cell that silently degraded to a different
# config, or a token count that isn't actually matched.
HPZ_MAX_PLAUSIBLE_RECOVERY = 2 / 3


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

    problems = []
    for r in runs:
        if r["run_id"] == reference:
            continue
        curve = {p["step"]: p["loss"] for p in r.get("loss_curve", [])}
        shared = sorted(set(curve) & set(ref_curve))
        if not shared:
            problems.append(f"{r['run_id']}: no overlapping logged steps with reference")
            continue
        worst = max(abs(curve[s] - ref_curve[s]) for s in shared)
        if worst > tol:
            problems.append(
                f"{r['run_id']}: max loss deviation {worst:.4f} over {len(shared)} "
                f"steps exceeds tol {tol}"
            )
    return problems


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
                    help="run_id whose loss curve is the correctness gate")
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
