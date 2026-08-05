"""Matrix driver: expand a matrix file into cells and run them.

Launch model is SSH-from-node0. For each cell the driver writes the generated
DeepSpeed + accelerate configs, starts the peer rank(s) over SSH, then runs
machine_rank 0 locally and waits. One command runs a whole matrix unattended.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from . import accel_config, config, ds_config
from . import placement as placement_mod
from . import strategies as strategies_mod


@dataclass
class Cell:
    spec: config.RunSpec
    strategy: strategies_mod.Strategy
    place: placement_mod.Placement

    @property
    def run_id(self) -> str:
        return self.spec.run_id


# ---------------------------------------------------------------------------
# Matrix expansion
# ---------------------------------------------------------------------------
def expand(matrix: dict[str, Any], base: config.RunSpec) -> tuple[list[Cell], list[str]]:
    """Cartesian product over the axes in a matrix file.

    Axes are `strategies`, `placements`, and `tokens_per_gpu`. Anything under
    `defaults` overrides the base spec for every cell in this matrix.
    """
    overrides = dict(matrix.get("defaults") or {})
    base = replace(base, **overrides) if overrides else base

    axis_strategies = matrix.get("strategies") or [base.strategy]
    axis_placements = matrix.get("placements") or [base.placement]
    axis_tokens = matrix.get("tokens_per_gpu") or [None]

    cells: list[Cell] = []
    skipped: list[str] = []

    for s_name, p_name, tokens in itertools.product(
        axis_strategies, axis_placements, axis_tokens
    ):
        strategy = strategies_mod.get(s_name)
        place = placement_mod.get(p_name)

        kwargs: dict[str, Any] = {"strategy": s_name, "placement": p_name}
        if tokens is not None:
            if isinstance(tokens, dict):
                # A 2C cell may pin its own seq_len -- 2048 tokens/GPU is
                # unreachable at seq_len 4096, since micro_batch must be >= 1.
                kwargs.update(tokens)
            else:
                kwargs["tokens_per_gpu"] = tokens

        try:
            spec = replace(base, **kwargs)
        except SystemExit as e:
            skipped.append(f"{s_name}/{p_name}/{tokens}: {e}")
            continue

        reasons = placement_mod.validate(strategy, place)
        if reasons:
            skipped.append(f"{spec.run_id}: {'; '.join(reasons)}")
            continue

        cells.append(Cell(spec, strategy, place))

    return cells, skipped


# ---------------------------------------------------------------------------
# Environment activation
#
# machine_rank 0 runs under the shell the driver was started from, so it
# inherits an already-activated conda env for free. A peer rank does not: ssh
# runs `bash -lc`, and `conda init` installs its shell hook into ~/.bashrc,
# which returns early for non-interactive shells. `conda` is then undefined on
# the peer, `accelerate` resolves to system Python or not at all, and rank 0
# sits in the rendezvous until it times out. So the activation is made explicit
# and emitted into every launch command -- peers *and* rank 0, so both ranks
# provably run the same interpreter.
# ---------------------------------------------------------------------------
def _detect_conda() -> tuple[str | None, str | None]:
    """(base, env) of the conda env the driver itself is running in."""
    env = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("CONDA_PREFIX")

    base = os.environ.get("CONDA_EXE")
    base = str(Path(base).parent.parent) if base else None
    if not base:
        # No CONDA_EXE (a bare `source activate`): CONDA_PREFIX_1 is the stack's
        # root, and in the base env itself CONDA_PREFIX already is the root.
        base = os.environ.get("CONDA_PREFIX_1") or (
            os.environ.get("CONDA_PREFIX") if env == "base" else None
        )
    return base, env


def _activate_cmd(base: str, env: str) -> str:
    """Shell snippet that activates `env`, hook first.

    Sourcing conda.sh directly is what makes this work in a non-interactive
    shell; `conda activate` accepts either a name or a full prefix path.
    """
    hook = Path(base) / "etc" / "profile.d" / "conda.sh"
    return f". {shlex.quote(str(hook))} && conda activate {shlex.quote(env)}"


# ---------------------------------------------------------------------------
# Launching
# ---------------------------------------------------------------------------
def _write_cell_configs(cell: Cell, gen_dir: Path) -> tuple[Path, Path]:
    """Write the cell's launcher configs under a *repo-relative* directory.

    Relative on purpose: the same path has to resolve on every node after
    `cd <repo_dir>`, and the peer nodes get this directory rsync'd to them
    before launch. An absolute path here would point at node0's filesystem.
    """
    if gen_dir.is_absolute():
        raise SystemExit(
            f"--generated-dir must be repo-relative (got {gen_dir}); peer nodes "
            "resolve it after cd'ing into their own copy of the repo"
        )
    d = gen_dir / cell.run_id
    ds_path = (
        ds_config.write(cell.spec, cell.strategy, d)
        if cell.strategy.uses_deepspeed
        else None
    )
    # machine_rank is supplied on the launch command line, so one accelerate
    # config serves every node in the cell.
    accel_path = accel_config.write(cell.spec, cell.strategy, cell.place, d, ds_path)
    return accel_path, d


# Forwarded on every command even when they match the default: these are what
# identify the cell, and a silent default change must not be able to make a
# launched run differ from the run_id it is filed under.
_ALWAYS_FORWARD = ("strategy", "placement", "seq_len", "micro_batch", "out_dir")

# Set on the command line by the driver itself, or not a train-time concept.
_NEVER_FORWARD = {"run_id", "tokens_per_gpu_step", "buckets", "extra", "machine_rank"}


def _train_flags(spec: config.RunSpec) -> list[str]:
    """Flags forwarded to cluster_bench.train."""
    proto = config.RunSpec()
    flags: list[str] = []

    for name, value in spec.to_dict().items():
        if name in _NEVER_FORWARD or not hasattr(proto, name):
            continue
        # micro_batch is derived from tokens_per_gpu; sending both is
        # redundant and invites the two drifting apart.
        if name == "micro_batch" and spec.tokens_per_gpu is not None:
            continue
        if value is None:
            continue
        if name not in _ALWAYS_FORWARD and str(value) == str(getattr(proto, name)):
            continue

        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            flags.append(flag if value else "--no-" + name.replace("_", "-"))
        else:
            flags += [flag, str(value)]
    return flags


def _launch_cmd(
    cell: Cell,
    accel_path: Path,
    machine_rank: int,
    repo_dir: Path,
    env_file: Path,
    nccl_debug: str,
    activate: str | None = None,
) -> str:
    devices = cell.place.cuda_visible_devices(machine_rank)
    flags = _train_flags(cell.spec) + ["--machine-rank", str(machine_rank)]
    inner = [
        "accelerate",
        "launch",
        "--config_file",
        str(accel_path),
        "--machine_rank",
        str(machine_rank),
        "-m",
        "cluster_bench.train",
        *flags,
    ]
    parts = [] if activate is None else [activate]
    return " && ".join(
        [
            *parts,
            f"cd {shlex.quote(str(repo_dir))}",
            f". {shlex.quote(str(env_file))}",
            # env.sh may be left at NCCL_DEBUG=INFO after topology work; timing
            # runs need WARN or the logging itself shows up in the p95.
            f"export NCCL_DEBUG={nccl_debug}",
            f"export CUDA_VISIBLE_DEVICES={devices}",
            shlex.join(inner),
        ]
    )


def _push_configs(host: str, gen_dir: Path, remote_dir: Path) -> None:
    """Copy this cell's generated configs to a peer node.

    Without this the peer cd's into its own checkout and finds no
    results/generated/<run_id>/ -- accelerate then fails on a missing config
    file, or worse, picks up a stale one from a previous cell.
    """
    remote_parent = remote_dir / gen_dir.parent
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {shlex.quote(str(remote_parent))}"],
        check=True,
    )
    subprocess.run(
        ["scp", "-q", "-r", str(gen_dir), f"{host}:{remote_parent}/"],
        check=True,
    )


def preflight(host: str, args: argparse.Namespace) -> None:
    """Check a peer can reach the repo and the env before any cell starts.

    One ssh per peer, once per sweep. Without it, a peer whose env didn't
    activate shows up as rank 0 hanging in the rendezvous for the full NCCL
    timeout, once per cell, with the real error buried in a rank log.
    """
    parts = ([args.activate] if args.activate else []) + [
        f"cd {shlex.quote(str(args.remote_dir))}",
        "python -c 'import accelerate, cluster_bench'",
        "command -v accelerate",
    ]
    p = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, f"bash -lc {shlex.quote(' && '.join(parts))}"],
        capture_output=True,
        text=True,
    )
    if p.returncode != 0:
        raise SystemExit(
            f"preflight failed on {host} (exit {p.returncode}):\n"
            f"{(p.stderr or p.stdout).strip()}\n"
            f"The peer runs `bash -lc`, which does not pick up conda's shell "
            f"hook from ~/.bashrc. Check --conda-base/--conda-env and that "
            f"{args.remote_dir} is a checkout with the env installed."
        )
    where = (p.stdout.strip().splitlines() or ["ok"])[-1]
    print(f"  preflight {host}: {where}")


def run_cell(cell: Cell, args: argparse.Namespace) -> dict[str, Any]:
    accel_path, gen_dir = _write_cell_configs(cell, Path(args.generated_dir))
    started = time.time()

    local_cmd = _launch_cmd(
        cell,
        accel_path,
        0,
        Path(args.repo_dir),
        Path(args.env_file),
        args.nccl_debug,
        args.activate,
    )
    peer_cmds = {
        rank: _launch_cmd(
            cell,
            accel_path,
            rank,
            Path(args.remote_dir),
            Path(args.env_file),
            args.nccl_debug,
            args.activate,
        )
        for rank in range(1, cell.place.num_machines)
    }

    if args.dry_run:
        for rank, cmd in peer_cmds.items():
            print(f"  ssh {args.hosts[rank]} (machine_rank {rank}):\n     {cmd}")
        print(f"  local (machine_rank 0):\n     {local_cmd}")
        return {"run_id": cell.run_id, "status": "dry-run"}

    peers: list[tuple[int, subprocess.Popen]] = []
    logs: list[Any] = []
    for rank, cmd in peer_cmds.items():
        host = args.hosts[rank]
        print(f"  -> ssh {host} (machine_rank {rank})")
        _push_configs(host, gen_dir, Path(args.remote_dir))
        log = (gen_dir / f"rank{rank}.log").open("w")
        logs.append(log)
        peers.append(
            (
                rank,
                subprocess.Popen(
                    ["ssh", "-o", "BatchMode=yes", host, f"bash -lc {shlex.quote(cmd)}"],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                ),
            )
        )

    print("  -> local (machine_rank 0)")
    rc = subprocess.run(["bash", "-lc", local_cmd]).returncode

    # A peer that dies at startup (missing env, missing config) doesn't fail
    # rank 0 -- rank 0 blocks in the rendezvous until it times out, which reads
    # like a hang rather than an error. Record the peer's exit code so the
    # sweep log says which rank actually broke.
    peer_rcs: dict[str, int] = {}
    for rank, p in peers:
        try:
            p.wait(timeout=args.peer_timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        peer_rcs[str(rank)] = p.returncode
    for f in logs:
        f.close()

    for rank, prc in peer_rcs.items():
        if prc != 0:
            print(f"  peer rank {rank} exited {prc}; see {gen_dir}/rank{rank}.log")

    ok = rc == 0 and all(prc == 0 for prc in peer_rcs.values())
    return {
        "run_id": cell.run_id,
        "status": "ok" if ok else "failed",
        "returncode": rc,
        "peer_returncodes": peer_rcs,
        "elapsed_s": time.time() - started,
    }


def _matches(cell: Cell, keys: list[str]) -> bool:
    """Select cells by strategy or placement name, matched exactly.

    Plain substring matching on run_id is too loose to be safe: `zero3` is a
    substring of `tp2-zero3`, so the documented narrow-start invocation would
    quietly pull in a TP cell and turn a 4-cell run into 5. Strategy and
    placement names therefore match exactly; anything else falls back to
    substring so tags and token counts stay selectable.
    """
    known = set(strategies_mod.STRATEGIES) | set(placement_mod.PLACEMENTS)
    for k in keys:
        if k in known:
            if k in (cell.strategy.name, cell.place.name):
                return True
        elif k in cell.run_id:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("matrix", type=Path, help="configs/matrix/*.yaml")
    ap.add_argument("--hosts", nargs="+", default=["localhost", "node1"],
                    help="one hostname per machine_rank; index 0 is this node")
    ap.add_argument("--repo-dir", default=str(Path.cwd()))
    ap.add_argument("--remote-dir", default=None,
                    help="repo path on peer nodes (defaults to --repo-dir)")
    ap.add_argument("--env-file", default="configs/env.sh")
    ap.add_argument("--generated-dir", default="results/generated")
    ap.add_argument("--nccl-debug", default="WARN")
    ap.add_argument("--peer-timeout", type=float, default=300.0)
    default_base, default_env = _detect_conda()
    ap.add_argument("--conda-env", default=default_env,
                    help="conda env name or prefix to activate on every rank "
                         f"(default: the driver's own, {default_env!r})")
    ap.add_argument("--conda-base", default=default_base,
                    help="conda install root, holding etc/profile.d/conda.sh "
                         f"(default: {default_base!r})")
    ap.add_argument("--no-conda", dest="use_conda", action="store_false",
                    help="don't activate anything; the shell rc on every node "
                         "is expected to put the right python on PATH")
    ap.add_argument("--no-preflight", dest="preflight", action="store_false",
                    help="skip the per-peer import check before the first cell")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these strategies/placements (exact names), "
                         "or cells whose run_id contains the given text")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true", help="print cells and exit")
    # RunSpec's own flags (--strategy, --seq-len, --model-path, ...) act as the
    # base every matrix cell is built from, so a one-off override needs no
    # edit to the matrix file. --out-dir comes from here too.
    config.add_arguments(ap)
    args = ap.parse_args()
    args.remote_dir = args.remote_dir or args.repo_dir

    args.activate = None
    if args.use_conda:
        if not (args.conda_base and args.conda_env):
            raise SystemExit(
                "could not detect the conda env to activate on the peer nodes "
                f"(base={args.conda_base!r}, env={args.conda_env!r}). Run the "
                "sweep from inside `conda activate cluster-bench`, pass "
                "--conda-base/--conda-env, or pass --no-conda if the peers' "
                "shell rc already puts the right python on PATH."
            )
        args.activate = _activate_cmd(args.conda_base, args.conda_env)

    base = config.from_args(args)
    matrix = yaml.safe_load(Path(args.matrix).read_text())
    cells, skipped = expand(matrix, base)

    if args.only:
        cells = [c for c in cells if _matches(c, args.only)]

    needed = max((c.place.num_machines for c in cells), default=1)
    if len(args.hosts) < needed:
        raise SystemExit(
            f"--hosts lists {len(args.hosts)} host(s) but this matrix has cells "
            f"spanning {needed} machines; hosts[0] must be this node"
        )

    print(f"matrix {args.matrix}: {len(cells)} cells, {len(skipped)} skipped")
    for reason in skipped:
        print(f"  skip  {reason}")
    for c in cells:
        print(f"  cell  {c.run_id}  [{c.place.links}]")
    if args.list:
        return

    if args.preflight and not args.dry_run:
        for host in args.hosts[1:needed]:
            preflight(host, args)

    results = []
    for i, cell in enumerate(cells, 1):
        print(f"\n[{i}/{len(cells)}] {cell.run_id}")
        results.append(run_cell(cell, args))

    log_path = base.out_dir / f"sweep_{Path(args.matrix).stem}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({"skipped": skipped, "cells": results}, indent=2))
    print(f"\nwrote {log_path}")

    failed = [r for r in results if r.get("status") == "failed"]
    if failed:
        print(f"{len(failed)} cells failed: {', '.join(r['run_id'] for r in failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
