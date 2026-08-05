"""Everything needed to explain a number six months from now.

A step-time delta is meaningless without the NCCL build, the driver, and the
repo state that produced it -- a NCCL bump alone can move topology detection
enough to reorder the whole matrix.
"""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from typing import Any

import torch


def _cmd(*argv: str) -> str | None:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None if out.returncode == 0 else None


def _git() -> dict[str, Any]:
    sha = _cmd("git", "rev-parse", "HEAD")
    status = _cmd("git", "status", "--porcelain")
    return {"sha": sha, "dirty": bool(status), "branch": _cmd("git", "rev-parse", "--abbrev-ref", "HEAD")}


def _nccl_version() -> str | None:
    try:
        v = torch.cuda.nccl.version()
    except Exception:
        return None
    return ".".join(str(x) for x in v) if isinstance(v, tuple) else str(v)


def _gpus() -> list[dict[str, Any]]:
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        out.append(
            {"index": i, "name": p.name, "total_mem_gb": p.total_memory / 1024**3}
        )
    return out


# NCCL env vars are the Tier-1 result being held constant; record which ones
# were actually in effect rather than trusting that env.sh was sourced.
_ENV_PREFIXES = ("NCCL_", "TORCH_NCCL_", "GLOO_", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "PYTORCH_CUDA_ALLOC_CONF")


def collect() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "nccl": _nccl_version(),
        "driver": _cmd(
            "nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"
        ),
        "gpus": _gpus(),
        "git": _git(),
        "packages": _versions(),
        "env": {
            k: v
            for k, v in sorted(os.environ.items())
            if k.startswith(_ENV_PREFIXES)
        },
    }


def _versions() -> dict[str, str | None]:
    import importlib.metadata as md

    out: dict[str, str | None] = {}
    for pkg in ("transformers", "trl", "accelerate", "deepspeed", "datasets", "liger-kernel"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = None
    return out
