"""RunSpec: every knob for one benchmark cell, in one serializable object.

The old `sft_gemma.py` hardcoded all of this inline, which made a sweep
impossible. Everything that can differ between two cells of the matrix lives
here and gets written verbatim into the results JSON, so a run is reproducible
from its own output.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bucket pinning.
#
# DeepSpeed's "auto" resolves these from the *live* model's hidden_size:
#     reduce_bucket_size                  = hidden**2
#     stage3_prefetch_bucket_size         = 0.9 * hidden**2
#     stage3_param_persistence_threshold  = 10 * hidden
# which means a 4B proxy (hidden 2560) and the 31B target (hidden 5120) get
# message sizes 4x apart -- silently breaking the comparison the proxy exists
# to make. We pin all three to what auto *would* have produced at the target's
# hidden size, so the proxy moves target-sized buckets.
# ---------------------------------------------------------------------------
TARGET_HIDDEN_SIZE = 5120


def buckets_for_hidden(hidden: int) -> dict[str, int]:
    return {
        "reduce_bucket_size": hidden * hidden,
        "stage3_prefetch_bucket_size": int(0.9 * hidden * hidden),
        "stage3_param_persistence_threshold": 10 * hidden,
    }


@dataclass
class RunSpec:
    # --- identity ---
    strategy: str = "zero3"
    placement: str = "full"
    tag: str = ""
    out_dir: Path = Path("results")

    # --- model ---
    model_path: str = "/opt/gpudata/models/Qwen/Qwen3.5-4B"
    attn_impl: str = "sdpa"
    dtype: str = "bfloat16"
    # Abort if the model config looks like a mixture of experts: expert
    # all-to-all is a different comm pattern and invalidates the dense proxy.
    require_dense: bool = True

    # --- data ---
    # "synthetic" generates exactly seq_len tokens per sample with no
    # tokenizer variance -- the right choice for timing. A real dataset path
    # is for the loss-curve correctness gate.
    dataset: str = "synthetic"
    dataset_config: str = "en"
    dataset_split: str = "train"
    prompt_field: str = "Question"
    reasoning_field: str = "Complex_CoT"
    completion_field: str = "Response"

    # --- THE critical control -------------------------------------------
    # per_device_train_batch_size * max_length = tokens/GPU/step. Pinned to
    # 8192 to match the 31B operating point even though it wastes most of a
    # 4B card. Let the 4B run at its natural micro-batch and everything
    # becomes compute-bound and all differences vanish into noise.
    seq_len: int = 4096
    micro_batch: int = 2
    grad_accum: int = 1
    # If set, overrides micro_batch as micro_batch = tokens_per_gpu / seq_len.
    # Used by Matrix 2C.
    tokens_per_gpu: int | None = None

    # --- trainer ---
    packing: bool = True
    packing_strategy: str = "bfd"
    padding_free: bool = True
    liger: bool = True
    gradient_checkpointing: bool = True
    completion_only_loss: bool = True
    average_tokens_across_devices: bool = True
    learning_rate: float = 6e-6
    seed: int = 42
    dataloader_num_workers: int = 8
    freeze_vision: bool = True

    # --- measurement ---
    warmup_steps: int = 20  # NCCL channel setup + allocator warmup
    measure_steps: int = 80
    # Barrier + cuda sync around every step. Costs a little absolute
    # throughput but is applied identically to every config, and without it
    # step boundaries smear across async collectives.
    sync_each_step: bool = True

    # --- deepspeed tuning (pinned, see module docstring) ---
    bucket_ref_hidden: int = TARGET_HIDDEN_SIZE
    stage3_max_live_parameters: int = 600_000_000
    stage3_max_reuse_distance: int = 600_000_000
    zero_quantized_weights: bool = False
    zero_quantized_gradients: bool = False
    overlap_comm: bool = True
    wall_clock_breakdown: bool = False

    # --- launch ---
    main_process_ip: str = "10.32.12.33"
    main_process_port: int = 29500
    machine_rank: int = 0

    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        if self.tokens_per_gpu is not None:
            if self.tokens_per_gpu % self.seq_len:
                raise SystemExit(
                    f"tokens_per_gpu={self.tokens_per_gpu} is not a multiple of "
                    f"seq_len={self.seq_len}; pick a seq_len that divides it "
                    "(Matrix 2C's 2048-token cell needs seq_len=2048, not 4096)"
                )
            self.micro_batch = self.tokens_per_gpu // self.seq_len

    @property
    def tokens_per_gpu_step(self) -> int:
        return self.micro_batch * self.seq_len * self.grad_accum

    @property
    def run_id(self) -> str:
        parts = [
            self.strategy,
            self.placement,
            f"t{self.tokens_per_gpu_step}",
            f"s{self.seq_len}",
        ]
        if self.tag:
            parts.append(self.tag)
        return "__".join(parts)

    @property
    def buckets(self) -> dict[str, int]:
        return buckets_for_hidden(self.bucket_ref_hidden)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["out_dir"] = str(self.out_dir)
        d["run_id"] = self.run_id
        d["tokens_per_gpu_step"] = self.tokens_per_gpu_step
        d["buckets"] = self.buckets
        return d


# ---------------------------------------------------------------------------
# CLI. Every field above is settable; env vars CB_<FIELD> act as defaults so a
# cell can be pinned from a launcher script without rewriting the command.
# ---------------------------------------------------------------------------
_BOOL_FIELDS = {
    f.name for f in dataclasses.fields(RunSpec) if f.type in ("bool", bool)
}


def _env_default(name: str, default: Any) -> Any:
    raw = os.environ.get(f"CB_{name.upper()}")
    if raw is None:
        return default
    if isinstance(default, bool) or name in _BOOL_FIELDS:
        return raw.lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


def add_arguments(ap: argparse.ArgumentParser) -> None:
    proto = RunSpec()
    for f in dataclasses.fields(RunSpec):
        if f.name == "extra":
            continue
        flag = "--" + f.name.replace("_", "-")
        default = _env_default(f.name, getattr(proto, f.name))

        if f.name in _BOOL_FIELDS:
            ap.add_argument(
                flag,
                dest=f.name,
                default=default,
                action=argparse.BooleanOptionalAction,
            )
        elif f.name == "tokens_per_gpu":
            ap.add_argument(flag, dest=f.name, type=int, default=default)
        elif f.name == "out_dir":
            ap.add_argument(flag, dest=f.name, type=Path, default=default)
        elif isinstance(default, int):
            ap.add_argument(flag, dest=f.name, type=int, default=default)
        elif isinstance(default, float):
            ap.add_argument(flag, dest=f.name, type=float, default=default)
        else:
            ap.add_argument(flag, dest=f.name, default=default)


def from_args(args: argparse.Namespace) -> RunSpec:
    names = {f.name for f in dataclasses.fields(RunSpec)} - {"extra"}
    return RunSpec(**{k: v for k, v in vars(args).items() if k in names})
