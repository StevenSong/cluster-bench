"""Step-time measurement.

Reports p50 *and* p95: the mean hides stragglers, and on a rail-aligned fabric
a single mis-routed ring shows up as a fat tail rather than a shifted centre.

Step time is the max across ranks, not rank 0's local view. Every rank blocks
on the next collective anyway, so the slowest rank is the step time; rank 0
alone would flatter any config whose stragglers sit elsewhere.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from transformers import TrainerCallback


def _all_reduce_max(value: float) -> float:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([value], dtype=torch.float64, device=_device())
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _device() -> torch.device:
    return (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


@dataclass
class BenchCallback(TrainerCallback):
    warmup_steps: int
    measure_steps: int
    tokens_per_gpu_step: int
    sync_each_step: bool = True

    step_times: list[float] = field(default_factory=list)
    loss_curve: list[dict[str, float]] = field(default_factory=list)
    _t0: float = 0.0
    _cuda: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._cuda = torch.cuda.is_available()

    # -- lifecycle -----------------------------------------------------
    def _sync(self) -> None:
        if self.sync_each_step and self._cuda:
            torch.cuda.synchronize()

    def on_train_begin(self, args, state, control, **kw):  # noqa: ANN001
        if self._cuda:
            torch.cuda.reset_peak_memory_stats()

    def on_step_begin(self, args, state, control, **kw):  # noqa: ANN001
        self._sync()
        self._t0 = time.perf_counter()

    def on_step_end(self, args, state, control, **kw):  # noqa: ANN001
        self._sync()
        dt = _all_reduce_max(time.perf_counter() - self._t0)
        self.step_times.append(dt)

        n = len(self.step_times)
        if n == self.warmup_steps and self._cuda:
            # Warmup covered NCCL channel setup and allocator growth; the peak
            # we care about is steady-state, so start the high-water mark over.
            torch.cuda.reset_peak_memory_stats()
        if n >= self.warmup_steps + self.measure_steps:
            control.should_training_stop = True
        return control

    def on_log(self, args, state, control, logs=None, **kw):  # noqa: ANN001
        if logs and "loss" in logs:
            self.loss_curve.append(
                {"step": float(state.global_step), "loss": float(logs["loss"])}
            )

    # -- results -------------------------------------------------------
    def summary(self, world_size: int) -> dict[str, Any]:
        measured = self.step_times[self.warmup_steps :]
        if not measured:
            return {
                "error": "no measured steps -- run ended during warmup",
                "steps_seen": len(self.step_times),
                "warmup_steps": self.warmup_steps,
            }

        ordered = sorted(measured)
        p50 = statistics.median(ordered)
        p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]

        out: dict[str, Any] = {
            "steps_measured": len(measured),
            "steps_discarded": self.warmup_steps,
            "step_time_p50_s": p50,
            "step_time_p95_s": p95,
            "step_time_mean_s": statistics.fmean(measured),
            "step_time_stdev_s": (
                statistics.stdev(measured) if len(measured) > 1 else 0.0
            ),
            "step_time_min_s": ordered[0],
            "step_time_max_s": ordered[-1],
            # p95/p50 is the straggler signal, unit-free and comparable
            # across token counts.
            "straggler_ratio": p95 / p50 if p50 else None,
            "tokens_per_gpu_step": self.tokens_per_gpu_step,
            "tokens_per_s_per_gpu": self.tokens_per_gpu_step / p50 if p50 else None,
            "tokens_per_s_total": (
                self.tokens_per_gpu_step * world_size / p50 if p50 else None
            ),
            "step_times_s": measured,
            "loss_curve": self.loss_curve,
        }
        if self._cuda:
            out["peak_mem_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            out["peak_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
        return out
