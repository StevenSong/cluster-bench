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


def _all_reduce(value: float, op) -> float:  # noqa: ANN001
    if not (dist.is_available() and dist.is_initialized()):
        return value
    t = torch.tensor([value], dtype=torch.float64, device=_device())
    dist.all_reduce(t, op=op)
    return float(t.item())


def _all_reduce_max(value: float) -> float:
    return _all_reduce(value, dist.ReduceOp.MAX)


def _all_reduce_sum(value: float) -> float:
    return _all_reduce(value, dist.ReduceOp.SUM)


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
    # Ranks per tensor-parallel group. Under TP every rank of a group is fed
    # the same micro-batch, so a per-rank tally counts each token tp_size
    # times; `summary` divides it back out. 1 for every non-TP config.
    tp_size: int = 1

    step_times: list[float] = field(default_factory=list)
    step_tokens: list[int] = field(default_factory=list)
    loss_curve: list[dict[str, float]] = field(default_factory=list)
    _t0: float = 0.0
    _pending_tokens: int = field(default=0, init=False)
    _cuda: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._cuda = torch.cuda.is_available()

    # -- token accounting ----------------------------------------------
    def count(self, inputs: dict) -> None:
        """Tokens in one micro-batch, called from the trainer's training_step.

        On real text the token count is a property of the corpus and the
        packing strategy, not of the flags -- so it gets measured. `wrapped`
        packing should make this exactly micro_batch * seq_len every step; a
        deviation means the control the whole study rests on is not holding,
        and `summary` reports it rather than leaving it to be discovered in the
        throughput numbers.

        Counted here and not in the collator because `dataloader_num_workers`
        puts the collator in a subprocess, where the tally would be lost.
        """
        mask = inputs.get("attention_mask")
        if mask is not None:
            self._pending_tokens += int(mask.sum().item())
            return
        # padding-free / packed batches carry position_ids instead: every
        # element of input_ids is a real token.
        ids = inputs.get("input_ids")
        if ids is not None:
            self._pending_tokens += int(ids.numel())

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
        # One entry per optimizer step, so grad_accum micro-batches are already
        # summed in by the time we get here.
        self.step_tokens.append(self._pending_tokens)
        self._pending_tokens = 0

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

        # Throughput is reported against the tokens that actually reached the
        # model. Nominal is kept alongside it so a broken control is visible as
        # a discrepancy instead of silently rescaling every number.
        tokens = self.step_tokens[self.warmup_steps :]
        tok_local = statistics.median(tokens) if tokens else 0.0
        # tp_size ranks share one copy of the batch, so the sum over ranks has
        # counted every token that many times. Dividing it out makes this the
        # count of *distinct* tokens the step consumed -- which is the quantity
        # the control pins, and what keeps a TP cell in the same comparison
        # group as the DDP baseline it is reported against.
        tok_all = _all_reduce_sum(float(tok_local)) / self.tp_size
        tok_per_gpu = tok_all / world_size if world_size else tok_local
        drift = (
            abs(tok_per_gpu - self.tokens_per_gpu_step) / self.tokens_per_gpu_step
            if self.tokens_per_gpu_step
            else 0.0
        )

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
            "tokens_per_gpu_step_measured": tok_per_gpu,
            # Raw per-rank counts, *not* de-duplicated: under TP they are
            # tp_size x the distinct tokens above, because that is genuinely
            # how many tokens went through this rank's shard of the model.
            "tokens_per_step_min": min(tokens) if tokens else None,
            "tokens_per_step_max": max(tokens) if tokens else None,
            "tp_size": self.tp_size,
            # >1% off nominal means wrapped packing did not deliver exact
            # chunks and tokens/GPU/step is no longer the invariant every
            # cross-cell comparison assumes. Treat the cell as suspect.
            "tokens_per_step_drift": drift,
            "tokens_per_step_control_held": drift <= 0.01,
            "tokens_per_s_per_gpu": tok_per_gpu / p50 if p50 else None,
            "tokens_per_s_total": tok_all / p50 if p50 else None,
            "step_times_s": measured,
            "step_tokens": tokens,
            "loss_curve": self.loss_curve,
        }
        if self._cuda:
            out["peak_mem_allocated_gb"] = torch.cuda.max_memory_allocated() / 1024**3
            out["peak_mem_reserved_gb"] = torch.cuda.max_memory_reserved() / 1024**3
        return out
