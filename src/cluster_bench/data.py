"""Datasets for the two things this repo measures.

`synthetic` is for timing: pre-tokenized samples of *exactly* seq_len tokens,
so tokens/GPU/step is an invariant rather than a packing outcome. bfd packing
on real text makes the step count -- and therefore tokens per step -- vary
between configs, which is fine for training and useless for a controlled
comparison.

A real dataset path is for the correctness gate: the loss curve at a fixed
seed must match the reference config, which is what catches packing-mask and
loss-normalization bugs that a fast-but-wrong config would otherwise hide.
"""

from __future__ import annotations

import numpy as np
from datasets import Dataset, load_dataset

from .config import RunSpec


def _synthetic(spec: RunSpec, vocab_size: int, n_samples: int) -> Dataset:
    rng = np.random.default_rng(spec.seed)
    # Stay clear of the low ids where special/reserved tokens live.
    lo = min(1000, max(1, vocab_size // 100))
    ids = rng.integers(lo, vocab_size, size=(n_samples, spec.seq_len), dtype=np.int64)

    # A fixed 25% prompt / 75% completion split so the number of loss-carrying
    # tokens is the same in every cell.
    #
    # Labels are pre-built rather than left to `completion_mask`: on the
    # skip_prepare_dataset path trl 1.8 never runs the step that turns a mask
    # into labels, and rejects the dataset instead of silently training on the
    # prompt. Masking here is the same arithmetic trl would have done
    # (-100 wherever the mask is 0) and keeps the split intact.
    prompt_len = spec.seq_len // 4
    labels = ids.copy()
    labels[:, :prompt_len] = -100

    return Dataset.from_dict(
        {
            "input_ids": [row.tolist() for row in ids],
            "labels": [row.tolist() for row in labels],
        }
    )


def _real(spec: RunSpec) -> Dataset:
    ds = load_dataset(spec.dataset, spec.dataset_config, split=spec.dataset_split)

    # Conversational prompt-completion rather than `messages`: the target
    # chat template does not expose {% generation %} tags, so assistant_only_loss
    # is unavailable and prompt-completion masking is how loss gets confined to
    # the reasoning + answer text.
    def to_prompt(ex: dict) -> dict:
        return {
            "prompt": [{"role": "user", "content": ex[spec.prompt_field]}],
            "completion": [
                {
                    "role": "assistant",
                    "reasoning": ex[spec.reasoning_field],
                    "content": ex[spec.completion_field],
                }
            ],
            "chat_template_kwargs": {"enable_thinking": True},
        }

    return ds.map(to_prompt, remove_columns=ds.column_names)


def build(spec: RunSpec, vocab_size: int, world_size: int) -> tuple[Dataset, bool]:
    """Return (dataset, is_synthetic).

    Sized so a run has enough samples for warmup + measurement with margin,
    and never so many that dataset construction dominates a 40-second cell.
    """
    if spec.dataset != "synthetic":
        return _real(spec), False

    steps = spec.warmup_steps + spec.measure_steps
    needed = steps * spec.micro_batch * spec.grad_accum * world_size
    return _synthetic(spec, vocab_size, int(needed * 1.25) + world_size), True
