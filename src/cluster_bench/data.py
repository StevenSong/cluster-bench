"""Datasets for the two things this repo measures.

Timing and the correctness gate both run on the **real** dataset. Real text is
what the 31B target will actually be trained on, so the tokenizer's length
distribution, the packing path and the completion mask are all exercised by the
numbers we report rather than only by the gate.

What real text costs is the control: `tokens/GPU/step` must be an invariant, and
bfd packing emits sequences of *at most* max_length, so tokens per step -- and
therefore the denominator of every throughput number -- would drift between
cells. `wrapped` packing is the way out: it concatenates the tokenized corpus
and cuts it into chunks of *exactly* max_length, so micro_batch * seq_len is
still the literal token count. That is why `packing_strategy` defaults to
`wrapped` for timing runs; `metrics.BenchCallback` counts the tokens that
actually reach the model so the invariant is verified, not assumed.

`synthetic` is kept as an escape hatch (`--dataset synthetic`): pre-tokenized
random ids of exactly seq_len, no tokenizer and no corpus needed. It is useful
for bringing up a node before the dataset is staged, and for isolating a
suspected data-path problem from a comm problem.
"""

from __future__ import annotations

import numpy as np
from datasets import Dataset, load_dataset

from .config import RunSpec

# Examples tokenized to estimate the corpus's mean length when auto-sizing.
_PROBE_SAMPLES = 128
# Chat template, special tokens and the reasoning wrapper all add tokens the
# probe's raw-field concatenation does not see.
_TEMPLATE_OVERHEAD = 1.15
# Margin on top of the computed requirement: packing residue, drop_last, and
# an uneven split across ranks all lose a few samples.
_SIZING_MARGIN = 1.5


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


def _mean_tokens_per_example(spec: RunSpec, ds: Dataset, tokenizer) -> float:
    """Rough tokens per example, from a probe of the head of the split."""
    n = min(_PROBE_SAMPLES, len(ds))
    if not n or tokenizer is None:
        return float(spec.seq_len)

    probe = ds.select(range(n))
    total = 0
    for ex in probe:
        text = " ".join(
            str(ex.get(f) or "")
            for f in (spec.prompt_field, spec.reasoning_field, spec.completion_field)
        )
        total += len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return max(1.0, (total / n) * _TEMPLATE_OVERHEAD)


def _n_examples_needed(spec: RunSpec, ds: Dataset, tokenizer, world_size: int) -> int:
    """How much of the split a cell has to touch.

    A 40-second cell has no business tokenizing a 25k-example corpus nine times
    over -- dataset preparation would dominate the thing being measured. Sizing
    is by tokens when packing (the corpus is one stream, sliced into chunks) and
    by examples when it is off (one example per sequence).
    """
    steps = spec.warmup_steps + spec.measure_steps
    sequences = steps * spec.micro_batch * spec.grad_accum * world_size

    if not spec.packing:
        needed = sequences
    else:
        per_example = _mean_tokens_per_example(spec, ds, tokenizer)
        needed = int(sequences * spec.seq_len / per_example) + 1

    return min(len(ds), int(needed * _SIZING_MARGIN) + world_size)


def _real(spec: RunSpec, tokenizer, world_size: int) -> Dataset:
    ds = load_dataset(spec.dataset, spec.dataset_config, split=spec.dataset_split)

    n = spec.dataset_num_samples or _n_examples_needed(spec, ds, tokenizer, world_size)
    # Head of the split rather than a shuffled sample: deterministic without
    # depending on datasets' shuffle implementation, and every cell in the
    # matrix therefore sees byte-identical text in identical order.
    if n < len(ds):
        ds = ds.select(range(n))

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


def build(
    spec: RunSpec, tokenizer, vocab_size: int, world_size: int
) -> tuple[Dataset, bool]:
    """Return (dataset, is_synthetic).

    Sized so a run has enough samples for warmup + measurement with margin,
    and never so many that dataset construction dominates a 40-second cell.
    """
    if spec.dataset != "synthetic":
        return _real(spec, tokenizer, world_size), False

    steps = spec.warmup_steps + spec.measure_steps
    needed = steps * spec.micro_batch * spec.grad_accum * world_size
    return _synthetic(spec, vocab_size, int(needed * 1.25) + world_size), True
