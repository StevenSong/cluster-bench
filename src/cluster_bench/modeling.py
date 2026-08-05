"""Model loading plus the pre-flight checks the proxy study depends on.

The 4B proxy is only a valid stand-in for dense 31B because ZeRO-3 comm
(~6P bytes/GPU/step) and compute (~8*P*tokens FLOPs) both scale with P, so P
cancels and the comm:compute ratio is set by tokens/GPU/step. Two architecture
choices break that argument, and each gets a guard here:

* **Mixture of experts.** Expert all-to-all is a different comm pattern with no
  counterpart in the dense target. Breaks the numerator.

* **Linear attention** (gated delta, Mamba, RWKV, and the hybrids that
  interleave such layers with real attention). Breaks the *denominator*, which
  is the subtler failure. Comm is untouched -- ZeRO-3 gathers bytes and does
  not care what the layer computes -- but the fused kernels live in
  `flash-linear-attention` / `causal-conv1d`, neither of which this repo
  installs, so transformers silently falls back to an eager chunked recurrence
  and inflates compute in every cell by the same large factor. Step times all
  rise together, `comm_overhead` is a ratio, and every overhead in Matrix 2A
  gets divided toward zero. The matrix stays correctly *ordered* while its
  magnitudes go quietly wrong, and Matrix 2C -- whose entire output is the
  tokens/GPU where comm stops mattering -- reports a crossover well below the
  truth. It is the same contamination as letting the 4B run at its natural
  micro-batch, arriving through a different door.

  The fallback announces itself only as a transformers warning about a "fast
  path", buried in whatever else rank 0 prints. Hence a hard guard: the study
  wants a plain dense full-attention model, and anything else should stop the
  run rather than land in a results JSON.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .config import RunSpec

_MOE_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "moe_layer_freq",
)

# Config keys that only exist on a model carrying linear-attention or
# state-space layers. Presence of any one of them is enough.
_LINEAR_ATTN_KEYS = (
    "linear_attn_config",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "mamba_d_state",
    "mamba_expand",
    "mamba_n_heads",
    "state_size",
    "conv_kernel",
    "attn_layer_indices",
    "full_attention_interval",
)

# `layer_types` is the modern spelling: one entry per layer. Softmax attention
# comes in several flavours and all of them are fine here -- sliding-window and
# chunked attention still run through the flash/sdpa kernels and still have a
# counterpart in the dense target. Anything outside this set is either linear
# attention or something new, and both should stop the run: a name we do not
# recognise is exactly the case where a silent eager fallback would hide.
_SOFTMAX_LAYER_TYPES = frozenset(
    {"full_attention", "sliding_attention", "chunked_attention"}
)


def describe(model_path: str) -> dict[str, Any]:
    """Architecture facts worth recording with every run."""
    cfg = AutoConfig.from_pretrained(model_path)
    text_cfg = getattr(cfg, "text_config", cfg)

    moe_evidence = {
        k: getattr(text_cfg, k) for k in _MOE_KEYS if getattr(text_cfg, k, None)
    }

    layer_types = getattr(text_cfg, "layer_types", None)
    linear_evidence = {
        k: getattr(text_cfg, k) for k in _LINEAR_ATTN_KEYS if getattr(text_cfg, k, None)
    }
    non_softmax = sorted(set(layer_types or []) - _SOFTMAX_LAYER_TYPES)
    if non_softmax:
        linear_evidence["layer_types"] = non_softmax

    return {
        "model_type": getattr(cfg, "model_type", None),
        # Counts, not the raw per-layer list: a 36-entry array of strings in
        # every results JSON buys nothing that {"full_attention": 36} does not.
        "layer_type_counts": (
            {t: layer_types.count(t) for t in sorted(set(layer_types))}
            if layer_types
            else None
        ),
        "is_linear_attention": bool(linear_evidence),
        "linear_attention_evidence": linear_evidence,
        "num_hidden_layers": getattr(text_cfg, "num_hidden_layers", None),
        "hidden_size": getattr(text_cfg, "hidden_size", None),
        "intermediate_size": getattr(text_cfg, "intermediate_size", None),
        "num_attention_heads": getattr(text_cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(text_cfg, "num_key_value_heads", None),
        "vocab_size": getattr(text_cfg, "vocab_size", None),
        "tie_word_embeddings": getattr(cfg, "tie_word_embeddings", None),
        "is_moe": bool(moe_evidence),
        "moe_evidence": moe_evidence,
        "has_vision_tower": hasattr(cfg, "vision_config"),
    }


def check_dense(info: dict[str, Any], model_path: str) -> None:
    if info["is_moe"]:
        raise SystemExit(
            f"{model_path} looks like a mixture of experts ({info['moe_evidence']}).\n"
            "Expert all-to-all is a different comm pattern, so this model is not a\n"
            "valid proxy for dense 31B and the whole sharding study would be\n"
            "measuring the wrong thing. Pick a dense ~4B, or pass --no-require-dense\n"
            "if you have decided to measure MoE deliberately."
        )


def check_full_attention(info: dict[str, Any], model_path: str) -> None:
    """Refuse hybrids and linear-attention models. See the module docstring.

    Cheap to check and expensive to miss: the symptom is a one-line transformers
    warning, and the consequence is a whole matrix of plausible-looking overhead
    numbers that are all too small.
    """
    if not info["is_linear_attention"]:
        return
    raise SystemExit(
        f"{model_path} has linear-attention or state-space layers "
        f"({info['linear_attention_evidence']}).\n"
        "Those layers have no fused kernel in this environment, so transformers\n"
        "falls back to an eager implementation -- the giveaway is its warning\n"
        "about a 'fast path' not being available. That inflates compute in every\n"
        "cell equally, and since comm_overhead is a ratio of step times, it\n"
        "shrinks every overhead in the matrix toward zero and moves Matrix 2C's\n"
        "crossover below the truth. Comm itself is unaffected, which is what\n"
        "makes the corruption hard to spot in the results.\n"
        "Use a dense, full-attention ~4B instead. If you have installed\n"
        "flash-linear-attention and causal-conv1d on *every* node and want to\n"
        "measure this architecture deliberately, pass --no-require-full-attention."
    )


def check_attn_available(attn_impl: str) -> None:
    """Fail on the flag, not fifteen seconds later on the weights.

    A sweep that dies partway through loading the model on rank 3 of an 8-rank
    cell leaves the other ranks in the rendezvous until it times out, and the
    message arrives buried in seven copies of a traceback. flash-attn is also
    the one dependency env.yaml cannot install by itself, so it is the one most
    likely to be missing on a freshly staged peer node.
    """
    if "flash" not in attn_impl:
        return
    try:
        import flash_attn  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            f"attn_impl={attn_impl!r} but flash-attn is not importable ({e}).\n"
            "It is a separate install step, on every node:\n"
            "    pip install -r requirements-flash.txt --no-build-isolation\n"
            "Or pass --attn-impl sdpa: step times are unaffected (identical\n"
            "FLOPs) but attention will cross packed document boundaries and the\n"
            "absolute loss stops being a training loss."
        ) from e


def load(spec: RunSpec) -> tuple[Any, Any, dict[str, Any]]:
    check_attn_available(spec.attn_impl)
    info = describe(spec.model_path)
    if spec.require_dense:
        check_dense(info, spec.model_path)
    if spec.require_full_attention:
        check_full_attention(info, spec.model_path)

    tokenizer = AutoTokenizer.from_pretrained(spec.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        spec.model_path,
        dtype=getattr(torch, spec.dtype),
        attn_implementation=spec.attn_impl,
    )

    frozen = 0
    if spec.freeze_vision and info["has_vision_tower"]:
        # Text-only workload: freezing the tower + projector saves optimizer
        # state and avoids drifting multimodal alignment for no benefit. It
        # also changes the sharded parameter count, so it is recorded.
        for name, p in model.named_parameters():
            if "vision_tower" in name or "embed_vision" in name:
                p.requires_grad_(False)
                frozen += p.numel()

    info["params_total"] = sum(p.numel() for p in model.parameters())
    info["params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    info["params_frozen_vision"] = frozen
    return model, tokenizer, info
