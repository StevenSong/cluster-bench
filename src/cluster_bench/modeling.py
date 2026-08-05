"""Model loading plus the pre-flight check the proxy study depends on.

The 4B proxy is only a valid stand-in for dense 31B because ZeRO-3 comm
(~6P bytes/GPU/step) and compute (~8*P*tokens FLOPs) both scale with P, so P
cancels and the comm:compute ratio is set by tokens/GPU/step. That argument
collapses if the proxy is a mixture of experts -- expert all-to-all is a
different comm pattern with no counterpart in the dense target.
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


def describe(model_path: str) -> dict[str, Any]:
    """Architecture facts worth recording with every run."""
    cfg = AutoConfig.from_pretrained(model_path)
    text_cfg = getattr(cfg, "text_config", cfg)

    moe_evidence = {
        k: getattr(text_cfg, k) for k in _MOE_KEYS if getattr(text_cfg, k, None)
    }
    return {
        "model_type": getattr(cfg, "model_type", None),
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


def load(spec: RunSpec) -> tuple[Any, Any, dict[str, Any]]:
    info = describe(spec.model_path)
    if spec.require_dense:
        check_dense(info, spec.model_path)

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
