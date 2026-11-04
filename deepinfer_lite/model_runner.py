"""A thin prefill/decode runner around a Transformers causal language model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class KVCache:
    """Per-request cache handle passed from Prefill into each Decode step.

    ``past_key_values`` is the Transformers cache implementation for now. The
    runner owns the logical sequence length so callers do not need to depend
    on a particular Transformers Cache API. A future cache manager can replace
    this payload while keeping the Prefill/Decode call contract stable.
    """

    past_key_values: Any
    sequence_length: int
    batch_size: int
    device: torch.device
    attention_mask: torch.Tensor


@dataclass
class PrefillOutput:
    """Result of processing a complete prompt once."""

    logits: torch.Tensor
    kv_cache: KVCache


@dataclass
class DecodeOutput:
    """Result of processing exactly one new token."""

    logits: torch.Tensor
    kv_cache: KVCache


class ModelRunner:
    """Run a model as one full-prompt Prefill followed by one-token Decode."""

    def __init__(self, model: Any, tokenizer: Any, device: str | torch.device):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device: str | torch.device = "cuda:0",
        dtype: torch.dtype | None = None,
        local_files_only: bool = False,
    ) -> "ModelRunner":
        """Load tokenizer and model from a local directory or model ID."""
        resolved_device = torch.device(device)
        if dtype is None:
            dtype = torch.float16 if resolved_device.type == "cuda" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=local_files_only
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir), dtype=dtype, local_files_only=local_files_only
        )
        return cls(model, tokenizer, resolved_device)

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> PrefillOutput:
        """Process all prompt tokens and initialize the request's KV cache."""
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape [batch, non-empty sequence]")
        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(device=self.device, dtype=torch.long)
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must have the same shape as input_ids")

        sequence_length = input_ids.shape[1]
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_position=torch.arange(sequence_length, device=self.device),
            use_cache=True,
            return_dict=True,
        )
        if outputs.past_key_values is None:
            raise RuntimeError("model did not return past_key_values; check use_cache support")

        cache = KVCache(
            past_key_values=outputs.past_key_values,
            sequence_length=sequence_length,
            batch_size=input_ids.shape[0],
            device=self.device,
            attention_mask=attention_mask,
        )
        return PrefillOutput(logits=outputs.logits[:, -1, :], kv_cache=cache)

    @torch.inference_mode()
    def decode(self, input_ids: torch.Tensor, kv_cache: KVCache) -> DecodeOutput:
        """Process exactly one token per batch row and append its KV state."""
        if input_ids.ndim != 2 or input_ids.shape[1] != 1:
            raise ValueError("Decode input_ids must have shape [batch, 1]")
        if input_ids.shape[0] != kv_cache.batch_size:
            raise ValueError("Decode batch size must match the Prefill cache")
        if kv_cache.device != self.device:
            raise ValueError("KV cache belongs to a different device")

        input_ids = input_ids.to(self.device)
        next_length = kv_cache.sequence_length + 1
        attention_mask = torch.cat(
            (
                kv_cache.attention_mask,
                torch.ones((kv_cache.batch_size, 1), dtype=torch.long, device=self.device),
            ),
            dim=1,
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=kv_cache.past_key_values,
            cache_position=torch.tensor([kv_cache.sequence_length], device=self.device),
            use_cache=True,
            return_dict=True,
        )
        if outputs.past_key_values is None:
            raise RuntimeError("model did not return updated past_key_values")

        updated_cache = KVCache(
            past_key_values=outputs.past_key_values,
            sequence_length=next_length,
            batch_size=kv_cache.batch_size,
            device=self.device,
            attention_mask=attention_mask,
        )
        return DecodeOutput(logits=outputs.logits[:, -1, :], kv_cache=updated_cache)
