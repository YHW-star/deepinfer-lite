"""基于 Transformers 因果语言模型封装的轻量 Prefill/Decode Runner。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class KVCache:
    """在 Prefill 和后续每次 Decode 之间传递的单请求缓存句柄。

    当前 ``past_key_values`` 使用 Transformers 的缓存实现。Runner 自行记录逻辑序列长度，
    调用方无需依赖某个特定的 Transformers Cache API。后续可替换底层缓存内容，同时保持
    Prefill/Decode 的调用接口不变。
    """

    past_key_values: Any
    sequence_length: int
    batch_size: int
    device: torch.device
    attention_mask: torch.Tensor


@dataclass
class PrefillOutput:
    """完整处理一次提示词后的结果。"""

    logits: torch.Tensor
    kv_cache: KVCache


@dataclass
class DecodeOutput:
    """恰好处理一个新 token 后的结果。"""

    logits: torch.Tensor
    kv_cache: KVCache


class ModelRunner:
    """以完整提示词 Prefill、随后逐 token Decode 的方式运行模型。"""

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
        """从本地目录或模型 ID 加载 tokenizer 和模型。"""
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
        """处理全部提示词 token，并初始化该请求的 KV Cache。"""
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
        """每个 batch 行恰好处理一个 token，并将对应的 KV 状态追加到缓存。"""
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
