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

    @staticmethod
    def _key_value_pairs(past_key_values: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            return list(zip(past_key_values.key_cache, past_key_values.value_cache))
        if hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()
        return [(layer[0], layer[1]) for layer in past_key_values]

    @staticmethod
    def _validate_storage(
        past_key_values: Any,
        expected_length: int,
        expected_batch_size: int,
    ) -> None:
        pairs = KVCache._key_value_pairs(past_key_values)
        if not pairs:
            raise RuntimeError("KV Cache 中没有任何层的 Key/Value Tensor")
        for layer_index, (key, value) in enumerate(pairs):
            if key.shape != value.shape:
                raise RuntimeError(f"第 {layer_index} 层的 Key/Value 形状不一致")
            if key.ndim != 4 or key.shape[0] != expected_batch_size:
                raise RuntimeError(f"第 {layer_index} 层的 KV Tensor 形状不是 [batch, heads, seq, dim]")
            if key.shape[-2] != expected_length:
                raise RuntimeError(
                    f"第 {layer_index} 层的 KV 长度为 {key.shape[-2]}，预期为 {expected_length}"
                )
            if not key.is_contiguous() or not value.is_contiguous():
                raise RuntimeError(f"第 {layer_index} 层的 KV Tensor 不是连续存储")

    @property
    def key_value_tensors(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """返回各层连续存储的 Key/Value Tensor。"""
        return self._key_value_pairs(self.past_key_values)

    @property
    def allocated_bytes(self) -> int:
        """返回当前 KV Tensor 的实际字节数，不含 allocator 缓存。"""
        return sum(key.numel() * key.element_size() + value.numel() * value.element_size()
                   for key, value in self.key_value_tensors)

    def append(
        self,
        past_key_values: Any,
        attention_mask: torch.Tensor,
    ) -> None:
        """接收模型追加一个 token 后的 KV Tensor，并更新本请求缓存。"""
        next_length = self.sequence_length + 1
        self._validate_storage(past_key_values, next_length, self.batch_size)
        if attention_mask.shape != (self.batch_size, next_length):
            raise ValueError("追加后的 attention_mask 长度与 KV Cache 不一致")
        self.past_key_values = past_key_values
        self.sequence_length = next_length
        self.attention_mask = attention_mask


@dataclass
class GenerationOutput:
    """贪心生成的 token、最终缓存和停止状态。"""

    token_ids: torch.Tensor
    kv_cache: KVCache
    decode_calls: int
    eos_reached: bool


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
            raise ValueError("input_ids 的形状必须是 [batch, 非空序列]")
        input_ids = input_ids.to(self.device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(device=self.device, dtype=torch.long)
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask 的形状必须与 input_ids 相同")

        sequence_length = input_ids.shape[1]
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cache_position=torch.arange(sequence_length, device=self.device),
            use_cache=True,
            return_dict=True,
        )
        if outputs.past_key_values is None:
            raise RuntimeError("模型没有返回 past_key_values，请检查是否支持 use_cache")

        self._validate_cache(outputs.past_key_values, sequence_length, input_ids.shape[0])

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
            raise ValueError("Decode 的 input_ids 形状必须是 [batch, 1]")
        if input_ids.shape[0] != kv_cache.batch_size:
            raise ValueError("Decode 的 batch size 必须与 Prefill cache 一致")
        if kv_cache.device != self.device:
            raise ValueError("KV Cache 所属设备与当前 Runner 不一致")

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
            raise RuntimeError("模型没有返回更新后的 past_key_values")

        kv_cache.append(outputs.past_key_values, attention_mask)
        return DecodeOutput(logits=outputs.logits[:, -1, :], kv_cache=kv_cache)

    @staticmethod
    def _validate_cache(past_key_values: Any, sequence_length: int, batch_size: int) -> None:
        KVCache._validate_storage(past_key_values, sequence_length, batch_size)

    @torch.inference_mode()
    def generate_greedy(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        max_new_tokens: int = 32,
    ) -> GenerationOutput:
        """使用 argmax 贪心采样生成，并遵守 EOS 与最大 token 数限制。"""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens 必须大于或等于 0")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("当前朴素 Runner 的生成接口一次只处理一个请求")

        prefill = self.prefill(input_ids, attention_mask)
        cache = prefill.kv_cache
        if max_new_tokens == 0:
            return GenerationOutput(
                token_ids=input_ids[:, :0].to(self.device),
                kv_cache=cache,
                decode_calls=0,
                eos_reached=False,
            )

        eos_token_id = getattr(self.model.generation_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(self.model.config, "eos_token_id", None)
        eos_ids = set(eos_token_id if isinstance(eos_token_id, (list, tuple, set))
                      else ([] if eos_token_id is None else [eos_token_id]))

        token_ids: list[torch.Tensor] = []
        decode_calls = 0
        eos_reached = False
        next_token = prefill.logits.argmax(dim=-1, keepdim=True)
        for token_index in range(max_new_tokens):
            token_ids.append(next_token)
            eos_reached = int(next_token.item()) in eos_ids
            if eos_reached or token_index + 1 == max_new_tokens:
                break
            decoded = self.decode(next_token, cache)
            cache = decoded.kv_cache
            decode_calls += 1
            next_token = decoded.logits.argmax(dim=-1, keepdim=True)

        return GenerationOutput(
            token_ids=torch.cat(token_ids, dim=1),
            kv_cache=cache,
            decode_calls=decode_calls,
            eos_reached=eos_reached,
        )
