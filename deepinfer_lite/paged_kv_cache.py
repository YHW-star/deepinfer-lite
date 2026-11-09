"""基于 Block Pool/Block Table 的分页 KV Cache Reference Path。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import torch

from .block_pool import BlockPool
from .block_table import BlockTable
from .model_runner import DecodeOutput, KVCache, ModelRunner, PrefillOutput


class PagedKVCacheError(RuntimeError):
    """分页 KV Cache 操作的基类异常。"""


@dataclass
class PagedKVStorage:
    """所有请求共享的物理 KV Block Tensor。"""

    key_blocks: tuple[torch.Tensor, ...]
    value_blocks: tuple[torch.Tensor, ...]
    block_size: int
    num_blocks: int

    @classmethod
    def allocate(
        cls,
        pool: BlockPool,
        pairs: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> "PagedKVStorage":
        """根据一份连续 KV 的形状，为整个 Pool 分配物理 Block Tensor。"""
        if not pairs:
            raise ValueError("至少需要一层 Key/Value Tensor")
        key_blocks: list[torch.Tensor] = []
        value_blocks: list[torch.Tensor] = []
        for layer_index, (key, value) in enumerate(pairs):
            if key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1:
                raise PagedKVCacheError(
                    f"第 {layer_index} 层 KV 必须是形状 [1, heads, seq, dim] 且 Key/Value 一致"
                )
            if key.device != value.device or key.dtype != value.dtype:
                raise PagedKVCacheError("Key/Value 必须位于同一设备且使用相同 dtype")
            shape = (pool.num_blocks, pool.block_size, key.shape[1], key.shape[3])
            key_blocks.append(torch.empty(shape, dtype=key.dtype, device=key.device))
            value_blocks.append(torch.empty(shape, dtype=value.dtype, device=value.device))
        return cls(tuple(key_blocks), tuple(value_blocks), pool.block_size, pool.num_blocks)


@dataclass
class PagedKVCache:
    """单请求分页 KV Cache，底层物理存储由 PagedKVStorage 提供。"""

    pool: BlockPool
    block_table: BlockTable
    storage: PagedKVStorage
    attention_mask: torch.Tensor

    @classmethod
    def from_contiguous(
        cls,
        past_key_values: Any,
        pool: BlockPool,
        request_id: str,
        *,
        storage: PagedKVStorage | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> "PagedKVCache":
        """将 Prefill 返回的连续 KV 按 token 写入分页 Block。"""
        pairs = KVCache._key_value_pairs(past_key_values)
        if not pairs:
            raise PagedKVCacheError("Prefill 没有返回任何 Key/Value Tensor")
        sequence_length = pairs[0][0].shape[-2]
        if sequence_length < 1:
            raise ValueError("KV Cache 的序列长度必须大于 0")
        for layer_index, (key, value) in enumerate(pairs):
            if key.shape[-2] != sequence_length or value.shape[-2] != sequence_length:
                raise PagedKVCacheError(f"第 {layer_index} 层 KV 序列长度不一致")
            if key.shape[0] != 1 or value.shape[0] != 1:
                raise PagedKVCacheError("当前分页 Reference Path 只支持 batch size=1")
            if key.shape != value.shape or key.device != value.device or key.dtype != value.dtype:
                raise PagedKVCacheError(f"第 {layer_index} 层 Key/Value 元数据不一致")
        if storage is None:
            storage = PagedKVStorage.allocate(pool, pairs)
        cls._validate_storage_shape(storage, pairs, pool)

        table = BlockTable(pool, request_id, sequence_length=sequence_length)
        if attention_mask is None:
            attention_mask = torch.ones((1, sequence_length), dtype=torch.long, device=pairs[0][0].device)
        if attention_mask.shape != (1, sequence_length):
            table.release()
            raise ValueError("attention_mask 形状必须是 [1, sequence_length]")
        cache = cls(pool=pool, block_table=table, storage=storage, attention_mask=attention_mask)
        try:
            cache._write_contiguous(pairs, start=0, source_start=0, num_tokens=sequence_length)
        except Exception:
            table.release()
            raise
        return cache

    @property
    def sequence_length(self) -> int:
        """返回当前请求已经写入的 token 数。"""
        return self.block_table.sequence_length

    @property
    def block_size(self) -> int:
        """返回物理 Block 的 token 容量。"""
        return self.pool.block_size

    @property
    def device(self) -> torch.device:
        """返回物理 KV Tensor 所在设备。"""
        return self.storage.key_blocks[0].device

    @property
    def dtype(self) -> torch.dtype:
        """返回物理 KV Tensor 的 dtype。"""
        return self.storage.key_blocks[0].dtype

    def append_from_contiguous(self, past_key_values: Any) -> None:
        """接收 Decode 后的完整连续 KV，只追加最后一个 token。"""
        pairs = KVCache._key_value_pairs(past_key_values)
        if len(pairs) != len(self.storage.key_blocks):
            raise PagedKVCacheError("Decode 返回的层数与分页存储不一致")
        expected_length = self.sequence_length + 1
        for layer_index, (key, value) in enumerate(pairs):
            if key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1:
                raise PagedKVCacheError(f"第 {layer_index} 层 Decode KV 形状非法")
            if key.shape[-2] != expected_length or value.shape[-2] != expected_length:
                raise PagedKVCacheError(
                    f"Decode KV 长度为 {key.shape[-2]}，预期为 {expected_length}"
                )
            if key.device != self.device or value.device != self.device:
                raise PagedKVCacheError(f"第 {layer_index} 层 Decode KV 设备不匹配")
            if key.dtype != self.storage.key_blocks[layer_index].dtype:
                raise PagedKVCacheError(f"第 {layer_index} 层 Decode Key dtype 不匹配")
            if value.dtype != self.storage.value_blocks[layer_index].dtype:
                raise PagedKVCacheError(f"第 {layer_index} 层 Decode Value dtype 不匹配")
            if key.shape[1] != self.storage.key_blocks[layer_index].shape[2]:
                raise PagedKVCacheError(f"第 {layer_index} 层 Decode KV head 数不匹配")
            if key.shape[3] != self.storage.key_blocks[layer_index].shape[3]:
                raise PagedKVCacheError(f"第 {layer_index} 层 Decode KV head_dim 不匹配")
        position = self.sequence_length
        self.block_table.append_token()
        self._write_contiguous(
            pairs,
            start=position,
            source_start=expected_length - 1,
            num_tokens=1,
        )
        self.attention_mask = torch.cat(
            (self.attention_mask, torch.ones((1, 1), dtype=self.attention_mask.dtype, device=self.attention_mask.device)),
            dim=1,
        )

    def to_contiguous(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """按逻辑 token 顺序读取并还原连续 KV。"""
        block_ids = self.block_table.blocks()
        pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for key_blocks, value_blocks in zip(self.storage.key_blocks, self.storage.value_blocks):
            key_chunks: list[torch.Tensor] = []
            value_chunks: list[torch.Tensor] = []
            remaining = self.sequence_length
            for physical_block in block_ids:
                count = min(remaining, self.block_size)
                if count == 0:
                    break
                key_chunks.append(key_blocks[physical_block, :count].permute(1, 0, 2))
                value_chunks.append(value_blocks[physical_block, :count].permute(1, 0, 2))
                remaining -= count
            key = torch.cat(key_chunks, dim=1).unsqueeze(0).contiguous()
            value = torch.cat(value_chunks, dim=1).unsqueeze(0).contiguous()
            pairs.append((key, value))
        return tuple(pairs)

    def release(self) -> list[int]:
        """释放请求占用的全部物理 Block。"""
        return self.block_table.release()

    def _write_contiguous(
        self,
        pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        start: int,
        source_start: int,
        num_tokens: int,
    ) -> None:
        """把连续 KV 的 token 区间写入 Block Table 指向的物理位置。"""
        pairs_list = list(pairs)
        end = start + num_tokens
        if end > self.sequence_length:
            raise PagedKVCacheError("写入范围超过当前 Block Table 的序列长度")
        for layer_index, (key, value) in enumerate(pairs_list):
            key_blocks = self.storage.key_blocks[layer_index]
            value_blocks = self.storage.value_blocks[layer_index]
            for target_position in range(start, end):
                source_position = source_start + target_position - start
                physical_block = self.block_table.physical_for_token(target_position)
                offset = target_position % self.block_size
                key_blocks[physical_block, offset].copy_(key[0, :, source_position, :])
                value_blocks[physical_block, offset].copy_(value[0, :, source_position, :])

    @staticmethod
    def _validate_storage_shape(
        storage: PagedKVStorage,
        pairs: list[tuple[torch.Tensor, torch.Tensor]],
        pool: BlockPool,
    ) -> None:
        if storage.block_size != pool.block_size or storage.num_blocks != pool.num_blocks:
            raise PagedKVCacheError("物理存储与 Block Pool 的容量配置不一致")
        if len(storage.key_blocks) != len(pairs) or len(storage.value_blocks) != len(pairs):
            raise PagedKVCacheError("物理存储层数与 KV 层数不一致")
        for layer_index, (key, value) in enumerate(pairs):
            expected_shape = (pool.num_blocks, pool.block_size, key.shape[1], key.shape[3])
            if storage.key_blocks[layer_index].shape != expected_shape:
                raise PagedKVCacheError(f"第 {layer_index} 层 Key 物理 Block 形状不一致")
            if storage.value_blocks[layer_index].shape != expected_shape:
                raise PagedKVCacheError(f"第 {layer_index} 层 Value 物理 Block 形状不一致")
            if storage.key_blocks[layer_index].dtype != key.dtype or storage.value_blocks[layer_index].dtype != value.dtype:
                raise PagedKVCacheError(f"第 {layer_index} 层物理 KV dtype 不一致")
            if storage.key_blocks[layer_index].device != key.device or storage.value_blocks[layer_index].device != value.device:
                raise PagedKVCacheError(f"第 {layer_index} 层物理 KV 设备不一致")


@dataclass
class PagedPrefillOutput:
    """分页 Prefill 的 logits 和缓存句柄。"""

    logits: torch.Tensor
    kv_cache: PagedKVCache


@dataclass
class PagedDecodeOutput:
    """分页 Decode 的 logits 和更新后的缓存句柄。"""

    logits: torch.Tensor
    kv_cache: PagedKVCache


class PagedModelRunner:
    """将现有 ModelRunner 接到分页 KV Reference Path。"""

    def __init__(
        self,
        runner: ModelRunner,
        pool: BlockPool,
        *,
        storage: PagedKVStorage | None = None,
    ) -> None:
        self.runner = runner
        self.pool = pool
        self.storage = storage

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        request_id: str,
        attention_mask: torch.Tensor | None = None,
    ) -> PagedPrefillOutput:
        """执行连续模型 Prefill，并将全部 KV 写入分页存储。"""
        result: PrefillOutput = self.runner.prefill(input_ids, attention_mask)
        paged_cache = PagedKVCache.from_contiguous(
            result.kv_cache.past_key_values,
            self.pool,
            request_id,
            storage=self.storage,
            attention_mask=result.kv_cache.attention_mask,
        )
        self.storage = paged_cache.storage
        return PagedPrefillOutput(result.logits, paged_cache)

    @torch.inference_mode()
    def decode(self, input_ids: torch.Tensor, paged_cache: PagedKVCache) -> PagedDecodeOutput:
        """还原连续 KV 执行单 token Decode，再将新增 KV 写回分页存储。"""
        contiguous_cache = KVCache(
            past_key_values=paged_cache.to_contiguous(),
            sequence_length=paged_cache.sequence_length,
            batch_size=1,
            device=paged_cache.device,
            attention_mask=paged_cache.attention_mask,
        )
        result: DecodeOutput = self.runner.decode(input_ids, contiguous_cache)
        paged_cache.append_from_contiguous(result.kv_cache.past_key_values)
        return PagedDecodeOutput(result.logits, paged_cache)
