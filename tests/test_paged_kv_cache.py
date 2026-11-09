"""第 10 天：验证分页 KV Reference Path 与连续 KV 的一致性。"""

from __future__ import annotations

import pytest
import torch

from deepinfer_lite import (
    BlockPool,
    BlockPoolExhaustedError,
    PagedKVCache,
    PagedKVCacheError,
)


def _synthetic_past(sequence_length: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """构造可追踪的连续 KV，Key 和 Value 使用不同偏移便于发现错位。"""
    layers = []
    for layer_index in range(2):
        key = torch.arange(1 * 2 * sequence_length * 4, dtype=torch.float32).reshape(
            1, 2, sequence_length, 4
        ) + layer_index * 10000
        value = key + 5000
        layers.append((key, value))
    return tuple(layers)


def _append_synthetic_token(
    past: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """保留已有 KV，仅构造并追加一个新的末尾 token。"""
    layers = []
    for layer_index, (key, value) in enumerate(past):
        next_key = torch.full_like(key[:, :, :1, :], 100000 + layer_index)
        next_value = torch.full_like(value[:, :, :1, :], 200000 + layer_index)
        layers.append((torch.cat((key, next_key), dim=2), torch.cat((value, next_value), dim=2)))
    return tuple(layers)


def _assert_same(left: tuple[tuple[torch.Tensor, torch.Tensor], ...], right: tuple[tuple[torch.Tensor, torch.Tensor], ...]) -> None:
    assert len(left) == len(right)
    for (left_key, left_value), (right_key, right_value) in zip(left, right):
        assert torch.equal(left_key, right_key)
        assert torch.equal(left_value, right_value)


@pytest.mark.parametrize("block_size", [2, 4, 8])
@pytest.mark.parametrize("sequence_length", [1, 2, 3, 7, 8, 9, 17])
def test_prefill_writes_and_reads_across_block_boundaries(block_size: int, sequence_length: int) -> None:
    """Prefill 跨越不同 Block 边界后，读取结果必须逐元素一致。"""
    contiguous = _synthetic_past(sequence_length)
    pool = BlockPool(num_blocks=32, block_size=block_size)
    cache = PagedKVCache.from_contiguous(contiguous, pool, "request-a")

    _assert_same(cache.to_contiguous(), contiguous)
    assert cache.sequence_length == sequence_length
    assert cache.block_table.logical_block_count() == (sequence_length + block_size - 1) // block_size

    cache.release()
    assert pool.free_count() == 32
    assert pool.allocated_count() == 0


def test_decode_appends_only_new_token_and_preserves_previous_blocks() -> None:
    """Decode 追加完整连续结果的最后一个 token，并正确跨越尾 Block。"""
    initial_length = 4
    full_length = initial_length + 1
    initial = _synthetic_past(initial_length)
    decoded = _append_synthetic_token(initial)
    pool = BlockPool(num_blocks=4, block_size=4)
    cache = PagedKVCache.from_contiguous(initial, pool, "request-a")

    cache.append_from_contiguous(decoded)

    _assert_same(cache.to_contiguous(), decoded)
    assert cache.sequence_length == full_length
    assert cache.block_table.blocks() == [0, 1]
    cache.release()
    assert pool.free_count() == 4


def test_decode_failure_does_not_change_paged_cache() -> None:
    """非法 Decode KV 不应修改序列长度、映射或物理内容。"""
    initial = _synthetic_past(3)
    invalid = _synthetic_past(5)
    pool = BlockPool(num_blocks=4, block_size=4)
    cache = PagedKVCache.from_contiguous(initial, pool, "request-a")
    before = cache.to_contiguous()
    before_blocks = cache.block_table.blocks()

    with pytest.raises(PagedKVCacheError, match="预期为 4"):
        cache.append_from_contiguous(invalid)

    _assert_same(cache.to_contiguous(), before)
    assert cache.sequence_length == 3
    assert cache.block_table.blocks() == before_blocks
    cache.release()


def test_decode_block_exhaustion_keeps_cache_unchanged() -> None:
    """尾 Block 满且 Pool 无空闲 Block 时，Decode 追加失败不改变缓存。"""
    initial = _synthetic_past(4)
    decoded = _synthetic_past(5)
    pool = BlockPool(num_blocks=1, block_size=4)
    cache = PagedKVCache.from_contiguous(initial, pool, "request-a")
    before = cache.to_contiguous()

    with pytest.raises(BlockPoolExhaustedError):
        cache.append_from_contiguous(decoded)

    _assert_same(cache.to_contiguous(), before)
    assert cache.sequence_length == 4
    assert cache.block_table.blocks() == [0]
    assert pool.allocated_count() == 1
    cache.release()
    assert pool.free_count() == 1


def test_prefill_allocation_failure_does_not_leak_blocks() -> None:
    """Prefill 申请 Block 失败时，已有请求和 Pool 状态保持完整。"""
    pool = BlockPool(num_blocks=1, block_size=4)
    initial = _synthetic_past(5)

    with pytest.raises(BlockPoolExhaustedError, match="Block 不足"):
        PagedKVCache.from_contiguous(initial, pool, "request-a")

    assert pool.free_count() == 1
    assert pool.allocated_count() == 0
    assert pool.allocation_count() == 0
