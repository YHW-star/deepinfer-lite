"""第 9 天：验证逻辑 Block 到物理 Block 的映射和资源回收。"""

from __future__ import annotations

import random

import pytest

from deepinfer_lite import BlockPool, BlockPoolExhaustedError, BlockTable, BlockTableReleasedError


def test_append_allocates_new_block_at_boundary() -> None:
    pool = BlockPool(num_blocks=4, block_size=4)
    table = BlockTable(pool, "request-a")

    assert table.append_tokens(1) == [0]
    assert table.append_tokens(3) == []
    assert table.append_token() == [1]
    assert table.sequence_length == 5
    assert table.blocks() == [0, 1]
    assert table.logical_to_physical(0) == 0
    assert table.logical_to_physical(1) == 1
    assert table.physical_for_token(0) == 0
    assert table.physical_for_token(3) == 0
    assert table.physical_for_token(4) == 1
    assert table.internal_waste_tokens() == 3


def test_initial_length_allocates_required_mapping() -> None:
    pool = BlockPool(num_blocks=5, block_size=8)
    table = BlockTable(pool, "request-a", sequence_length=17)

    assert table.blocks() == [0, 1, 2]
    assert table.logical_block_count() == 3
    assert table.allocated_tokens() == 24
    assert table.internal_waste_tokens() == 7
    assert table.snapshot().as_dict()["sequence_length"] == 17


def test_release_returns_all_blocks_and_rejects_future_access() -> None:
    pool = BlockPool(num_blocks=4, block_size=4)
    table = BlockTable(pool, "request-a", sequence_length=9)
    allocated = table.blocks()

    assert table.release() == allocated
    assert pool.free_count() == 4
    assert table.snapshot().released is True
    assert table.snapshot().logical_blocks == 0
    assert table.snapshot().allocated_tokens == 0
    with pytest.raises(BlockTableReleasedError):
        table.append_token()
    with pytest.raises(BlockTableReleasedError):
        table.release()
    with pytest.raises(BlockTableReleasedError):
        table.logical_block_count()


def test_empty_request_can_finish_without_allocating_blocks() -> None:
    pool = BlockPool(num_blocks=2, block_size=8)
    table = BlockTable(pool, "empty")

    assert table.release() == []
    assert pool.snapshot().free_blocks == 2


def test_random_100_request_lifecycles_restore_pool_capacity() -> None:
    randomizer = random.Random(42)
    pool = BlockPool(num_blocks=64, block_size=8)
    initial_free = pool.free_count()

    for request_index in range(100):
        table = BlockTable(pool, f"request-{request_index}")
        target_length = randomizer.randint(0, 128)
        while table.sequence_length < target_length:
            step = min(randomizer.randint(1, 9), target_length - table.sequence_length)
            table.append_tokens(step)

        assert table.sequence_length == target_length
        assert table.logical_block_count() == (target_length + 7) // 8
        table.release()
        assert pool.free_count() == initial_free
        assert pool.allocated_count() == 0
        assert pool.allocation_count() == 0


def test_mapping_bounds_are_checked() -> None:
    pool = BlockPool(num_blocks=2, block_size=4)
    table = BlockTable(pool, "request-a", sequence_length=4)

    with pytest.raises(IndexError):
        table.logical_to_physical(1)
    with pytest.raises(IndexError):
        table.physical_for_token(4)


def test_growth_fails_atomically_when_pool_is_exhausted() -> None:
    pool = BlockPool(num_blocks=2, block_size=4)
    table = BlockTable(pool, "request-a", sequence_length=4)
    pool.allocate(1, request_id="request-b")
    before = table.snapshot()

    with pytest.raises(BlockPoolExhaustedError, match="Block 不足"):
        table.append_token()

    assert table.snapshot() == before
    assert pool.blocks_for("request-a") == [0]
    assert pool.blocks_for("request-b") == [1]
