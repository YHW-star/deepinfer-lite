"""第 8 天：验证固定 Block Pool 的申请、释放和错误处理。"""

from __future__ import annotations

import pytest

from deepinfer_lite import (
    BlockDoubleFreeError,
    BlockOwnershipError,
    BlockPool,
    BlockPoolExhaustedError,
    DuplicateAllocationError,
)


def test_all_blocks_can_be_allocated_and_released() -> None:
    pool = BlockPool(num_blocks=8, block_size=16)

    allocated = pool.allocate(8, request_id="request-1")

    assert allocated == list(range(8))
    assert pool.free_count() == 0
    assert pool.allocated_count() == 8
    assert pool.allocation_count() == 1
    assert pool.snapshot().as_dict() == {
        "num_blocks": 8,
        "block_size": 16,
        "free_blocks": 0,
        "allocated_blocks": 8,
        "allocation_count": 1,
    }

    assert pool.release("request-1") == allocated
    assert pool.free_count() == 8
    assert pool.allocated_count() == 0
    assert pool.allocation_count() == 0


def test_interleaved_allocate_release_reuses_freed_blocks() -> None:
    pool = BlockPool(num_blocks=6, block_size=8)
    request_a = pool.allocate(2, request_id="a")
    request_b = pool.allocate(2, request_id="b")

    pool.release("a")
    request_c = pool.allocate(2, request_id="c")

    assert request_a == [0, 1]
    assert request_b == [2, 3]
    assert request_c == [0, 1]
    assert pool.blocks_for("b") == [2, 3]
    assert pool.owner_of(0) == "c"
    assert pool.owner_of(4) is None
    assert pool.free_count() == 2

    pool.release("b")
    pool.release("c")
    assert pool.snapshot().free_blocks == 6


def test_insufficient_capacity_is_atomic() -> None:
    pool = BlockPool(num_blocks=3, block_size=16)
    pool.allocate(2, request_id="a")
    before = pool.snapshot()

    with pytest.raises(BlockPoolExhaustedError, match="需要 2 个"):
        pool.allocate(2, request_id="b")

    assert pool.snapshot() == before
    assert pool.blocks_for("a") == [0, 1]


def test_request_larger_than_total_capacity_is_rejected() -> None:
    pool = BlockPool(num_blocks=3, block_size=16)

    with pytest.raises(BlockPoolExhaustedError, match="需要 4 个，当前只有 3 个可用"):
        pool.allocate(4, request_id="too-large")

    assert pool.free_count() == 3
    assert pool.allocated_count() == 0


def test_duplicate_allocation_and_double_release_are_rejected() -> None:
    pool = BlockPool(num_blocks=4, block_size=8)
    pool.allocate(1, request_id="a")

    with pytest.raises(DuplicateAllocationError, match="已经持有"):
        pool.allocate(1, request_id="a")
    with pytest.raises(BlockDoubleFreeError, match="没有可释放"):
        pool.release("missing")
    pool.release("a")
    with pytest.raises(BlockDoubleFreeError, match="没有可释放"):
        pool.release("a")


def test_partial_release_checks_owner_and_preserves_mapping() -> None:
    pool = BlockPool(num_blocks=5, block_size=8)
    block_ids = pool.allocate(3, request_id="a")

    with pytest.raises(BlockOwnershipError, match="不能由请求"):
        pool.release_blocks([block_ids[0]], request_id="b")
    assert pool.blocks_for("a") == block_ids

    pool.release_blocks([block_ids[1]], request_id="a")
    assert pool.blocks_for("a") == [block_ids[0], block_ids[2]]
    assert pool.free_count() == 3
    pool.release("a")
    assert pool.allocated_count() == 0


def test_anonymous_allocation_can_be_released_by_block_ids() -> None:
    pool = BlockPool(num_blocks=3, block_size=8)
    block_ids = pool.allocate(2)

    pool.release_blocks(block_ids)

    assert pool.free_count() == 3
    assert pool.allocation_count() == 0


def test_invalid_release_batch_does_not_partially_free_blocks() -> None:
    pool = BlockPool(num_blocks=4, block_size=8)
    block_ids = pool.allocate(2, request_id="a")

    with pytest.raises(BlockDoubleFreeError, match="未分配或已经释放"):
        pool.release_blocks([block_ids[0], 3], request_id="a")
    with pytest.raises(BlockDoubleFreeError, match="包含重复 Block"):
        pool.release_blocks([block_ids[0], block_ids[0]], request_id="a")

    assert pool.blocks_for("a") == block_ids
    assert pool.allocated_count() == 2
    assert pool.free_count() == 2


@pytest.mark.parametrize(
    ("num_blocks", "block_size"),
    [(0, 8), (-1, 8), (4, 0), (4, -1)],
)
def test_pool_configuration_must_be_positive(num_blocks: int, block_size: int) -> None:
    with pytest.raises(ValueError):
        BlockPool(num_blocks=num_blocks, block_size=block_size)
