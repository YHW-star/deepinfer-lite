"""固定大小 KV Block 的申请、释放和生命周期管理。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable


class BlockPoolError(RuntimeError):
    """Block Pool 操作的基类异常。"""


class BlockPoolExhaustedError(BlockPoolError):
    """可用 block 数不足。"""

    def __init__(self, requested: int, available: int) -> None:
        self.requested = requested
        self.available = available
        super().__init__(f"Block 不足：需要 {requested} 个，当前只有 {available} 个可用")


class DuplicateAllocationError(BlockPoolError):
    """同一个请求重复申请了一组 block。"""


class BlockDoubleFreeError(BlockPoolError):
    """释放了未知或已经释放的 block。"""


class BlockOwnershipError(BlockPoolError):
    """请求释放了不属于自己的 block。"""


@dataclass(frozen=True)
class BlockPoolSnapshot:
    """Block Pool 的只读状态快照。"""

    num_blocks: int
    block_size: int
    free_blocks: int
    allocated_blocks: int
    allocation_count: int

    def as_dict(self) -> dict[str, int]:
        """返回适合写入 JSON 的字典。"""
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "free_blocks": self.free_blocks,
            "allocated_blocks": self.allocated_blocks,
            "allocation_count": self.allocation_count,
        }


class BlockPool:
    """管理固定数量、固定 token 容量的物理 Block ID。

    当前类只管理 Block 的 ID 和生命周期，不持有 GPU Tensor。第 10 天接入 KV
    写入时，可将 Block ID 映射到预分配的物理 KV Tensor。
    """

    def __init__(self, num_blocks: int, block_size: int) -> None:
        if not isinstance(num_blocks, int) or isinstance(num_blocks, bool) or num_blocks < 1:
            raise ValueError("num_blocks 必须是大于等于 1 的整数")
        if not isinstance(block_size, int) or isinstance(block_size, bool) or block_size < 1:
            raise ValueError("block_size 必须是大于等于 1 的整数")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free_blocks: deque[int] = deque(range(num_blocks))
        self._allocated_blocks: dict[str, list[int]] = {}
        self._block_owners: dict[int, str] = {}
        self._anonymous_id = 0

    def allocate(self, num_blocks: int = 1, request_id: str | None = None) -> list[int]:
        """为一个请求申请连续逻辑序列对应数量的物理 Block ID。

        物理 ID 不要求连续；返回顺序表示逻辑 Block 到物理 Block 的映射顺序。
        申请失败时不修改 Pool 状态。
        """
        if not isinstance(num_blocks, int) or isinstance(num_blocks, bool) or num_blocks < 1:
            raise ValueError("申请数量必须是大于等于 1 的整数")
        owner = request_id if request_id is not None else self._new_anonymous_id()
        if owner in self._allocated_blocks:
            raise DuplicateAllocationError(f"请求 {owner!r} 已经持有 Block")
        available = len(self._free_blocks)
        if num_blocks > available:
            raise BlockPoolExhaustedError(num_blocks, available)

        block_ids = [self._free_blocks.popleft() for _ in range(num_blocks)]
        self._allocated_blocks[owner] = block_ids
        self._block_owners.update({block_id: owner for block_id in block_ids})
        return list(block_ids)

    def release(self, request_id: str) -> list[int]:
        """释放一个请求持有的全部 Block，并返回被释放的物理 ID。"""
        if request_id not in self._allocated_blocks:
            raise BlockDoubleFreeError(f"请求 {request_id!r} 没有可释放的 Block")
        block_ids = list(self._allocated_blocks[request_id])
        self.release_blocks(block_ids, request_id=request_id)
        return block_ids

    def extend(self, num_blocks: int, request_id: str) -> list[int]:
        """为已经持有 Block 的请求追加物理 Block。"""
        if not isinstance(num_blocks, int) or isinstance(num_blocks, bool) or num_blocks < 1:
            raise ValueError("追加数量必须是大于等于 1 的整数")
        if request_id not in self._allocated_blocks:
            raise BlockDoubleFreeError(f"请求 {request_id!r} 没有可扩展的 Block")
        available = len(self._free_blocks)
        if num_blocks > available:
            raise BlockPoolExhaustedError(num_blocks, available)

        block_ids = [self._free_blocks.popleft() for _ in range(num_blocks)]
        self._allocated_blocks[request_id].extend(block_ids)
        self._block_owners.update({block_id: request_id for block_id in block_ids})
        return list(block_ids)

    def release_blocks(self, block_ids: Iterable[int], request_id: str | None = None) -> None:
        """释放一部分或全部 Block，先完成校验再修改状态。"""
        ids = list(block_ids)
        if not ids:
            raise ValueError("至少需要释放一个 Block")
        if len(set(ids)) != len(ids):
            raise BlockDoubleFreeError("同一批释放请求中包含重复 Block")
        for block_id in ids:
            owner = self._block_owners.get(block_id)
            if owner is None:
                raise BlockDoubleFreeError(f"Block {block_id} 未分配或已经释放")
            if request_id is not None and owner != request_id:
                raise BlockOwnershipError(
                    f"Block {block_id} 属于请求 {owner!r}，不能由请求 {request_id!r} 释放"
                )

        owners = {self._block_owners[block_id] for block_id in ids}
        if request_id is None and len(owners) != 1:
            raise BlockOwnershipError("一次 release_blocks 只能释放同一个请求的 Block")
        for owner in owners:
            remaining = [
                block_id
                for block_id in self._allocated_blocks[owner]
                if block_id not in ids
            ]
            if remaining:
                self._allocated_blocks[owner] = remaining
            else:
                del self._allocated_blocks[owner]
        for block_id in ids:
            del self._block_owners[block_id]
        self._free_blocks.extend(sorted(ids))

    def free_count(self) -> int:
        """返回当前可申请的 Block 数量。"""
        return len(self._free_blocks)

    def allocated_count(self) -> int:
        """返回当前已经分配的 Block 数量。"""
        return len(self._block_owners)

    def allocation_count(self) -> int:
        """返回当前持有 Block 的请求数量。"""
        return len(self._allocated_blocks)

    def blocks_for(self, request_id: str) -> list[int]:
        """查询请求当前持有的 Block 映射，返回副本。"""
        if request_id not in self._allocated_blocks:
            raise BlockDoubleFreeError(f"请求 {request_id!r} 没有已分配的 Block")
        return list(self._allocated_blocks[request_id])

    def owner_of(self, block_id: int) -> str | None:
        """查询 Block 所属请求；空闲 Block 返回 None。"""
        return self._block_owners.get(block_id)

    def is_allocated(self, block_id: int) -> bool:
        """判断 Block 是否正在使用。"""
        return block_id in self._block_owners

    def snapshot(self) -> BlockPoolSnapshot:
        """返回当前资源状态快照。"""
        return BlockPoolSnapshot(
            num_blocks=self.num_blocks,
            block_size=self.block_size,
            free_blocks=self.free_count(),
            allocated_blocks=self.allocated_count(),
            allocation_count=self.allocation_count(),
        )

    def _new_anonymous_id(self) -> str:
        while True:
            self._anonymous_id += 1
            candidate = f"__anonymous_{self._anonymous_id}"
            if candidate not in self._allocated_blocks:
                return candidate
