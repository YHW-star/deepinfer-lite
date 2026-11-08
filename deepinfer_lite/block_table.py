"""单请求逻辑 Block 到物理 Block 的映射。"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .block_pool import BlockPool, BlockPoolError


class BlockTableError(RuntimeError):
    """Block Table 操作的基类异常。"""


class BlockTableReleasedError(BlockTableError):
    """请求已经结束并释放了 Block。"""


@dataclass(frozen=True)
class BlockTableSnapshot:
    """单请求 Block Table 的只读状态。"""

    request_id: str
    sequence_length: int
    block_size: int
    logical_blocks: int
    allocated_tokens: int
    internal_waste_tokens: int
    released: bool

    def as_dict(self) -> dict[str, int | str | bool]:
        """返回适合写入 JSON 的字典。"""
        return {
            "request_id": self.request_id,
            "sequence_length": self.sequence_length,
            "block_size": self.block_size,
            "logical_blocks": self.logical_blocks,
            "allocated_tokens": self.allocated_tokens,
            "internal_waste_tokens": self.internal_waste_tokens,
            "released": self.released,
        }


class BlockTable:
    """维护一个请求的逻辑 block 到物理 block 映射。"""

    def __init__(
        self,
        pool: BlockPool,
        request_id: str,
        sequence_length: int = 0,
    ) -> None:
        if not request_id:
            raise ValueError("request_id 不能为空")
        if not isinstance(sequence_length, int) or isinstance(sequence_length, bool) or sequence_length < 0:
            raise ValueError("sequence_length 必须是大于等于 0 的整数")
        self.pool = pool
        self.request_id = request_id
        self.sequence_length = sequence_length
        self._block_ids: list[int] = []
        self._released = False
        required_blocks = self._required_blocks(sequence_length)
        if required_blocks:
            self._block_ids = pool.allocate(required_blocks, request_id=request_id)

    def append_token(self) -> list[int]:
        """追加一个 token，必要时申请新的物理 Block。"""
        return self.append_tokens(1)

    def append_tokens(self, num_tokens: int) -> list[int]:
        """追加 token，并返回本次新申请的物理 Block ID。"""
        self._ensure_active()
        if not isinstance(num_tokens, int) or isinstance(num_tokens, bool) or num_tokens < 1:
            raise ValueError("追加 token 数必须是大于等于 1 的整数")
        target_length = self.sequence_length + num_tokens
        required_blocks = self._required_blocks(target_length)
        new_block_count = required_blocks - len(self._block_ids)
        if new_block_count > 0:
            if self._block_ids:
                new_block_ids = self.pool.extend(new_block_count, self.request_id)
            else:
                new_block_ids = self.pool.allocate(new_block_count, self.request_id)
            self._block_ids.extend(new_block_ids)
        self.sequence_length = target_length
        return list(self._block_ids[-new_block_count:]) if new_block_count else []

    def release(self) -> list[int]:
        """释放该请求的全部物理 Block，并标记请求结束。"""
        self._ensure_active()
        block_ids = list(self._block_ids)
        if block_ids:
            self.pool.release(self.request_id)
        self._block_ids.clear()
        self._released = True
        return block_ids

    def logical_to_physical(self, logical_block_id: int) -> int:
        """查询逻辑 block 对应的物理 Block ID。"""
        self._ensure_active()
        if not isinstance(logical_block_id, int) or isinstance(logical_block_id, bool):
            raise TypeError("logical_block_id 必须是整数")
        if logical_block_id < 0 or logical_block_id >= len(self._block_ids):
            raise IndexError(f"逻辑 Block {logical_block_id} 超出范围")
        return self._block_ids[logical_block_id]

    def physical_for_token(self, token_position: int) -> int:
        """查询 token 位置所在的物理 Block。"""
        self._ensure_active()
        if not isinstance(token_position, int) or isinstance(token_position, bool):
            raise TypeError("token_position 必须是整数")
        if token_position < 0 or token_position >= self.sequence_length:
            raise IndexError(f"token 位置 {token_position} 超出范围")
        return self._block_ids[token_position // self.pool.block_size]

    def blocks(self) -> list[int]:
        """返回当前逻辑到物理的映射副本。"""
        self._ensure_active()
        return list(self._block_ids)

    def logical_block_count(self) -> int:
        """返回当前逻辑 block 数量。"""
        self._ensure_active()
        return len(self._block_ids)

    def allocated_tokens(self) -> int:
        """返回已分配 block 能容纳的 token 数。"""
        self._ensure_active()
        return len(self._block_ids) * self.pool.block_size

    def internal_waste_tokens(self) -> int:
        """返回尾部 block 尚未使用的 token 容量。"""
        self._ensure_active()
        return self.allocated_tokens() - self.sequence_length

    def snapshot(self) -> BlockTableSnapshot:
        """返回当前请求状态快照。"""
        if self._released:
            return BlockTableSnapshot(
                request_id=self.request_id,
                sequence_length=self.sequence_length,
                block_size=self.pool.block_size,
                logical_blocks=0,
                allocated_tokens=0,
                internal_waste_tokens=0,
                released=True,
            )
        return BlockTableSnapshot(
            request_id=self.request_id,
            sequence_length=self.sequence_length,
            block_size=self.pool.block_size,
            logical_blocks=self.logical_block_count(),
            allocated_tokens=self.allocated_tokens(),
            internal_waste_tokens=self.internal_waste_tokens(),
            released=self._released,
        )

    def _required_blocks(self, sequence_length: int) -> int:
        return ceil(sequence_length / self.pool.block_size)

    def _ensure_active(self) -> None:
        if self._released:
            raise BlockTableReleasedError(f"请求 {self.request_id!r} 已经释放")
