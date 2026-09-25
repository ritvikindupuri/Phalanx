"""
Block Manager: PagedAttention Virtual Memory Manager for KV-Cache
Implements virtual memory paging, block tables, physical memory pools,
reference-counted shared blocks (Copy-on-Write), and preemption eviction.
"""

from typing import List, Dict, Optional, Set
from dataclasses import dataclass


@dataclass
class PhysicalBlock:
    block_id: int
    block_size: int
    ref_count: int = 0
    is_allocated: bool = False
    prefix_hash: Optional[int] = None  # Used for prefix caching


class BlockTable:
    """Logical to physical block mapping for a single request sequence."""
    def __init__(self, request_id: str, block_size: int):
        self.request_id = request_id
        self.block_size = block_size
        self.physical_block_ids: List[int] = []
        self.num_tokens: int = 0

    @property
    def num_blocks(self) -> int:
        return len(self.physical_block_ids)

    @property
    def last_block_capacity(self) -> int:
        remainder = self.num_tokens % self.block_size
        return (self.block_size - remainder) if remainder != 0 else 0


class MemoryPool:
    """
    Pre-allocated Physical KV Cache Pool.
    Manages physical blocks, free lists, reference counting, and utilization metrics.
    """
    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks: Dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_id=i, block_size=block_size) for i in range(num_blocks)
        }
        self.free_block_ids: List[int] = list(range(num_blocks))
        self.allocated_block_ids: Set[int] = set()

    def get_num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def get_memory_utilization(self) -> float:
        return 1.0 - (len(self.free_block_ids) / self.num_blocks)

    def allocate_block(self, prefix_hash: Optional[int] = None) -> Optional[int]:
        if not self.free_block_ids:
            return None
        block_id = self.free_block_ids.pop(0)
        block = self.blocks[block_id]
        block.is_allocated = True
        block.ref_count = 1
        block.prefix_hash = prefix_hash
        self.allocated_block_ids.add(block_id)
        return block_id

    def retain_block(self, block_id: int):
        """Increments reference counter for shared prefix blocks (Copy-on-Write)."""
        block = self.blocks[block_id]
        block.ref_count += 1

    def free_block(self, block_id: int):
        block = self.blocks[block_id]
        if block.ref_count > 0:
            block.ref_count -= 1
        
        if block.ref_count == 0:
            block.is_allocated = False
            block.prefix_hash = None
            if block_id in self.allocated_block_ids:
                self.allocated_block_ids.remove(block_id)
            self.free_block_ids.append(block_id)


class BlockManager:
    """
    OS-style Virtual Memory Manager for LLM KV Cache.
    Translates per-sequence logical token offsets to physical memory slots.
    """
    def __init__(self, num_gpu_blocks: int = 256, block_size: int = 16):
        self.block_size = block_size
        self.gpu_pool = MemoryPool(num_blocks=num_gpu_blocks, block_size=block_size)
        self.block_tables: Dict[str, BlockTable] = {}

    def can_allocate(self, seq_len: int) -> bool:
        """Determines if enough physical blocks exist for a sequence."""
        needed_blocks = (seq_len + self.block_size - 1) // self.block_size
        return self.gpu_pool.get_num_free_blocks() >= needed_blocks

    def can_append_slot(self, request_id: str) -> bool:
        """Checks if a decode step can write a new token (checks if current block is full)."""
        block_table = self.block_tables[request_id]
        # If last block still has room, we don't need a new physical block
        if block_table.last_block_capacity > 0:
            return True
        # Otherwise we need at least 1 free physical block
        return self.gpu_pool.get_num_free_blocks() >= 1

    def allocate_sequence(self, request_id: str, prompt_len: int, shared_blocks: Optional[List[int]] = None) -> BlockTable:
        """Allocates physical blocks for prompt prefill, reusing shared prefix blocks if provided."""
        table = BlockTable(request_id, self.block_size)
        table.num_tokens = prompt_len

        # Re-use prefix blocks if available
        if shared_blocks:
            for b_id in shared_blocks:
                self.gpu_pool.retain_block(b_id)
                table.physical_block_ids.append(b_id)

        tokens_already_covered = len(table.physical_block_ids) * self.block_size
        remaining_tokens = max(0, prompt_len - tokens_already_covered)
        needed_new_blocks = (remaining_tokens + self.block_size - 1) // self.block_size

        for _ in range(needed_new_blocks):
            b_id = self.gpu_pool.allocate_block()
            if b_id is None:
                # OOM - roll back
                self.free_sequence(request_id)
                raise MemoryError(f"KV Cache OOM: Out of physical blocks for sequence {request_id}")
            table.physical_block_ids.append(b_id)

        self.block_tables[request_id] = table
        return table

    def append_slot(self, request_id: str) -> Optional[int]:
        """
        Allocates space for 1 new decoded token.
        Returns the physical block ID where the token is stored.
        """
        table = self.block_tables[request_id]
        if table.last_block_capacity == 0:
            # Need a new physical block
            b_id = self.gpu_pool.allocate_block()
            if b_id is None:
                return None  # Preemption needed
            table.physical_block_ids.append(b_id)
        
        table.num_tokens += 1
        return table.physical_block_ids[-1]

    def free_sequence(self, request_id: str):
        """Reclaims all physical memory blocks owned by a finished sequence."""
        if request_id not in self.block_tables:
            return
        table = self.block_tables.pop(request_id)
        for b_id in table.physical_block_ids:
            self.gpu_pool.free_block(b_id)

    def get_memory_stats(self) -> Dict[str, float]:
        """Provides real-time KV cache memory metrics."""
        total = self.gpu_pool.num_blocks
        free = self.gpu_pool.get_num_free_blocks()
        used = total - free
        utilization = (used / total) * 100.0 if total > 0 else 0.0

        # Calculate virtual fragmentation vs contiguous allocation
        # In contiguous allocation, maximum sequence allocation leads to severe internal waste
        total_tokens_stored = sum(t.num_tokens for t in self.block_tables.values())
        allocated_capacity_tokens = used * self.block_size
        internal_waste = (allocated_capacity_tokens - total_tokens_stored) if allocated_capacity_tokens > 0 else 0
        waste_pct = (internal_waste / allocated_capacity_tokens * 100.0) if allocated_capacity_tokens > 0 else 0.0

        return {
            "total_blocks": total,
            "used_blocks": used,
            "free_blocks": free,
            "utilization_pct": round(utilization, 2),
            "active_sequences": len(self.block_tables),
            "tokens_stored": total_tokens_stored,
            "internal_waste_pct": round(waste_pct, 2)
        }
