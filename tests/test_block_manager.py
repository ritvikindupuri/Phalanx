"""
Unit Tests for PagedAttention Virtual Memory Manager
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from vllm_core.block_manager import BlockManager, BlockTable


def test_block_allocation_and_free():
    # 4 blocks, block_size = 4 (Total 16 token slots)
    mgr = BlockManager(num_gpu_blocks=4, block_size=4)
    assert mgr.gpu_pool.get_num_free_blocks() == 4

    # Allocate sequence of 9 tokens -> requires 3 blocks (4 + 4 + 1)
    table = mgr.allocate_sequence("req_1", prompt_len=9)
    assert len(table.physical_block_ids) == 3
    assert mgr.gpu_pool.get_num_free_blocks() == 1
    assert table.last_block_capacity == 3  # 12 allocated slots - 9 tokens = 3 remaining

    # Append 3 slots without needing a new physical block
    for _ in range(3):
        assert mgr.can_append_slot("req_1") is True
        b_id = mgr.append_slot("req_1")
        assert b_id == table.physical_block_ids[-1]

    assert table.last_block_capacity == 0  # Block full! (12 tokens)

    # Next append requires 1 new block (uses the last free block)
    assert mgr.can_append_slot("req_1") is True
    mgr.append_slot("req_1")  # Token 13
    assert len(table.physical_block_ids) == 4
    assert mgr.gpu_pool.get_num_free_blocks() == 0

    # Block 4 still has 3 slots remaining
    assert mgr.can_append_slot("req_1") is True
    mgr.append_slot("req_1")  # Token 14
    mgr.append_slot("req_1")  # Token 15
    mgr.append_slot("req_1")  # Token 16 (Block 4 is now completely full)
    assert table.last_block_capacity == 0

    # Now all 4 blocks are 100% full and free list is empty -> cannot append!
    assert mgr.can_append_slot("req_1") is False

    # Free sequence -> all 4 blocks return to free list
    mgr.free_sequence("req_1")
    assert mgr.gpu_pool.get_num_free_blocks() == 4
    assert len(mgr.block_tables) == 0


def test_shared_prefix_blocks_refcounting():
    mgr = BlockManager(num_gpu_blocks=4, block_size=4)

    # Sequence 1 allocates 2 blocks
    t1 = mgr.allocate_sequence("req_1", prompt_len=8)
    shared_blocks = t1.physical_block_ids
    assert mgr.gpu_pool.get_num_free_blocks() == 2

    # Sequence 2 shares the same prefix blocks
    t2 = mgr.allocate_sequence("req_2", prompt_len=8, shared_blocks=shared_blocks)
    # No new blocks needed, shared blocks retained
    assert mgr.gpu_pool.get_num_free_blocks() == 2
    assert mgr.gpu_pool.blocks[shared_blocks[0]].ref_count == 2

    # Freeing Sequence 1 does not free the physical blocks because Sequence 2 still holds them
    mgr.free_sequence("req_1")
    assert mgr.gpu_pool.get_num_free_blocks() == 2
    assert mgr.gpu_pool.blocks[shared_blocks[0]].ref_count == 1

    # Freeing Sequence 2 finally returns them to free pool
    mgr.free_sequence("req_2")
    assert mgr.gpu_pool.get_num_free_blocks() == 4
    print("[PASS] test_shared_prefix_blocks_refcounting passed!")


if __name__ == "__main__":
    test_block_allocation_and_free()
    print("[PASS] test_block_allocation_and_free passed!")
    test_shared_prefix_blocks_refcounting()
    print("All block manager tests passed successfully!")

