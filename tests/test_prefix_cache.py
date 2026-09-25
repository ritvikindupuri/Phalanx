import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from vllm_core.prefix_cache import PrefixCache


def test_prefix_matching_and_caching():
    cache = PrefixCache(block_size=4)

    system_prompt_tokens = [101, 102, 103, 104, 201, 202, 203, 204]  # 2 full blocks of 4
    assigned_blocks = [42, 99]

    # First lookup: Miss
    matched_blocks, matched_tokens = cache.match_prefix(system_prompt_tokens)
    assert matched_blocks == []
    assert matched_tokens == 0
    assert cache.cache_misses == 1

    # Insert into prefix tree
    cache.insert_prefix(system_prompt_tokens, assigned_blocks)
    assert cache.total_cached_blocks == 2

    # Second lookup with identical prefix: Full Hit
    matched_blocks, matched_tokens = cache.match_prefix(system_prompt_tokens + [301, 302])
    assert matched_blocks == [42, 99]
    assert matched_tokens == 8
    assert cache.cache_hits == 1

    # Third lookup with partial 1-block match
    partial_tokens = [101, 102, 103, 104, 999, 999, 999, 999]
    matched_blocks, matched_tokens = cache.match_prefix(partial_tokens)
    assert matched_blocks == [42]
    assert matched_tokens == 4
    assert cache.cache_hits == 2
    print("[PASS] test_prefix_matching_and_caching passed!")


if __name__ == "__main__":
    test_prefix_matching_and_caching()
    print("All prefix cache tests passed successfully!")

