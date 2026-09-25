"""
Automatic Prefix Caching (Radix Tree / LRU Trie)
Enables instantaneous TTFT (Time-To-First-Token) by reusing pre-computed
KV-cache physical blocks for common system prompts and conversational histories.
"""

from typing import List, Tuple, Dict, Optional, Any
import time


class PrefixTreeNode:
    def __init__(self, token_chunk: Tuple[int, ...], block_id: int):
        self.token_chunk = token_chunk  # Chunk of tokens corresponding to 1 physical block
        self.block_id = block_id
        self.children: Dict[Tuple[int, ...], 'PrefixTreeNode'] = {}
        self.last_accessed: float = time.time()
        self.access_count: int = 1


class PrefixCache:
    """
    Radix-tree based prefix cache matching vLLM & SGLang's Automatic Prefix Caching.
    Maps token prefix sequences directly to reusable physical block IDs.
    """
    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.root: Dict[Tuple[int, ...], PrefixTreeNode] = {}
        self.total_cached_blocks: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0

    def match_prefix(self, token_ids: List[int]) -> Tuple[List[int], int]:
        """
        Finds the longest sequence of cached physical blocks matching the input token sequence.
        Returns: (matching_block_ids, matched_token_count)
        """
        matched_blocks: List[int] = []
        matched_tokens = 0
        curr_children = self.root

        # Chunk input tokens by block_size
        num_full_chunks = len(token_ids) // self.block_size
        for i in range(num_full_chunks):
            chunk = tuple(token_ids[i * self.block_size : (i + 1) * self.block_size])
            if chunk in curr_children:
                node = curr_children[chunk]
                node.last_accessed = time.time()
                node.access_count += 1
                matched_blocks.append(node.block_id)
                matched_tokens += self.block_size
                curr_children = node.children
            else:
                break

        if matched_tokens > 0:
            self.cache_hits += 1
        else:
            self.cache_misses += 1

        return matched_blocks, matched_tokens

    def insert_prefix(self, token_ids: List[int], physical_block_ids: List[int]):
        """
        Inserts newly computed physical blocks into the prefix tree for future request reuse.
        """
        num_chunks = min(len(token_ids) // self.block_size, len(physical_block_ids))
        if num_chunks == 0:
            return

        curr_children = self.root
        for i in range(num_chunks):
            chunk = tuple(token_ids[i * self.block_size : (i + 1) * self.block_size])
            block_id = physical_block_ids[i]

            if chunk not in curr_children:
                new_node = PrefixTreeNode(chunk, block_id)
                curr_children[chunk] = new_node
                self.total_cached_blocks += 1
                curr_children = new_node.children
            else:
                curr_node = curr_children[chunk]
                curr_node.last_accessed = time.time()
                curr_children = curr_node.children

    def get_stats(self) -> Dict[str, Any]:
        total_queries = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_queries * 100.0) if total_queries > 0 else 0.0
        return {
            "cached_blocks": self.total_cached_blocks,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_pct": round(hit_rate, 2)
        }
