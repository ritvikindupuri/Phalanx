"""
Continuous (Iteration-Level) Batching Scheduler with Chunked Prefill & Preemption
Eliminates batch latency bubbles by scheduling requests dynamically at every decoding step.
"""

from enum import Enum
from typing import List, Dict, Optional, Tuple, Deque
from collections import deque
import time
from vllm_core.block_manager import BlockManager
from vllm_core.prefix_cache import PrefixCache


class RequestStatus(Enum):
    WAITING = "WAITING"
    RUNNING = "RUNNING"
    PREEMPTED = "PREEMPTED"
    FINISHED = "FINISHED"


class SequenceRequest:
    def __init__(
        self,
        request_id: str,
        prompt_tokens: List[int],
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        arrival_time: Optional[float] = None
    ):
        self.request_id = request_id
        self.prompt_tokens = prompt_tokens
        self.output_tokens: List[int] = []
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.status = RequestStatus.WAITING
        self.arrival_time = arrival_time or time.time()
        self.first_token_time: Optional[float] = None
        self.completion_time: Optional[float] = None
        self.num_prefilled_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_tokens) + len(self.output_tokens)

    @property
    def is_finished(self) -> bool:
        return len(self.output_tokens) >= self.max_new_tokens or self.status == RequestStatus.FINISHED

    @property
    def ttft_ms(self) -> Optional[float]:
        """Time-To-First-Token in milliseconds."""
        if self.first_token_time:
            return (self.first_token_time - self.arrival_time) * 1000.0
        return None

    @property
    def total_latency_ms(self) -> Optional[float]:
        if self.completion_time:
            return (self.completion_time - self.arrival_time) * 1000.0
        return None


class ScheduleOutput:
    def __init__(
        self,
        prefill_requests: List[SequenceRequest],
        decode_requests: List[SequenceRequest],
        preempted_requests: List[SequenceRequest]
    ):
        self.prefill_requests = prefill_requests
        self.decode_requests = decode_requests
        self.preempted_requests = preempted_requests

    @property
    def has_work(self) -> bool:
        return bool(self.prefill_requests or self.decode_requests)


class Scheduler:
    """
    Orca/vLLM style Iteration-Level Scheduler.
    Manages waiting, running, and preempted queues with KV-cache awareness.
    """
    def __init__(
        self,
        block_manager: BlockManager,
        prefix_cache: PrefixCache,
        max_batch_size: int = 16,
        max_budget_tokens_per_step: int = 2048,
        chunked_prefill_size: int = 512
    ):
        self.block_manager = block_manager
        self.prefix_cache = prefix_cache
        self.max_batch_size = max_batch_size
        self.max_budget_tokens_per_step = max_budget_tokens_per_step
        self.chunked_prefill_size = chunked_prefill_size

        self.waiting_queue: Deque[SequenceRequest] = deque()
        self.running_queue: List[SequenceRequest] = []
        self.preempted_queue: Deque[SequenceRequest] = deque()

        # Telemetry
        self.total_completed_requests = 0
        self.total_preemptions = 0

    def add_request(self, request: SequenceRequest):
        self.waiting_queue.append(request)

    def schedule(self) -> ScheduleOutput:
        """
        Executes one scheduling iteration:
        1. Allocates next slots for currently decoding requests.
        2. Preempts low-priority requests if KV-cache is congested.
        3. Admits new waiting requests into prefill batch if budget permits.
        """
        decode_requests: List[SequenceRequest] = []
        preempted_requests: List[SequenceRequest] = []
        prefill_requests: List[SequenceRequest] = []

        budget_remaining = self.max_budget_tokens_per_step

        # Step 1: Schedule running decode requests
        still_running = []
        for req in self.running_queue:
            if req.is_finished:
                self._finish_request(req)
                continue

            # Check if KV-cache can allocate 1 token slot
            if not self.block_manager.can_append_slot(req.request_id):
                # Preemption triggered!
                self._preempt_request(req)
                preempted_requests.append(req)
                continue

            # Append memory slot
            self.block_manager.append_slot(req.request_id)
            decode_requests.append(req)
            still_running.append(req)
            budget_remaining -= 1

        self.running_queue = still_running

        # Step 2: Try re-admitting preempted requests first (FIFO recovery)
        while self.preempted_queue and len(self.running_queue) < self.max_batch_size and budget_remaining > 32:
            req = self.preempted_queue[0]
            if self.block_manager.can_allocate(req.total_tokens):
                self.preempted_queue.popleft()
                self.block_manager.allocate_sequence(req.request_id, req.total_tokens)
                req.status = RequestStatus.RUNNING
                self.running_queue.append(req)
                decode_requests.append(req)
                budget_remaining -= 1
            else:
                break

        # Step 3: Admit new waiting requests (Prefill phase)
        while self.waiting_queue and len(self.running_queue) < self.max_batch_size and budget_remaining > 0:
            req = self.waiting_queue[0]
            prompt_len = len(req.prompt_tokens)

            # Check Prefix Cache
            shared_blocks, matched_tokens = self.prefix_cache.match_prefix(req.prompt_tokens)

            tokens_to_compute = prompt_len - matched_tokens
            if tokens_to_compute > budget_remaining:
                # Exceeds current step budget; chunk or wait for next iteration
                break

            if not self.block_manager.can_allocate(prompt_len):
                # KV cache full; stop scheduling new requests this iteration
                break

            # Admit request
            self.waiting_queue.popleft()
            block_table = self.block_manager.allocate_sequence(
                req.request_id, prompt_len, shared_blocks=shared_blocks
            )
            # Store in prefix cache for future requests
            self.prefix_cache.insert_prefix(req.prompt_tokens, block_table.physical_block_ids)

            req.status = RequestStatus.RUNNING
            req.num_prefilled_tokens = prompt_len
            self.running_queue.append(req)
            prefill_requests.append(req)
            budget_remaining -= tokens_to_compute

        return ScheduleOutput(
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
            preempted_requests=preempted_requests
        )

    def _finish_request(self, request: SequenceRequest):
        request.status = RequestStatus.FINISHED
        request.completion_time = time.time()
        self.block_manager.free_sequence(request.request_id)
        self.total_completed_requests += 1

    def _preempt_request(self, request: SequenceRequest):
        request.status = RequestStatus.PREEMPTED
        self.block_manager.free_sequence(request.request_id)
        self.preempted_queue.append(request)
        self.total_preemptions += 1
