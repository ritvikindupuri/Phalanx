import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from vllm_core.block_manager import BlockManager
from vllm_core.prefix_cache import PrefixCache
from vllm_core.scheduler import Scheduler, SequenceRequest, RequestStatus


def test_continuous_batching_lifecycle():
    block_mgr = BlockManager(num_gpu_blocks=16, block_size=4)
    prefix_cache = PrefixCache(block_size=4)
    scheduler = Scheduler(block_manager=block_mgr, prefix_cache=prefix_cache, max_batch_size=2)

    req1 = SequenceRequest("r1", prompt_tokens=[1, 2, 3], max_new_tokens=2)
    req2 = SequenceRequest("r2", prompt_tokens=[4, 5], max_new_tokens=2)
    req3 = SequenceRequest("r3", prompt_tokens=[6, 7, 8], max_new_tokens=2)

    scheduler.add_request(req1)
    scheduler.add_request(req2)
    scheduler.add_request(req3)

    # Step 1: Schedule first batch (max_batch_size = 2, so r1 and r2 admitted into prefill)
    out1 = scheduler.schedule()
    assert len(out1.prefill_requests) == 2
    assert len(out1.decode_requests) == 0
    assert len(scheduler.waiting_queue) == 1  # r3 waiting

    # Simulate token generation for r1 and r2
    req1.output_tokens.append(10)
    req2.output_tokens.append(20)

    # Step 2: Next iteration -> r1 and r2 are now decode requests
    out2 = scheduler.schedule()
    assert len(out2.decode_requests) == 2
    assert len(out2.prefill_requests) == 0

    # Simulate r1 completing
    req1.output_tokens.append(11)  # max_new_tokens reached!
    req2.output_tokens.append(21)

    # Step 3: r1 finishes and frees its blocks; r3 is admitted into running batch dynamically!
    out3 = scheduler.schedule()
    assert req1.status == RequestStatus.FINISHED
    assert len(out3.prefill_requests) == 1
    assert out3.prefill_requests[0].request_id == "r3"
    print("[PASS] test_continuous_batching_lifecycle passed!")


if __name__ == "__main__":
    test_continuous_batching_lifecycle()
    print("All scheduler tests passed successfully!")

