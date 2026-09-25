"""
Serving Benchmark: Real Concurrent LLM Inference Telemetry
Measures TTFT (Time-To-First-Token), ITL (Inter-Token Latency), and Tokens/Sec under concurrent load.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
from rich.console import Console
from rich.table import Table

from vllm_core.engine import NanoLLMEngine

console = Console()


def run_serving_benchmark(model_name: str = "gpt2", num_requests: int = 8, output_len: int = 16):
    console.print(f"\n[bold cyan]=== Live Serving Benchmark: {model_name} ===[/bold cyan]\n")
    console.print(f"Concurrent requests: {num_requests} | Output tokens per request: {output_len}")

    engine = NanoLLMEngine(
        model_name_or_path=model_name,
        num_blocks=256,
        block_size=16,
        max_batch_size=8,
        device="cpu"
    )

    test_prompts = [
        "In artificial intelligence infrastructure, high throughput serving requires",
        "The key-value cache memory bottleneck in large language models occurs because",
        "Continuous batching eliminates latency bubbles by dynamically scheduling",
        "Speculative decoding utilizes a lightweight draft model to predict future tokens",
        "Operating system virtual memory paging inspired PagedAttention to eliminate",
        "Tensor parallelism splits weight matrices across multiple computing units to",
        "FlashAttention optimizes memory bandwidth by computing attention in SRAM tiles",
        "Automatic prefix caching saves redundant computation by indexing token blocks"
    ]

    # Enqueue requests
    for i in range(num_requests):
        prompt = test_prompts[i % len(test_prompts)]
        engine.add_request(f"bench_req_{i}", prompt, max_new_tokens=output_len, temperature=0.7)

    # Telemetry collectors
    step_latencies = []
    total_tokens_generated = 0
    start_time = time.perf_counter()

    console.print("[*] Processing continuous batch requests through PyTorch engine...")

    while engine.has_unfinished_requests():
        step_out = engine.step()
        num_new = len(step_out.new_tokens)
        total_tokens_generated += num_new
        if step_out.step_latency_ms > 0:
            step_latencies.append(step_out.step_latency_ms)

    total_time_s = time.perf_counter() - start_time
    throughput_tokens_per_sec = total_tokens_generated / total_time_s if total_time_s > 0 else 0.0

    # Collect TTFT and E2E latencies
    ttfts = [req.ttft_ms for req in engine.requests_map.values() if req.ttft_ms is not None]
    e2es = [req.total_latency_ms for req in engine.requests_map.values() if req.total_latency_ms is not None]

    p50_ttft = np.percentile(ttfts, 50) if ttfts else 0.0
    p90_ttft = np.percentile(ttfts, 90) if ttfts else 0.0
    p99_ttft = np.percentile(ttfts, 99) if ttfts else 0.0

    p50_e2e = np.percentile(e2es, 50) if e2es else 0.0
    p90_e2e = np.percentile(e2es, 90) if e2es else 0.0

    avg_step_ms = np.mean(step_latencies) if step_latencies else 0.0

    # Output Rich Table
    table = Table(title=f"Serving Benchmark Results ({model_name} on CPU)", show_header=True, header_style="bold magenta")
    table.add_column("Benchmark Metric", style="dim")
    table.add_column("Measured Value", justify="right")

    table.add_row("Total Requests Processed", f"{num_requests}")
    table.add_row("Total Output Tokens Generated", f"{total_tokens_generated}")
    table.add_row("Total Elapsed Serving Time", f"{total_time_s:.2f} s")
    table.add_row("System Serving Throughput", f"[bold green]{throughput_tokens_per_sec:.2f} tokens/sec[/bold green]")
    table.add_row("Average Inter-Token Latency (ITL)", f"{avg_step_ms:.1f} ms / step")
    table.add_row("Time-To-First-Token (TTFT) P50", f"{p50_ttft:.1f} ms")
    table.add_row("Time-To-First-Token (TTFT) P90", f"{p90_ttft:.1f} ms")
    table.add_row("Time-To-First-Token (TTFT) P99", f"{p99_ttft:.1f} ms")
    table.add_row("End-to-End Latency P50", f"{p50_e2e:.1f} ms")
    table.add_row("End-to-End Latency P90", f"{p90_e2e:.1f} ms")

    console.print(table)


if __name__ == "__main__":
    run_serving_benchmark()
