"""
Live End-to-End Demo: Continuous Batching & PagedAttention with Real Hugging Face Model
Streams concurrent prompts through the iteration-level scheduler with physical KV-cache allocations.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live

from vllm_core.engine import NanoLLMEngine

console = Console()


def run_live_demo(model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", num_blocks: int = 256, block_size: int = 16):
    console.print(Panel.fit(
        "[bold cyan]nano-vLLM: High-Throughput LLM Serving Engine[/bold cyan]\n"
        f"Model: [bold green]{model_name}[/bold green] | PagedAttention KV-Pool: [bold yellow]{num_blocks} blocks x {block_size} slots[/bold yellow]\n"
        "[dim]Real PyTorch execution, iteration-level continuous batching, and zero mock data.[/dim]",
        border_style="cyan"
    ))

    # Initialize real engine
    engine = NanoLLMEngine(
        model_name_or_path=model_name,
        num_blocks=num_blocks,
        block_size=block_size,
        max_batch_size=8,
        device="cpu"
    )

    # Real concurrent conversational prompts
    prompts = [
        ("req_1", "The fundamental architecture of modern AI infrastructure relies on", 24),
        ("req_2", "In distributed systems, the primary bottleneck for LLM inference is", 28),
        ("req_3", "PagedAttention eliminates KV cache memory fragmentation by", 24),
        ("req_4", "Operating system virtual memory paging inspired", 20)
    ]

    console.print("\n[bold yellow][*] Enqueuing 4 concurrent requests into continuous batching queue...[/bold yellow]\n")
    for req_id, prompt, max_tokens in prompts:
        engine.add_request(req_id, prompt, max_new_tokens=max_tokens, temperature=0.7)

    # Tracking metrics
    iteration = 0
    start_time = time.perf_counter()
    total_tokens_generated = 0

    console.print("[bold green][*] Starting Continuous Batch Iteration Loop...[/bold green]\n")

    while engine.has_unfinished_requests():
        iteration += 1
        step_out = engine.step()

        num_new = len(step_out.new_tokens)
        total_tokens_generated += num_new

        mem = step_out.memory_stats
        console.print(
            f"[bold cyan]Step {iteration:02d}[/bold cyan] | "
            f"Active Batch: {mem['active_sequences']} reqs | "
            f"KV Blocks: {mem['used_blocks']}/{mem['total_blocks']} ({mem['utilization_pct']}%) | "
            f"Step Latency: {step_latency_ms:.1f}ms" if (step_latency_ms := step_out.step_latency_ms) else ""
        )

        for req_id, (token_id, token_str) in step_out.new_tokens.items():
            clean_str = repr(token_str)
            console.print(f"  [dim]-> [{req_id}][/dim] Generated: [green]{clean_str}[/green]")

        if step_out.finished_requests:
            for req_id in step_out.finished_requests:
                console.print(f"  [bold magenta][DONE] Finished: {req_id}[/bold magenta]")

    elapsed = time.perf_counter() - start_time
    throughput = total_tokens_generated / elapsed if elapsed > 0 else 0.0

    console.print("\n[bold cyan]=== Final Serving Telemetry ===[/bold cyan]\n")

    summary_table = Table(title="Inference Engine Telemetry", show_header=True, header_style="bold magenta")
    summary_table.add_column("Serving Metric", style="dim")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("Model Evaluated", model_name)
    summary_table.add_row("Total Generation Time", f"{elapsed:.2f} s")
    summary_table.add_row("Total Output Tokens", str(total_tokens_generated))
    summary_table.add_row("Throughput (Tokens/Sec)", f"[bold green]{throughput:.2f} tok/s[/bold green]")
    summary_table.add_row("Continuous Iterations", str(iteration))
    summary_table.add_row("KV Memory Virtual Blocks Used", f"{mem['used_blocks']} / {mem['total_blocks']}")
    console.print(summary_table)

    console.print("\n[bold cyan]Completed Request Outputs:[/bold cyan]\n")
    for req_id, req in engine.requests_map.items():
        full_text = engine.tokenizer.decode(req.prompt_tokens + req.output_tokens)
        ttft_str = f"{req.ttft_ms:.1f} ms" if req.ttft_ms else "N/A"
        console.print(Panel(f"[bold]{full_text}[/bold]\n\n[dim]TTFT: {ttft_str} | Generated Tokens: {len(req.output_tokens)}[/dim]", title=req_id, border_style="green"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-vLLM Live Inference Demo")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model name")
    parser.add_argument("--blocks", type=int, default=256, help="Number of physical KV blocks")
    parser.add_argument("--block-size", type=int, default=16, help="Tokens per block")
    args = parser.parse_args()

    run_live_demo(model_name=args.model, num_blocks=args.blocks, block_size=args.block_size)
