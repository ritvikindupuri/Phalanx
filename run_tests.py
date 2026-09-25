"""
Master Test Runner: Executes all unit, integration, and security test suites.
Produces clean, rich console diagnostics for CI/CD and developer verification.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def run_test_module(name: str, fn):
    t0 = time.perf_counter()
    try:
        fn()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return True, f"{elapsed_ms:.1f} ms", None
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return False, f"{elapsed_ms:.1f} ms", str(e)


def main():
    console.print(Panel.fit(
        "[bold cyan]Phalanx: Automated Test Suite Runner[/bold cyan]\n"
        "[dim]Running all Unit, Memory Paging, Scheduling, and AI Infrastructure Security Tests[/dim]",
        border_style="cyan"
    ))

    # Import test suites
    from tests.test_block_manager import test_block_allocation_and_free, test_shared_prefix_blocks_refcounting
    from tests.test_prefix_cache import test_prefix_matching_and_caching
    from tests.test_scheduler import test_continuous_batching_lifecycle
    from tests.test_ai_security import (
        test_prompt_injection_defense,
        test_pii_and_secret_redaction,
        test_malicious_model_deserialization_scanner
    )

    test_cases = [
        # Systems & Memory Management
        ("PagedAttention: Virtual Block Allocation & Free", test_block_allocation_and_free, "Core Memory"),
        ("PagedAttention: Copy-on-Write & Ref Counting", test_shared_prefix_blocks_refcounting, "Core Memory"),
        ("Prefix Cache: Radix Tree Prefix Matching", test_prefix_matching_and_caching, "Core Memory"),
        ("Scheduler: Iteration-Level Continuous Batching", test_continuous_batching_lifecycle, "Scheduling"),
        
        # AI Infrastructure Security & Supply Chain
        ("Gateway Firewall: Prompt Injection & Jailbreak Defense", test_prompt_injection_defense, "AI Security"),
        ("Gateway Privacy: Secret & PII Auto-Redaction", test_pii_and_secret_redaction, "AI Security"),
        ("Supply Chain: Malicious Pickle Opcode Detection", test_malicious_model_deserialization_scanner, "AI Security"),
    ]

    table = Table(title="Test Execution Results", show_header=True, header_style="bold magenta")
    table.add_column("Test Suite", style="bold")
    table.add_column("Subsystem", style="dim")
    table.add_column("Status", justify="center")
    table.add_column("Execution Time", justify="right")
    table.add_column("Diagnostic")

    passed_count = 0
    failed_count = 0

    for name, fn, subsystem in test_cases:
        passed, duration, err = run_test_module(name, fn)
        if passed:
            passed_count += 1
            status_str = "[bold green]PASS[/bold green]"
            diag_str = "[green]All assertions verified[/green]"
        else:
            failed_count += 1
            status_str = "[bold red]FAIL[/bold red]"
            diag_str = f"[red]{err}[/red]"

        table.add_row(name, subsystem, status_str, duration, diag_str)

    console.print("\n")
    console.print(table)

    summary_color = "green" if failed_count == 0 else "red"
    console.print(f"\n[bold {summary_color}]Results: {passed_count} passed, {failed_count} failed out of {len(test_cases)} tests.[/bold {summary_color}]\n")

    if failed_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
