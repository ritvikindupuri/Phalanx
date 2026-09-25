"""
AI Security Guardrails & Prompt Injection Firewall
Inspects incoming prompts for jailbreak attempts, system override attacks,
and sensitive PII/secrets before forwarding requests to GPU/CPU compute.
"""

import re
from typing import Dict, Any, List, Tuple

# Patterns identifying adversarial prompt injection & jailbreak vectors
JAILBREAK_PATTERNS = [
    (re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions?", re.IGNORECASE), "Instruction Override Attack"),
    (re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+(rules|prompts|constraints)", re.IGNORECASE), "Constraint Disregard Attack"),
    (re.compile(r"you\s+are\s+now\s+(DAN|unfiltered|jailbroken|an\s+AI\s+without\s+rules)", re.IGNORECASE), "Persona Hijack / DAN Jailbreak"),
    (re.compile(r"(system\s+prompt|developer\s+mode)\s+(leak|reveal|output|display)", re.IGNORECASE), "System Prompt Extraction Probe"),
    (re.compile(r"(reveal|leak|display|output|show|print)\s+(the\s+)?(system\s+prompt|developer\s+mode|initial\s+instructions)", re.IGNORECASE), "System Prompt Extraction Probe"),
    (re.compile(r"base64\s+decode\s+and\s+execute", re.IGNORECASE), "Obfuscation Execution Vector"),
    (re.compile(r"<\s*script\s*>", re.IGNORECASE), "HTML/XSS Injection Vector")
]

# Sensitive secrets and PII patterns
SECRET_PATTERNS = [
    (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS Access Key ID"),
    (re.compile(r"ghp_[a-zA-Z0-9]{36}"), "GitHub Personal Access Token"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,15}\b"), "Email Address"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "US Social Security Number")
]


class GuardrailResult:
    def __init__(self, is_allowed: bool, reason: str, violations: List[str], sanitized_text: str):
        self.is_allowed = is_allowed
        self.reason = reason
        self.violations = violations
        self.sanitized_text = sanitized_text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_allowed": self.is_allowed,
            "reason": self.reason,
            "violations": self.violations,
            "sanitized_text": self.sanitized_text
        }


def evaluate_prompt_safety(text: str, mask_secrets: bool = True) -> GuardrailResult:
    """
    Evaluates prompt for adversarial injection and sensitive secret leakage.
    Returns GuardrailResult specifying whether to permit or block execution.
    """
    violations = []

    # 1. Check for Jailbreak / Prompt Injection
    for pattern, label in JAILBREAK_PATTERNS:
        if pattern.search(text):
            violations.append(f"Prompt Injection Detected: {label}")

    if violations:
        return GuardrailResult(
            is_allowed=False,
            reason="Blocked by AI Gateway Security Guardrail",
            violations=violations,
            sanitized_text=text
        )

    # 2. Check for PII / Secret leaks and optionally mask
    sanitized = text
    pii_found = []
    for pattern, label in SECRET_PATTERNS:
        matches = pattern.findall(sanitized)
        if matches:
            pii_found.append(f"Sensitive Data Detected: {label} ({len(matches)} instance(s))")
            if mask_secrets:
                sanitized = pattern.sub(f"[REDACTED_{label.upper().replace(' ', '_')}]", sanitized)

    return GuardrailResult(
        is_allowed=True,
        reason="Prompt approved for inference",
        violations=pii_found,
        sanitized_text=sanitized
    )
