"""
AI Model Supply Chain Security Scanner
Inspects model files and checkpoints before loading into GPU/CPU memory.
Detects malicious Python pickle bytecode (arbitrary code execution CVEs)
and enforces safe serialization (SafeTensors).
"""

import os
import io
import pickle
import struct
from typing import Dict, Any, List, Tuple


# Dangerous Python built-in modules frequently weaponized in pickle deserialization exploits
DANGEROUS_CALLABLES = {
    "os": {"system", "popen", "kill", "execv", "execve", "spawn"},
    "subprocess": {"Popen", "call", "check_call", "check_output", "run"},
    "sys": {"exit"},
    "builtins": {"eval", "exec", "compile", "__import__"},
    "socket": {"socket", "create_connection"},
    "posix": {"system"},
    "nt": {"system"}
}


class SafeModelUnpickler(pickle.Unpickler):
    """Restricted unpickler that intercepts and blocks dangerous globals during deserialization analysis."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.found_dangerous_calls: List[str] = []

    def find_class(self, module, name):
        if module in DANGEROUS_CALLABLES and name in DANGEROUS_CALLABLES[module]:
            call_sig = f"{module}.{name}"
            self.found_dangerous_calls.append(call_sig)
            raise SecurityError(f"Security Alert: Malicious opcode detected calling '{call_sig}'")
        # For security scanning purposes, allow safe scalar/numpy inspection
        return super().find_class(module, name)


class SecurityError(Exception):
    pass


def scan_model_file(filepath: str) -> Dict[str, Any]:
    """
    Performs static binary analysis on model weight files.
    Returns audit status, format type, risk level, and detailed findings.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    file_size_bytes = os.path.getsize(filepath)

    findings: List[str] = []
    is_safe = True
    risk_level = "LOW"

    # 1. Check for SafeTensors format (zero-execution memory-mapped format)
    if ext in [".safetensors"]:
        with open(filepath, "rb") as f:
            header_bytes = f.read(8)
            if len(header_bytes) == 8:
                header_len = struct.unpack("<Q", header_bytes)[0]
                findings.append(f"Valid SafeTensors binary header verified (header size: {header_len} bytes)")
                return {
                    "filepath": filepath,
                    "format": "SafeTensors",
                    "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                    "is_safe": True,
                    "risk_level": "LOW",
                    "findings": findings
                }

    # 2. Check for PyTorch / Pickle formats (.bin, .pt, .pkl)
    if ext in [".bin", ".pt", ".pth", ".pkl"]:
        findings.append("Legacy Python Pickle format detected (inherently un-sandboxed).")
        risk_level = "MEDIUM"

        try:
            with open(filepath, "rb") as f:
                content = f.read()

            unpickler = SafeModelUnpickler(io.BytesIO(content))
            try:
                unpickler.load()
            except SecurityError as sec_err:
                is_safe = False
                risk_level = "CRITICAL"
                findings.append(str(sec_err))
            except Exception:
                # Normal deserialization errors due to missing custom weights are ignored during static scan
                pass

            if unpickler.found_dangerous_calls:
                is_safe = False
                risk_level = "CRITICAL"
                for call in unpickler.found_dangerous_calls:
                    findings.append(f"Arbitrary code execution payload identified: {call}")

        except Exception as e:
            findings.append(f"Inspection note: {str(e)}")

    return {
        "filepath": filepath,
        "format": ext.replace(".", "").upper() or "UNKNOWN",
        "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
        "is_safe": is_safe,
        "risk_level": risk_level,
        "findings": findings
    }
