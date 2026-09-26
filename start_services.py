"""
Phalanx Service Orchestrator:
Launches the Inference Core (port 8000), AI Security Gateway (port 8080),
and Standalone Prometheus Server (port 9090).
"""

import sys
import os
import time
import subprocess
import requests

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROM_EXE = os.path.join(BASE_DIR, "prometheus_bin", "prometheus-3.15.0.windows-amd64", "prometheus.exe")
PROM_CONFIG = os.path.join(BASE_DIR, "prometheus_config.yml")

def main():
    print("=" * 70)
    print("  Phalanx + Prometheus Service Orchestrator")
    print("=" * 70)

    # 1. Start Inference Core (:8000)
    print("[1/3] Starting Phalanx Inference Engine on http://localhost:8000 ...")
    engine_proc = subprocess.Popen(
        [sys.executable, "-m", "server.app", "--port", "8000"],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    # Wait for engine to initialize
    print("      Waiting for model weights to load...")
    engine_ready = False
    for _ in range(40):
        try:
            r = requests.get("http://localhost:8000/v1/models", timeout=2)
            if r.status_code == 200:
                engine_ready = True
                print("      [+] Inference Engine is READY on :8000!")
                break
        except Exception:
            pass
        time.sleep(1)

    if not engine_ready:
        print("[!] Warning: Engine startup timed out. Proceeding anyway...")

    # 2. Start AI Security Gateway (:8080)
    print("[2/3] Starting AI Security Gateway on http://localhost:8080 ...")
    gateway_proc = subprocess.Popen(
        [sys.executable, "-m", "gateway.proxy", "--port", "8080", "--engine-url", "http://localhost:8000"],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    gateway_ready = False
    for _ in range(15):
        try:
            r = requests.get("http://localhost:8080/healthz", timeout=1)
            if r.status_code == 200:
                gateway_ready = True
                print("      [+] AI Security Gateway is READY on :8080!")
                break
        except Exception:
            pass
        time.sleep(0.5)

    # 3. Start Prometheus Server (:9090)
    print("[3/3] Starting Prometheus Server on http://localhost:9090 ...")
    prom_proc = subprocess.Popen(
        [PROM_EXE, f"--config.file={PROM_CONFIG}", "--web.listen-address=0.0.0.0:9090"],
        cwd=BASE_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    prom_ready = False
    for _ in range(15):
        try:
            r = requests.get("http://localhost:9090/-/ready", timeout=1)
            if r.status_code == 200:
                prom_ready = True
                print("      [+] Prometheus Web UI is READY on http://localhost:9090!")
                break
        except Exception:
            pass
        time.sleep(0.5)

    print("\n" + "=" * 70)
    print("  ALL SERVICES ARE RUNNING!")
    print("  - AI Security Gateway: http://localhost:8080")
    print("  - Gateway Prometheus Metrics: http://localhost:8080/metrics")
    print("  - Inference Core Engine: http://localhost:8000")
    print("  - Engine Prometheus Metrics: http://localhost:8000/metrics")
    print("  - Prometheus Web UI / Dashboard: http://localhost:9090")
    print("=" * 70)

    # Keep running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services...")
        engine_proc.terminate()
        gateway_proc.terminate()
        prom_proc.terminate()

if __name__ == "__main__":
    main()
