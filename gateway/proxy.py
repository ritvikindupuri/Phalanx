"""
Zero-Trust AI Gateway & Ingress Reverse Proxy
Acts as an AI-aware API Gateway in front of inference pods:
Validates prompts, blocks prompt injections, sanitizes PII/secrets,
and exports security metrics to Prometheus.
"""

from typing import List, Optional, Dict, Any, Union
import time
import argparse
import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from gateway.guardrails import evaluate_prompt_safety

app = FastAPI(title="Zero-Trust AI Security Gateway", version="1.0.0")

UPSTREAM_ENGINE_URL = "http://localhost:8000"

# Prometheus metrics state
METRICS = {
    "total_requests": 0,
    "injections_blocked": 0,
    "pii_sanitized": 0,
    "forwarded_to_inference": 0
}


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=32, ge=1, le=512)
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    prompt: str
    max_tokens: int = Field(default=32, ge=1, le=512)
    temperature: float = 0.7
    top_p: float = 0.9
    stream: bool = False


@app.get("/healthz")
def healthz():
    return {"status": "healthy", "service": "ai-security-gateway"}


@app.get("/metrics")
def get_prometheus_metrics():
    """Exposes AI Gateway security telemetry in Prometheus format."""
    lines = [
        "# HELP ai_gateway_requests_total Total requests processed by the AI Gateway",
        "# TYPE ai_gateway_requests_total counter",
        f"ai_gateway_requests_total {METRICS['total_requests']}",
        "# HELP ai_gateway_injections_blocked_total Total prompt injection and jailbreak attacks neutralized",
        "# TYPE ai_gateway_injections_blocked_total counter",
        f"ai_gateway_injections_blocked_total {METRICS['injections_blocked']}",
        "# HELP ai_gateway_pii_sanitized_total Total sensitive PII and secrets redacted",
        "# TYPE ai_gateway_pii_sanitized_total counter",
        f"ai_gateway_pii_sanitized_total {METRICS['pii_sanitized']}",
        "# HELP ai_gateway_forwarded_to_inference_total Requests passed to backend GPU/CPU engine",
        "# TYPE ai_gateway_forwarded_to_inference_total counter",
        f"ai_gateway_forwarded_to_inference_total {METRICS['forwarded_to_inference']}"
    ]
    return Response(content="\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.post("/v1/chat/completions")
async def secure_chat_completions(req: ChatRequest):
    METRICS["total_requests"] += 1

    # 1. Aggregate prompt texts from incoming messages
    full_prompt = " ".join([m.content for m in req.messages])

    # 2. Evaluate prompt safety through Guardrail Firewall
    guardrail_result = evaluate_prompt_safety(full_prompt, mask_secrets=True)

    if not guardrail_result.is_allowed:
        METRICS["injections_blocked"] += 1
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": {
                    "type": "security_policy_violation",
                    "code": "PROMPT_INJECTION_DETECTED",
                    "message": "Request blocked by AI Security Gateway firewall.",
                    "details": guardrail_result.violations
                }
            }
        )

    # 3. Sanitize PII/secrets in messages before forwarding
    if guardrail_result.violations:
        METRICS["pii_sanitized"] += len(guardrail_result.violations)
        for m in req.messages:
            eval_single = evaluate_prompt_safety(m.content, mask_secrets=True)
            m.content = eval_single.sanitized_text

    METRICS["forwarded_to_inference"] += 1

    # 4. Reverse-proxy request to upstream inference engine pod
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                f"{UPSTREAM_ENGINE_URL}/v1/chat/completions",
                json=req.dict()
            )
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except httpx.ConnectError:
            raise HTTPException(
                status_code=status.HTTP_533_SERVICE_UNAVAILABLE if hasattr(status, 'HTTP_533_SERVICE_UNAVAILABLE') else 503,
                detail="Inference engine backend unreachable"
            )


@app.post("/v1/completions")
async def secure_completions(req: CompletionRequest):
    METRICS["total_requests"] += 1

    guardrail_result = evaluate_prompt_safety(req.prompt, mask_secrets=True)

    if not guardrail_result.is_allowed:
        METRICS["injections_blocked"] += 1
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={
                "error": {
                    "type": "security_policy_violation",
                    "code": "PROMPT_INJECTION_DETECTED",
                    "message": "Request blocked by AI Security Gateway firewall.",
                    "details": guardrail_result.violations
                }
            }
        )

    if guardrail_result.violations:
        METRICS["pii_sanitized"] += len(guardrail_result.violations)
        req.prompt = guardrail_result.sanitized_text

    METRICS["forwarded_to_inference"] += 1

    # Proxy to upstream engine
    chat_payload = {
        "model": req.model,
        "messages": [{"role": "user", "content": req.prompt}],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "stream": req.stream
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.post(
                f"{UPSTREAM_ENGINE_URL}/v1/chat/completions",
                json=chat_payload
            )
            if resp.status_code == 200:
                data = resp.json()
                text = ""
                if "choices" in data and len(data["choices"]) > 0:
                    text = data["choices"][0].get("message", {}).get("content", "")
                return JSONResponse(status_code=200, content={"text": text, "raw": data})
            return JSONResponse(status_code=resp.status_code, content=resp.json())
        except httpx.ConnectError:
            raise HTTPException(
                status_code=503,
                detail="Inference engine backend unreachable"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Trust AI Security Gateway")
    parser.add_argument("--port", type=int, default=8080, help="Gateway listening port")
    parser.add_argument("--engine-url", type=str, default="http://localhost:8000", help="Upstream inference core URL")
    args = parser.parse_args()

    UPSTREAM_ENGINE_URL = args.engine_url
    print(f"[*] Starting AI Security Gateway on port {args.port} -> Upstream: {UPSTREAM_ENGINE_URL}")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
