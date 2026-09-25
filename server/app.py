"""
OpenAI-Compatible FastAPI Server for nano-vLLM
Implements /v1/chat/completions with streaming Server-Sent Events (SSE),
/v1/models, and /metrics for Prometheus monitoring.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Optional, Dict, Any, Union
import time
import json
import uuid
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from vllm_core.engine import NanoLLMEngine

app = FastAPI(title="nano-vLLM High-Throughput Serving Engine", version="1.0.0")

# Global engine instance
engine: Optional[NanoLLMEngine] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=64, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = False


@app.on_event("startup")
def startup_event():
    global engine
    engine = NanoLLMEngine(
        model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
        num_blocks=256,
        block_size=16,
        max_batch_size=16,
        device="cpu"
    )


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": engine.model_name if engine else "gpt2",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nano-vllm"
            }
        ]
    }


@app.get("/metrics")
def get_metrics():
    if not engine:
        raise HTTPException(status_code=503, detail="Engine initializing")
    mem_stats = engine.block_manager.get_memory_stats()
    prefix_stats = engine.prefix_cache.get_stats()
    return {
        "kv_cache_utilization_pct": mem_stats["utilization_pct"],
        "kv_used_blocks": mem_stats["used_blocks"],
        "kv_total_blocks": mem_stats["total_blocks"],
        "active_concurrent_sequences": mem_stats["active_sequences"],
        "tokens_stored_in_cache": mem_stats["tokens_stored"],
        "prefix_cache_hit_rate_pct": prefix_stats["hit_rate_pct"],
        "scheduler_total_completed": engine.scheduler.total_completed_requests,
        "scheduler_total_preemptions": engine.scheduler.total_preemptions
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not loaded")

    # Format chat prompt
    full_prompt = "\n".join([f"{m.role}: {m.content}" for m in request.messages]) + "\nassistant:"
    request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    req = engine.add_request(
        request_id=request_id,
        prompt=full_prompt,
        max_new_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )

    if request.stream:
        async def stream_generator():
            while not req.is_finished:
                # Step engine asynchronously
                engine.step()
                if req.output_tokens:
                    last_token = req.output_tokens[-1]
                    token_str = engine.tokenizer.decode([last_token])
                    chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": token_str},
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                await asyncio.sleep(0.01)

            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_generator(), media_type="text/event-stream")

    # Non-streaming mode: step until finished
    while not req.is_finished:
        engine.step()
        await asyncio.sleep(0.005)

    generated_text = engine.tokenizer.decode(req.output_tokens)
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": generated_text.strip()},
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(req.prompt_tokens),
            "completion_tokens": len(req.output_tokens),
            "total_tokens": req.total_tokens
        },
        "system_telemetry": {
            "ttft_ms": round(req.ttft_ms, 2) if req.ttft_ms else None,
            "total_latency_ms": round(req.total_latency_ms, 2) if req.total_latency_ms else None
        }
    }
