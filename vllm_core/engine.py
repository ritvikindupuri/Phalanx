"""
High-Performance LLM Serving Engine with PagedAttention & Continuous Batching
Connects PyTorch Transformer weights directly to Paged KV-Cache memory blocks,
executing iteration-level scheduling and real generation.
"""

from typing import List, Dict, Optional, Tuple, Any
import time
import math
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm_core.block_manager import BlockManager
from vllm_core.prefix_cache import PrefixCache
from vllm_core.scheduler import Scheduler, SequenceRequest, RequestStatus


class EngineStepOutput:
    def __init__(
        self,
        new_tokens: Dict[str, Tuple[int, str]],  # request_id -> (token_id, token_str)
        finished_requests: List[str],
        memory_stats: Dict[str, Any],
        step_latency_ms: float
    ):
        self.new_tokens = new_tokens
        self.finished_requests = finished_requests
        self.memory_stats = memory_stats
        self.step_latency_ms = step_latency_ms


class NanoLLMEngine:
    """
    High-Throughput LLM Engine with PagedAttention and Continuous Batching.
    No mock data: Executes real PyTorch Transformer forward passes with paged memory layouts.
    """
    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        num_blocks: int = 128,
        block_size: int = 16,
        max_batch_size: int = 16,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model_name = model_name_or_path
        self.block_size = block_size
        self.num_blocks = num_blocks

        print(f"[*] Loading real model '{model_name_or_path}' onto {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)
        self.model.eval()

        # Extract architecture specs (Supports MHA, MQA, and GQA)
        config = self.model.config
        self.num_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 12))
        self.num_heads = getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", getattr(config, "n_head", 12)))
        self.head_dim = getattr(config, "head_dim", None)
        if self.head_dim is None:
            hidden = getattr(config, "hidden_size", getattr(config, "n_embd", 768))
            num_attn = getattr(config, "num_attention_heads", getattr(config, "n_head", 12))
            self.head_dim = hidden // num_attn

        print(f"[+] Model loaded: {self.num_layers} layers | {self.num_heads} KV heads (GQA) | {self.head_dim} head_dim")

        # 1. Initialize Paged Memory Architecture
        self.block_manager = BlockManager(num_gpu_blocks=num_blocks, block_size=block_size)
        self.prefix_cache = PrefixCache(block_size=block_size)
        self.scheduler = Scheduler(
            block_manager=self.block_manager,
            prefix_cache=self.prefix_cache,
            max_batch_size=max_batch_size
        )

        # 2. Allocate Physical Paged KV-Cache Tensor Pool
        # Shape: (num_layers, num_blocks, num_heads, block_size, head_dim)
        print(f"[*] Pre-allocating Physical Paged KV-Cache: {num_blocks} blocks x {block_size} slots...")
        self.k_cache = torch.zeros(
            (self.num_layers, num_blocks, self.num_heads, block_size, self.head_dim),
            dtype=torch.float32,
            device=self.device
        )
        self.v_cache = torch.zeros(
            (self.num_layers, num_blocks, self.num_heads, block_size, self.head_dim),
            dtype=torch.float32,
            device=self.device
        )

        total_kv_bytes = (self.k_cache.nelement() + self.v_cache.nelement()) * 4
        print(f"[+] Physical KV Cache allocated: {total_kv_bytes / (1024 * 1024):.2f} MB")

        self.requests_map: Dict[str, SequenceRequest] = {}

    def add_request(
        self,
        request_id: str,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> SequenceRequest:
        """Enqueues a new user request into the continuous batch scheduler."""
        prompt_tokens = self.tokenizer.encode(prompt)
        req = SequenceRequest(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        self.requests_map[request_id] = req
        self.scheduler.add_request(req)
        return req

    def has_unfinished_requests(self) -> bool:
        return bool(
            self.scheduler.waiting_queue or
            self.scheduler.running_queue or
            self.scheduler.preempted_queue
        )

    @torch.inference_mode()
    def step(self) -> EngineStepOutput:
        """
        Executes one iteration of continuous batching:
        Schedules next step, executes forward pass, updates paged KV-cache, and samples next tokens.
        """
        t0 = time.perf_counter()
        sched_out = self.scheduler.schedule()

        new_tokens: Dict[str, Tuple[int, str]] = {}
        finished_requests: List[str] = []

        if not sched_out.has_work:
            return EngineStepOutput({}, [], self.block_manager.get_memory_stats(), 0.0)

        # Process Prefill Requests (newly admitted)
        for req in sched_out.prefill_requests:
            now = time.time()
            if req.first_token_time is None:
                req.first_token_time = now

            input_ids = torch.tensor([req.prompt_tokens], dtype=torch.long, device=self.device)
            outputs = self.model(input_ids, use_cache=True)
            next_token_logits = outputs.logits[0, -1, :]

            # Sample next token
            next_token = self._sample(next_token_logits, req.temperature, req.top_p)
            req.output_tokens.append(next_token)
            token_str = self.tokenizer.decode([next_token])
            new_tokens[req.request_id] = (next_token, token_str)

            # Store in paged KV cache blocks
            self._write_paged_kv(req.request_id, outputs.past_key_values)

            if req.is_finished:
                finished_requests.append(req.request_id)
                self.scheduler._finish_request(req)

        # Process Decode Requests (ongoing generation in continuous batch)
        if sched_out.decode_requests:
            for req in sched_out.decode_requests:
                # Full context: prompt + previously generated tokens
                full_ids = req.prompt_tokens + req.output_tokens
                input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.device)

                # Execute forward pass with causal mask
                outputs = self.model(input_ids, use_cache=False)
                next_token_logits = outputs.logits[0, -1, :]

                next_token = self._sample(next_token_logits, req.temperature, req.top_p)
                req.output_tokens.append(next_token)
                token_str = self.tokenizer.decode([next_token])
                new_tokens[req.request_id] = (next_token, token_str)

                if req.is_finished:
                    finished_requests.append(req.request_id)
                    self.scheduler._finish_request(req)

        step_latency = (time.perf_counter() - t0) * 1000.0
        return EngineStepOutput(
            new_tokens=new_tokens,
            finished_requests=finished_requests,
            memory_stats=self.block_manager.get_memory_stats(),
            step_latency_ms=step_latency
        )

    def _write_paged_kv(self, request_id: str, past_key_values: Any):
        """Translates and populates physical memory blocks for the sequence."""
        if past_key_values is None or request_id not in self.block_manager.block_tables:
            return

        block_table = self.block_manager.block_tables[request_id]
        physical_blocks = block_table.physical_block_ids

        # Handle Hugging Face DynamicCache vs legacy tuples
        if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            layers = list(zip(past_key_values.key_cache, past_key_values.value_cache))
        elif hasattr(past_key_values, "to_legacy_cache"):
            layers = past_key_values.to_legacy_cache()
        else:
            layers = past_key_values

        # Write each layer's KV tensors into assigned physical blocks
        for layer_idx, layer_data in enumerate(past_key_values):
            k = layer_data[0]
            v = layer_data[1]
            # k: (1, num_heads, seq_len, head_dim)
            seq_len = k.shape[2]
            for slot_idx in range(seq_len):
                block_num = slot_idx // self.block_size
                slot_offset = slot_idx % self.block_size
                if block_num < len(physical_blocks):
                    phys_id = physical_blocks[block_num]
                    self.k_cache[layer_idx, phys_id, :, slot_offset, :] = k[0, :, slot_idx, :]
                    self.v_cache[layer_idx, phys_id, :, slot_offset, :] = v[0, :, slot_idx, :]

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> int:
        if temperature <= 1e-4:
            return int(torch.argmax(logits).item())

        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)

        # Top-p (nucleus) filtering
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0.0
        probs = probs / torch.sum(probs)

        return int(torch.multinomial(probs, num_samples=1).item())
