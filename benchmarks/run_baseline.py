"""Run a deterministic single-request Qwen3 inference baseline.

The baseline deliberately runs requests serially. It measures the model path
that later Paged KV Cache and Continuous Batching implementations must beat.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class RequestResult:
    request_id: int
    input_tokens: int
    generated_tokens: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    throughput_tokens_per_s: float
    peak_memory_allocated_mib: float
    peak_memory_reserved_mib: float
    eos_hit: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/day2_baseline")
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-requests", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def cuda_sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)


def build_fixed_prompt(tokenizer: Any, input_tokens: int) -> torch.Tensor:
    if input_tokens < 1:
        raise ValueError("--input-tokens must be >= 1")

    seed_text = (
        "Explain the purpose of a key value cache in autoregressive language "
        "model inference. It stores prior attention states and avoids repeating "
        "work for tokens that have already been processed. "
    )
    seed_ids = tokenizer(
        seed_text,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    repeats = (input_tokens + seed_ids.numel() - 1) // seed_ids.numel()
    prompt_ids = seed_ids.repeat(repeats)[:input_tokens]
    return prompt_ids.unsqueeze(0)


def is_eos(token_id: int, eos_token_id: Any) -> bool:
    if eos_token_id is None:
        return False
    if isinstance(eos_token_id, int):
        return token_id == eos_token_id
    return token_id in set(eos_token_id)


@torch.inference_mode()
def run_request(
    model: Any,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    device: str,
    request_id: int,
) -> RequestResult:
    input_tokens = int(prompt_ids.shape[1])
    input_ids = prompt_ids.to(device)
    attention_mask = torch.ones_like(input_ids)

    cuda_sync(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()

    # Prefill: process the complete prompt once and retain its KV states.
    prefill = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    next_token = prefill.logits[:, -1, :].argmax(dim=-1)
    cuda_sync(device)
    first_token_time = time.perf_counter()

    past_key_values = prefill.past_key_values
    generated = [int(next_token.item())]
    eos_hit = is_eos(generated[0], model.config.eos_token_id)

    # Decode: one token per model call. We intentionally continue after EOS so
    # every benchmark request has exactly max_new_tokens generated tokens.
    for step in range(1, max_new_tokens):
        current_length = input_tokens + step
        decode_mask = torch.ones(
            (1, current_length),
            dtype=attention_mask.dtype,
            device=device,
        )
        outputs = model(
            input_ids=next_token[:, None],
            attention_mask=decode_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        generated.append(int(next_token.item()))
        eos_hit = eos_hit or is_eos(generated[-1], model.config.eos_token_id)

    cuda_sync(device)
    end = time.perf_counter()
    ttft_ms = (first_token_time - start) * 1000
    e2e_ms = (end - start) * 1000
    generated_tokens = len(generated)
    tpot_ms = (
        (end - first_token_time) * 1000 / (generated_tokens - 1)
        if generated_tokens > 1
        else 0.0
    )
    throughput = generated_tokens / max(end - start, 1e-12)

    if device.startswith("cuda"):
        peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20
        peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20
    else:
        peak_allocated = 0.0
        peak_reserved = 0.0

    return RequestResult(
        request_id=request_id,
        input_tokens=input_tokens,
        generated_tokens=generated_tokens,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        e2e_ms=e2e_ms,
        throughput_tokens_per_s=throughput,
        peak_memory_allocated_mib=peak_allocated,
        peak_memory_reserved_mib=peak_reserved,
        eos_hit=eos_hit,
    )


def summarize(results: list[RequestResult], total_wall_s: float) -> dict[str, Any]:
    def mean(key: str) -> float:
        return sum(getattr(item, key) for item in results) / len(results)

    def percentile(key: str, fraction: float) -> float:
        values = sorted(getattr(item, key) for item in results)
        index = min(len(values) - 1, max(0, int(round((len(values) - 1) * fraction))))
        return values[index]

    total_generated = sum(item.generated_tokens for item in results)
    return {
        "requests": len(results),
        "total_generated_tokens": total_generated,
        "total_wall_s": total_wall_s,
        "aggregate_throughput_tokens_per_s": total_generated / max(total_wall_s, 1e-12),
        "ttft_ms_mean": mean("ttft_ms"),
        "ttft_ms_p50": percentile("ttft_ms", 0.50),
        "ttft_ms_p99": percentile("ttft_ms", 0.99),
        "tpot_ms_mean": mean("tpot_ms"),
        "tpot_ms_p50": percentile("tpot_ms", 0.50),
        "tpot_ms_p99": percentile("tpot_ms", 0.99),
        "e2e_ms_mean": mean("e2e_ms"),
        "e2e_ms_p50": percentile("e2e_ms", 0.50),
        "e2e_ms_p99": percentile("e2e_ms", 0.99),
        "peak_memory_allocated_mib": max(item.peak_memory_allocated_mib for item in results),
        "peak_memory_reserved_mib": max(item.peak_memory_reserved_mib for item in results),
    }


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 1 or args.num_requests < 1 or args.warmup < 0:
        raise ValueError("max-new-tokens and num-requests must be >= 1; warmup must be >= 0")
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA is unavailable; choose --device cpu for a smoke test")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
    ).to(args.device).eval()
    prompt_ids = build_fixed_prompt(tokenizer, args.input_tokens)

    print(
        f"model={args.model_dir} device={args.device} input_tokens={args.input_tokens} "
        f"max_new_tokens={args.max_new_tokens} requests={args.num_requests}"
    )
    for warmup_id in range(args.warmup):
        run_request(model, prompt_ids, args.max_new_tokens, args.device, -warmup_id - 1)
        print(f"warmup {warmup_id + 1}/{args.warmup} complete")

    cuda_sync(args.device)
    benchmark_start = time.perf_counter()
    results = [
        run_request(model, prompt_ids, args.max_new_tokens, args.device, request_id)
        for request_id in range(args.num_requests)
    ]
    cuda_sync(args.device)
    total_wall_s = time.perf_counter() - benchmark_start

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "model_dir": str(args.model_dir),
        "device": args.device,
        "device_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu",
        "dtype": str(next(model.parameters()).dtype),
        "seed": args.seed,
        "sampling": {"method": "greedy", "do_sample": False, "temperature": 0.0, "top_p": 1.0},
        "workload": {
            "input_tokens": args.input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "warmup": args.warmup,
            "num_requests": args.num_requests,
            "mode": "closed_loop_serial",
        },
    }
    summary = summarize(results, total_wall_s)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "requests.json").write_text(
        json.dumps([asdict(item) for item in results], indent=2), encoding="utf-8"
    )
    with (output_dir / "requests.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(item) for item in results)

    print(json.dumps(summary, indent=2))
    print(f"results written to {output_dir}")


if __name__ == "__main__":
    main()
