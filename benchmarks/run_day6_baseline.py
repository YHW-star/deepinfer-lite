"""运行第 6 天的 Prefill、Decode、显存和静态批处理基线。"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default="/kaggle/working/day6_baseline")
    parser.add_argument("--contexts", default="128,512,2048")
    parser.add_argument("--batch-sizes", default="1,4,8")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_int_list(value: str, name: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"{name} 必须是逗号分隔的整数") from error
    if not values or any(item < 1 for item in values):
        raise ValueError(f"{name} 中的值必须大于或等于 1")
    return values


def cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_fixed_prompt(tokenizer: Any, input_tokens: int, batch_size: int = 1) -> torch.Tensor:
    """生成固定 token 输入，避免不同 trial 的文本长度发生变化。"""
    seed_text = (
        "Explain the purpose of a key value cache in autoregressive language "
        "model inference. It stores prior attention states and avoids repeating "
        "work for tokens that have already been processed. "
    )
    seed_ids = tokenizer(seed_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    repeats = (input_tokens + seed_ids.numel() - 1) // seed_ids.numel()
    prompt = seed_ids.repeat(repeats)[:input_tokens].unsqueeze(0)
    return prompt.repeat(batch_size, 1)


def peak_memory(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return (
        torch.cuda.max_memory_allocated(device) / 2**20,
        torch.cuda.max_memory_reserved(device) / 2**20,
    )


@torch.inference_mode()
def prefill(model: Any, prompt_ids: torch.Tensor, device: torch.device) -> Any:
    attention_mask = torch.ones_like(prompt_ids, device=device)
    positions = torch.arange(prompt_ids.shape[1], device=device)
    return model(
        input_ids=prompt_ids,
        attention_mask=attention_mask,
        cache_position=positions,
        use_cache=True,
        return_dict=True,
    )


@torch.inference_mode()
def decode_one(
    model: Any,
    next_token: torch.Tensor,
    past_key_values: Any,
    sequence_length: int,
    batch_size: int,
    device: torch.device,
) -> Any:
    attention_mask = torch.ones(
        (batch_size, sequence_length + 1), dtype=torch.long, device=device
    )
    return model(
        input_ids=next_token,
        attention_mask=attention_mask,
        cache_position=torch.tensor([sequence_length], device=device),
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )


@torch.inference_mode()
def measure_prefill(
    model: Any,
    tokenizer: Any,
    context_length: int,
    device: torch.device,
    trials: int,
    warmup: int,
) -> list[dict[str, Any]]:
    prompt_ids = build_fixed_prompt(tokenizer, context_length).to(device)
    for _ in range(warmup):
        warmup_outputs = prefill(model, prompt_ids, device)
        del warmup_outputs
    records = []
    for trial in range(1, trials + 1):
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        cuda_sync(device)
        start = time.perf_counter()
        outputs = prefill(model, prompt_ids, device)
        cuda_sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        allocated_mib, reserved_mib = peak_memory(device)
        records.append({
            "metric": "prefill",
            "context_tokens": context_length,
            "batch_size": 1,
            "trial": trial,
            "latency_ms": elapsed_ms,
            "throughput_tokens_per_s": context_length / max(elapsed_ms / 1000, 1e-12),
            "peak_memory_allocated_mib": allocated_mib,
            "peak_memory_reserved_mib": reserved_mib,
            "generated_tokens": 0,
            "status": "ok" if outputs.past_key_values is not None else "failed",
        })
        del outputs
    return records


@torch.inference_mode()
def measure_decode(
    model: Any,
    tokenizer: Any,
    context_length: int,
    device: torch.device,
    trials: int,
    warmup: int,
) -> list[dict[str, Any]]:
    prompt_ids = build_fixed_prompt(tokenizer, context_length).to(device)
    for _ in range(warmup):
        outputs = prefill(model, prompt_ids, device)
        decode_one(
            model,
            outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True),
            outputs.past_key_values,
            context_length,
            1,
            device,
        )
    records = []
    for trial in range(1, trials + 1):
        outputs = prefill(model, prompt_ids, device)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cuda_sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        decoded = decode_one(
            model, next_token, outputs.past_key_values, context_length, 1, device
        )
        cuda_sync(device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        allocated_mib, reserved_mib = peak_memory(device)
        records.append({
            "metric": "decode_one",
            "context_tokens": context_length,
            "batch_size": 1,
            "trial": trial,
            "latency_ms": elapsed_ms,
            "throughput_tokens_per_s": 1000 / max(elapsed_ms, 1e-12),
            "peak_memory_allocated_mib": allocated_mib,
            "peak_memory_reserved_mib": reserved_mib,
            "generated_tokens": 1,
            "status": "ok" if decoded.past_key_values is not None else "failed",
        })
    return records


@torch.inference_mode()
def measure_static_batch(
    model: Any,
    tokenizer: Any,
    context_length: int,
    batch_size: int,
    max_new_tokens: int,
    device: torch.device,
    trials: int,
    warmup: int,
) -> list[dict[str, Any]]:
    prompt_ids = build_fixed_prompt(tokenizer, context_length, batch_size).to(device)

    def run_once() -> tuple[float, float, float]:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        cuda_sync(device)
        start = time.perf_counter()
        outputs = prefill(model, prompt_ids, device)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past_key_values = outputs.past_key_values
        for step in range(1, max_new_tokens):
            decoded = decode_one(
                model,
                next_token,
                past_key_values,
                context_length + step - 1,
                batch_size,
                device,
            )
            past_key_values = decoded.past_key_values
            next_token = decoded.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cuda_sync(device)
        elapsed_s = time.perf_counter() - start
        allocated_mib, reserved_mib = peak_memory(device)
        return elapsed_s, allocated_mib, reserved_mib

    for _ in range(warmup):
        run_once()
    records = []
    for trial in range(1, trials + 1):
        elapsed_s, allocated_mib, reserved_mib = run_once()
        generated_tokens = batch_size * max_new_tokens
        records.append({
            "metric": "static_batch",
            "context_tokens": context_length,
            "batch_size": batch_size,
            "trial": trial,
            "latency_ms": elapsed_s * 1000,
            "throughput_tokens_per_s": generated_tokens / max(elapsed_s, 1e-12),
            "peak_memory_allocated_mib": allocated_mib,
            "peak_memory_reserved_mib": reserved_mib,
            "generated_tokens": generated_tokens,
            "status": "ok",
        })
    return records


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (record["metric"], record["context_tokens"], record["batch_size"])
        groups[key].append(record)
    summary = []
    for (metric, context_tokens, batch_size), items in groups.items():
        def stats(field: str) -> tuple[float, float]:
            values = [float(item[field]) for item in items]
            return mean(values), stdev(values) if len(values) > 1 else 0.0

        latency_mean, latency_std = stats("latency_ms")
        throughput_mean, throughput_std = stats("throughput_tokens_per_s")
        allocated_mean, allocated_std = stats("peak_memory_allocated_mib")
        reserved_mean, reserved_std = stats("peak_memory_reserved_mib")
        summary.append({
            "metric": metric,
            "context_tokens": context_tokens,
            "batch_size": batch_size,
            "trials": len(items),
            "latency_ms_mean": latency_mean,
            "latency_ms_std": latency_std,
            "throughput_tokens_per_s_mean": throughput_mean,
            "throughput_tokens_per_s_std": throughput_std,
            "peak_memory_allocated_mib_mean": allocated_mean,
            "peak_memory_allocated_mib_std": allocated_std,
            "peak_memory_reserved_mib_mean": reserved_mean,
            "peak_memory_reserved_mib_std": reserved_std,
        })
    return sorted(summary, key=lambda item: (item["metric"], item["context_tokens"], item["batch_size"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, summary: list[dict[str, Any]]) -> None:
    """输出四联图；绘图依赖只在实际生成图表时导入。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    prefill_rows = [row for row in summary if row["metric"] == "prefill"]
    decode_rows = [row for row in summary if row["metric"] == "decode_one"]
    batch_rows = [row for row in summary if row["metric"] == "static_batch"]

    axes[0, 0].plot(
        [row["context_tokens"] for row in prefill_rows],
        [row["latency_ms_mean"] for row in prefill_rows],
        marker="o",
        label="Prefill",
    )
    axes[0, 0].set_title("Prefill 延迟与上下文长度")
    axes[0, 0].set_xlabel("上下文 token 数")
    axes[0, 0].set_ylabel("平均延迟（ms）")

    axes[0, 1].plot(
        [row["context_tokens"] for row in decode_rows],
        [row["latency_ms_mean"] for row in decode_rows],
        marker="o",
        color="tab:orange",
        label="Decode",
    )
    axes[0, 1].set_title("单 token Decode 延迟与上下文长度")
    axes[0, 1].set_xlabel("上下文 token 数")
    axes[0, 1].set_ylabel("平均延迟（ms）")

    for batch_size in sorted({row["batch_size"] for row in batch_rows}):
        rows = [row for row in batch_rows if row["batch_size"] == batch_size]
        axes[1, 0].plot(
            [row["context_tokens"] for row in rows],
            [row["throughput_tokens_per_s_mean"] for row in rows],
            marker="o",
            label=f"batch={batch_size}",
        )
    axes[1, 0].set_title("静态批处理吞吐")
    axes[1, 0].set_xlabel("上下文 token 数")
    axes[1, 0].set_ylabel("生成 token/s")
    axes[1, 0].legend()

    for metric, label, color in (
        ("prefill", "Prefill", "tab:blue"),
        ("decode_one", "Decode", "tab:orange"),
    ):
        rows = [row for row in summary if row["metric"] == metric]
        axes[1, 1].plot(
            [row["context_tokens"] for row in rows],
            [row["peak_memory_allocated_mib_mean"] for row in rows],
            marker="o",
            color=color,
            label=label,
        )
    axes[1, 1].set_title("峰值 allocated 显存")
    axes[1, 1].set_xlabel("上下文 token 数")
    axes[1, 1].set_ylabel("MiB")
    axes[1, 1].legend()

    for axis in axes.flat:
        axis.grid(alpha=0.25)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    contexts = parse_int_list(args.contexts, "--contexts")
    batch_sizes = parse_int_list(args.batch_sizes, "--batch-sizes")
    if args.trials < 1 or args.warmup < 0 or args.max_new_tokens < 1:
        raise ValueError("trials 和 max-new-tokens 必须大于等于 1，warmup 不能小于 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请检查 Kaggle GPU 设置")

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        local_files_only=True,
    ).to(device).eval()

    records: list[dict[str, Any]] = []
    for context_length in contexts:
        records.extend(measure_prefill(model, tokenizer, context_length, device, args.trials, args.warmup))
        records.extend(measure_decode(model, tokenizer, context_length, device, args.trials, args.warmup))
        for batch_size in batch_sizes:
            records.extend(measure_static_batch(
                model, tokenizer, context_length, batch_size, args.max_new_tokens,
                device, args.trials, args.warmup,
            ))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(records)
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "model_dir": str(args.model_dir),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": str(next(model.parameters()).dtype),
        "seed": args.seed,
        "sampling": {"method": "greedy", "do_sample": False},
        "contexts": contexts,
        "batch_sizes": batch_sizes,
        "max_new_tokens": args.max_new_tokens,
        "trials": args.trials,
        "warmup": args.warmup,
        "batch_mode": "static_prefill_then_decode",
        "stop_behavior": "fixed_output_length_ignore_eos",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(output_dir / "baseline_trials.csv", records)
    write_csv(output_dir / "baseline_summary.csv", summary)
    write_plot(output_dir / "baseline_plots.png", summary)
    print(json.dumps({"output_dir": str(output_dir), "summary_rows": len(summary)}, indent=2))
    print(f"结果已写入：{output_dir}")


if __name__ == "__main__":
    main()
