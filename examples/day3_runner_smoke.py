"""Verify one Prefill and at least 32 single-token Decode calls."""

from __future__ import annotations

import argparse
import json

import torch

from deepinfer_lite import ModelRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--prompt", default="Explain KV Cache in one sentence.")
    args = parser.parse_args()
    if args.decode_steps < 32:
        parser.error("--decode-steps must be at least 32 for Day 3 acceptance")

    runner = ModelRunner.from_pretrained(args.model_dir, device=args.device)
    encoded = runner.tokenizer(args.prompt, return_tensors="pt")
    prefill = runner.prefill(encoded["input_ids"], encoded.get("attention_mask"))
    next_token = prefill.logits.argmax(dim=-1, keepdim=True)
    cache = prefill.kv_cache

    for _ in range(args.decode_steps):
        decoded = runner.decode(next_token, cache)
        cache = decoded.kv_cache
        next_token = decoded.logits.argmax(dim=-1, keepdim=True)

    expected_length = int(encoded["input_ids"].shape[1]) + args.decode_steps
    if cache.sequence_length != expected_length:
        raise RuntimeError(
            f"cache length mismatch: expected {expected_length}, got {cache.sequence_length}"
        )

    print(json.dumps({
        "status": "passed",
        "device": str(runner.device),
        "prefill_tokens": int(encoded["input_ids"].shape[1]),
        "decode_calls": args.decode_steps,
        "cache_sequence_length": cache.sequence_length,
        "next_token_id": int(next_token.item()),
        "dtype": str(next(runner.model.parameters()).dtype),
    }, indent=2))


if __name__ == "__main__":
    main()
