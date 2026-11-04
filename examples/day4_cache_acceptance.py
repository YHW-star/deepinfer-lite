"""验证朴素 KV Cache 与 Transformers 贪心生成结果一致。"""

from __future__ import annotations

import argparse
import json

import torch

from deepinfer_lite import ModelRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt", default="Explain KV Cache in one sentence.")
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens 必须至少为 1")

    runner = ModelRunner.from_pretrained(args.model_dir, device=args.device)
    encoded = runner.tokenizer(args.prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(runner.device)
    attention_mask = encoded["attention_mask"].to(runner.device)

    reference = runner.model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        use_cache=True,
    )[:, input_ids.shape[1]:]
    actual = runner.generate_greedy(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
    )
    matches = torch.equal(actual.token_ids, reference)
    result = {
        "status": "passed" if matches else "failed",
        "device": str(runner.device),
        "prompt_tokens": int(input_ids.shape[1]),
        "requested_max_new_tokens": args.max_new_tokens,
        "reference_generated_tokens": int(reference.shape[1]),
        "runner_generated_tokens": int(actual.token_ids.shape[1]),
        "tokens_match": matches,
        "decode_calls": actual.decode_calls,
        "eos_reached": actual.eos_reached,
        "cache_sequence_length": actual.kv_cache.sequence_length,
        "kv_allocated_mib": actual.kv_cache.allocated_bytes / 2**20,
        "dtype": str(next(runner.model.parameters()).dtype),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not matches:
        raise SystemExit("Runner 输出与 Transformers generate 不一致")


if __name__ == "__main__":
    main()
