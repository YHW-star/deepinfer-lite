"""第 5 天：验证朴素 KV Cache、生成正确性和边界行为。"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from deepinfer_lite import KVCache, ModelRunner


MODEL_DIR = Path(
    os.environ.get(
        "DEEPINFER_MODEL_DIR",
        "/kaggle/input/datasets/yhw1027/qwen3-0-6b-model/Qwen3-0.6B",
    )
)
DEVICE = os.environ.get("DEEPINFER_TEST_DEVICE", "cuda:0")


@pytest.fixture(scope="session")
def runner() -> ModelRunner:
    """在测试会话中只加载一次模型；环境不满足时明确跳过模型测试。"""
    if not (MODEL_DIR / "config.json").is_file():
        pytest.skip(f"找不到模型目录：{MODEL_DIR}")
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip(f"测试设备 {DEVICE} 不可用")
    return ModelRunner.from_pretrained(MODEL_DIR, device=DEVICE, local_files_only=True)


def make_input(runner: ModelRunner, length: int, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """生成可复现的 token IDs，避免 tokenizer 文本长度造成测试偏差。"""
    vocab_size = int(runner.model.config.vocab_size)
    tokens = (torch.arange(length, dtype=torch.long) * 37 + 101 + offset) % vocab_size
    input_ids = tokens.unsqueeze(0)
    return input_ids, torch.ones_like(input_ids)


@pytest.mark.parametrize("input_length", [1, 128, 512, 2048])
@pytest.mark.parametrize("output_length", [1, 16, 128])
def test_generate_matches_transformers_for_length_matrix(
    runner: ModelRunner,
    input_length: int,
    output_length: int,
) -> None:
    """覆盖计划要求的输入/输出长度组合，并逐 token 对比参考生成。"""
    input_ids, attention_mask = make_input(runner, input_length)
    reference = runner.model.generate(
        input_ids=input_ids.to(runner.device),
        attention_mask=attention_mask.to(runner.device),
        do_sample=False,
        max_new_tokens=output_length,
        use_cache=True,
    )[:, input_length:]
    actual = runner.generate_greedy(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=output_length,
    )

    assert torch.equal(actual.token_ids, reference)
    assert actual.token_ids.shape == (1, output_length)
    assert actual.kv_cache.sequence_length == input_length + output_length - 1
    assert actual.decode_calls == output_length - 1


def test_two_requests_can_be_interleaved(runner: ModelRunner) -> None:
    """两个请求各自持有独立 cache，并以交错顺序逐 token 执行。"""
    prompts = [make_input(runner, 23, 0), make_input(runner, 37, 503)]
    output_limit = 16
    references = []
    for input_ids, attention_mask in prompts:
        full = runner.model.generate(
            input_ids=input_ids.to(runner.device),
            attention_mask=attention_mask.to(runner.device),
            do_sample=False,
            max_new_tokens=output_limit,
            use_cache=True,
        )
        references.append(full[:, input_ids.shape[1]:])

    states = []
    for input_ids, attention_mask in prompts:
        prefill = runner.prefill(input_ids, attention_mask)
        states.append({
            "next_token": prefill.logits.argmax(dim=-1, keepdim=True),
            "cache": prefill.kv_cache,
            "tokens": [],
            "finished": False,
        })

    eos_id = getattr(runner.model.generation_config, "eos_token_id", None)
    eos_ids = set(eos_id if isinstance(eos_id, (list, tuple, set)) else ([] if eos_id is None else [eos_id]))
    for _ in range(output_limit):
        for state in states:
            if state["finished"]:
                continue
            token = state["next_token"]
            state["tokens"].append(token)
            if int(token.item()) in eos_ids or len(state["tokens"]) == output_limit:
                state["finished"] = True
                continue
            decoded = runner.decode(token, state["cache"])
            state["cache"] = decoded.kv_cache
            state["next_token"] = decoded.logits.argmax(dim=-1, keepdim=True)

    for state, reference, (input_ids, _) in zip(states, references, prompts):
        actual_tokens = torch.cat(state["tokens"], dim=1)
        assert torch.equal(actual_tokens, reference)
        assert state["cache"].sequence_length == input_ids.shape[1] + len(state["tokens"]) - 1

    assert states[0]["cache"] is not states[1]["cache"]
    assert states[0]["cache"].past_key_values is not states[1]["cache"].past_key_values


def test_eos_stops_generation_immediately(runner: ModelRunner) -> None:
    """把首个贪心 token 临时设为 EOS，验证 Runner 和参考实现都立即停止。"""
    input_ids, attention_mask = make_input(runner, 19, 211)
    prefill = runner.prefill(input_ids, attention_mask)
    expected_eos = int(prefill.logits.argmax(dim=-1).item())
    previous_eos = runner.model.generation_config.eos_token_id
    runner.model.generation_config.eos_token_id = expected_eos
    try:
        reference = runner.model.generate(
            input_ids=input_ids.to(runner.device),
            attention_mask=attention_mask.to(runner.device),
            do_sample=False,
            max_new_tokens=128,
            use_cache=True,
        )[:, input_ids.shape[1]:]
        actual = runner.generate_greedy(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
        )
    finally:
        runner.model.generation_config.eos_token_id = previous_eos

    assert torch.equal(actual.token_ids, reference)
    assert actual.token_ids.shape == (1, 1)
    assert actual.eos_reached
    assert actual.decode_calls == 0
    assert actual.kv_cache.sequence_length == input_ids.shape[1]


@pytest.mark.parametrize("block_size", [8, 16, 32])
@pytest.mark.parametrize("target_length", ["before", "full", "after"])
def test_contiguous_cache_append_at_block_boundaries(block_size: int, target_length: str) -> None:
    """以未来分页缓存的块边界长度检查当前连续缓存追加语义。"""
    boundary_length = {"before": block_size - 1, "full": block_size, "after": block_size + 1}[target_length]
    initial_length = boundary_length - 1
    initial = _synthetic_past(initial_length)
    cache = KVCache(
        past_key_values=initial,
        sequence_length=initial_length,
        batch_size=1,
        device=torch.device("cpu"),
        attention_mask=torch.ones((1, initial_length), dtype=torch.long),
    )

    cache.append(
        _synthetic_past(boundary_length),
        torch.ones((1, boundary_length), dtype=torch.long),
    )

    assert cache.sequence_length == boundary_length
    assert all(key.shape[-2] == boundary_length for key, _ in cache.key_value_tensors)
    assert all(key.is_contiguous() and value.is_contiguous() for key, value in cache.key_value_tensors)
    expected_bytes = sum(key.numel() * key.element_size() + value.numel() * value.element_size()
                         for key, value in cache.key_value_tensors)
    assert cache.allocated_bytes == expected_bytes


def test_invalid_append_does_not_change_cache() -> None:
    """追加长度不合法时拒绝更新，保证原 cache 状态完整。"""
    cache = KVCache(
        past_key_values=_synthetic_past(7),
        sequence_length=7,
        batch_size=1,
        device=torch.device("cpu"),
        attention_mask=torch.ones((1, 7), dtype=torch.long),
    )
    original_past = cache.past_key_values
    original_mask = cache.attention_mask

    with pytest.raises(RuntimeError, match="KV 长度"):
        cache.append(_synthetic_past(9), torch.ones((1, 8), dtype=torch.long))

    assert cache.sequence_length == 7
    assert cache.past_key_values is original_past
    assert cache.attention_mask is original_mask


def _synthetic_past(sequence_length: int) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """构造形状正确的连续 KV Tensor，供不依赖模型的缓存单元测试使用。"""
    return tuple(
        (
            torch.zeros((1, 2, sequence_length, 4), dtype=torch.float16),
            torch.zeros((1, 2, sequence_length, 4), dtype=torch.float16),
        )
        for _ in range(2)
    )
