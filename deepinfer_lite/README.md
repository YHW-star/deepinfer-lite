# 模型 Runner

`ModelRunner` 为 Transformers 因果语言模型提供明确的 Prefill/Decode 边界。Prefill 接收完整
提示词，返回用于预测下一个 token 的 logits 和 `KVCache` 句柄。Decode 每个 batch 行只接收
一个 token 和该句柄，然后返回下一 token 的 logits 及更新后的句柄。

第 4 天的 `KVCache` 是朴素连续缓存：每个请求独立保存各层的 Key/Value Tensor，形状为
`[batch, heads, seq, head_dim]`，并在每次 Decode 后把序列长度加 1。当前底层由
Transformers 返回和追加 `past_key_values`，Runner 会检查每层 Key/Value 的形状、长度和
连续存储属性；尚未分配分页缓存块。

`generate_greedy()` 使用 logits 的 `argmax`，遇到 EOS 立即停止，最多生成
`max_new_tokens` 个 token。返回的 `decode_calls` 是实际送入模型的单 token 次数，因此
最后一个刚由 logits 选出的 token 尚未再次送入模型时，最终 cache 长度会比完整输出少 1。

## Block Pool

`BlockPool(num_blocks, block_size)` 管理固定容量的 Block ID。它支持按请求申请/释放、按 ID 部分释放、查询请求映射和 owner，以及查看 free/allocated 快照。容量不足、重复申请、重复释放和越权释放都会抛出明确异常；容量不足时 Pool 状态保持不变。

当前 Block Pool 只管理元数据，不分配 KV Tensor。真实的分页 KV 写入和 attention 读取将在后续阶段接入。

运行独立单元测试：

```python
pytest -q tests/test_block_pool.py
```

## Kaggle 验收运行

挂载模型 Dataset 后，在仓库根目录运行：

```python
%env PYTHONPATH=/kaggle/working/deepinfer-lite
!python examples/day3_runner_smoke.py \
  --model-dir /kaggle/input/datasets/yhw1027/qwen3-0-6b-model/Qwen3-0.6B \
  --device cuda:0 \
  --decode-steps 32
```

预期输出包含 `"status": "passed"` 和 `"tokens_match": true`，表示 Runner 的贪心结果
与 Transformers `generate` 完全一致。
