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
