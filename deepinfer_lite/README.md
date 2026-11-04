# 模型 Runner

`ModelRunner` 为 Transformers 因果语言模型提供明确的 Prefill/Decode 边界。Prefill 接收完整
提示词，返回用于预测下一个 token 的 logits 和 `KVCache` 句柄。Decode 每个 batch 行只接收
一个 token 和该句柄，然后返回下一 token 的 logits 及更新后的句柄。

当前 `KVCache` 保存 Transformers 的 `past_key_values`、逻辑序列长度、batch size、设备和
attention mask。目前尚未分配分页缓存块；第 4 天将实现朴素缓存，后续可在不修改调用位置的
情况下替换底层缓存内容。

## Kaggle 验收运行

挂载模型 Dataset 后，在仓库根目录运行：

```python
%env PYTHONPATH=/kaggle/working/deepinfer-lite
!python examples/day3_runner_smoke.py \
  --model-dir /kaggle/input/datasets/yhw1027/qwen3-0-6b-model/Qwen3-0.6B \
  --device cuda:0 \
  --decode-steps 32
```

预期输出包含 `"status": "passed"` 和 `"decode_calls": 32`，且
`cache_sequence_length` 等于提示词 token 数加 32。
