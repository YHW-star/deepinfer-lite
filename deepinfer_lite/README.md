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

## Block Table

`BlockTable` 为一个请求维护逻辑 Block 到物理 Block 的有序映射。它不直接持有 KV
Tensor，因此可以先独立验证分页分配器的边界和生命周期，再在后续阶段接入真正的 KV
写入路径。

```python
from deepinfer_lite import BlockPool, BlockTable

pool = BlockPool(num_blocks=64, block_size=16)
table = BlockTable(pool, request_id="request-1", sequence_length=17)
assert table.logical_block_count() == 2
first_physical_block = table.logical_to_physical(0)
table.append_token()  # 只有跨过 16-token 边界时才会申请新 Block
table.release()       # 释放该请求的全部物理 Block
```

主要行为：

- 初始序列长度按 `ceil(sequence_length / block_size)` 申请物理 Block；
- `append_token()` 和 `append_tokens(n)` 只在跨越 Block 边界时扩容；
- `logical_to_physical()` 查询逻辑 Block 映射，`physical_for_token()` 查询 token 所在物理 Block；
- Pool 不足时抛出 `BlockPoolExhaustedError`，请求的长度、映射和 Pool 状态保持不变；
- `release()` 回收全部物理 Block，之后的访问抛出 `BlockTableReleasedError`；
- `snapshot()` 可记录序列长度、Block 数量和尾部内部碎片。

运行第 9 天单元测试：

```python
pytest -q tests/test_block_table.py
```

测试包含 Block 边界扩容、初始长度分配、映射查询、空请求、释放后的访问保护、Pool
不足时的原子性，以及固定随机种子的 100 次请求生命周期回收验收。

## Paged KV Cache Reference Path

`PagedKVStorage` 为每层分配形状为 `[num_blocks, block_size, kv_heads, head_dim]` 的
物理 Tensor；`PagedKVCache` 使用 `BlockTable` 保存逻辑到物理映射。Prefill 通过
`PagedKVCache.from_contiguous()` 将完整连续 KV 写入对应 Block，Decode 通过
`append_from_contiguous()` 只追加完整结果中的最后一个 token；`to_contiguous()` 按逻辑
顺序 gather 回 Transformers 可接受的连续 KV。

`PagedModelRunner` 是模型对照适配器：每轮 Decode 前 gather 连续 KV，调用现有
`ModelRunner.decode()`，再把返回 KV 的新增 token 写回分页存储。因此它验证分页管理和
读写的正确性。对于当前 Transformers 5.x，gather 结果会先封装为 `DynamicCache`，再
传给 Qwen3 前向；这一步是模型 API 适配，不改变分页存储布局。该路径不能代表 Attention
Kernel 已经直接按 Block Table 读取，也不能据此宣称获得了 PagedAttention 性能收益。

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
