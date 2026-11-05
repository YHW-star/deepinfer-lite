# 第 7 天：推理缓存设计说明

本文说明 DeepInfer-Lite 当前的 KV Cache 参考实现、Prefill/Decode 的性能特征，以及从连续缓存迁移到 Paged KV Cache 所需的接口变化。本文的数字以 Qwen3-0.6B、FP16、单张 Tesla T4 为基准。

## 1. KV Cache 显存计算

### 1.1 单 token、单请求

每层 KV Cache 保存一个 Key Tensor 和一个 Value Tensor，因此单 token 的字节数为：

```text
bytes_per_token
  = 2                         # K 和 V
  × num_layers
  × num_kv_heads
  × head_dim
  × bytes_per_element
```

当前模型配置为：

| 参数 | 数值 |
|---|---:|
| `num_layers` | 28 |
| `num_kv_heads` | 8 |
| `head_dim` | 128 |
| FP16 字节数 | 2 |

```text
2 × 28 × 8 × 128 × 2
= 114,688 bytes/token
≈ 112 KiB/token
```

因此，长度为 `T` 的单请求 KV Tensor 理论大小为：

```text
kv_bytes(T) = T × 114,688
```

| 上下文长度 | KV Tensor 理论大小 |
|---:|---:|
| 128 | 14.00 MiB |
| 512 | 56.00 MiB |
| 2048 | 224.00 MiB |
| 8192 | 896.00 MiB |
| 16384 | 1.75 GiB |

这些数字只包含 KV Tensor，不包含模型权重、临时激活、attention workspace、输入输出 Tensor、CUDA runtime 和 PyTorch allocator 缓存。Day 6 中测得的 allocated 显存明显高于 KV Tensor 理论值，差额主要来自模型权重和运行时开销。

### 1.2 多请求和生成阶段

如果有 `N` 个请求，第 `i` 个请求当前序列长度为 `T_i`，则 KV 总量为：

```text
kv_bytes_total = bytes_per_token × Σ(T_i)
```

Decode 每生成一个 token，就为对应请求增加 `bytes_per_token`。如果 batch 中请求长度不同，连续缓存需要分别维护每个请求的实际长度；不能用最长请求长度乘以 batch size 作为真实需求，否则会高估显存。

### 1.3 Paged KV Cache 的容量和内部浪费

设 block size 为 `B` token，一个请求长度为 `T` 时需要：

```text
num_blocks = ceil(T / B)
allocated_tokens = num_blocks × B
internal_waste_tokens = allocated_tokens - T
```

其物理 KV 分配量为：

```text
paged_bytes(T, B) = ceil(T / B) × B × bytes_per_token
```

Paged Cache 不会消除最后一个 block 的尾部浪费，但会避免为每个请求预留一段过大的连续地址空间，并允许不同请求共享同一个 Block Pool。内部浪费率定义为：

```text
internal_fragmentation =
  (Σ allocated_tokens - Σ actual_tokens) / Σ allocated_tokens
```

Block size 越小，尾部浪费通常越低，但 block table 项、分配管理和 kernel 索引开销会增加；block size 越大，管理更简单，但短请求和不满 block 的尾部浪费更高。第 8 天以后用 8、16、32 做对比，不根据理论值直接宣称性能收益。

## 2. Prefill 和 Decode 的计算特征

### 2.1 Prefill

Prefill 一次处理完整输入序列，主要工作是：

1. 对输入 token 并行执行 embedding、QKV 投影和 attention；
2. 计算每个位置的隐藏状态和 logits；
3. 将所有位置的 K/V 写入请求的 cache；
4. 使用最后一个位置的 logits 选择首个生成 token。

Prefill 通常更接近 compute-bound，矩阵乘规模随输入长度增加，显存读写和临时 activation 也会增加。当前 Day 6 实测：

| 输入长度 | Prefill 平均延迟 |
|---:|---:|
| 128 | 53.67 ms |
| 512 | 88.48 ms |
| 2048 | 830.48 ms |

从 512 增至 2048 token 时，Prefill 延迟增加到约 9.4 倍，是当前长上下文的主要瓶颈。这里的曲线受模型、GPU、Transformers 版本和实现路径共同影响，不能简单视为严格的理论复杂度曲线。

### 2.2 Decode

Decode 每次只处理一个新 token：

1. 读取已有请求的 K/V；
2. 为当前 token 计算 Q、K、V；
3. 用 Q 与历史 K 做 attention；
4. 写入当前 token 的 K/V；
5. 计算 logits 并选择下一 token。

Decode 的矩阵乘规模很小，通常更接近 memory-bound。序列越长，读取的历史 KV 越多；batch 增大后，可以用更多请求摊薄部分 kernel 启动和权重读取开销，但 KV 读写压力也会增加。当前 Day 6 单 token Decode 平均延迟为 42.45–45.58 ms，变化小于 Prefill；这说明当前模型和 workload 下 Decode 延迟的主要变化没有来自 Prefill 计算。

静态 batch 结果体现了吞吐和显存的权衡：

| 上下文长度 | batch=1 | batch=4 | batch=8 |
|---:|---:|---:|---:|
| 128 | 22.59 tokens/s | 89.12 tokens/s | 169.97 tokens/s |
| 512 | 22.58 tokens/s | 73.65 tokens/s | 106.20 tokens/s |
| 2048 | 14.39 tokens/s | 19.71 tokens/s | 21.28 tokens/s |

这是固定 batch 的参考数据，不包含请求动态进入、提前结束、排队和 P99；后续 Continuous Batching 必须使用相同输入/输出长度和采样配置重新测量。

## 3. 当前实现和限制

当前 `deepinfer_lite/model_runner.py` 的边界如下：

- `prefill(input_ids, attention_mask)` 接收完整 prompt，返回最后位置 logits 和 `KVCache`；
- `decode(input_ids, kv_cache)` 每个 batch row 接收一个 token，并返回更新后的 logits/cache；
- `generate_greedy()` 只支持单请求，使用 `argmax`，处理 EOS 和 `max_new_tokens`；
- `KVCache` 保存 Transformers 的 `past_key_values`、逻辑长度、batch size、设备和 attention mask；
- 每次追加都会检查 Key/Value 形状、序列长度和连续存储属性；
- 当前没有独立的 GPU KV 内存池，不能记录物理 block 的申请、释放和碎片率；
- 当前没有 Block Table，不能让 attention kernel 直接按逻辑 block 映射读取物理 block；
- 当前没有 Continuous Batching、请求取消、流式输出、Prefix Cache 或跨请求共享 KV；
- 当前 `past_key_values` 的具体追加和存储仍由 Transformers 管理，不能把它表述为已经实现生产级 PagedAttention；
- 当前 batch 性能脚本是静态 batch，所有请求使用相同上下文长度和固定输出长度；
- 第 1 天和第 6 天的 Transformers/PyTorch 版本不同，跨版本数字不能作为严格 A/B 结论。

因此，当前实现的正确结论是“完成了可验证的连续 KV Reference Path”，而不是“已经完成分页 KV Cache”。

## 4. Paged KV Cache 改造点

### 4.1 保持不变的上层调用接口

Prefill/Decode 的调用方不应直接操作物理显存地址。建议保持以下语义：

```python
prefill(input_ids, attention_mask) -> PrefillOutput(logits, cache_handle)
decode(input_ids, cache_handle) -> DecodeOutput(logits, cache_handle)
```

将 `KVCache` 从“直接持有 `past_key_values`”改为“持有逻辑序列状态和 block table”：

```python
@dataclass
class PagedKVCache:
    block_table: list[int]
    sequence_length: int
    batch_size: int
    device: torch.device
    attention_mask: torch.Tensor
```

### 4.2 新增 Block Pool

Block Pool 负责固定大小的物理 KV Tensor：

```text
free_blocks: deque[int]
allocated_blocks: dict[request_id, list[int]]
block_size: int
```

每个物理 block 的形状可抽象为：

```text
[num_layers, 2, block_size, num_kv_heads, head_dim]
```

实际布局应根据 attention kernel 的访存方式确定，并在 metadata 中记录 block size、dtype 和 layout。Block Pool 至少提供：

- `allocate(num_blocks)`：申请足够的物理 block，不足时返回明确的 OOM；
- `append_token(cache_handle)`：尾部 block 未满时直接写入，满时申请新 block；
- `release(cache_handle)`：请求结束后释放全部 block；
- `free_count()`、`allocated_count()`：供 benchmark 和泄漏测试使用。

### 4.3 Prefill 写入分页缓存

Prefill 得到每层完整 K/V 后，按 token 范围切分到物理 block：

```text
logical block 0 -> physical block p0
logical block 1 -> physical block p1
...
```

同时更新 `block_table`。尾部不满的 block 只能由当前请求写入，不能直接作为 Prefix Cache 共享对象。Prefill 过程中需要保证 block 分配完成后再写入，发生异常时释放已申请 block，避免半初始化请求泄漏。

### 4.4 Decode 读取和追加

每轮 Decode：

1. 根据 `sequence_length // block_size` 找到逻辑尾 block；
2. 若尾 block 未满，写入当前位置；
3. 若尾 block 已满，从 Block Pool 申请新 block 并更新 `block_table`；
4. attention kernel 根据 `block_table` 读取历史 K/V，不把所有 block 拼接成新的连续 Tensor；
5. 生成完成、EOS 或取消时立即释放 block。

如果暂时不能改造模型 attention，使其直接读取 block table，则先完成 Block Pool/Block Table 的 Reference Path，并明确它只验证逻辑映射和生命周期，不声称已经获得物理分页 kernel 的性能收益。

### 4.5 调度器和指标改造

Paged Cache 接入后，调度器需要在每轮执行前估算所需 block 数，并拒绝无法满足显存预算的请求。每轮至少记录：

```text
iteration, prefill_request_ids, decode_request_ids,
free_blocks, allocated_blocks, block_size,
logical_tokens, allocated_tokens, internal_fragmentation
```

核心正确性检查：

- block 不重复分配；
- 请求结束后 free block 数恢复到初始值；
- 尾部跨 block 边界时 token 不丢失、不重复；
- 逻辑 token 序列与连续 Reference Path 完全一致；
- OOM 或异常路径释放已经申请的 block；
- Prefix Cache 共享 block 只有在引用计数归零后才可回收。

## 5. 第 8 天的直接输入

第 8 天先实现独立的 `BlockPool`，不立即修改模型 attention：

1. 用 block size 8、16、32 做申请/释放单元测试；
2. 用随机长度请求验证 block table 映射；
3. 记录实际 token 数、分配 token 数和内部碎片率；
4. 将 `PagedKVCache` 接口接到当前 `ModelRunner` 的 Reference Path；
5. 只有 Reference Path 通过 token 一致性和资源回收测试后，再决定是否接入真实分页 attention kernel。

