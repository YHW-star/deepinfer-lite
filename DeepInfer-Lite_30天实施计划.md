# DeepInfer-Lite：面向 Qwen3 的单卡 LLM 推理引擎

> 目标：在 30 天内实现一个可以运行、可以压测、可以解释性能权衡的轻量级推理引擎。
>
> 主线机制：**Paged KV Cache → Prefill/Decode 分离 → Continuous Batching → Prefix Caching → 性能 Profiling**。
>
> 推荐模型路线：Qwen3-0.6B 用于早期正确性调试，Qwen3-4B 用于主实验，Qwen3-8B（7B 档）用于最终扩展性验证。

---

## 1. 项目定位

### 1.1 要解决的问题

朴素的 LLM 推理实现通常为每个请求分配一块连续 KV Cache，并等待整个 batch 完成后再处理下一批请求。这种方式在长上下文和请求长度不一致时会产生明显问题：

- KV Cache 连续分配导致显存碎片；
- 短请求必须等待长请求完成；
- Decode 阶段 GPU 利用率低；
- 共享系统提示词时重复执行 Prefill；
- 显存容量无法支撑足够的并发请求。

本项目实现一个小型推理引擎，研究以下问题：

> 在单张 GPU 上，分页 KV Cache 和连续批处理能否在长上下文混合负载下提高吞吐、降低显存占用，同时控制 TTFT 和 P99 延迟？

### 1.2 项目最终交付物

- 可运行的 Python/PyTorch 推理引擎；
- Paged KV Cache 和 Block Pool；
- Prefill/Decode 分离的调度器；
- Continuous Batching；
- Prefix Caching；
- FastAPI 流式推理服务；
- 自动化 benchmark 和压测脚本；
- P50/P99、TTFT、TPOT、吞吐、峰值显存、碎片率报告；
- 一份包含设计、实验、结果和 trade-off 的技术报告；
- 一个 3 分钟可展示的 Demo。

### 1.3 明确不纳入 30 天主线的内容

以下内容只作为后续方向，不进入本项目的完成标准：

- 完整复刻 DeepSeek-V3/R1 的 MoE、MLA 和跨节点通信；
- 多 GPU Tensor Parallel；
- 完整 P/D 分离集群；
- 大规模生产级容错和服务治理；
- 同时实现 Speculative Decoding、量化 Kernel 和分布式调度。

30 天项目只选择一个进阶方向：**Triton Decode Attention Kernel 或 KV Cache 量化**。

---

## 2. 模型、硬件和环境假设

### 2.1 模型梯度

| 阶段 | 模型 | 用途 |
|---|---|---|
| 正确性调试 | Qwen3-0.6B 或 1.7B | 快速定位实现错误 |
| 主实验 | Qwen3-4B | 完成全部核心机制和主要 benchmark |
| 最终展示 | Qwen3-8B（7B 档） | 验证在更大模型和显存压力下的扩展性 |

如果机器显存不足以运行 Qwen3-8B，最终版本使用 Qwen3-4B；不要为了运行 8B 而牺牲实验完整性。

### 2.2 显存建议

| GPU 显存 | 推荐配置 |
|---|---|
| 8GB | Qwen3-0.6B/1.7B，FP16 或 INT4 |
| 12GB | Qwen3-4B，INT4 或限制上下文长度 |
| 16GB | Qwen3-4B FP16，或 Qwen3-8B INT4 |
| 24GB | Qwen3-8B INT4/FP16 |
| 48GB+ | Qwen3-8B FP16，增加并发和上下文长度 |

Qwen3-8B FP16 权重通常需要 16GB 以上显存，且还需要为 KV Cache、CUDA Workspace 和运行时预留空间。16GB 显卡优先使用 INT4 权重。

### 2.3 固定推理设置

主 benchmark 使用确定性配置，避免采样参数造成结果不可比：

```yaml
model: Qwen3-4B-Instruct
dtype: float16
enable_thinking: false
temperature: 0.0
top_p: 1.0
max_new_tokens: 128
seed: 42
```

Qwen3 的 thinking 模式会显著改变生成 token 数和 Decode 时间。主实验关闭 thinking；最后单独增加一组 thinking mode 实验，并记录实际生成 token 数。

### 2.4 软件依赖

- Python 3.10 或更高版本；
- PyTorch 2.x；
- Transformers；
- FastAPI、Uvicorn；
- Triton（如果选择 Kernel 进阶方向）；
- Locust 或 asyncio 压测脚本；
- PyTorch Profiler；
- 条件允许时使用 Nsight Systems。

---

## 3. 系统设计

### 3.1 整体架构

```text
                 ┌────────────────────┐
 HTTP / Python ─▶│ Request Manager     │
                 └─────────┬──────────┘
                           │
                 ┌─────────▼──────────┐
                 │ Scheduler           │
                 │ waiting/running     │
                 │ prefill/decode      │
                 └─────┬─────────┬─────┘
                       │         │
             ┌─────────▼───┐ ┌──▼────────────┐
             │ Block Pool   │ │ Model Runner  │
             │ Block Table  │ │ Prefill       │
             │ Prefix Cache │ │ Decode        │
             └──────────────┘ └──────┬────────┘
                                     │
                              ┌──────▼───────┐
                              │ Token Stream │
                              └──────────────┘
```

### 3.2 请求状态

每个请求维护以下字段：

```python
class SequenceState:
    request_id: str
    input_ids: Tensor
    generated_ids: Tensor
    block_table: list[int]
    num_cached_tokens: int
    num_prefill_tokens: int
    num_generated_tokens: int
    status: str  # waiting/running/finished/aborted
    arrival_time: float
    first_token_time: float | None
    finish_time: float | None
```

### 3.3 Block Pool

Block Pool 负责管理固定大小的物理 KV Block：

```text
free_blocks: deque[int]
allocated_blocks: dict[request_id, list[int]]
block_size: 8 / 16 / 32 tokens
```

每个逻辑 Block 映射到一个物理 Block：

```text
logical block 0 -> physical block 17
logical block 1 -> physical block 3
logical block 2 -> physical block 24
```

请求结束时立即回收其物理 Block。需要防止以下错误：

- Block 重复分配；
- 请求结束后没有回收；
- 多个请求错误共享非 Prefix Cache Block；
- 追加 token 时越界；
- Prefill 和 Decode 同时写入同一位置。

### 3.4 调度策略

第一版使用简单、容易解释的 token budget：

```yaml
max_prefill_tokens_per_iteration: 4096
max_decode_batch_size: 16
max_running_requests: 32
```

每轮调度顺序：

1. 回收已经完成或取消的请求；
2. 为 waiting 请求尝试分配 Block；
3. 选取最多 `max_prefill_tokens_per_iteration` 个 Prefill token；
4. 选取最多 `max_decode_batch_size` 个 Decode 请求；
5. 执行模型前向；
6. 采样并更新每个请求状态；
7. 将生成 token 写入流式输出队列。

第一版优先保障 Decode 请求，避免 Prefill 长时间阻塞已经运行的请求。之后通过实验调整 Prefill budget。

### 3.5 Prefix Cache

对完整 KV Block 的 token 序列计算哈希：

```text
prefix_hash -> physical_block_id
```

只缓存完整 Block，不缓存未填满的尾部 Block。这样可以避免频繁修改共享 Block，也能简化引用计数管理。

最低要求：

- 使用 token IDs 计算稳定 hash；
- 维护引用计数；
- 请求释放时减少引用计数；
- 使用 LRU 或简单 FIFO 淘汰；
- 统计命中率和节省的 Prefill token 数。

---

## 4. 代码目录规划

```text
deepinfer-lite/
├─ README.md
├─ requirements.txt
├─ configs/
│  ├─ qwen3_4b_fp16.yaml
│  └─ qwen3_8b_int4.yaml
├─ deepinfer/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ request.py
│  ├─ tokenizer.py
│  ├─ model_runner.py
│  ├─ kv_cache.py
│  ├─ block_manager.py
│  ├─ prefix_cache.py
│  ├─ scheduler.py
│  ├─ sampler.py
│  ├─ metrics.py
│  ├─ engine.py
│  └─ server.py
├─ kernels/
│  └─ paged_decode_attention.py  # 进阶方向
├─ benchmarks/
│  ├─ run_single.py
│  ├─ run_concurrency.py
│  ├─ run_prefix_cache.py
│  ├─ workload.py
│  └─ summarize.py
├─ tests/
│  ├─ test_block_manager.py
│  ├─ test_kv_correctness.py
│  ├─ test_scheduler.py
│  └─ test_prefix_cache.py
├─ reports/
│  ├─ figures/
│  └─ final_report.md
└─ scripts/
   ├─ download_model.py
   ├─ run_server.py
   └─ run_all_benchmarks.py
```

---

## 5. 30 天执行计划

### 第 1 天：冻结范围和环境

任务：

- 确认 GPU、显存、CUDA、PyTorch 版本；
- 下载 Qwen3-0.6B 或 1.7B；
- 记录目标模型的 `num_layers`、`num_kv_heads`、`head_dim`、dtype；
- 初始化 Git 仓库；
- 创建最小 README。

产出：

- `environment.txt`；
- `configs/qwen3_*.yaml`；
- 模型配置和显存估算表。

验收：可以运行一次 Transformers 推理并保存输出。

#### 第 1 天实际环境记录（Kaggle，2026-10-29）

本次 Kaggle 会话已确认使用双 T4：

| 项目 | 实际值 |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| PyTorch 编译 CUDA | 12.8 |
| NVIDIA Driver | 580.178.04 |
| `nvidia-smi` CUDA Version | 13.0 |
| 可见 GPU 数量 | 2 |
| GPU 0 | Tesla T4，约 14.6 GiB，Compute Capability 7.5 |
| GPU 1 | Tesla T4，约 14.6 GiB，Compute Capability 7.5 |
| 初始 GPU 显存占用 | 每张约 3 MiB |
| 主实验设备 | `cuda:0` |
| 推荐 dtype | `torch.float16` |
| 模型目录 | `/kaggle/working/models/Qwen3-0.6B` |

说明：`torch.version.cuda=12.8` 表示 PyTorch 构建版本；`nvidia-smi` 显示的 13.0 表示驱动支持的 CUDA 上限，两者不需要相同。两张 T4 的显存默认不合并，主线实验将一张卡作为单卡推理设备，另一张卡用于并行基线或复现实验。

#### 第 1 天已完成事项

- [x] Kaggle GPU Accelerator 已启用；
- [x] 确认两张 Tesla T4 可见且没有其他进程占用；
- [x] Qwen3-0.6B 已下载到 `/kaggle/working/models/Qwen3-0.6B`；
- [x] 模型已使用 `cuda:0` 和 FP16 成功加载；
- [x] `enable_thinking=False` 的 Chat Template 调用成功；
- [x] 贪心生成 smoke test 成功返回文本；
- [x] 将 Transformers 的精确版本补录到 `environment.json`；
- [x] 保存本次环境快照和 smoke test 输出；
- [x] 记录模型配置中的 `num_layers`、`num_kv_heads`、`head_dim`。

#### 第 1 天最终快照结果

本次快照已经实际生成并下载：

```text
/kaggle/working/day1_snapshot/environment.json
/kaggle/working/day1_snapshot/smoke_test.json
/kaggle/working/day1_snapshot.zip
```

关键版本和模型配置如下：

| 项目 | 实际值 |
|---|---:|
| Transformers | 5.16.1 |
| Hugging Face Hub | 1.29.0 |
| Qwen3 模型类型 | `qwen3` |
| Hidden size | 1024 |
| Layers | 28 |
| Attention heads | 16 |
| KV heads | 8 |
| Head dimension | 128 |
| FP16 KV Cache / token | 114,688 bytes，约 112 KiB |
| Smoke test 输入 token | 24 |
| Smoke test 输出 token | 103 |
| Smoke test 状态 | passed |

KV Cache 单 token 的计算为：

```text
2 × num_layers × num_kv_heads × head_dim × 2 bytes
= 2 × 28 × 8 × 128 × 2
= 114,688 bytes/token
```

仅估算 KV Tensor 本体时，8K tokens 约占 896 MiB，16K tokens 约占 1.75 GiB；实际运行还要额外预留模型权重、Activation、CUDA Workspace 和框架开销。这组数字将作为后续 Block Size、KV Pool 容量和长上下文实验的初始估算。

本次 smoke test 使用的关键代码为：

```python
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    dtype=torch.float16,
).to("cuda:0").eval()

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)

with torch.inference_mode():
    output = model.generate(
        **tokenizer(prompt, return_tensors="pt").to("cuda:0"),
        max_new_tokens=64,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
```

本次生成成功，说明模型下载、Tokenizer、Qwen3 Chat Template、FP16 模型加载和基础 GPU 推理链路均已打通。该结果只作为环境 smoke test，不作为性能基线。

#### 依赖警告处理结果

安装依赖初期曾出现 `gradio`、`diffusers` 与 `huggingface-hub 0.36.2` 的版本冲突提示。重启并完成模型恢复后，最终快照显示：

```text
Transformers: 5.16.1
huggingface-hub: 1.29.0
```

当前模型下载、Tokenizer、Chat Template 和 GPU 推理均已成功，因此这组依赖在本项目中可用。后续 benchmark 固定记录这两个版本，不要在实验中途随意升级；如果 Kaggle 新会话恢复为不同版本，应重新生成 `environment.json`，并将版本差异作为实验元数据记录。

#### Notebook 重启后的恢复步骤

重启 Kernel 会清空 Python 变量，但同一 Kaggle 会话中的 `/kaggle/working` 文件通常仍保留。先运行下面的恢复单元，重新设置 `model_dir` 并创建 `tokenizer`、`model`；若本地目录中没有模型配置文件，则自动重新下载：

```python
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir = "/kaggle/working/models/Qwen3-0.6B"
model_path = Path(model_dir)

if not (model_path / "config.json").exists():
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="Qwen/Qwen3-0.6B",
        local_dir=model_dir,
    )

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    dtype=torch.float16,
).to("cuda:0").eval()

print("Loaded:", model_dir)
print("Device:", next(model.parameters()).device)
print("Dtype:", next(model.parameters()).dtype)
```

看到 `Device: cuda:0` 后，再运行下面的“第 1 天收尾单元”保存环境和 smoke test。若恢复单元提示找不到文件且下载失败，检查 Notebook 的 Internet 设置。

#### 第 1 天收尾单元

模型和 Tokenizer 已在当前 Notebook 内加载为 `model` 和 `tokenizer` 后，运行下面这个完整单元。它会重新执行一次固定 smoke test，读取模型配置，并在 `/kaggle/working/day1_snapshot/` 保存 `environment.json` 与 `smoke_test.json`，最后打包为 zip 并显示下载链接：

```python
import json
import sys
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
import huggingface_hub

assert torch.cuda.is_available(), "CUDA is unavailable"
assert "model" in globals() and "tokenizer" in globals(), (
    "请先运行模型加载单元，确保 model 和 tokenizer 已加载"
)

device = "cuda:0"
model_dir = "/kaggle/working/models/Qwen3-0.6B"
config = model.config
num_layers = int(config.num_hidden_layers)
num_attention_heads = int(config.num_attention_heads)
num_kv_heads = int(
    getattr(config, "num_key_value_heads", num_attention_heads)
)
head_dim = int(
    getattr(config, "head_dim", config.hidden_size // num_attention_heads)
)

messages = [{
    "role": "user",
    "content": "用三句话解释大语言模型推理中的 KV Cache。",
}]
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

torch.cuda.synchronize()
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
torch.cuda.synchronize()

input_token_count = int(inputs["input_ids"].shape[-1])
generated_token_ids = output[0, input_token_count:].detach().cpu().tolist()
generated_text = tokenizer.decode(
    generated_token_ids,
    skip_special_tokens=True,
)

devices = []
for i in range(torch.cuda.device_count()):
    prop = torch.cuda.get_device_properties(i)
    devices.append({
        "index": i,
        "name": prop.name,
        "memory_gib": round(prop.total_memory / 1024**3, 2),
        "compute_capability": f"{prop.major}.{prop.minor}",
    })

metadata = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "transformers": transformers.__version__,
    "huggingface_hub": huggingface_hub.__version__,
    "cuda_available": torch.cuda.is_available(),
    "device_count": torch.cuda.device_count(),
    "devices": devices,
    "model": "Qwen/Qwen3-0.6B",
    "model_dir": "/kaggle/working/models/Qwen3-0.6B",
    "device": device,
    "dtype": str(next(model.parameters()).dtype),
    "enable_thinking": False,
    "seed": 42,
    "model_config": {
        "model_type": config.model_type,
        "hidden_size": int(config.hidden_size),
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "kv_bytes_per_token_fp16": (
            2 * num_layers * num_kv_heads * head_dim * 2
        ),
    },
    "smoke_test": {
        "status": "passed",
        "prompt_tokens": input_token_count,
        "generated_tokens": len(generated_token_ids),
    },
}

smoke_test = {
    "timestamp_utc": metadata["timestamp_utc"],
    "model": metadata["model"],
    "prompt": messages[0]["content"],
    "prompt_tokens": input_token_count,
    "generated_tokens": len(generated_token_ids),
    "max_new_tokens": 128,
    "do_sample": False,
    "enable_thinking": False,
    "output": generated_text,
}

snapshot_dir = Path("/kaggle/working/day1_snapshot")
snapshot_dir.mkdir(parents=True, exist_ok=True)
(snapshot_dir / "environment.json").write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)
(snapshot_dir / "smoke_test.json").write_text(
    json.dumps(smoke_test, indent=2, ensure_ascii=False, default=str),
    encoding="utf-8",
)
zip_path = shutil.make_archive(
    "/kaggle/working/day1_snapshot",
    "zip",
    root_dir=snapshot_dir,
)

print("Environment and model config:")
print(json.dumps(metadata, indent=2, ensure_ascii=False, default=str))
print("\nSmoke test output:")
print(generated_text)
print("\nSaved files:", list(snapshot_dir.iterdir()))
print("Zip:", zip_path)

from IPython.display import FileLink, display
display(FileLink(zip_path))
```

点击单元输出的 zip 下载链接即可保存到本地；也可以在 Kaggle 的 Notebook Output 中找到 `day1_snapshot.zip`。之后每个 benchmark 都复制 `environment.json`，并额外记录 Block Size、Prefill Token Budget、Decode Batch Size 和 workload 配置。模型配置中的 `num_layers` 对应 `num_hidden_layers`，`num_kv_heads` 对应 `num_key_value_heads`；若配置没有单独提供 `head_dim`，代码按 `hidden_size // num_attention_heads` 计算。

### 第 2 天：建立统一 benchmark

任务：

- 实现固定输入长度和输出长度的 workload 生成器；
- 记录 TTFT、TPOT、E2E latency、吞吐和峰值显存；
- 输出 JSON 和 CSV；
- 固定随机种子和采样参数。

验收：相同配置连续运行 3 次，结果波动可解释。

#### 第 2 天实际环境记录（Kaggle，2026-11-03）

本日使用第 1 天保存的 Kaggle 双 T4 环境和已挂载的 Qwen3-0.6B Dataset。模型已在 Notebook 当前 Python Kernel 中加载，因此基准函数复用现有 `model` 与 `tokenizer`，没有为每个 workload 再加载模型副本。

| 项目 | 实际配置 |
|---|---|
| Python | 3.13.15 |
| PyTorch | 2.11.0+cu128 |
| PyTorch 编译 CUDA | 12.8 |
| Transformers | 5.16.1 |
| GPU | Tesla T4 × 2，每张约 14.56 GiB |
| 推理设备 | `cuda:0` |
| 模型 | Qwen/Qwen3-0.6B |
| 模型 Dataset 路径 | `/kaggle/input/datasets/yhw1027/qwen3-0-6b-model/Qwen3-0.6B` |
| 权重 dtype | FP16 |
| 解码方式 | Greedy，`do_sample=False` |
| Thinking 模式 | 关闭 |
| 随机种子 | 42 |

#### 第 2 天实现和运行方式

仓库中的统一基线实现位于：

```text
benchmarks/run_baseline.py
benchmarks/README.md
```

脚本采用串行单请求 Reference Benchmark：对完整 prompt 执行一次 Prefill，后续每步只输入一个新 token，并通过 `past_key_values` 继续 Decode。它记录 TTFT、TPOT、E2E、吞吐和 CUDA 峰值显存，并将配置与原始逐请求数据写入 JSON/CSV。

本次使用 Notebook 内已加载模型的调用方式：

```python
from benchmarks.run_baseline import run_benchmark

result = run_benchmark(
    model=model,
    tokenizer=tokenizer,
    model_dir="/kaggle/input/datasets/yhw1027/qwen3-0-6b-model/Qwen3-0.6B",
    output_dir="/kaggle/working/day2_short",
    input_tokens=128,
    max_new_tokens=128,
    warmup=2,
    num_requests=10,
    device="cuda:0",
    seed=42,
)
```

`run_benchmark` 接收当前 Kernel 中已经加载的 `model` 和 `tokenizer`，不会执行 `from_pretrained()`。注意：如果 Kernel 重启，变量会清空，需先从 Dataset 路径重新加载模型；模型文件仍由 Kaggle Dataset 提供，不需要重新从 Hugging Face 下载。

#### 第 2 天已完成事项

- [x] 基线脚本支持固定输入 token 数、输出 token 数、请求数和 warmup 次数；
- [x] 固定 greedy decoding、`seed=42` 和 thinking 关闭；
- [x] 实现 Prefill 后逐 token Decode，并记录 TTFT、TPOT、E2E；
- [x] 记录峰值 allocated/reserved CUDA 显存；
- [x] 输出 `metadata.json`、`summary.json`、`requests.json` 和 `requests.csv`；
- [x] 支持复用 Notebook 当前进程中加载的模型；
- [x] 完成 32 输入 / 16 输出、2 个请求的 smoke benchmark；
- [x] 完成输入长度 128、512、2048 tokens 的首轮实测；
- [x] 每种 workload 完成三个独立 trial，并生成汇总 CSV；
- [x] 下载并持久化 `day2_complete_results` 汇总结果；
- [ ] 补齐每个 trial 的逐请求 `requests.json` 和 `requests.csv` 原始归档。

Smoke benchmark 配置为 32 个输入 tokens、最多生成 16 个 tokens、warmup 1 次、正式请求 2 个。汇总吞吐约 20.21 tokens/s、TTFT mean 56.63 ms、TPOT mean 48.95 ms/token、E2E mean 790.83 ms、峰值 allocated 显存 1161.22 MiB。结果保存于 `/kaggle/working/day2_loaded_model`。该小样本只用于确认 benchmark 函数能执行，不纳入正式 workload 对比。

#### 第 2 天首轮实测结果

三种 workload 均使用 warmup 2 次、正式请求 10 个、每个请求生成 128 tokens、greedy decoding、`seed=42`，结果保存于：

```text
/kaggle/working/day2_short
/kaggle/working/day2_medium
/kaggle/working/day2_long
```

| workload | 输入 tokens | 吞吐 tokens/s | TTFT mean | TTFT P99 | TPOT mean | E2E mean | 峰值 allocated | 峰值 reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 128 | 21.174 | 52.67 ms | 55.01 ms | 47.18 ms | 6044.42 ms | 1219 MiB | 1252 MiB |
| medium | 512 | 21.071 | 85.21 ms | 86.67 ms | 47.16 ms | 6073.91 ms | 1397 MiB | 1470 MiB |
| long | 2048 | 18.948 | 787.56 ms | 853.74 ms | 46.98 ms | 6754.42 ms | 2062 MiB | 2768 MiB |

#### 初步观察

1. 输入长度从 128 增至 512 时，TTFT 从 52.67 ms 增至 85.21 ms；增至 2048 时达到 787.56 ms，长输入的主要新增耗时出现在 Prefill。
2. 三组 TPOT mean 都约为 47 ms/token，说明本轮测试中 Decode 每 token 延迟对输入长度变化不敏感。
3. 串行吞吐从约 21.17 tokens/s 降至 18.95 tokens/s，长 Prefill 拉低了整体生成吞吐。
4. 峰值显存随输入长度上升。allocated/reserved 包含模型、KV Cache、临时 Tensor 和 PyTorch allocator 行为，不能解释为 KV Cache 单独占用。
5. 本脚本采用串行 closed-loop 请求。汇总吞吐不是服务 QPS；这些数据也不足以代表在线并发 P99。

首轮数据之后已完成每种 workload 的两个重复 trial，三次 trial 汇总保存在：

```text
C:\Users\86176\Desktop\AI Infra\day2_complete_results\day2_all_trials.csv
```

该 CSV 包含 short、medium、long 各 3 次 trial。以下统计使用样本标准差（`n-1`）和相对标准差 `std / mean`：

| workload | 吞吐 mean | 吞吐 std | 吞吐相对 std | TTFT mean | TTFT 相对 std | TPOT mean | E2E mean | allocated 显存 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short | 24.103 tokens/s | 0.080 | 0.33% | 46.559 ms | 1.05% | 41.445 ms/token | 5310.115 ms | 1219 MiB |
| medium | 23.931 tokens/s | 0.112 | 0.47% | 83.215 ms | 4.69% | 41.456 ms/token | 5348.093 ms | 1397 MiB |
| long | 20.956 tokens/s | 0.121 | 0.58% | 811.707 ms | 1.77% | 41.699 ms/token | 6107.514 ms | 2062 MiB |

重复 trial 的结果表明：

1. 三组吞吐相对标准差均小于 0.6%，串行 closed-loop 基线的吞吐较稳定；
2. TPOT mean 均约 41.4～41.7 ms/token，输入长度变化没有造成明显 Decode 波动；
3. TTFT 随输入长度从 128 增至 2048 tokens 明显增加，long 的 mean 约 811.7 ms，是后续 Prefill 优化的主要观察点；
4. allocated 显存三次完全一致，分别约为 1219、1397、2062 MiB；
5. short 和 medium 的 reserved 显存分别出现 1252/2768 MiB、1470/2768 MiB 的变化，主要反映 PyTorch CUDA allocator 的缓存和复用状态，不应直接解释为 KV Cache 需求变化；long 的 reserved 显存稳定在 2768 MiB。

本次结果可用于第 2 天基线结论，但仍是单请求串行执行，不能代表连续批处理或在线服务 QPS/P99。后续 Paged KV Cache 和调度器实验需保持相同模型、GPU、输入长度、输出长度和采样设置，才能公平对比。

#### 保存 Kaggle 结果

在 Kaggle 中可将结果目录打包，下载或保存为 Notebook Output：

```python
import shutil

for name in ["short", "medium", "long"]:
    source = f"/kaggle/working/day2_{name}"
    shutil.make_archive(source, "zip", root_dir=source)
```

每个结果目录应包含：

```text
metadata.json
summary.json
requests.json
requests.csv
```

第 2 天进入下一阶段前的退出条件：三种 workload 各完成至少三个 trial、汇总 CSV 已保存、版本和设备信息可追溯，已满足。当前下载的 `day2_complete_results` 只包含汇总 CSV，逐请求原始 JSON/CSV 尚未归档；此项作为补档任务，不影响进入第 3 天。

### 第 3 天：实现模型 Runner 骨架

#### 实现内容

- `deepinfer_lite/model_runner.py` 提供 `ModelRunner.from_pretrained()`，负责加载 tokenizer 和 causal LM，并将模型放到指定设备；默认 CUDA 使用 FP16，CPU 使用 FP32。
- `ModelRunner.prefill(input_ids, attention_mask=None)` 一次处理完整 prompt，返回 `PrefillOutput(logits, kv_cache)`；`logits` 是 prompt 最后位置用于预测首 token 的 logits。
- `ModelRunner.decode(input_ids, kv_cache)` 每个 batch row 只接收一个 token，传入已有 cache，返回 `DecodeOutput(logits, kv_cache)`。调用方用 logits 选择下一 token，再进入下一次 Decode。
- `KVCache` 包装 Transformers `past_key_values`，并记录逻辑序列长度、batch size、device 和 attention mask。当前 cache 后端仍由 Transformers 管理，尚未实现独立 KV Tensor 或分页分配。
- `examples/day3_runner_smoke.py` 提供 Kaggle 验收入口，运行一次 Prefill 和至少 32 次单 token Decode，并检查最终 cache 长度。

#### Kaggle 验收

确认 Notebook 当前目录是仓库根目录，模型 Dataset 已挂载，然后执行：

```python
%env PYTHONPATH=/kaggle/working/deepinfer-lite
!python examples/day3_runner_smoke.py \
  --model-dir /kaggle/input/datasets/yhw1027/qwen3-0-6b-model/Qwen3-0.6B \
  --device cuda:0 \
  --decode-steps 32
```

预期 JSON 中 `status` 为 `passed`、`decode_calls` 为 `32`，且 `cache_sequence_length` 等于 `prefill_tokens + 32`。模型下载和加载只发生在该 smoke 脚本进程中；重复验收时应确认模型仍从 Kaggle Dataset 本地路径读取。

#### 完成状态

- [x] tokenizer 和模型加载接口；
- [x] Prefill 全 prompt 前向；
- [x] Decode 单 token 前向和 cache 回传；
- [x] 明确 KV Cache handle 的输入输出字段；
- [x] 添加 32 步 smoke 验收脚本并通过 Python 语法检查；
- [x] 在 Kaggle Tesla T4 (`cuda:0`) 上完成实际模型验收。

#### 第 3 天 Kaggle 实测结果（2026-11-04）

使用 Qwen3-0.6B FP16，在 `cuda:0` 上运行 `day3_runner_smoke.py --decode-steps 32`，输出如下：

```json
{
  "status": "passed",
  "device": "cuda:0",
  "prefill_tokens": 8,
  "decode_calls": 32,
  "cache_sequence_length": 40,
  "next_token_id": 30,
  "dtype": "torch.float16"
}
```

验收结论：单请求完成一次 Prefill 和 32 次 Decode，且 `cache_sequence_length = prefill_tokens + decode_calls = 40`，第 3 天退出条件满足。

### 第 4 天：实现朴素 KV Cache

#### 实现内容

- `KVCache` 为每个请求保存独立的 `past_key_values`、逻辑序列长度、batch size、设备和 attention mask。
- 每层 Key/Value 都检查为 `[batch, heads, seq, head_dim]`，Key 与 Value 形状一致，序列维度等于逻辑长度，并且 Tensor 是连续存储；当前底层仍使用 Transformers 返回的连续 KV Tensor。
- `KVCache.append()` 接收一次 Decode 后的完整 KV 状态，校验长度增加 1 后再更新缓存，形成朴素的连续追加路径。
- `ModelRunner.generate_greedy()` 使用 logits `argmax` 采样，遇到模型配置中的 EOS 立即停止，并限制最多生成 `max_new_tokens` 个 token。
- `examples/day4_cache_acceptance.py` 在同一模型和同一输入上分别运行 Transformers `generate(do_sample=False)` 与 Runner 逐 token 生成，比较完整生成 token 序列。

#### Kaggle 验收

```python
%env PYTHONPATH=/kaggle/working/deepinfer-lite
!python examples/day4_cache_acceptance.py \
  --model-dir /kaggle/input/datasets/yhw1027/qwen3-0-6b-model/Qwen3-0.6B \
  --device cuda:0 \
  --max-new-tokens 32
```

验收脚本还会输出实际生成 token 数、Decode 调用次数、EOS 状态、最终 cache 长度和 KV Tensor 字节数。

#### 完成状态

- [x] 为单请求保存连续 KV Tensor，并校验形状和长度；
- [x] 支持 Decode 后追加 KV 状态；
- [x] 实现 greedy sampling；
- [x] 处理 EOS 和 `max_new_tokens`；
- [x] 添加 Transformers `generate` 一致性验收脚本；
- [x] 在 Kaggle T4 上完成 Day 4 实际验收并保存 JSON 输出。

验收退出条件：Runner 生成的 token 序列与 Transformers `generate` 在 greedy 模式下完全一致。

#### 第 4 天 Kaggle 实测结果（2026-11-04）

使用 Qwen3-0.6B FP16，在 `cuda:0` 上运行 `day4_cache_acceptance.py --max-new-tokens 32`，输出如下：

```json
{
  "status": "passed",
  "device": "cuda:0",
  "prompt_tokens": 8,
  "requested_max_new_tokens": 32,
  "reference_generated_tokens": 32,
  "runner_generated_tokens": 32,
  "tokens_match": true,
  "decode_calls": 31,
  "eos_reached": false,
  "cache_sequence_length": 39,
  "kv_allocated_mib": 4.265625,
  "dtype": "torch.float16"
}
```

验收结论：Runner 与 Transformers `generate(do_sample=False)` 生成的 32 个 token 完全一致。`decode_calls` 为 31，是因为最后一个 token 由最后一次 logits 直接选出，尚未再次送入模型；因此最终 cache 长度为 `8 + 31 = 39`。第 4 天退出条件满足。

### 第 5 天：补充正确性测试

#### 测试文件

产出：`tests/test_kv_correctness.py`。测试默认读取 Kaggle Dataset 中的 Qwen3-0.6B；可用 `DEEPINFER_MODEL_DIR` 和 `DEEPINFER_TEST_DEVICE` 覆盖模型路径和设备。

```python
%env PYTHONPATH=/kaggle/working/deepinfer-lite
!DEEPINFER_MODEL_DIR=/kaggle/input/datasets/yhw1027/qwen3-0-6b-model/Qwen3-0.6B \
  DEEPINFER_TEST_DEVICE=cuda:0 \
  pytest -q tests/test_kv_correctness.py
```

测试：

- 输入长度 1、128、512、2048；
- 输出长度 1、16、128；
- 单请求和两个请求交错执行；
- EOS 提前结束；
- 未来分页缓存 block 前、满、后的连续追加边界；
- 非法追加不会破坏原有 cache 状态。

其中长度矩阵共 12 组，逐 token 对比 Runner 和 Transformers `generate(do_sample=False)` 的完整输出；模型相关测试在没有模型或 CUDA 时会跳过，缓存边界单元测试不依赖 GPU。

#### 完成状态

- [x] 建立长度矩阵和 greedy 输出一致性测试；
- [x] 建立两个独立请求交错 Decode 测试；
- [x] 建立 EOS 提前结束测试；
- [x] 建立连续 KV Cache 的 block 边界和非法追加测试；
- [x] 在 Kaggle T4 上运行完整测试，24 项全部通过。

#### 第 5 天 Kaggle 测试结果（2026-11-05）

在 Kaggle Notebook 的 Qwen3-0.6B / Tesla T4 环境运行 `pytest -q tests/test_kv_correctness.py`：

```text
24 passed in 69.84s
```

24 项包括 12 组输入/输出长度组合、两个请求交错 Decode、EOS 提前停止、9 组 block 边界追加，以及非法追加后的状态保持。长度矩阵中 Runner 与 Transformers greedy 输出逐 token 一致。第 5 天退出条件满足。

### 第 6 天：完成基础显存和性能基线

#### 实现内容

新增 `benchmarks/run_day6_baseline.py`，一次加载模型后固定使用 greedy 解码，并对每个配置运行多个 trial：

- 单请求 Prefill 延迟；
- 单 token Decode 延迟；
- 1/4/8 并发下的吞吐；
- 不同上下文长度下的峰值显存。

这里的 batch 1/4/8 是静态批处理：同一批请求先完成 Prefill，再逐轮 Decode，作为连续批处理的参考基线，不代表动态请求到达场景。Decode 计时只覆盖单 token 前向，但峰值显存包含测量开始前已经存在的 KV Cache。

#### Kaggle 运行

```python
%env PYTHONPATH=/kaggle/working/deepinfer-lite
!python benchmarks/run_day6_baseline.py \
  --model-dir /kaggle/input/datasets/yhw1027/qwen3-0-6b-model/Qwen3-0.6B \
  --output-dir /kaggle/working/day6_baseline \
  --contexts 128,512,2048 \
  --batch-sizes 1,4,8 \
  --max-new-tokens 32 \
  --trials 3 \
  --warmup 1 \
  --device cuda:0 \
  --seed 42
```

产出目录包含 `metadata.json`、逐 trial 原始数据 `baseline_trials.csv`、汇总表 `baseline_summary.csv`、`summary.json` 和四联图 `baseline_plots.png`。

#### 完成状态

- [x] 实现单请求 Prefill 延迟测量；
- [x] 实现单 token Decode 延迟测量；
- [x] 实现静态 batch 1/4/8 吞吐测量；
- [x] 记录不同上下文长度的峰值显存；
- [x] 输出 CSV、JSON 和 PNG 图表；
- [x] 在 Kaggle T4 上运行完整 baseline 并把结果表写回本节。

#### 第 6 天 Kaggle 实测结果（2026-11-05）

结果归档：

```text
C:\Users\86176\Desktop\AI Infra\day6_baseline.zip
C:\Users\86176\Desktop\AI Infra\day6_baseline\day6_baseline\baseline_summary.csv
C:\Users\86176\Desktop\AI Infra\day6_baseline\day6_baseline\baseline_plots.png
reports/day6_baseline/
```

本次实测配置为 Qwen3-0.6B FP16、Tesla T4、`cuda:0`、输入长度 128/512/2048、静态 batch 1/4/8、输出长度 32、3 次 trial、1 次 warmup。汇总均值如下：

**单请求 Prefill、单 token Decode 和显存**

| 上下文 token | Prefill 平均延迟 | Prefill allocated 显存 | Decode 平均延迟 | Decode allocated 显存 |
|---:|---:|---:|---:|---:|
| 128 | 53.67 ms | 1197 MiB | 43.88 ms | 1212 MiB |
| 512 | 88.48 ms | 1350 MiB | 42.45 ms | 1412 MiB |
| 2048 | 830.48 ms | 2062 MiB | 45.58 ms | 2177 MiB |

**静态批处理生成吞吐**

| 上下文 token | batch=1 | batch=4 | batch=8 |
|---:|---:|---:|---:|
| 128 | 22.59 tokens/s | 89.12 tokens/s | 169.97 tokens/s |
| 512 | 22.58 tokens/s | 73.65 tokens/s | 106.20 tokens/s |
| 2048 | 14.39 tokens/s | 19.71 tokens/s | 21.28 tokens/s |

观察结论：

1. Prefill 是长上下文的主要延迟瓶颈，输入从 512 增至 2048 token 时平均延迟从 88.48 ms 增至 830.48 ms；
2. 单 token Decode 延迟在三个上下文长度下约为 42–45.6 ms，变化明显小于 Prefill；
3. 短上下文适合通过静态 batch 提升吞吐，128 token 时 batch=8 达到 169.97 tokens/s，约为 batch=1 的 7.5 倍；
4. 2048 token 时 batch=8 的吞吐只有 21.28 tokens/s，且 allocated 显存达到 8363 MiB，说明长上下文下显存和带宽压力限制了批处理扩展；
5. 峰值 reserved 显存受 PyTorch allocator 缓存影响，A/B 对比应优先关注 allocated 显存，同时保留两者。

本次 `metadata.json` 记录的环境是 Python 3.13.16、PyTorch 2.11.0+cu130、Transformers 5.20.2；第 1 天快照为 PyTorch 2.11.0+cu128、Transformers 5.16.1。两次结果不应直接作为严格 A/B 数字，后续基线与 Paged KV Cache 对比必须在同一 Kaggle 环境和同一代码版本下重跑。

第 6 天退出条件满足：原始 trial、汇总 CSV、JSON 元数据和 PNG 图表均已保存。

### 第 7 天：整理设计文档

产出：`docs/day7_design.md`。

文档已整理以下内容：

- KV Cache 的显存计算；
- Prefill 和 Decode 的计算特征；
- 当前实现的限制；
- Paged KV Cache 的改造点。

#### 已记录的关键结论

- Qwen3-0.6B FP16 的 KV Cache 理论占用为 `114,688 bytes/token`，计算式为 `2 × 28 × 8 × 128 × 2`；
- Prefill 主要处理完整输入并写入全部 KV，长上下文下更容易成为 compute-bound 瓶颈；
- Decode 每轮处理一个 token，主要读取历史 KV，更接近 memory-bound；
- 当前代码是可验证的连续 KV Reference Path，不包含独立 Block Pool、Block Table、Continuous Batching 或真实分页 attention kernel；
- Paged KV Cache 的改造重点是 Block Pool、逻辑到物理 block 映射、尾 block 追加、释放和碎片率统计，同时保持 Prefill/Decode 上层接口稳定。

验收：文档可以在 5 分钟内讲清当前代码、实测证据、限制和下一步设计；第 8 天直接从 Block Pool 和 Block Table 开始实现。

### 第 8 天：实现 Block Pool

#### 实现内容

新增 `deepinfer_lite/block_pool.py`，当前只管理固定大小物理 Block 的 ID 和生命周期，不持有 GPU KV Tensor：

- 实现固定 Block Size；
- `BlockPool(num_blocks, block_size)` 创建固定容量的 Pool，Block ID 为稳定的整数；
- `allocate(num_blocks, request_id)` 申请一组 Block，并返回逻辑顺序对应的物理 ID；
- `release(request_id)` 释放请求持有的全部 Block，`release_blocks()` 支持部分释放；
- `free_count()`、`allocated_count()`、`allocation_count()` 和 `snapshot()` 查询资源状态；
- 记录请求到 Block ID 的映射，并检查重复申请、重复释放、越权释放和容量不足；
- 申请失败先完成容量校验，保证失败不会改变 Pool 状态。

Block Pool 目前不代表真实 PagedAttention：第 10 天接入 KV Tensor 后，才会将物理 ID 映射到实际 GPU Block。

测试：

- 全部申请后释放；
- 交错申请和释放；
- Block 不足时返回明确错误；
- 申请量超过总容量时状态保持不变；
- 重复申请、重复释放和越权释放；
- 查询 owner、请求映射和快照。

产出：`tests/test_block_pool.py`。

#### 完成状态

- [x] 实现固定 Block Size 和确定性的 Block ID；
- [x] 实现申请、释放、部分释放和查询；
- [x] 记录 free/allocated/request allocation 数量；
- [x] 增加重复释放、重复申请、越权释放和超额申请检查；
- [x] 添加全部申请释放、交错申请释放和容量不足测试；
- [x] 在 Kaggle 运行第 8 天测试，12 项全部通过。

#### 第 8 天测试结果（2026-11-08）

在 Kaggle 仓库根目录运行 `pytest -q tests/test_block_pool.py`：

```text
12 passed in 7.65s
```

结果覆盖全量申请释放、交错申请释放与 block 复用、容量不足和超额申请原子性、重复申请/释放、越权释放、匿名申请回收及 Pool 配置检查。第 8 天退出条件满足。

### 第 9 天：实现 Block Table

#### 实现内容

新增 `deepinfer_lite/block_table.py`，以一个请求为边界维护逻辑 Block 到物理 Block 的
有序映射。Block Table 只负责分页元数据和生命周期，不直接操作 GPU KV Tensor，便于在
接入分页 KV 写入前先验证分配器的正确性。

- `BlockTable(pool, request_id, sequence_length=0)` 按
  `ceil(sequence_length / block_size)` 申请初始物理 Block；空请求不申请 Block；
- `append_token()` 和 `append_tokens(n)` 更新序列长度，只有跨过当前尾 Block 边界时
  才通过 `BlockPool.extend()` 申请新物理 Block；
- `logical_to_physical(logical_block_id)` 提供逻辑 Block 查询，
  `physical_for_token(token_position)` 提供 token 到物理 Block 的查询；
- `release()` 回收请求持有的全部物理 Block，并将请求标记为结束；释放后的请求不能再
  追加、释放或查询映射；
- `snapshot()` 记录序列长度、逻辑 Block 数、已分配 token 容量、尾部内部碎片和释放状态；
- `BlockPool.extend()` 为已分配请求追加物理 Block；申请数量不足时先校验容量，保证
  BlockTable 长度和映射不会被部分修改；
- `deepinfer_lite/__init__.py` 导出 `BlockTable`、快照和释放状态异常，后续测试可直接
  从包根目录导入。

#### 测试覆盖

新增 `tests/test_block_table.py`，覆盖：

- 追加 token 跨越 Block 边界时自动扩容；
- 初始序列长度的 Block 数和尾部浪费计算；
- 逻辑 Block、token 位置和物理 Block 的映射查询；
- 空请求结束时不泄漏 Pool 容量；
- 释放后访问保护和释放状态快照；
- Pool 不足时扩容失败的原子性；
- 固定随机种子下 100 次随机长度请求的创建、追加和释放。

#### 验收命令

在 Kaggle Notebook 的仓库根目录运行：

```python
%env PYTHONPATH=/kaggle/working/deepinfer-lite
!pytest -q tests/test_block_table.py
```

随机生命周期测试的强验收条件是每个请求释放后均满足：

```text
free_count == 64
allocated_count == 0
allocation_count == 0
```

本地已完成 Python 编译检查和 `git diff --check`。

#### 第 9 天测试结果（2026-11-08）

在 Kaggle 仓库根目录运行 `pytest -q tests/test_block_table.py`：

```text
.......                                                                  [100%]
7 passed in 7.57s
```

7 项测试覆盖 Block 边界扩容、初始分配、映射查询、释放后访问保护、空请求、容量不足
时的原子性，以及随机长度请求运行 100 次后的 Pool 容量恢复。第 9 天退出条件满足。

### 第 10 天：分页 KV 写入

#### 实现内容

新增 `deepinfer_lite/paged_kv_cache.py`，完成 PyTorch Tensor 上的分页 KV Reference Path：

- `PagedKVStorage` 为每层分配 `[num_blocks, block_size, kv_heads, head_dim]` 的物理 KV
  Block，所有请求共享同一个物理存储；
- `PagedKVCache.from_contiguous()` 在 Prefill 后按 Block Table 将完整连续 KV 写入物理
  Block，并检查层数、形状、dtype、设备和 batch size；
- `append_from_contiguous()` 在 Decode 后只取完整 KV 的最后一个 token，追加到当前尾
  Block；跨越边界时由 BlockTable 自动申请新 Block；
- `to_contiguous()` 按逻辑 token 顺序读取 Block，gather 成 Transformers 使用的连续
  `[1, kv_heads, sequence_length, head_dim]` KV；
- `PagedModelRunner` 将现有 `ModelRunner` 接入该 Reference Path：每轮 Decode 前 gather
  连续 KV，完成模型前向后再写回新增 token；
- Prefill 写入异常、Decode KV 形状错误和 Block 不足均保持原状态或释放已申请资源；
- 导出 `PagedKVCache`、`PagedKVStorage`、`PagedModelRunner` 及相关输出/异常类型。

#### 测试覆盖

新增 `tests/test_paged_kv_cache.py`，覆盖 3/4/8 Block Size 下的 Prefill 写入、跨边界
读取、Decode 追加、非法 Decode 原子性、Prefill 申请失败无泄漏和释放回收。新增模型
测试 `test_paged_reference_generation_matches_contiguous_runner`，用 Qwen3-0.6B 对比
分页 Reference Path 与现有连续 Runner 的 greedy token。

#### Kaggle 验收命令

在仓库根目录执行：

```python
%env PYTHONPATH=/kaggle/working/deepinfer-lite
%env DEEPINFER_MODEL_DIR=/kaggle/input/datasets/yhw1027/qwen3-0-6b-model/Qwen3-0.6B
%env DEEPINFER_TEST_DEVICE=cuda:0
!pytest -q tests/test_paged_kv_cache.py
!pytest -q tests/test_kv_correctness.py -k paged_reference
```

验收标准：分页路径生成 token 与连续 Runner 完全一致；跨 Block 边界没有 token 丢失或
重复；请求释放后 Pool 的 `free_count` 恢复初始值，`allocated_count` 和
`allocation_count` 均为 0。该实现是 gather 型 Reference Path，不宣称 Attention Kernel
已经直接读取分页 KV。

#### 第 10 天模型一致性结果（2026-11-09）

在 Kaggle 使用 Qwen3-0.6B、`cuda:0` 运行：

```text
1 passed, 24 deselected in 12.72s
```

`test_paged_reference_generation_matches_contiguous_runner` 通过，说明分页 Reference
Path 与连续 Runner 的 greedy 生成 token 完全一致。

#### 第 10 天分页读写测试结果（2026-11-09）

在 Kaggle 仓库根目录运行 `pytest -q tests/test_paged_kv_cache.py`：

```text
.........................                                                [100%]
25 passed in 7.66s
```

测试覆盖 Prefill 按 Block 写入及 gather 还原、不同 Block Size 和跨边界场景、Decode
只追加新 token、非法输入及 Pool 耗尽时的状态保持、分配失败无泄漏和释放回收。结合
上面的模型 greedy 对照测试，第 10 天验收条件满足。当前仍是 gather 型 Reference Path，
不代表 Attention Kernel 直接读取分页 KV。

### 第 11 天：不同 Block Size 实验

测试 Block Size：8、16、32。

记录：

- 内部碎片率；
- Block Table 大小；
- Prefill/Decode 延迟；
- 峰值显存；
- Block 管理 CPU 开销。

选择一个默认值，通常从 16 开始。

### 第 12 天：压力测试分页管理

任务：

- 长短请求随机混合；
- 频繁创建和结束请求；
- 反复分配和释放；
- 人为制造显存接近上限的场景。

验收：无越界、无泄漏、无错误复用。

### 第 13 天：分页 KV 第一阶段里程碑

完成：

- Paged KV Cache；
- Block Pool；
- Block Table；
- 正确性测试；
- Block Size 对比图。

Git checkpoint：

```text
milestone/paged-kv-correct
```

### 第 14 天：实现请求状态机

状态：

```text
WAITING -> RUNNING -> FINISHED
                   -> ABORTED
```

任务：

- 实现 waiting/running/finished 队列；
- 记录 arrival、first token、finish 时间；
- 实现请求取消和超时的基本处理。

### 第 15 天：实现 Decode Batch

任务：

- 将多个 running 请求组织成 Decode batch；
- 每轮为每个请求生成一个 token；
- 完成请求立即从 batch 移除；
- 支持不同请求具有不同历史长度。

验收：4 个不同长度请求可以在同一调度循环中正确完成。

### 第 16 天：实现动态加入请求

任务：

- Decode 过程中接受新请求；
- 仅在有足够 Block 时加入；
- 新请求先进入 waiting；
- 调度器每轮更新 running 集合。

### 第 17 天：加入 Prefill Token Budget

参数：

```yaml
max_prefill_tokens_per_iteration: 2048 / 4096 / 8192
max_decode_batch_size: 8 / 16
```

分别测试不同预算下的吞吐、TTFT 和 P99。

### 第 18 天：Continuous Batching 里程碑

完成：

- 动态请求进入和退出；
- Prefill/Decode 调度；
- 基础 token budget；
- 并发正确性测试。

Git checkpoint：

```text
milestone/continuous-batching
```

### 第 19 天：实现 Prefix Cache 索引

任务：

- 对完整 Block 的 token IDs 计算 hash；
- 建立 hash 到物理 Block 的索引；
- 增加引用计数；
- 实现命中和未命中路径。

### 第 20 天：实现 Prefix Cache 淘汰

第一版使用简单 LRU 或 FIFO。

需要防止：

- 正在被请求使用的 Block 被淘汰；
- 引用计数为零后仍长期占用；
- 共享 Block 被错误写入。

### 第 21 天：Prefix Cache 实验

测试：

- 共享 512 token 前缀；
- 共享 2K token 前缀；
- 共享 8K token 前缀；
- 完全随机前缀；
- 50% 请求共享前缀。

记录命中率、TTFT、Prefill token 数和峰值显存。

### 第 22 天：FastAPI 服务

实现：

```text
POST /generate
POST /generate_stream
GET  /metrics
```

`/generate_stream` 使用 SSE 或逐 token JSON 输出。

### 第 23 天：服务稳定性

增加：

- 最大并发限制；
- 请求超时；
- 请求取消；
- 错误响应；
- request ID；
- 基本日志。

### 第 24 天：压测脚本

实现开放式请求发生器，支持：

- 固定并发；
- 固定到达率；
- 随机输入长度；
- 随机输出长度；
- 共享前缀比例。

输出 P50/P90/P99 延迟和吞吐曲线。

### 第 25 天：服务化里程碑

完成：

- 流式接口；
- 动态批处理；
- 并发限制；
- 压测脚本；
- 基本 metrics。

Git checkpoint：

```text
milestone/servable-engine
```

### 第 26 天：Profiling

使用 PyTorch Profiler 观察：

- Prefill GEMM；
- Decode Attention；
- KV Cache 读写；
- Tensor 拼接和 `.contiguous()`；
- CPU Scheduler；
- Host-to-Device 同步。

输出至少一张优化前 timeline。

### 第 27 天：完成一个进阶方向

二选一：

#### 方向 A：Triton Decode Attention

- 根据 `block_table` 直接读取分页 KV；
- 对比先拼接连续 KV 的 reference path；
- 测试 batch size 和上下文长度变化。

#### 方向 B：KV Cache 量化

- 选择 INT8 或 FP8；
- 比较显存、TPOT、吞吐和输出质量；
- 保持权重格式不变，避免混淆变量。

如果第 26 天发现瓶颈主要在 GEMM 或模型权重加载，选择 KV Cache 量化更稳妥；如果瓶颈明显在分页 Decode Attention，选择 Triton。

### 第 28 天：Qwen3-4B 完整实验

固定实验矩阵：

| 输入长度 | 输出长度 | 并发 |
|---:|---:|---:|
| 128 | 128 | 1/4/8/16 |
| 2K | 128 | 1/4/8/16 |
| 8K | 256 | 1/4/8 |
| 16K | 256 | 1/4/8，显存允许时 |

对比：

1. Transformers 基线；
2. 朴素 KV + 静态 batch；
3. Paged KV + Continuous Batching；
4. 完整版本。

### 第 29 天：Qwen3-8B 扩展实验

使用 Qwen3-8B INT4 或机器能承受的最大模型，至少完成：

- 512 token 输入；
- 2K token 输入；
- 8K token 输入；
- 并发 1/4/8；
- Paged KV 与朴素 KV 对比。

如果显存不够，使用 Qwen3-4B 完成最终实验，并把 8B 作为未完成的扩展项，不要伪造结果。

### 第 30 天：整理报告和 Demo

完成：

- README；
- 设计文档；
- benchmark 结果；
- 图表；
- 限制和后续工作；
- 3 分钟 Demo；
- 简历 bullet。

Git checkpoint：

```text
v1.0-reproducible-benchmark
```

---

## 6. Benchmark 设计

### 6.1 必须记录的指标

#### TTFT

```text
TTFT = first_token_time - request_arrival_time
```

表示用户等待首 token 的时间，主要受 Prefill 和排队影响。

#### TPOT

```text
TPOT = (finish_time - first_token_time) / (generated_tokens - 1)
```

表示 Decode 阶段平均每 token 时间。

#### E2E Latency

```text
E2E = finish_time - request_arrival_time
```

#### 吞吐

```text
throughput = total_generated_tokens / wall_time
```

#### 内部碎片率

```text
fragmentation = 1 - valid_tokens / allocated_token_capacity
```

#### KV 利用率

```text
kv_utilization = valid_tokens / allocated_token_capacity
```

### 6.2 负载类型

至少使用四类 workload：

1. **短请求**：128 input / 128 output；
2. **长输入**：8K input / 128 output；
3. **长输出**：512 input / 1K output；
4. **混合请求**：随机输入长度和输出长度。

Prefix Cache 另外使用：

- 共享长 system prompt；
- 共享文档前缀；
- 完全随机 prompt；
- 50% 请求共享前缀。

### 6.3 结果图表

最终报告至少包含：

- 吞吐随并发变化曲线；
- TTFT 随输入长度变化曲线；
- TPOT 随并发变化曲线；
- P50/P99 延迟曲线；
- 峰值显存对比；
- Block Size 对碎片率的影响；
- Prefix Cache 命中率与 TTFT 的关系；
- Profiling 优化前后时间线。

---

## 7. 正确性验收标准

### 7.1 输出正确性

- greedy 模式下，分页 KV 与连续 KV 生成 token 一致；
- 单请求和多请求结果一致；
- 不同 Block Size 结果一致；
- Prefix Cache 命中和未命中结果一致；
- thinking mode 关闭时输出长度和停止条件可控。

### 7.2 资源正确性

- 100 次请求后没有显存持续增长；
- 请求完成后所有非共享 Block 被回收；
- Prefix Cache 的引用计数始终非负；
- Block 不足时请求进入等待状态，不发生非法访问；
- 取消请求后对应资源被回收。

### 7.3 服务正确性

- 流式接口可以逐 token 返回；
- 并发请求不会互相污染输出；
- 超时请求可以结束；
- 服务重启后不会依赖旧的内存状态。

---

## 8. 预期目标和解释方式

下面是合理的目标区间，不是承诺值，最终简历必须替换成实测结果：

| 指标 | 目标区间 |
|---|---:|
| 混合并发吞吐提升 | 1.3～2.5 倍 |
| 长上下文峰值显存降低 | 20%～50% |
| 内部碎片率 | 5%～15% |
| Prefix Cache 命中时 TTFT 降低 | 20%～60% |
| 短请求单请求延迟 | 不明显劣于基线 |

必须解释收益边界：

- 单请求短文本时，Continuous Batching 几乎没有优势；
- Prefill 较长时，Prefill 会影响 Decode P99；
- Prefix Cache 在前缀随机时收益很小；
- Block 太小会增加管理和寻址开销；
- Block 太大则会增加内部碎片；
- KV Cache 量化可能节省显存，但会增加反量化开销或引入质量损失；
- Qwen3-8B INT4 的权重收益不能直接等同于 KV Cache 优化收益。

---

## 9. 风险和降级方案

### 风险 1：Qwen3-8B 无法在当前 GPU 运行

降级：

- 使用 Qwen3-4B 完成主实验；
- 使用 Qwen3-8B INT4 仅运行短上下文和低并发；
- 在报告中明确硬件限制。

### 风险 2：Triton Kernel 调试超时

降级：

- 保留 PyTorch reference path；
- 选择 KV Cache 量化作为进阶方向；
- 不为了 Kernel 牺牲 Paged KV 和 Continuous Batching 的完整性。

### 风险 3：连续批处理结果不一致

降级：

- 先固定 greedy sampling；
- 暂时关闭 thinking mode；
- 先支持“一个 Prefill batch + 多轮 Decode”，再加入动态 Prefill；
- 逐步扩大到真正的 iteration-level batching。

### 风险 4：优化收益不明显

处理方式：

- 先检查 workload 是否包含足够长上下文和并发；
- 使用 profiler 确认瓶颈是否真的在 KV 管理；
- 报告负收益场景，分析原因；
- 不修改实验数据，只调整假设和结论。

---

## 10. 最终报告结构

```text
1. 背景和问题定义
2. Qwen3 推理链路和 KV Cache 显存分析
3. 朴素 KV Cache 基线
4. Paged KV Cache 设计
5. Continuous Batching 调度设计
6. Prefix Caching 设计
7. 服务化实现
8. 实验环境和 workload
9. 性能结果
10. Profiling 和优化过程
11. Trade-off 和失败实验
12. 限制与后续工作
```

报告中必须保留失败实验，例如：

- Block Size 8 在小 batch 下管理开销过高；
- Prefix Cache 在随机 prompt 下没有收益；
- 过大的 Prefill budget 导致 P99 Decode 延迟恶化；
- KV Cache 量化节省显存，但 TPOT 变差。

这些内容比单纯展示最高吞吐更能体现工程判断力。

---

## 11. 简历表述模板

完成实测后，可以使用以下模板：

> 独立实现面向 Qwen3-4B/8B 的单卡 LLM 推理引擎，支持 Paged KV Cache、Continuous Batching 和 Prefix Caching；在 **XK 输入、Y 并发**的混合负载下，相比朴素 KV Cache 基线，吞吐提升 **X%**，峰值显存降低 **Y%**，内部碎片率控制在 **Z%**，并通过 Block Size、Prefill Token Budget 和 Prefix Cache 命中率分析不同调度策略的适用边界。

如果完成 Triton：

> 使用 Triton 实现基于 Block Table 的 Decode Attention Kernel，避免分页 KV Cache 的连续拼接路径，在 **batch size=X、上下文长度=Y** 时将 Decode TPOT 降低 **Z%**。

如果完成 KV Cache 量化：

> 实现 FP8/INT8 KV Cache 实验，在长上下文场景下节省 **X%** KV 显存，并量化比较显存、Decode 延迟和生成质量之间的权衡。

---

## 12. 每日工作节奏

每天结束时必须完成三件事：

1. 提交一次可以运行的代码；
2. 保存一份实验结果或失败日志；
3. 更新 `README.md` 或设计文档中的进展。

每天只允许存在一个主要未解决问题。例如当天目标是 Block Table，就不要同时开始 Prefix Cache 或 Kernel 优化。

最终项目的验收标准不是代码量，而是能完整回答：

1. 最初的瓶颈假设是什么？
2. 哪些 profiling 数据支持这个假设？
3. 为什么选择当前 Block Size 和调度预算？
4. 优化收益在哪些负载下出现？
5. 哪些场景下优化会失效？
6. 如果显存再减少一半，下一步会怎么改？

---

## 13. 无本地 GPU 时的 Kaggle Notebook 执行方案

### 13.1 是否可行

可以。Kaggle Notebook 可以承担本项目的 GPU 推理、压测和 profiling 实验；本地只负责代码编辑、单元测试、文档和结果整理。Kaggle 的 GPU 型号、可用时长和配额会变化，因此最终报告必须记录每次实验的实际 Accelerator、显存和 CUDA/PyTorch 版本。

推荐工作方式：

```text
本地：写代码、运行 CPU 单元测试、维护 Git/文档
Kaggle：运行模型、GPU benchmark、显存测试、Profiler
Kaggle Dataset：保存模型权重、代码快照和实验输入
本地：下载结果 CSV/JSON/图片并生成最终报告
```

### 13.2 模型选择调整

Kaggle 常见的 16GB 级别 GPU 更适合以下安排：

| 模型 | Kaggle 用途 |
|---|---|
| Qwen3-0.6B | 全部机制的快速调试和回归测试 |
| Qwen3-1.7B | 主要正确性和调度实验 |
| Qwen3-4B | 主性能实验，建议 FP16、较低并发或 4-bit 权重 |
| Qwen3-8B | 只做 INT4、短上下文、低并发扩展验证 |

在 16GB 显存上，不要默认 Qwen3-8B FP16 可以稳定运行。Qwen3-4B 是更稳妥的主实验模型；Qwen3-8B 只用于证明扩展性，无法运行时使用 Qwen3-4B 完成最终报告。

### 13.3 Kaggle Notebook 初始化

每次新建 Notebook 后，按以下顺序执行：

1. 在 Notebook 设置中选择 GPU Accelerator；
2. 记录 `nvidia-smi`、Python、PyTorch、CUDA 和 Transformers 版本；
3. 安装固定版本依赖；
4. 从 Kaggle Dataset 或允许联网的环境加载模型；
5. 设置 `HF_HOME` 或模型缓存目录到 `/kaggle/working`；
6. 运行一个 0.6B/1.7B smoke test；
7. 再运行正式 benchmark。

建议把安装、环境检查和模型加载写成一个可重复执行的 `notebooks/bootstrap.ipynb`，避免手工操作导致不同实验环境不一致。

### 13.4 代码和模型持久化

Kaggle Notebook 的运行环境是临时的，不能把 `/kaggle/working` 当作长期存储。每次会话结束前保存：

- benchmark 的 CSV/JSON；
- profiler trace；
- 生成的图表；
- 运行日志；
- 当前代码压缩包或 Git 快照。

推荐建立两个 Kaggle Dataset：

```text
deepinfer-code     # 代码、配置、测试输入
qwen3-model-cache  # 模型权重或可访问的模型缓存
```

如果模型下载耗时较长，先在一个 Notebook 中下载并保存为 Dataset，后续实验直接挂载该 Dataset。不要把模型权重提交到普通 Git 仓库。

#### 推荐方案：把模型保存成 Kaggle Dataset

模型权重约 1.5GB，最适合单独创建一个 Kaggle Dataset，后续 Notebook 作为输入数据集挂载。Dataset 挂载后位于 `/kaggle/input`，是只读目录，但模型可以直接从那里加载，不需要复制到 `/kaggle/working`。

第一次创建模型 Dataset：

1. 在当前 Notebook 完成模型下载，确认目录为 `/kaggle/working/models/Qwen3-0.6B`；
2. 在 Kaggle 页面选择保存 Notebook 版本，并将该目录作为输出或数据集保存；
3. 将数据集命名为清晰的 slug，例如 `qwen3-0-6b-model`；
4. 在之后的 Notebook 中通过右侧 **Add Input / Add Data** 添加这个 Dataset；
5. 添加后检查实际挂载路径：`/kaggle/input/qwen3-0-6b-model`。

不同 Kaggle 界面的保存入口名称可能是“Save as Dataset”“Create Dataset”或从 Notebook Output 创建 Dataset。关键是最终 Dataset 中直接包含 `config.json`、Tokenizer 文件和模型权重文件，而不是只有一个无法自动解压的压缩包。

后续 Notebook 使用以下代码自动选择持久化 Dataset；只有找不到 Dataset 时才下载：

```python
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

dataset_model_dir = Path("/kaggle/input/qwen3-0-6b-model")
working_model_dir = Path("/kaggle/working/models/Qwen3-0.6B")

if (dataset_model_dir / "config.json").exists():
    model_dir = str(dataset_model_dir)
    print("Using persistent Kaggle Dataset:", model_dir)
else:
    model_dir = str(working_model_dir)
    if not (working_model_dir / "config.json").exists():
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id="Qwen/Qwen3-0.6B",
            local_dir=model_dir,
        )
    print("Using session-local model directory:", model_dir)

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_dir,
    dtype=torch.float16,
).to("cuda:0").eval()
```

模型 Dataset 挂载成功后，新的 Kaggle 会话只需读取 `/kaggle/input/qwen3-0-6b-model`，不会再次从 Hugging Face 下载。`/kaggle/working` 仍用于保存 benchmark 结果、Profiler trace 和 `day1_snapshot.zip`。

#### 三种持久化层级

| 场景 | 位置 | 是否需要重新下载 |
|---|---|---|
| 同一会话重启 Kernel | `/kaggle/working` | 通常不需要 |
| 新会话或新 Notebook | Kaggle Dataset `/kaggle/input/...` | 不需要 |
| 只保存压缩包 | `/kaggle/input/...zip` | 需要每次解压到 working，速度较慢 |

优先使用“Dataset 中保存独立模型文件”的方式。不要把下载后的模型目录只留在 Notebook 的临时输出里，也不要每次启动都无条件调用 `snapshot_download`。

### 13.5 Kaggle 上的实验拆分

由于 Notebook 会话可能被中断，实验按短任务拆分，每个任务控制在一次会话可以完成的范围内：

```text
01_smoke_test.ipynb       # 模型加载和单请求正确性
02_naive_baseline.ipynb   # 朴素 KV Cache 基线
03_paged_kv.ipynb         # Block Pool、Block Table、碎片率
04_scheduler.ipynb        # Continuous Batching
05_prefix_cache.ipynb     # 前缀命中率和 TTFT
06_profiling.ipynb        # PyTorch Profiler/Triton
07_final_benchmark.ipynb  # 最终矩阵和结果导出
```

每个 Notebook 都应该：

- 从固定配置加载；
- 先运行一个小 smoke test；
- 将结果写入带时间戳的目录；
- 输出 GPU 型号和 Git commit；
- 支持从中断点重新执行。

### 13.6 Kaggle 上的服务化替代方案

Kaggle 适合做性能实验，不适合长期暴露一个稳定的公网推理服务。第 22—25 天的服务化部分调整为：

- 保留 `FastAPI` 和 `/generate`、`/generate_stream` 接口；
- 在 Notebook 内启动 Uvicorn，使用本地 asyncio/HTTP 客户端压测；
- 不把公网可访问性作为验收标准；
- 重点验证请求排队、动态 Batch、流式输出和 P50/P99；
- 如果 Kaggle 会话不适合启动服务，则直接调用 Engine API，使用同一个请求生成器完成压测。

报告中明确写成“单进程服务原型”，不要声称已经完成生产级部署。

### 13.7 Kaggle 版本的最低完成标准

没有本地 GPU 时，项目仍应至少完成：

1. Qwen3-0.6B/1.7B 上的完整正确性测试；
2. Qwen3-4B 上的 Paged KV Cache 和 Continuous Batching 实验；
3. 至少两种 Block Size 的碎片率对比；
4. 至少一种共享前缀 workload 的 Prefix Cache 实验；
5. TTFT、TPOT、吞吐、P50/P99 和峰值显存结果；
6. 一份记录实际 GPU 型号和会话限制的可复现实验报告。

如果只拿到 CPU 或 GPU 配额不足，先完成所有机制和 correctness，再把性能结果集中在 Qwen3-0.6B/1.7B；不要用不同硬件的结果直接互相比较。

### 13.8 Kaggle 实验记录模板

每个结果文件旁边保存一个元数据 JSON：

```json
{
  "model": "Qwen3-4B",
  "dtype": "float16",
  "accelerator": "实际填写，例如 T4/P100",
  "gpu_memory_gb": 16,
  "torch_version": "实际填写",
  "cuda_version": "实际填写",
  "block_size": 16,
  "max_prefill_tokens_per_iteration": 4096,
  "max_decode_batch_size": 16,
  "enable_thinking": false,
  "seed": 42
}
```

这样可以避免 Kaggle 换 GPU 或重启会话后，结果失去可比性。

### 13.9 双 T4 配置的具体用法

如果 Kaggle 提供 **2 × NVIDIA T4**，每张卡通常有约 16GB 显存。两张卡的显存默认是彼此独立的，不能把它们当成一张拥有 32GB 连续显存的 GPU 使用。除非专门实现 Tensor Parallel 或 Pipeline Parallel，否则模型权重和 KV Cache 必须完整放在其中一张卡上。

#### 推荐分工

主线项目仍然保持单卡语义，避免把双卡通信开销混入 Paged KV Cache 和调度实验：

| GPU | 用途 |
|---|---|
| `cuda:0` | 运行 DeepInfer-Lite 主实验 |
| `cuda:1` | 并行运行 Transformers/朴素 KV 基线，或复跑另一组参数 |

这样可以缩短实验时间，同时保证同一组结果的硬件条件一致。每组结果仍要记录 GPU 编号，不能把两张卡上的吞吐简单相加后当成单卡性能。

#### 模型安排

| 模型 | 双 T4 上的建议 |
|---|---|
| Qwen3-0.6B/1.7B | 单卡 FP16，完成全部正确性和调度测试 |
| Qwen3-4B | 单卡 FP16，作为主实验模型 |
| Qwen3-8B | 优先 INT4 单卡；FP16 只有在专门做双卡模型并行时尝试 |

Qwen3-8B 的 FP16 权重接近单张 T4 的显存上限，加载后还要预留 KV Cache、CUDA Workspace 和运行时空间。因此不建议把 Qwen3-8B FP16 作为 30 天主线。双卡并不能自动解决这个问题，除非额外实现模型切分。

#### 代码中的设备控制

所有 benchmark 都显式指定设备，避免两个进程抢同一张卡：

```python
device = "cuda:0"  # 主实验
# device = "cuda:1"  # 第二个独立实验进程
```

并行跑两组实验时，每个进程只设置一个可见设备，例如：

```text
进程 A：CUDA_VISIBLE_DEVICES=0
进程 B：CUDA_VISIBLE_DEVICES=1
```

结果文件名带上设备和配置，例如：

```text
qwen3_4b_paged_bs16_cuda0.json
qwen3_4b_baseline_bs16_cuda1.json
```

#### 双卡可选扩展，不作为主线验收项

第 29 天如果主线已经稳定，可以增加一个小型扩展实验：

- 两张 T4 各运行一个独立推理进程，比较总吞吐与单卡吞吐；或
- 用两个进程模拟路由层，把请求按轮询方式分发到两张卡；或
- 研究 Qwen3-8B 的简单 Pipeline/Tensor Parallel，但只报告模型并行通信开销，不把它和单卡 Paged KV 结果混在同一张主图中。

最推荐第一种：双进程独立服务。它实现简单、结论清楚，可以回答“增加一张卡后系统总吞吐如何变化”，同时不会引入复杂的模型切分代码。

#### 双 T4 下的最终实验矩阵

主报告保留单卡结果：

1. `cuda:0` 上的 Transformers 基线；
2. `cuda:0` 上的朴素 KV Cache；
3. `cuda:0` 上的 Paged KV + Continuous Batching；
4. `cuda:0` 上的完整版本。

附加报告增加双卡结果：

| 配置 | 目的 |
|---|---|
| 1 进程 / 1 GPU | 单卡基线 |
| 2 进程 / 2 GPU | 独立副本扩展吞吐 |
| 1 模型 / 2 GPU | 可选模型并行探索 |

双卡实验必须额外记录：

- 单卡吞吐和双卡总吞吐；
- 每张卡的显存占用；
- 请求分发策略；
- 进程间或卡间通信时间；
- 双卡扩展效率：

```text
scaling_efficiency = dual_gpu_throughput / (2 × single_gpu_throughput)
```

如果采用两个独立进程，扩展效率主要受到请求分发和负载不均衡影响；如果采用模型并行，扩展效率还会受到卡间通信影响。两种结果必须分开解释。

---

## 14. 实现边界：自己实现什么，复用什么

本项目的研究重点是 **KV Cache 管理、请求调度和测量方法**。30 天内不重新实现 Qwen3 的所有 Transformer 层，也不把 HuggingFace `generate()` 改名后称为自研引擎。

### 14.1 复用范围

- Transformers：加载模型权重、Tokenizer 和模型配置；
- PyTorch：执行 Qwen3 Attention、MLP、Norm、RoPE 等模型算子；
- Triton：只有选择 Kernel 进阶方向时才使用；
- FastAPI：只承担 HTTP 接口，不承担推理调度逻辑。

### 14.2 自己实现的范围

- `Request` 生命周期、队列和状态机；
- Prefill/Decode 请求组织和迭代调度；
- KV Block 元数据、Block Table、分配和释放；
- Prefix Cache 元数据、引用计数和淘汰；
- Benchmark workload、统计和结果落盘；
- 服务请求到 Engine 的适配。

### 14.3 分阶段模型执行路径

实现分两阶段，以降低工程风险：

**阶段 A：Engine 参考实现。** Transformers/PyTorch 执行模型前向；项目负责请求管理和调度。分页 KV 可先在 PyTorch Tensor 上实现逻辑 Block 映射，并在 Reference Path 中 gather 成连续 KV 后调用模型。这一阶段主要验证管理和调度正确性，不能把 gather 版本的速度当作优化后的分页 Kernel 性能。

**阶段 B：真实分页访问。** 若 HuggingFace 模型接口与版本允许，提供 paged cache adapter，使 Attention 读取 `block_table` 指向的 KV；否则只在可控的 Decode Attention 子路径上实现 Triton kernel，其他模型算子仍复用 Transformers/PyTorch。报告要准确标出真实分页读取覆盖的路径。

如果到第 10 天仍无法稳定接入自定义 Cache，优先完成完整 Reference Engine，并将“分页 KV 逻辑管理”和“Kernel 直接分页访问”作为两个不同结果陈述。不得仅凭不连续的 Block 元数据声称已经消除了 GPU KV Cache 的物理碎片或获得了生产级 PagedAttention 性能。

---

## 15. 核心模块接口草案

先固定接口，再分模块实现，可以减少后期改动。类型标注可以按实际代码调整，但职责要保持清楚。

### 15.1 Request 与生成结果

```python
@dataclass
class GenerateRequest:
    request_id: str
    input_ids: torch.Tensor
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    enable_thinking: bool = False

@dataclass
class GenerateResult:
    request_id: str
    output_ids: list[int]
    ttft_s: float
    tpot_s: float
    e2e_latency_s: float
```

不要在 Engine 内隐式 tokenize 用户文本；Serving 层可以调用 Tokenizer，Engine 的核心输入保持 token IDs。这样 benchmark 不会把 tokenizer 时间误算成 GPU 推理时间。

### 15.2 Block Manager

建议接口：

```python
allocate(request_id, num_blocks) -> list[int]
append_token(request_id, token_position) -> (physical_block, offset)
get_block_table(request_id) -> list[int]
free(request_id) -> None
stats() -> BlockStats
```

`append_token` 负责在跨越 Block 边界时申请新 Block。分配失败应该返回明确状态供 Scheduler 暂缓请求，不能在分配失败后继续写 KV。

建议状态指标：

```text
total_blocks
free_blocks
active_blocks
prefix_cached_blocks
valid_token_slots
allocated_token_slots
allocation_failures
```

### 15.3 Model Runner

```python
prefill(batch_input_ids, seq_lens, cache_context) -> next_token_logits
decode(batch_input_ids, positions, block_tables, cache_context) -> next_token_logits
```

模型 Runner 只做模型前向和 KV 读写，不决定哪些请求应该进入 batch。Scheduler 只产出本轮执行计划，不直接修改模型权重或采样参数。

### 15.4 Scheduler

```python
submit(request) -> request_id
schedule_once() -> SchedulePlan
on_model_output(plan, logits) -> list[TokenEvent]
cancel(request_id) -> None
```

`SchedulePlan` 至少应包含：

- 本轮 Prefill 请求及 token 数；
- 本轮 Decode 请求及当前 token；
- 每个请求的 position IDs；
- 对应 Block Table；
- 本轮新分配和待释放 Block。

---

## 16. 张量布局和内存估算

### 16.1 KV Cache 形状

逻辑上先按以下概念表示 KV：

```text
K/V: [num_layers, num_blocks, block_size, num_kv_heads, head_dim]
```

工程实现可以把 K 和 V 分开分配，也可以将两者合并一维；关键要求是 layout 明确、Block Table 对应关系稳定、读写位置可测试。

单 token 的 KV 字节数：

```text
bytes_per_token =
    2 * num_layers * num_kv_heads * head_dim * bytes_per_element
```

其中 2 表示 K 和 V。某请求长度为 `S` 时：

```text
kv_bytes(request) = S * bytes_per_token
```

Paged Cache 分配容量为：

```text
allocated_slots = ceil(S / block_size) * block_size
internal_waste = allocated_slots - S
```

这里仅估算 KV Tensor 本体，不包含 Block Table、Allocator、临时 Activation、模型权重、CUDA Workspace 和框架开销。实测显存以 `torch.cuda.max_memory_allocated()` 和 `max_memory_reserved()` 同时记录；两者不能混为一谈。

### 16.2 预分配与动态分配

优先预分配固定大小 Block Pool，避免 benchmark 中把大量 Python/Tensor 分配开销误判为模型开销。池大小由可用显存预算确定：

```text
kv_budget = free_gpu_memory - model_weights - runtime_reserve
num_blocks = floor(kv_budget / bytes_per_block)
bytes_per_block = block_size * bytes_per_token
```

实际设置要留出安全余量；不要把 `nvidia-smi` 显示的全部空闲显存都分给 KV Pool。

---

## 17. Scheduler 参考伪代码与调度规则

第一版使用简单 FIFO admission 和 decode 优先策略，避免一开始实现过多策略。每次迭代将计划生成、GPU 执行和状态更新分成三个阶段：

```text
while engine_is_running:
    release_finished_or_cancelled_requests()
    admit_waiting_requests_while_blocks_available()
    prefill = select_waiting_prefill(token_budget)
    decode = select_running_decode(max_decode_batch_size)
    plan = build_plan(prefill, decode)
    logits = model_runner.execute(plan)
    events = sample_and_update(plan, logits)
    publish(events)
```

### 17.1 调度规则

- waiting 请求按到达顺序进入，默认不做长度排序；
- 优先保障已运行请求的 Decode，减少它们被新 Prefill 抢占的时间；
- 每轮 Prefill token 不超过预算；
- Block 不足时，新请求保留在 waiting 队列，不部分进入 running；
- 请求生成 EOS 或达到上限后，释放私有 Block；
- Prefix Cache Block 只有在引用计数归零后才可淘汰；
- 固定 Batch 不够时允许尾部不足，不通过复制请求伪造 batch 大小。

### 17.2 必需的调度日志

每轮至少记录一行：

```text
iteration, timestamp, prefill_request_ids, prefill_tokens,
decode_request_ids, running_count, waiting_count,
free_blocks, allocated_blocks, gpu_step_ms
```

这些日志用于解释 TTFT/P99 的变化，也用于排查调度器是否饥饿某些请求。

### 17.3 防止 Prefill 饥饿 Decode

第一版可设置 `max_prefill_tokens_per_iteration`。若观察到 Decode TPOT/P99 随 Prefill budget 增大明显变差，则降低预算；若 GPU 空闲且 waiting 长时间积压，则逐步增大预算。最终选择要依据吞吐与尾延迟曲线，不凭单次测量决定。

---

## 18. benchmark 执行规程

### 18.1 每组 benchmark 的标准步骤

1. 记录 GPU 型号、驱动、PyTorch/CUDA/Transformers 版本和代码 commit；
2. 确认 GPU 没有其他 Notebook/进程占用；
3. 加载模型并执行 5 次 warmup；
4. 同步 CUDA，再重置峰值显存统计；
5. 运行至少 3 个重复 trial，每个 trial 至少 30 个请求；
6. 保存每请求原始延迟和每轮调度日志；
7. 报告中使用中位数和分位数，并保留所有重复 trial；
8. 发现 OOM、异常值或不完整 trial 时标记失败原因，不将其静默删除。

模型加载时间单独报告，不纳入稳定态生成延迟。GPU 计时边界使用 CUDA event 或明确同步，避免异步执行导致测时偏小。

### 18.2 对比公平性

基线与优化版本必须使用相同的：

- GPU 型号和 GPU 数量；
- 模型权重和权重 dtype/量化格式；
- Tokenizer、prompt token IDs 和输出长度；
- thinking 设置、采样参数和停止条件；
- 请求到达过程和并发定义；
- warmup 数、trial 数和测量边界。

如果用第二张 T4 并行跑基线，先做一次隔离复测，确认没有 GPU 共享、电力/温度或 CPU 资源争用。正式 A/B 数字尽量串行运行，或明确标记为并行实验。

### 18.3 请求到达模式

同时实现两种模式，不能只用“瞬间提交 N 个请求”代表在线服务：

- **Closed-loop 并发**：固定 N 个客户端，完成一个请求后立即发下一个；适合测饱和吞吐；
- **Open-loop 到达率**：按固定请求/秒到达；适合观察排队、SLO 和 P99。

报告每个结果使用哪种模式。Open-loop 需要记录目标到达率和实际到达率，避免过载时客户端发送速度下降掩盖排队。

### 18.4 结果目录约定

```text
reports/results/
  2026-10-28_t4x2_qwen3-4b/
    metadata.json
    requests.csv
    scheduler.csv
    summary.json
    profiler_trace/
    figures/
    run.log
```

每次运行使用新目录，不覆盖旧结果。`summary.json` 中记录失败状态和配置摘要。

---

## 19. 更严格的阶段退出条件

以下条件不满足时，不进入下一阶段堆新功能：

| 阶段 | 退出条件 |
|---|---|
| 基线 | 固定输入/输出下可重复运行；指标定义和单位统一；结果带元数据 |
| 朴素 Engine | greedy 输出与参考实现匹配；单请求、多请求生命周期结束后资源恢复 |
| Paged KV | Block 边界正确；随机申请释放无泄漏；Reference Path token 一致 |
| Continuous Batching | 新请求可在 decode 期间进入；完成请求立即退出；无请求永久等待 |
| Prefix Cache | 命中与未命中输出一致；引用计数正确；随机前缀负收益可被识别 |
| Serving | 流式接口逐 token 输出；取消请求会结束生成并回收资源 |
| Profiling | 至少确认一个真实瓶颈，并用单变量改动验证其影响 |
| 最终报告 | 原始数据、配置、图表和结论可以互相追溯；无未标明的硬件差异 |

### 19.1 关键失败判据

- 连续 3 次运行的吞吐变动超过 10%：先查 GPU 争用、时钟、warmup 和负载生成；
- 随机请求结束后 free Block 数少于初始值：先修资源生命周期，不进入 Prefix Cache；
- Prefix Cache 命中后输出与冷启动不同：先修 RoPE position、attention mask 或 cache position；
- 并发增加但吞吐下降且 GPU 利用率也低：先查 scheduler 和 CPU 同步，不先写 Kernel；
- 只有高并发极端设置获益、常见负载退化：报告适用边界，不把单点峰值作为结论。

---

## 20. 30 天日程的时间预算和优先级

本计划默认每天投入约 3—5 小时，合计约 100—140 小时。若每天只有 1—2 小时，建议把项目周期延长到 6—8 周；不要通过删掉正确性测试来挤进 30 天。

### P0：必须完成

- 固定 benchmark 和元数据；
- 朴素 KV Reference Engine；
- Block Manager 和分页 Cache Reference Path；
- iteration-level Continuous Batching；
- 正确性、资源回收和公平 A/B；
- 可复现实验报告。

### P1：时间允许时完成

- Prefix Cache；
- FastAPI 流式服务；
- 双 T4 两进程吞吐扩展实验。

### P2：仅在主线稳定后完成

- Triton Decode Attention；
- KV Cache 量化；
- Qwen3-8B 扩展。

如果进度落后，依次删减 P2、P1，保住 P0。对于求职项目，一个范围清楚、数据可信的 P0 完成版，比多个未经验证的功能更有价值。
