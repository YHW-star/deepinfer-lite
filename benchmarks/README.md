# 第 2 天：推理基线测试

`run_baseline.py` 是第一版可复现的参考测试脚本。它使用固定输入和输出 token 数、贪心解码，并按顺序逐个处理请求。该脚本为后续分页 KV Cache 和连续批处理实现建立对照基线。

## 在 Kaggle 中运行

先挂载包含 Qwen3 模型文件的 Dataset，并确认模型路径后运行：

```python
!python benchmarks/run_baseline.py \
  --model-dir /kaggle/input/qwen3-0-6b-model \
  --output-dir /kaggle/working/day2_baseline_128_128 \
  --input-tokens 128 \
  --max-new-tokens 128 \
  --warmup 2 \
  --num-requests 10 \
  --device cuda:0 \
  --seed 42
```

如果 Dataset 中多了一层目录，`--model-dir` 应指向包含 `config.json` 的目录，例如：

```text
/kaggle/input/qwen3-0-6b-model/Qwen3-0.6B
```

正式测试前先运行小规模 smoke benchmark：

```python
!python benchmarks/run_baseline.py \
  --model-dir /kaggle/input/qwen3-0-6b-model \
  --output-dir /kaggle/working/day2_smoke \
  --input-tokens 32 \
  --max-new-tokens 16 \
  --warmup 1 \
  --num-requests 2 \
  --device cuda:0 \
  --seed 42
```

## 输出文件

每次运行会生成以下文件：

- `metadata.json`：模型、GPU、软件版本、随机种子和 workload 配置；
- `summary.json`：吞吐，以及 TTFT、TPOT、E2E 的平均值、P50、P99 和峰值显存；
- `requests.json`：逐请求原始测量值；
- `requests.csv`：用于绘图和表格分析的原始数据。

当前脚本按顺序逐个处理请求，属于闭环串行测试。汇总吞吐不是在线服务 QPS，而是后续调度器优化的参考基线。

## 指标定义

- **TTFT（首 Token 延迟）**：Prefill 开始到生成第一个 token 的时间；
- **TPOT（单 Token 解码延迟）**：首 token 之后，每个生成 token 的平均耗时；
- **E2E（端到端延迟）**：Prefill 开始到最后一个 token 生成完成的总时间；
- **吞吐**：benchmark 总耗时内生成的 token 数除以总时间；
- **峰值显存**：PyTorch 统计到的 CUDA 峰值已分配显存和峰值预留显存。

为保证请求间可比，脚本遇到 EOS 后仍会继续 Decode，直到达到 `max_new_tokens`。`eos_hit` 字段记录是否遇到 EOS。
