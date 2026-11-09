# DeepInfer-Lite

面向 Qwen3 的轻量级单卡大语言模型推理引擎实验项目。

本项目围绕推理链路中的核心问题展开：KV Cache 如何管理、Prefill 和 Decode 如何拆分、请求如何动态组成批次，以及这些设计对吞吐、延迟和显存的影响。

## 项目目标

在单张 GPU 上逐步实现并测量：

- 朴素 KV Cache 基线；
- 分页 KV Cache；
- Prefill/Decode 分离；
- 连续批处理；
- 前缀缓存；
- 流式推理服务；
- 面向长上下文和混合并发负载的性能分析。

项目重点不是堆砌推理框架名词，而是记录每一个优化的瓶颈假设、实现方式、测量结果和适用边界。

## 当前进度

- [x] Kaggle 双 Tesla T4 环境验证
- [x] Qwen3-0.6B FP16 模型加载和生成验证
- [x] 环境快照与模型配置记录
- [x] 统一 benchmark 脚本骨架
- [x] 在 Kaggle 完成统一 benchmark 实测和重复 trial
- [x] 第 3 天 ModelRunner Prefill/Decode 接口和 KV Cache 句柄骨架
- [x] 在 Kaggle T4 上通过 32 步 Runner smoke 验收
- [x] 第 4 天朴素 KV Cache、贪心生成和一致性验收脚本
- [x] 在 Kaggle T4 上通过第 4 天 KV Cache 一致性验收
- [x] 第 5 天 KV Cache 正确性测试文件
- [x] 在 Kaggle T4 上运行第 5 天完整测试（24 项通过）
- [x] 第 6 天 Prefill/Decode/静态批处理基线脚本
- [x] 在 Kaggle T4 上完成第 6 天 baseline 实测
- [x] 第 7 天 KV Cache/Prefill/Decode/Paged Cache 设计文档
- [x] 第 8 天 Block Pool 生命周期管理器和单元测试
- [x] 在 Kaggle 通过第 8 天 Block Pool 测试（12 项）
- [x] 第 9 天 Block Table 映射、边界扩容和生命周期回收实现
- [x] 在 Kaggle 通过第 9 天随机 100 次请求回收验收（7 项）
- [x] 第 10 天分页 KV Reference Path 实现与 Qwen3 greedy 一致性验收
- [x] 在 Kaggle 通过第 10 天分页 KV 读写测试（25 项）
- [ ] 分页 KV Cache
- [ ] 连续批处理
- [ ] 前缀缓存
- [ ] FastAPI 流式服务
- [ ] Triton Decode Attention 或 KV Cache 量化

## 当前环境

- Python：3.13.15
- PyTorch：2.11.0+cu128
- PyTorch 编译 CUDA：12.8
- Transformers：5.16.1
- GPU：Kaggle 2 × Tesla T4
- 每张 GPU 显存：约 14.56 GiB
- 当前模型：Qwen/Qwen3-0.6B
- 主实验设备：`cuda:0`
- 推理类型：FP16

两张 T4 的显存默认不合并。主实验按单卡推理执行，另一张卡用于并行跑基线或复现实验。

## 第 2 天：统一基线测试

基线脚本位于：

```text
benchmarks/run_baseline.py
```

它采用固定输入长度、固定输出长度和贪心解码，逐请求执行 Prefill 与 Decode，记录：

- 首 Token 延迟（TTFT）；
- 单 Token 解码延迟（TPOT）；
- 端到端延迟（E2E）；
- 生成吞吐；
- GPU 峰值显存；
- 每个请求的原始结果。

### 在 Kaggle 中运行

先在 Notebook 中挂载包含 Qwen3 模型文件的 Dataset，然后执行：

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

确认小规模测试成功后，再运行正式基线：

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

如果 Dataset 多了一层模型目录，`--model-dir` 要指向包含 `config.json` 的目录，例如：

```text
/kaggle/input/qwen3-0-6b-model/Qwen3-0.6B
```

每次运行会生成：

```text
metadata.json   # 环境、GPU、模型和测试配置
summary.json    # 汇总指标
requests.json   # 每个请求的原始指标
requests.csv    # 方便后续绘图和分析
```

当前基线是串行请求，结果用于和后续分页 KV Cache、连续批处理版本做对比，不代表在线服务的最终 QPS。

## 项目目录

```text
deepinfer-lite/
├─ benchmarks/
│  ├─ run_baseline.py       # 统一基线测试
│  ├─ run_day6_baseline.py  # 第 6 天性能和显存基线
│  └─ README.md              # 基线测试说明
├─ deepinfer_lite/
│  ├─ block_pool.py          # 第 8 天固定 Block 生命周期管理
│  ├─ block_table.py         # 第 9 天逻辑 Block 到物理 Block 映射
│  ├─ paged_kv_cache.py      # 第 10 天分页 KV 写入、读取和模型适配器
│  ├─ model_runner.py       # Prefill/Decode Runner 与 KV Cache handle
│  └─ README.md              # Runner、Block Pool 和 Block Table 接口说明
├─ examples/
│  ├─ day3_runner_smoke.py  # 32 次 Decode 验收
│  └─ day4_cache_acceptance.py # 朴素 KV Cache 正确性验收
├─ tests/
│  ├─ test_kv_correctness.py # 第 5 天正确性测试
│  ├─ test_block_pool.py     # 第 8 天 Block Pool 测试
│  ├─ test_block_table.py    # 第 9 天 Block Table 测试
│  └─ test_paged_kv_cache.py # 第 10 天分页 KV 读写测试
├─ reports/
│  └─ day6_baseline/          # 第 6 天原始数据、汇总表和图表
├─ docs/
│  └─ day7_design.md          # 第 7 天设计说明
├─ DeepInfer-Lite_30天实施计划.md
└─ README.md
```

模型权重不提交到 Git 仓库。Kaggle 的模型 Dataset、实验结果和环境快照单独保存。

### 运行 KV Cache 正确性测试

在 Kaggle Notebook 的仓库根目录运行：

```python
%env PYTHONPATH=/kaggle/working/deepinfer-lite
!DEEPINFER_MODEL_DIR=/kaggle/input/datasets/yhw1027/qwen3-0-6b-model/Qwen3-0.6B \
  DEEPINFER_TEST_DEVICE=cuda:0 \
  pytest -q tests/test_kv_correctness.py
```

测试目录包含 `conftest.py`，即使没有设置 `PYTHONPATH`，直接从仓库目录运行 `pytest` 也能导入 `deepinfer_lite`。

### 运行第 9 天 Block Table 测试

```python
%env PYTHONPATH=/kaggle/working/deepinfer-lite
!pytest -q tests/test_block_table.py
```

验收重点是固定随机种子下创建并结束 100 个不同长度请求；每个请求结束后，Pool
必须恢复为 `free_count == 64`、`allocated_count == 0`、`allocation_count == 0`。

最近一次 Kaggle 验收结果：`7 passed in 7.57s`（2026-11-08）。

### 运行第 10 天分页 KV 测试

分页 Reference Path 先在 PyTorch Tensor 上管理物理 Block：Prefill 将完整 KV 按 token
写入 Block，Decode 将模型返回的完整 KV 的最后一个 token 写入尾 Block，模型下一轮
前向前再按 Block Table gather 成连续 KV。该路径用于验证映射、读写和输出一致性，尚未
实现 Attention Kernel 直接读取分页 Block。

```python
%env PYTHONPATH=/kaggle/working/deepinfer-lite
%env DEEPINFER_MODEL_DIR=/kaggle/input/datasets/yhw1027/qwen3-0-6b-model/Qwen3-0.6B
%env DEEPINFER_TEST_DEVICE=cuda:0
!pytest -q tests/test_paged_kv_cache.py
!pytest -q tests/test_kv_correctness.py -k paged_reference
```

模型一致性验收目标是分页路径和连续 Runner 的 greedy token 完全相同，并且请求释放后
`free_count` 恢复到 Pool 初始容量。

最近一次模型一致性结果：`1 passed, 24 deselected in 12.72s`（2026-11-09）。
分页 KV 读写与资源回收单元测试：`25 passed in 7.66s`（2026-11-09）。

### 运行第 6 天性能基线

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

## 项目计划

完整的 30 天实施计划见：

```text
DeepInfer-Lite_30天实施计划.md
```

计划包括模型选择、双 T4 使用方式、Paged KV Cache 设计、调度器接口、benchmark 规范、阶段退出条件和风险降级方案。

第 7 天设计说明见：

```text
docs/day7_design.md
```

## Git 提交

在项目目录执行：

```cmd
git add .
git commit -m "描述本次修改"
git push
```
