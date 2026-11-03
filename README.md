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
- [ ] 在 Kaggle 完成统一 benchmark 实测
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
│  └─ README.md              # 基线测试说明
├─ DeepInfer-Lite_30天实施计划.md
└─ README.md
```

模型权重不提交到 Git 仓库。Kaggle 的模型 Dataset、实验结果和环境快照单独保存。

## 项目计划

完整的 30 天实施计划见：

```text
DeepInfer-Lite_30天实施计划.md
```

计划包括模型选择、双 T4 使用方式、Paged KV Cache 设计、调度器接口、benchmark 规范、阶段退出条件和风险降级方案。

## Git 提交

在项目目录执行：

```cmd
git add .
git commit -m "描述本次修改"
git push
```
