# DeepInfer-Lite

面向 Qwen3 的轻量级单卡推理引擎实验项目。



## 当前状态

- [x] Kaggle 双 Tesla T4 环境验证

- [x] Qwen3-0.6B FP16 加载与生成 smoke test

- [ ] 统一 benchmark

- [ ] Paged KV Cache

- [ ] Continuous Batching

- [ ] Prefix Caching


## 环境

- Python 3.13.15

- PyTorch 2.11.0+cu128

- Transformers 5.16.1

- GPU：Kaggle 2 × Tesla T4

- 当前模型：Qwen/Qwen3-0.6B


## 项目计划


见同目录下的 `DeepInfer-Lite\_30天实施计划.md`。



## 运行环境



模型推理和 GPU benchmark 在 Kaggle Notebook 中执行。

模型权重不提交到 Git 仓库。

