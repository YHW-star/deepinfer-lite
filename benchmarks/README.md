# Day 2 baseline benchmark

`run_baseline.py` is the first reproducible reference benchmark. It runs one
request at a time with fixed token lengths and greedy decoding. It is intended
to establish a baseline before implementing Paged KV Cache or Continuous
Batching.

## Run in Kaggle

After mounting the Qwen3 model Dataset and restoring the model path, run:

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

If the Dataset contains an extra top-level directory, use the path that
contains `config.json`, for example:

```text
/kaggle/input/qwen3-0-6b-model/Qwen3-0.6B
```

Run a small smoke benchmark first:

```python
!python benchmarks/run_baseline.py \
  --model-dir /kaggle/input/qwen3-0-6b-model \
  --output-dir /kaggle/working/day2_smoke \
  --input-tokens 32 \
  --max-new-tokens 16 \
  --warmup 1 \
  --num-requests 2
```

## Output files

Each run creates:

- `metadata.json`: model, GPU, versions, seed and workload;
- `summary.json`: aggregate throughput, mean/P50/P99 TTFT, TPOT, E2E and peak memory;
- `requests.json`: raw per-request measurements;
- `requests.csv`: raw measurements for plotting.

The current benchmark uses serial closed-loop requests. Its aggregate
throughput is not an online-service QPS result; it is the baseline against
which the later scheduler is compared.

## Metric definitions

- **TTFT**: time from the start of Prefill to the first generated token;
- **TPOT**: average time per generated token after the first token;
- **E2E**: time from Prefill start until the final generated token;
- **Throughput**: generated tokens divided by total benchmark wall time;
- **Peak memory**: PyTorch peak allocated and reserved CUDA memory.

The script continues decoding after EOS so every request has exactly
`max_new_tokens` generated tokens. The `eos_hit` field records whether EOS was
encountered; this keeps different runs comparable while retaining the signal.
