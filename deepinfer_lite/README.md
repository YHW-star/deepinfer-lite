# Model Runner

`ModelRunner` wraps a Transformers causal language model with an explicit
Prefill/Decode boundary. Prefill accepts the full prompt and returns the logits
for its next token plus a `KVCache` handle. Decode accepts exactly one token
per batch row and that handle, then returns next-token logits and the updated
handle.

The current `KVCache` stores Transformers `past_key_values` and a logical
sequence length, batch size, device, and attention mask. It does not allocate
paged blocks yet; Day 4 will define the naive cache behavior, and later days
can replace the payload without changing the call sites.

## Kaggle acceptance run

From the repository root, after the model Dataset is attached:

```python
%env PYTHONPATH=/kaggle/working/deepinfer-lite
!python examples/day3_runner_smoke.py \
  --model-dir /kaggle/input/datasets/yhw1027/qwen3-0-6b-model/Qwen3-0.6B \
  --device cuda:0 \
  --decode-steps 32
```

Expected output includes `"status": "passed"`, `"decode_calls": 32`, and a
`cache_sequence_length` equal to the prompt token count plus 32.
