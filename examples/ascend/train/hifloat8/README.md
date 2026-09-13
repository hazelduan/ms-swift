# Qwen Dense HiFloat8 phase-1

This example keeps the ms-swift Trainer path unchanged. A custom DeepSpeed JSON
is passed through Transformers/Accelerate to the DeepSpeed engine, which converts
only the selected Qwen MLP projections after BF16 device placement and before
optimizer/ZeRO construction.

The implementation requires a torch_npu build containing
`torch_npu.utils.hifloat8_train` and native HiFloat8 quantize/matmul kernels.
DeepSpeed probes the native kernel before mutating the model and fails closed when
the current SoC/CANN combination does not support it.

Before training, validate the three real training GEMMs at a Qwen MLP shape. The
script records forward, `dX`, and `dW` cosine/NRMSE plus exact native-op counts,
and exits with status 2 when the native kernel is unavailable:

```bash
python examples/ascend/train/hifloat8/validate_hifloat8_linear.py \
  --output output/hifloat8/hifloat8_linear_validation.json
```

Prepare the deterministic 400/100 split:

```bash
python examples/ascend/train/hifloat8/prepare_alpaca_split.py \
  examples/ascend/train/hifloat8/data/alpaca_gpt4_en_500.jsonl \
  examples/ascend/train/hifloat8/data
```

Run the fixed full-parameter BF16 baseline from this repository root:

```bash
swift sft examples/ascend/train/hifloat8/qwen3_0_6b_full.yaml
```

Run the matched HiFloat8 candidate (the only intended differences are the
DeepSpeed config and output directory):

```bash
swift sft examples/ascend/train/hifloat8/qwen3_0_6b_full.yaml \
  --deepspeed examples/ascend/train/hifloat8/zero2_hifloat8.json \
  --output_dir output/hifloat8/full_hifloat8
```

For LoRA, use `qwen3_0_6b_lora.yaml` and the same two DeepSpeed configurations.
The HiFloat8 selector targets PEFT's `.base_layer` children and never the LoRA
A/B adapters. Performance runs should override `eval_strategy` and
`save_strategy` to `no`, use 40 steps, discard steps 1-5, and summarize
`step_metrics.jsonl`. Each step record also includes rank-0 HiFloat8 operator
counts; a passing end-to-end candidate must show one forward and the expected
`dX`/`dW` path per converted projection, rather than only passing a standalone
operator test.
