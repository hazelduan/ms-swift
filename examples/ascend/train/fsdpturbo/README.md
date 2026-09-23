# Independent FSDPTurbo SFT backend

`swift fsdpturbo sft` uses its own trainer. Swift loads model metadata, prepares templates and datasets; the optional `fsdp_turbo` package owns distributed meshes, wrapping, optimizer steps and gradient clipping. This route does not use HF Trainer/Accelerate FSDP or Megatron's training lifecycle.

## Installation and launch

Install Swift's requirements and a compatible FSDPTurbo package in the active environment. Current Swift preprocessing requires the `Json` dataset feature; validation uses datasets 4.8.4. FSDPTurbo remains an optional dependency. The validated package is the local FSDPTurbo revision `4defe05` based on `435b9fdd804ef0efe413f2dcff82a9616d85c1b4`; its TP/FSDP composition fixes are required for the tested TP configurations. A stock package with the same `0.1.0` version is not sufficient evidence of compatibility. See the experiment report for the exact wheel and revision.

Load your CANN environment, activate your training environment, and select eight available logical NPU devices. Then run from this checkout:

```bash
NPROC_PER_NODE=8 MASTER_PORT=29501 swift fsdpturbo sft \
  --model Qwen/Qwen3.5-35B-A3B \
  --dataset /path/to/train.jsonl \
  --output_dir /path/to/output \
  --tuner_type full --torch_dtype bfloat16 --attn_impl eager \
  --fsdp_size 8 --tp_size 1 --ep_size 4 --efsdp_size 2 \
  --fsdp_implementation native --ep_dispatcher eager \
  --forward_prefetch 0 --backward_prefetch 0 \
  --max_steps 20 --max_length 128 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 1 \
  --gradient_checkpointing true --vit_gradient_checkpointing true \
  --save_strategy no --eval_strategy no --split_dataset_ratio 0 \
  --report_to none --enable_npu_model_patch false
```

The outer CLI forwards to a secondary router which starts exactly one torchrun launcher. `enable_npu_model_patch=false` is set before Swift model imports because FSDPTurbo owns model parallelization. Run from the intended checkout or install that checkout explicitly to avoid selecting another Swift installation.

## Topology

`WORLD_SIZE` must be divisible by both `fsdp_size * tp_size` and `ep_size * efsdp_size`. These are overlapping meshes; do not multiply all four sizes to calculate world size. `efsdp_size > 1` requires `ep_size > 1`.

Eight-device validation configurations:

| FSDP | TP | EP | EFSDP | Local batch | Global batch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 1 | 1 | 1 | 8 |
| 8 | 1 | 4 | 2 | 1 | 8 |
| 4 | 2 | 1 | 1 | 2 | 8 |
| 4 | 2 | 4 | 2 | 2 | 8 |

## Current scope

The model-spec registry currently contains Qwen3.5 MoE only. Every required FSDP, TP, expert and recompute module pattern is checked against the actual model. EP/EFSDP applies to expert containers, not embeddings, attention or the language-model head.

This initial backend requires full causal-LM SFT, a map-style dataset, positive `max_steps`, gradient accumulation of one, AdamW, eager attention and `save_strategy=no`. It writes resolved arguments and training metrics. Checkpoint save/resume, evaluation, LoRA, PP, CP and packing are not validated; unsupported switches are rejected where exposed. CPU offload, other dispatchers, multimodal batches and CUDA need separate validation.

The eight-device smoke/trajectory results establish only the tested short-sequence full-SFT configurations. They do not establish long-context throughput, convergence or checkpoint correctness. See the experiment report for exact revisions, CANN version, data, loss/gradient curves and limitations.
