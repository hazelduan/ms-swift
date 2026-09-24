# Independent FSDPTurbo SFT backend

`swift fsdpturbo sft` uses its own trainer. Swift loads model metadata, prepares templates and datasets; the optional `fsdp_turbo` package owns distributed meshes, wrapping, optimizer steps and gradient clipping. This route does not use HF Trainer/Accelerate FSDP or Megatron's training lifecycle.

## Installation and launch

Install Swift's requirements and a compatible FSDPTurbo package in the active environment. Current Swift preprocessing requires the `Json` dataset feature; validation uses datasets 4.8.4. FSDPTurbo remains an optional dependency. The validated package is the local FSDPTurbo revision `4defe05` based on `435b9fdd804ef0efe413f2dcff82a9616d85c1b4`; its TP/FSDP composition fixes are required for the tested TP configurations. A stock package with the same `0.1.0` version is not sufficient evidence of compatibility. See the experiment report for the exact wheel and revision.

Load your CANN environment, activate your training environment, and select eight available logical NPU devices. Then run from this checkout:

```bash
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
NPROC_PER_NODE=8 MASTER_PORT=29501 swift fsdpturbo sft \
  --model Qwen/Qwen3.5-35B-A3B \
  --dataset /path/to/train.jsonl \
  --output_dir /path/to/output \
  --tuner_type full --torch_dtype bfloat16 --attn_impl eager \
  --fsdp_size 8 --tp_size 1 --ep_size 8 --efsdp_size 1 \
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

Eight-device topology configurations (see the capacity results below):

| FSDP | TP | EP | EFSDP | Local batch | Global batch |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 1 | 1 | 1 | 8 |
| 8 | 1 | 4 | 2 | 1 | 8 |
| 8 | 1 | 8 | 1 | 1 | 8 |
| 4 | 2 | 1 | 1 | 2 | 8 |
| 4 | 2 | 4 | 2 | 2 | 8 |

## Current scope

The model-spec registry currently contains Qwen3.5 MoE only. Every required FSDP, TP, expert and recompute module pattern is checked against the actual model. EP/EFSDP applies to expert containers, not embeddings, attention or the language-model head.

This initial backend requires full causal-LM SFT, a map-style dataset, positive `max_steps`, gradient accumulation of one, AdamW, eager attention and `save_strategy=no`. It writes resolved arguments and training metrics. Checkpoint save/resume, evaluation, LoRA, PP, CP and packing are not validated; unsupported switches are rejected where exposed. CPU offload is covered below; other dispatchers, multimodal batches and CUDA need separate validation.

The eight-device smoke/trajectory results establish only the tested short-sequence full-SFT configurations. They do not establish long-context throughput, convergence or checkpoint correctness. See the experiment report for exact revisions, CANN version, data, loss/gradient curves and limitations.

## CANN 9.1.0 capacity results

On eight Ascend910_9382 logical NPU devices, the full Qwen3.5-35B-A3B checkpoint completed 20 steps with FSDP8 and FSDP8+EP8+EFSDP1. Both used sequence length 128, recompute, disabled prefetch and expandable allocator segments. Without CPU offload, full-model FSDP8+EP4+EFSDP2, FSDP4+TP2 and FSDP4+TP2+EP4+EFSDP2 exhausted device memory on this setup. Those device-resident capacity limits remain; the offload configurations below pass.

The adapter initializes its loss-reduction communicator during setup, before backward fills the device allocator cache. This reserves HCCL communication buffers before first-step loss reduction; it does not change the loss, gradients or optimizer.

A separate 4-layer checkpoint derived from the pretrained model (4.828B parameters, original 256 experts and original projection widths) completed 20 steps on all four FSDP/TP/EP/EFSDP combinations in the table. Its paired loss and gradient-norm comparisons passed the predeclared thresholds. This fixture result establishes topology integration; the full-model capacity failures above remain known limitations.

## CPU offload

Add `--offload_params true` to move sharded parameters, gradients and AdamW state to host memory. For the EP4/EFSDP2 case, use `--fsdp_size 8 --tp_size 1 --ep_size 4 --efsdp_size 2`; for TP2, use FSDP4 and local batch 2 to retain global batch 8.

The adapter initializes a CPU Gloo backend alongside the detected accelerator backend before HF training arguments create their distributed state. FSDPTurbo's mesh groups inherit both backends, so CPU gradient-norm reductions use Gloo while NPU model collectives use HCCL. The default `offload_params=false` path is unchanged. Leave `ddp_backend` unset or use the accelerator's normal backend. An existing accelerator-only process group is rejected early rather than destroyed or silently replaced.

Full Qwen3.5-35B-A3B results on eight logical NPUs with offload enabled, using the same 128-token configuration above:

| FSDP | TP | EP | EFSDP | Optimizer steps | Peak allocated NPU memory across ranks |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1 | 1 | 1 | 20 | 23.12 GiB |
| 8 | 1 | 4 | 2 | 20 | 34.59 GiB |
| 4 | 2 | 1 | 1 | 20 | 38.80 GiB |
| 4 | 2 | 4 | 2 | 20 | 35.43 GiB |

All four runs completed forward, backward, optimizer updates and clean exit. Per-rank probes confirmed CPU gradients and optimizer state. A matched 20-step FSDP8 comparison against non-offload passed the predeclared loss/gradient-norm thresholds (maximum relative errors 1.33%/7.34%); paired EP4/EFSDP2, TP2 and combined checks on the 4-layer fixture also passed. CPU offload needs sufficient host RAM and adds host computation and transfers; these short runs do not establish long-context throughput or convergence.

The real distributed collective regression can be run from this checkout in the accelerator environment:

```bash
torchrun --nproc_per_node=8 --master_port=29501 tests/fsdpturbo/distributed_offload.py
```
