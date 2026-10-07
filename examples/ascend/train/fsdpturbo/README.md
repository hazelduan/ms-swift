# Independent FSDPTurbo SFT backend

`swift fsdpturbo sft` uses its own trainer. Swift loads model metadata, prepares templates and datasets; the optional `fsdp_turbo` package owns distributed meshes, wrapping, optimizer steps and gradient clipping. This route does not use HF Trainer/Accelerate FSDP or Megatron's training lifecycle.

For DeepSeek V4 Flash and Qwen3.5-122B multi-node deployment, use the [multi-node guide](MULTI_NODE.md).

## Installation and launch

Install Swift's requirements and a compatible FSDPTurbo package in the active environment. Current Swift preprocessing requires the `Json` dataset feature; validation uses datasets 4.8.4. FSDPTurbo remains an optional dependency. The latest validated package is [`ee117ad`](https://gitcode.com/hazeldxq/FSDPTurbo/commit/ee117ade23ab9b479df1b40b86780d8a8a64a251), on branch `swift-fsdpturbo-compat`, based on upstream `0a4b3bc`. Besides lazy optional quantization imports and replicated precision-sensitive parameters, it supplies the DeepSeek V4 bridge, mixed-device gradient clipping and same-layout DCP resume fixes. A stock package with the same `0.1.0` version is not sufficient evidence of compatibility.

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

The model-spec registry contains Qwen3.5 MoE and DeepSeek V4. Every required FSDP, TP, expert and recompute module pattern is checked against the actual model. EP/EFSDP applies to expert containers, not embeddings, attention or the language-model head. DeepSeek V4 uses instance-level RMSNorm and HyperConnection/Sinkhorn replacements; native masked attention is retained. Its nondifferentiable top-k indexer stays frozen for SFT.

This backend requires full causal-LM SFT, a map-style dataset, positive `max_steps`, gradient accumulation of one, AdamW and eager attention. It supports `save_strategy=no` or `steps`; DCP resume restores model, optimizer, scheduler, rank-local data cursor and per-rank RNG with the same parallel layout. It writes resolved arguments and training metrics. Evaluation, LoRA, PP, CP and packing are not validated; unsupported switches are rejected where exposed. CPU offload is covered below; other dispatchers, multimodal batches and CUDA need separate validation.

Only global rank 0 loads and converts pretrained CPU weights. All ranks construct on meta and apply the backend's sharding first; PyTorch DCP broadcasts one tensor at a time into local shards. FP8/FP4 checkpoint weights are dequantized to the requested training dtype while FP32 stability parameters are preserved. On resume, only tensor metadata is read before local-shard allocation; original pretrained weights are not loaded again.

The results establish only the tested short-sequence full-SFT configurations. They do not establish long-context throughput or convergence. The earlier eight-device experiments did not test checkpoint resume; the latest tests below do.

## Large-model startup validation (2026-10-08)

Tested code pair: Swift `0400d8c2d5c5c68cea4c6e166dcf5892781d47c5` (including upstream `0bd7b0aa1`) and FSDPTurbo `ee117ade23ab9b479df1b40b86780d8a8a64a251` (including upstream `0a4b3bc`). Later documentation-only commits do not change this code pair. Environment: Python 3.11.15, CANN 9.0.0, driver 26.0.rc1, torch 2.9.0+cpu, torch_npu 2.9.0, Transformers 5.9.0 and Accelerate 1.13.0.

| Validation | Result |
| --- | --- |
| Full official configurations on meta | DeepSeek: 43 layers / 284.332B parameters; Qwen122: 48 layers / 122.563B. All registered module patterns matched. |
| DeepSeek exact-width, depth-only reduction | 4 layers / 27.388B, original 256 experts and SL/CSA/HCA attention types; FSDP16/EP8/EFSDP2, 3 steps, exit 0. Rank-0 peak allocated memory 28.09 GB. Random fixture, not pretrained convergence. |
| Reduced DeepSeek BF16 / FP8+FP4 input / CPU offload | 20 steps each, exit 0; packed FP4 values and E8M0 scales checked numerically. |
| Qwen pretrained-derived 4-layer fixture | FSDP2/TP2/EP2/EFSDP2 and router auxiliary loss paths passed. |
| Rank-zero/meta/shard loading | 4-NPU regression reconstructed every FSDP/TP/EP/EFSDP parameter exactly; only global rank 0 called the pretrained loader. |
| Checkpoint resume | Step 18 save, metadata-only restore, steps 19–20 matched continuous-run loss and gradient norms exactly; changed-layout resume explicitly rejected. Unused Adam state regression passed. |
| Two torchrun agents | NNODES2 / NPROC2 each / WORLD_SIZE4 passed on one physical host; this is not a physical-cluster network test. |
| Focused regression / static checks | 55 tests passed; modified-file Ruff, compileall and diff-check passed. |

The 20-step precision gate was fixed in advance: loss relative error <= 5%, gradient-norm relative error <= 10%, at least 95% of points inside each limit, correlation >= 0.99. EP/EFSDP versus FSDP achieved maximum loss/gradient errors **0.01846% / 0.6652%**, correlations **0.999720 / 0.999826**. CPU offload versus device-resident EP achieved **0.02083% / 1.1518%**, correlations **0.999852 / 0.999703**. Both passed all points; thresholds were not relaxed. An earlier loss-casting failure is retained in the experiment logs.

Raw evidence on A3 is under `/home/dxq/experiments/swift_fsdpturbo_large_models_20261007`: `reviewed_exact_width`, `final_quant_fp32`, `fp32_loss_{reference,ep,offload}`, `reviewed_resume_restore`, `final_qwen_aux`, `comparison.json` and `final_validation.log`. Full physical-cluster training and full-checkpoint convergence remain deployment validations. See [MULTI_NODE.md](MULTI_NODE.md) for the reproducible environment and launch commands.

## Earlier CANN 9.0.0 upstream-refresh smoke

Swift `cf4bf59ae` (including upstream `8cb686833`) and FSDPTurbo `beb7457` (including upstream `0a4b3bc`) were revalidated on 16 Ascend910_9382 logical NPUs with torch/torch_npu 2.9.0, Transformers 5.9.0, sequence length 128 and two optimizer steps:

| FSDP | TP | EP | EFSDP | Loss step 1 -> 2 | Peak allocated memory reported by rank 0 |
| ---: | ---: | ---: | ---: | --- | ---: |
| 16 | 1 | 1 | 1 | 1.578125 -> 0.855469 | 31.47 GB |
| 16 | 1 | 8 | 2 | 1.578125 -> 0.859375 | 35.47 GB |
| 8 | 2 | 1 | 1 | 1.507812 -> 1.046875 | 55.06 GB |
| 8 | 2 | 8 | 2 | 1.500000 -> 1.046875 | 36.87 GB |

Qwen3.5 linear-attention `A_log` and gated-norm weights remain replicated FP32 parameters rather than being folded into a mixed-dtype FSDP group. A 16-rank probe found 60 such parameters per rank; synchronized gradients and post-step parameter sums had zero maximum cross-rank difference. These short runs prove the listed integration cells only, not long-run numerical equivalence or production throughput.

## Earlier CANN 9.1.0 capacity results

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
