# Independent FSDPTurbo SFT backend

`swift fsdpturbo sft` reuses Swift's arguments, tokenizer, templates and datasets. The optional `fsdp_turbo` package owns model parallelism, parameter wrapping, optimizer steps and clipping through its existing training lifecycle. This entry does not use HF Trainer or Megatron parallel wrapping.

## Source pair

Install this Swift checkout and pin FSDPTurbo to [`a2145cc546213dadd9404c9ae25ca1b1393f33fd`](https://gitcode.com/hazeldxq/FSDPTurbo/commit/a2145cc546213dadd9404c9ae25ca1b1393f33fd), based on upstream `6d69def`. The compatibility changes contain model/optional-dependency import fixes, native FSDP import isolation, eager MoE inverse-map and CUDA grouped-matmul fixes, parameter exclusions, and Engram dense parameter groups. Native V4.1 uses the upstream BF16 builder and sharded loader; it does not require a fork-specific checkpoint decoder or `load_device` argument. Transformers 5.16 or newer is required by the paired backend.

## Native DeepSeek V4.1

Prepare a dequantized BF16 Hugging Face checkpoint, its tokenizer, and a matching flat native `ModelArgs` JSON. A composite HF `config.json` and the native architecture JSON can be supplied separately. Training requires `dtype=bf16`, `expert_dtype=null`, `n_mtp_layers=0` and `dspark_block_size=0` in the native JSON. A reduced-depth experiment must retain the original widths/expert counts and declare only its instantiated Engram layers, preserving the corresponding hash buckets and table sizes.

For four available accelerator processes, the following illustrates the CLI. Choose device count and model size according to the available memory; this is an interface example, not a full 40-layer capacity result.

```bash
NPROC_PER_NODE=4 MASTER_PORT=29501 swift fsdpturbo sft \
  --model /path/to/v41-bf16 --model_type deepseek_v41 --template deepseek_v41 \
  --model_kwargs '{"config_path":"/path/to/native_model_args.json"}' \
  --dataset /path/to/train.jsonl --output_dir /path/to/output \
  --use_chat_template false --lazy_tokenize false --truncation_strategy split \
  --tuner_type full --torch_dtype bfloat16 --attn_impl eager \
  --fsdp_size 4 --tp_size 1 --cp_size 1 --ep_size 4 --efsdp_size 1 \
  --fsdp_implementation native --ep_dispatcher eager --offload_params true \
  --forward_prefetch 0 --backward_prefetch 0 \
  --max_steps 20 --max_length 1024 --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing false --vit_gradient_checkpointing false \
  --save_strategy no --eval_strategy no --split_dataset_ratio 0 \
  --add_version false --report_to none --enable_npu_model_patch false
```

The CLI starts one torchrun launcher. Set device visibility in the caller's CUDA or CANN environment. `enable_npu_model_patch=false` must take effect before Swift model imports because this backend owns parallel wrapping.

Swift calls the native builder, normalizes parameters to BF16 through `prepare_deepseek_v41_model_for_fsdp`, applies FSDPTurbo, then calls `model.from_pretrained(strict=True, engram_rows_per_chunk=4096)`. Parameter preparation uses CPU when CPU offload is enabled. Buffers retain their native dtypes; the head retains FP32 computation. Host Engram weights and active-row FP32 moments use the model's existing optimizer, while the dense optimizer accepts Swift's decay/no-decay groups.

## Parallel interfaces

The independent meshes overlap. `WORLD_SIZE` must be divisible by `fsdp_size * tp_size`, `ep_size * efsdp_size`, and `cp_size * tp_size`; multiplying all five sizes is incorrect. EFSDP requires EP greater than one, and TP requires native composable FSDP.

| Interface | Configuration |
| --- | --- |
| FSDP | `fsdp_size`; model-plan units are checked against actual modules |
| TP | `tp_size`; V4.1 uses shared-expert `w1/w3` column shards and `w2` row shards |
| EP / EFSDP | `ep_size` / `efsdp_size`; only expert containers enter these plans |
| CP | `cp_size` and `cp_implementation=auto/ulysses/kv_allgather/kcp`; the selected mode must match the model plan |

`auto` selects native KV-AllGather for V4.1 and Ulysses for the existing Qwen3.5 MoE plan. Native V4.1 CP delegates to `set_context_parallel_group` and model-owned loss, with no legacy attention/loss replacement. Its inputs are right-padded to CP/compression alignment, then token IDs, masks and global positions are split once; labels stay complete and unshifted. Text-only metadata takes this path even though HF metadata contains a vision tower.

V4.1 KV-AllGather CP requires TP=Ulysses=KCP=1 and eager operators. With `cp_size > 1`, its existing block recomputation can be enabled using both checkpointing switches. Ordinary V4.1 shared-cache recomputation remains unsupported. V4.1 attention-head TP is not implemented upstream; the TP plan above covers the shared experts.

The registered HF model path keeps its loader-selected FP32 stability parameters. Those parameters are excluded individually from FSDP, remain on the accelerator with CPU offload, and have gradients averaged before the optimizer step. An unused local branch still participates in synchronization; globally unused parameters retain `grad=None`.

## Data and lifecycle

The entry currently supports full causal-LM SFT, a map-style dataset, positive `max_steps`, GA1, AdamW and eager attention. V4.1 accepts text-only Swift batches; actual images and invalid text metadata fail explicitly. The previous V4 model plan is absent.

`save_strategy=no` or `steps` is accepted. The HF model path uses the existing DCP save/resume lifecycle with unchanged topology, optimizer groups, data cursor and RNG. Native V4.1 `resume_from_checkpoint` and `save_only_model` remain rejected; model-only files are not a Host Engram resume checkpoint. Evaluation, LoRA, PP and packing are not implemented by this entry.

For HF models, only rank zero loads original CPU weights; other ranks build on meta and load owner-local shards through DCP. Native V4.1 directly uses its native sharded initial-load API. Its raw FP8/FP4 checkpoint storage must be decoded separately before using this BF16 path.

## Validation

Run with the real Swift and optional FSDPTurbo dependencies:

```bash
python -m pytest -q tests/fsdpturbo
python -m torch.distributed.run --standalone --nproc_per_node=4 tests/fsdpturbo/distributed_loss.py
python -m torch.distributed.run --standalone --nproc_per_node=4 tests/fsdpturbo/distributed_replicated_gradients.py
```

The distributed commands use CPU/Gloo tensors. They establish metric and replica-gradient contracts, not accelerator model training. Actual model runs must separately verify initial weights, input/label hashes, loss/gradient curves, optimizer updates, topology, peak memory and clean exits. Results from earlier source pairs do not certify this pair.
