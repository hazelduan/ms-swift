# DeepSpeed AutoEP HiFloat8 validation

This example extends the Dense training chain with routed grouped GEMM and
shared-expert Linear GEMM. DeepSpeed owns EP dispatch, expert sharding, ZeRO-2,
optimizer groups and checkpoints. It does not enable Transformers EP or FSDP.
Use the matching `hif8_deepspeed` source branches of torch-npu, DeepSpeed and
Swift; torch-npu must expose native HiFloat8 quantization and grouped training.

Source the matching CANN environment and activate the editable-install conda
environment before running. Select only the devices allocated to the experiment.
On the validated Ascend950PR/CANN 9.1 configuration, a hardware-generated CANN
v2 rank table is required for singleton expert data-parallel groups.

```bash
export ASCEND_RT_VISIBLE_DEVICES=<four-allocated-devices>
export RANK_TABLE_FILE=<task-owned-v2-rank-table>
export HCCL_CONNECT_TIMEOUT=300
export HIF8_MODEL=<existing-qwen3.5-35b-a3b-directory>
export HIF8_DATASET=<fixed-training-jsonl>
export HIF8_OUTPUT_DIR=<fresh-output-directory>
bash examples/ascend/train/hifloat8/run_autoep.sh bf16
```

Repeat with `hifloat8` and a different output directory. Both JSON configs use
the same optimizer and communication dtype. Leave the scheduler with Swift:
the deferred scheduler binding preserves the normal cosine schedule when
DeepSpeed creates the optimizer. Do not add Transformers EP or an FSDP config.

The example is text-only, sequence length 256, EP4, one sample per rank, seed 42,
20 steps, LR 1e-5 and checkpoints every 10 steps. It freezes the model except
the routed/shared projections in MoE layers 0 and 39. AutoEP splits the two
fused routed weights into three Parameters: the resulting model has 12 trainable
projection Parameters per rank, and 64 local routed experts in each of 40 layers.
All other expert projections still execute forward/dX through the selected
backend. Attention, router projection/softmax, shared gate and communication
are not quantized. LoRA, AutoTP, PP, ZeRO-3 and topology-changing resume are not
validated by this example.

For strict per-rank validation, add:

```bash
--external_plugins examples/ascend/train/hifloat8/autoep_audit.py \
--callbacks autoep_hifloat8_audit
```

The audit checks topology, exact trainable counts, every trainable gradient and
actual updates. It enables input gradients so frozen earlier paths exercise dX,
and saves per-rank timing, memory, operation counts and final trainable weights.
Set `HIF8_PROFILE=1` for a separate short profiling run; profiler overhead must
not be included in the matched performance comparison. Python counts do not
replace native kernel evidence. Raw logs/checkpoints/profiles belong outside Git.

Compare identical initial weights and sample order, full loss curves, validation
loss/accuracy and same-topology fresh-process checkpoint resume. A launch or
finite loss alone is not an acceptance result. Disabling `hifloat8.enabled`
returns the BF16 AutoEP computation; the existing Dense example remains separate.
