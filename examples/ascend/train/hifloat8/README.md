# Qwen Dense HiFloat8 phase-1

This harness validates the complete Swift → Transformers/Accelerate → DeepSpeed
→ torch_npu training path. Parameters, optimizer states, adapters, attention,
normalization, loss, and communication remain BF16/FP32. Only the 28 Qwen MLP
`gate_proj`, `up_proj`, and `down_proj` GEMMs use native HiFloat8.

DeepSpeed converts modules after BF16 placement/broadcast and before parameter
collection or optimizer/ZeRO construction. It probes a real kernel before any
mutation, requires every selection pattern to match, and requires exactly 84
eligible modules. Full training and LoRA use separate configs because their
module names differ; LoRA A/B tensors are never selected.

## Environment

Run on a free, healthy Ascend950PR device with the matching CANN environment.
The runner is intentionally path-free: provide existing model, data, source,
environment, and output locations through variables. All three Python packages
must be editable installs from the supplied source checkouts.

```bash
export HIF8_MODEL=<qwen3-0.6b-directory>
export HIF8_SOURCE_DATA=<alpaca-500-jsonl>
export HIF8_OUTPUT_ROOT=<task-owned-output-directory>
export HIF8_CONDA_ENV=<isolated-conda-environment>
export HIF8_CANN_ENV=<cann-9.1-set-env-script>
export HIF8_DEEPSPEED_REPO=<deepspeed-checkout>
export HIF8_TORCH_NPU_REPO=<torch-npu-checkout>
export HIF8_DEVICE=4
export HIF8_CPUSET=<numa0-cpu-list>
export HIF8_MASTER_PORT=<free-local-port>
```

`run_phase1.sh` clears stale Ascend/CANN and distributed-launch variables,
selects the chosen environment directly, sources exactly one CANN setup script,
and rejects any device other
than physical NPU 4. Before every process it saves `npu-smi info` and proceeds
only when the card is healthy and its process table explicitly says it is free.
It also verifies Python `3.11.15`, torch `2.9.0+cpu`, torch_npu
`2.9.0.post4`, Transformers `5.16.1`, CANN `9.1.0`, clean editable source
trees, import paths, and revisions. Processes are pinned with `taskset`
to the supplied NUMA0 CPU list; the host does not require `numactl`.
The runner sets an explicit rank-0/world-size-1 local process group so
DeepSpeed never falls back to MPI discovery.
On this A5 host, `npu-smi`/DCMI returns permission error `-8005` for the `dxq`
account, so launch the harness as root while pointing it at the isolated existing
environment and task-owned work/output paths. The runner fails closed otherwise.

## Fixed experiment

- Data: deterministic seed-42 split, 400 train / 100 evaluation rows.
- Full: BF16 AdamW, LR `1e-5`, batch 4, gradient accumulation 1.
- LoRA: rank 8, alpha 16, dropout 0, gate/up/down only, LR `1e-4`.
- Accuracy: 100 steps; evaluation at 25/50/75/100; checkpoint at 50/100.
- Resume: a new output and process resume checkpoint 50 through step 100 for
  both BF16 and HiFloat8, full and LoRA.
- Performance: three independent 50-step runs per mode; discard steps 1–10.
- Profiler: separate four-step full and LoRA traces.

The YAML registers `HiFloat8StepTimer` from `step_timer.py`. It calls
`torch.npu.synchronize()` at both `on_step_begin` and `on_step_end`, so
`compute_s` and `e2e_s` are device-synchronized rather than enqueue time.

Run the validation ladder in order, or use `all`:

```bash
bash examples/ascend/train/hifloat8/run_phase1.sh preflight
bash examples/ascend/train/hifloat8/run_phase1.sh primitive
bash examples/ascend/train/hifloat8/run_phase1.sh accuracy
bash examples/ascend/train/hifloat8/run_phase1.sh resume
bash examples/ascend/train/hifloat8/run_phase1.sh perf
bash examples/ascend/train/hifloat8/run_phase1.sh profile
bash examples/ascend/train/hifloat8/run_phase1.sh report
```

For a durable sequential run:

```bash
mkdir -p "$HIF8_OUTPUT_ROOT"
nohup bash examples/ascend/train/hifloat8/run_phase1.sh all \
  >"$HIF8_OUTPUT_ROOT/launcher.log" 2>&1 &
echo $! >"$HIF8_OUTPUT_ROOT/launcher.pid"
```

The primitive stage covers both Qwen MLP directions at
`M={1,127,128,2560}` and requires finite forward/`dX`/`dW`, cosine ≥0.99,
and NRMSE ≤0.15. Accuracy runs require native per-step operation counts;
full training must execute base `dW`, while frozen LoRA bases must not.
For every LoRA accuracy and resumed step, the callback also records that all 84
base weights have `requires_grad=False`, all 168 A/B adapter tensors remain
trainable and produce finite gradients, and the HiFloat8 base `dW` counter is
exactly zero. Missing callback evidence fails the summary.

## Result gate

The summary command loads the complete curves, performs tensor-level checkpoint
comparison, reads all optimizer/scheduler tensors, scans native profiler files,
and emits JSON, CSV, and Markdown. It returns nonzero unless every correctness
gate passes; performance is always reported but is not a correctness gate.

```bash
python examples/ascend/train/hifloat8/summarize_phase1.py \
  "$HIF8_OUTPUT_ROOT" --output-dir "$HIF8_OUTPUT_ROOT/results"
```

The fixed training gates are loss Pearson ≥0.98, mean absolute loss delta ≤0.03,
P95 absolute loss delta ≤0.05, evaluation loss delta ≤0.05, token-accuracy
delta ≤0.01, resume next-step loss delta ≤`1e-5`, optimizer maximum tensor
delta ≤`1e-4`, and tensor-bitwise-identical final weights. A non-bitwise result
fails by default; the cosine ≥0.999999 fallback requires an explicit underlying
nondeterminism declaration and reason, which is recorded in the result.

Disabling the DeepSpeed `hifloat8` object returns to the ordinary BF16 path.
DeepSpeed Pipeline, AutoTP, custom MPU, ZeRO-3, and DeepSpeed MoE/EP are rejected;
MoE/EP integration is deliberately outside this Dense phase.
