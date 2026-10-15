<!-- Copyright (c) ModelScope Contributors. All rights reserved. -->

# DeepSpeed HiFloat8 configuration on Ascend

[`zero2_autoep_hifloat8.json`](zero2_autoep_hifloat8.json) uses DeepSpeed's optional
accelerator-selected TorchAO HiFloat8 APIs with BF16 parameters and communication,
ZeRO-2, and single-node AutoEP with four ranks. Swift reads this file through its
ordinary `--deepspeed` argument and passes the configuration to DeepSpeed. DeepSpeed
owns capability checks, the public `quantize_` conversion and optimizer creation;
no Swift backend plugin or kernel patch is needed.

From the repository root, add this argument to an appropriately configured training
command:

```text
--deepspeed examples/ascend/train/hifloat8/zero2_autoep_hifloat8.json
```

This is a configuration example, not a complete launch script. The JSON's `auto`
values are filled by Transformers/Accelerate from the training arguments. Use BF16
training arguments consistent with the file, including `--torch_dtype bfloat16`.

## Dependencies and configuration

Install matching, pinned revisions of the public TorchAO integration in
[DeepSpeed PR #8784](https://github.com/deepspeedai/DeepSpeed/pull/8784),
`torchao>=0.18,<0.19`, and the optional Ascend `torchao_npu` package that exposes
`torchao.hifloat8`. An upstream DeepSpeed release without that extension is not
sufficient. Prepare the paired PyTorch/TorchNPU versions, TorchAO, CANN/driver,
device permissions, and this machine's four-rank HCCL configuration separately.
Keep the extension's automatic activation enabled. No explicit `import torchao_npu`
or manual model conversion is required. Upstream TorchAO alone does not supply
the HiFloat8 implementation.

The source partial-training validation used Ascend950PR, CANN 9.1.0, driver
25.7.rc1, Python 3.11.15, PyTorch/TorchNPU 2.11.0, TorchAO 0.18.0,
Transformers 5.16.1, Accelerate 1.14.0, and Swift 4.6.0.dev0. Changing the stack
requires validation on the target machine.

`hifloat8.backend="torchao"` selects the accelerator's optional HiFloat8 APIs.
Linear and GroupedExperts are configured together by TorchAO `quantize_`.
DeepSpeed's normal initialization applies it after AutoEP replacement and device
placement, before optimizer setup. One `ModuleFqnToConfig` map uses AO's
`HiFloat8Config` for Linear and a small DeepSpeed handler to bind the same policy
and grouped GEMM to GroupedExperts. There is no provider conversion wrapper.
Its numerical configuration is optional: omitting `config`, as here, uses TorchAO's
validated BF16-input defaults (input/weight/gradient scale denominators 15/15/224). The
converted modules retain their input/output dtype at the module boundary even when
the working GEMM uses FP16. `expected_module_count` can be supplied as an additional
selection check once the converted module count has been established for the model.
The existing `hifloat8` section is retained; its old backend name `torchao_npu`
is normalized to `torchao`. The temporary `low_precision` section is rejected.

For a BF16 comparison, remove the `hifloat8` section and retain the same
AutoEP, ZeRO, communication, and training settings. The native BF16 grouped path
uses PyTorch/TorchNPU and does not require the optional TorchAO HiFloat8 package.

## Validation scope and parameter selection

The configuration is derived from a Qwen3.5-35B-A3B text-only partial-training run:
all 40 layers executed, while only the routed and shared experts in layers 0 and
39 were trainable (12 tensors per rank after EP4). The run used BF16 model weights,
batch size 1 per rank, gradient accumulation 1, sequence length 256, eager attention,
`--experts_impl eager`, `--enable_npu_model_patch false`, no activation checkpointing,
seed 42, and AdamW at `1e-5` with a cosine schedule.

The module patterns choose where low-precision computation runs; they do **not**
choose trainable parameters. To reproduce that scope, freeze all parameters and
explicitly unfreeze only the layer 0/39 expert parameters through Swift's
`freeze_parameters_regex` and `trainable_parameters_regex` arguments with
`--tuner_type full`. That tuner setting still requires the explicit freeze/unfreeze
selection to reproduce partial training. Confirm the actual parameter names and
trainable count for the loaded model before launching.
Set up the full BF16 model, dataset, four authorized NPUs, and local communication
configuration outside this example.

The HiFloat8 training contract is BF16 with ZeRO stages 0–2; this model evidence
covers EP4/ZeRO-2 only. Full-parameter SFT, LoRA, activation checkpointing, offload,
compile, EP8, multi-node training, and AutoEP checkpoint/resume remain untested.
ZeRO-3, pipeline/AutoTP/custom-MPU combinations, and quantized checkpoint ingestion
are unsupported by this integration. The original partial-training run disabled
checkpoint saving; it does not establish model checkpoint recovery.

The paired DeepSpeed integration includes Qwen3.5 text-wrapper compatibility
and NPU permutation index generation through the existing AutoEP extension points.
Neither requires additional fields in this configuration; vision input remains untested.
