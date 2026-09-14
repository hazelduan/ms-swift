"""Synchronized per-step timing and memory callback for phase-1 experiments."""

import json
import os
import resource
import time

import torch

from swift import TrainerCallback, callbacks_map


class HiFloat8StepTimer(TrainerCallback):
    @staticmethod
    def _is_target_projection(name):
        return any(
            f".{projection}." in name
            for projection in ("gate_proj", "up_proj", "down_proj")
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self.path = os.path.join(args.output_dir, "step_metrics.jsonl")
        self.previous_end = None
        self.previous_tokens = int(getattr(state, "num_input_tokens_seen", 0) or 0)
        if state.is_world_process_zero and state.global_step == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            open(self.path, "w", encoding="utf-8").close()
        model = kwargs.get("model")
        self.audit_hifloat8_contract = (
            os.environ.get("HIF8_AUDIT_OP_CONTRACT") == "1"
        )
        self.get_hifloat8_op_counts = None
        self.hifloat8_modules = []
        self.hifloat8_forward_hooks = []
        try:
            from torch_npu.utils.hifloat8_train.hifloat8_linear import (
                HiFloat8Linear,
                get_hifloat8_op_counts,
                reset_hifloat8_op_counts,
            )

            reset_hifloat8_op_counts()
            self.get_hifloat8_op_counts = get_hifloat8_op_counts
            if model is not None and self.audit_hifloat8_contract:
                self.hifloat8_modules = [
                    (name, module)
                    for name, module in model.named_modules()
                    if isinstance(module, HiFloat8Linear)
                ]
                self.hifloat8_forward_hooks = [
                    module.register_forward_pre_hook(
                        self._capture_hifloat8_input(name)
                    )
                    for name, module in self.hifloat8_modules
                ]
                if state.is_world_process_zero:
                    modules = [
                        {
                            "name": name,
                            "in_features": module.in_features,
                            "out_features": module.out_features,
                            "weight_numel": module.weight.numel(),
                            "weight_dtype": str(module.weight.dtype),
                            "weight_requires_grad": module.weight.requires_grad,
                        }
                        for name, module in self.hifloat8_modules
                    ]
                    census = {
                        "count": len(modules),
                        "matrix_elements": sum(
                            item["weight_numel"] for item in modules
                        ),
                        "names": [item["name"] for item in modules],
                        "modules": modules,
                    }
                    os.makedirs(args.output_dir, exist_ok=True)
                    with open(
                        os.path.join(args.output_dir, "hifloat8_module_census.json"),
                        "w",
                        encoding="utf-8",
                    ) as stream:
                        json.dump(census, stream, indent=2, sort_keys=True)
        except (ImportError, RuntimeError):
            pass
        parameters = list(model.named_parameters()) if model is not None else []
        self.lora_base_parameters = [
            (name, parameter)
            for name, parameter in parameters
            if name.endswith(".base_layer.weight") and self._is_target_projection(name)
        ]
        self.lora_adapter_parameters = [
            (name, parameter)
            for name, parameter in parameters
            if (".lora_A." in name or ".lora_B." in name)
            and self._is_target_projection(name)
        ]
        self.lora_static = None
        self.validate_lora_gradients = os.environ.get("HIF8_VALIDATE_LORA_GRADS") == "1"
        self.lora_gradient_names = set()
        self.lora_finite_checks = []
        self.lora_gradient_status = None
        self.lora_hook_handles = []
        if self.lora_adapter_parameters:
            self.lora_static = {
                "base_parameter_count": len(self.lora_base_parameters),
                "base_all_frozen": all(
                    not parameter.requires_grad
                    for _, parameter in self.lora_base_parameters
                ),
                "base_dtypes": sorted(
                    {str(parameter.dtype) for _, parameter in self.lora_base_parameters}
                ),
                "adapter_parameter_count": len(self.lora_adapter_parameters),
                "adapter_all_trainable": all(
                    parameter.requires_grad
                    for _, parameter in self.lora_adapter_parameters
                ),
                "adapter_dtypes": sorted(
                    {
                        str(parameter.dtype)
                        for _, parameter in self.lora_adapter_parameters
                    }
                ),
            }
            self.lora_static["pass"] = (
                self.lora_static["base_parameter_count"] == 84
                and self.lora_static["base_all_frozen"]
                and self.lora_static["base_dtypes"] == ["torch.bfloat16"]
                and self.lora_static["adapter_parameter_count"] == 168
                and self.lora_static["adapter_all_trainable"]
                and self.lora_static["adapter_dtypes"] == ["torch.bfloat16"]
            )
            if not self.lora_static["pass"]:
                raise RuntimeError(
                    f"LoRA parameter invariant failed: {self.lora_static}"
                )
            if state.is_world_process_zero:
                with open(
                    os.path.join(args.output_dir, "lora_invariants.json"),
                    "w",
                    encoding="utf-8",
                ) as stream:
                    json.dump(self.lora_static, stream, indent=2, sort_keys=True)
            if self.validate_lora_gradients:
                self.lora_hook_handles = [
                    parameter.register_hook(self._capture_lora_gradient(name))
                    for name, parameter in self.lora_adapter_parameters
                ]

    def _capture_lora_gradient(self, name):
        def capture(gradient):
            self.lora_gradient_names.add(name)
            self.lora_finite_checks.append(torch.isfinite(gradient).all())
            return gradient

        return capture

    def _capture_hifloat8_input(self, name):
        def capture(module, inputs):
            input_value = inputs[0] if inputs else None
            self.hifloat8_forward_inputs.append(
                {
                    "name": name,
                    "input_requires_grad": bool(
                        isinstance(input_value, torch.Tensor)
                        and input_value.requires_grad
                    ),
                    "weight_requires_grad": bool(module.weight.requires_grad),
                }
            )

        return capture

    @staticmethod
    def _synchronize():
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()

    def on_step_begin(self, args, state, control, **kwargs):
        self._synchronize()
        self.step_start = time.perf_counter()
        self.step_start_op_counts = (
            self.get_hifloat8_op_counts() if self.get_hifloat8_op_counts else {}
        )
        self.lora_gradient_status = None
        self.lora_gradient_names.clear()
        self.lora_finite_checks.clear()
        self.hifloat8_forward_inputs = []

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if not self.validate_lora_gradients or not self.lora_adapter_parameters:
            return
        expected = len(self.lora_adapter_parameters)
        present = len(self.lora_gradient_names)
        self.lora_gradient_status = {
            "expected": expected,
            "present": present,
            "all_finite": (
                present == expected
                and bool(self.lora_finite_checks)
                and bool(torch.stack(self.lora_finite_checks).all().item())
            ),
        }

    def on_step_end(self, args, state, control, **kwargs):
        self._synchronize()
        now = time.perf_counter()
        tokens = int(getattr(state, "num_input_tokens_seen", 0) or 0)
        delta_tokens = max(0, tokens - self.previous_tokens)
        compute_seconds = now - self.step_start
        record = {
            "step": state.global_step,
            "compute_s": compute_seconds,
            "e2e_s": None if self.previous_end is None else now - self.previous_end,
            "delta_input_tokens": delta_tokens,
            "tokens_s": delta_tokens / compute_seconds if delta_tokens else None,
            "host_max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024**2,
        }
        if self.get_hifloat8_op_counts:
            current_op_counts = self.get_hifloat8_op_counts()
            record["hifloat8_ops"] = {
                name: count - self.step_start_op_counts.get(name, 0)
                for name, count in current_op_counts.items()
                if count != self.step_start_op_counts.get(name, 0)
            }
            expected_ops = {
                "forward": len(self.hifloat8_forward_inputs),
                "backward_dx": sum(
                    item["input_requires_grad"]
                    for item in self.hifloat8_forward_inputs
                ),
                "backward_dw": sum(
                    item["weight_requires_grad"]
                    for item in self.hifloat8_forward_inputs
                ),
            }
            if self.audit_hifloat8_contract:
                observed_ops = record["hifloat8_ops"]
                record["hifloat8_expected_ops"] = expected_ops
                record["hifloat8_no_dx_modules"] = [
                    item["name"]
                    for item in self.hifloat8_forward_inputs
                    if not item["input_requires_grad"]
                ]
                record["hifloat8_op_contract"] = {
                    "pass": all(
                        int(observed_ops.get(name, 0)) == expected
                        for name, expected in expected_ops.items()
                    ),
                    "module_count": len(self.hifloat8_modules),
                }
        if self.lora_static is not None:
            record["lora_static"] = self.lora_static
            record["lora_adapter_gradients"] = self.lora_gradient_status
        if hasattr(torch, "npu") and torch.npu.is_available():
            record.update(
                {
                    "npu_allocated_gib": torch.npu.memory_allocated() / 1024**3,
                    "npu_reserved_gib": torch.npu.memory_reserved() / 1024**3,
                    "npu_peak_allocated_gib": torch.npu.max_memory_allocated()
                    / 1024**3,
                    "npu_peak_reserved_gib": torch.npu.max_memory_reserved() / 1024**3,
                }
            )
        if state.is_world_process_zero:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.previous_end = now
        self.previous_tokens = tokens

    def on_train_end(self, args, state, control, **kwargs):
        for handle in self.lora_hook_handles + self.hifloat8_forward_hooks:
            handle.remove()


callbacks_map["hifloat8_step_timer"] = HiFloat8StepTimer
