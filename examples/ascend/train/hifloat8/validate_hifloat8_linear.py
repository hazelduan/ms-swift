#!/usr/bin/env python3
"""Validate native HiFloat8 forward, dX, and dW against BF16 at a Qwen MLP shape."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu

from torch_npu.utils.hifloat8_train.hifloat8_linear import (
    _MatmulWithHiFloat8,
    assert_hifloat8_training_available,
    get_hifloat8_op_counts,
    reset_hifloat8_op_counts,
)


THRESHOLDS = {"cosine_min": 0.99, "nrmse_max": 0.15}


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_float = actual.detach().float().flatten()
    reference_float = reference.detach().float().flatten()
    return {
        "cosine": F.cosine_similarity(actual_float, reference_float, dim=0).item(),
        "nrmse": (torch.linalg.vector_norm(actual_float - reference_float)
                  / torch.linalg.vector_norm(reference_float).clamp_min(1e-12)).item(),
        "max_abs": (actual_float - reference_float).abs().max().item(),
    }


def write_result(result: dict, output: Path | None) -> None:
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


def validate_case(name: str, rows: int, in_features: int, out_features: int, seed: int, device: torch.device) -> dict:
    stage = f"{name}:allocate_inputs"
    try:
        torch.manual_seed(seed)
        input_value = torch.randn((rows, in_features), dtype=torch.bfloat16, device=device)
        weight_value = torch.randn((out_features, in_features), dtype=torch.bfloat16, device=device)
        upstream = torch.randn((rows, out_features), dtype=torch.bfloat16, device=device)

        stage = f"{name}:bf16_reference_forward_backward"
        reference_input = input_value.detach().clone().requires_grad_()
        reference_weight = weight_value.detach().clone().requires_grad_()
        reference_output = F.linear(reference_input, reference_weight)
        (reference_output * upstream).sum().backward()

        stage = f"{name}:hifloat8_forward_backward"
        reset_hifloat8_op_counts()
        actual_input = input_value.detach().clone().requires_grad_()
        actual_weight = weight_value.detach().clone().requires_grad_()
        actual_output = _MatmulWithHiFloat8.apply(actual_input, actual_weight)
        (actual_output * upstream).sum().backward()
        torch.npu.synchronize(device)
    except Exception as error:
        return {"status": "fail", "failure_stage": stage, "reason": str(error)}

    metrics = {
        "forward": error_metrics(actual_output, reference_output),
        "dX": error_metrics(actual_input.grad, reference_input.grad),
        "dW": error_metrics(actual_weight.grad, reference_weight.grad),
    }
    checks = {
        metric: values["cosine"] >= THRESHOLDS["cosine_min"]
        and values["nrmse"] <= THRESHOLDS["nrmse_max"]
        for metric, values in metrics.items()
    }
    expected_ops = {
        "forward": 1,
        "backward_dx": 1,
        "backward_dw": 1,
        "quantize_input": 1,
        "quantize_weight": 1,
        "quantize_grad": 1,
        "quant_matmul": 3,
    }
    observed_ops = get_hifloat8_op_counts()
    return {
        "status": "pass" if all(checks.values()) and observed_ops == expected_ops else "fail",
        "shape": {"m": rows, "k": in_features, "n": out_features},
        "checks": checks,
        "metrics": metrics,
        "expected_ops": expected_ops,
        "observed_ops": observed_ops,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    device = torch.device(args.device)
    result = {
        "device": str(device),
        "device_name": torch_npu.npu.get_device_name(device),
        "dtype": "bfloat16",
        "shapes": {
            "gate_up": {"m": args.rows, "k": args.hidden_size, "n": args.intermediate_size},
            "down": {"m": args.rows, "k": args.intermediate_size, "n": args.hidden_size},
        },
        "thresholds": THRESHOLDS,
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
    }
    try:
        assert_hifloat8_training_available(probe_kernel=True, device=device)
    except RuntimeError as error:
        result.update(status="blocked", reason=str(error))
        write_result(result, args.output)
        return 2

    cases = {
        "gate_up": validate_case(
            "gate_up", args.rows, args.hidden_size, args.intermediate_size, args.seed, device
        ),
        "down": validate_case(
            "down", args.rows, args.intermediate_size, args.hidden_size, args.seed + 1, device
        ),
    }
    result.update(status="pass" if all(case["status"] == "pass" for case in cases.values()) else "fail", cases=cases)
    write_result(result, args.output)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
