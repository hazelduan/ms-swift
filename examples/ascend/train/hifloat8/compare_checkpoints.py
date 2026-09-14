#!/usr/bin/env python3
"""Compare continuous and resumed DeepSpeed checkpoints without hiding small drift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_checkpoint(output: Path) -> Path:
    checkpoints = [
        path for path in output.glob("checkpoint-*") if path.name[11:].isdigit()
    ]
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint below {output}")
    return max(checkpoints, key=lambda path: int(path.name[11:]))


def one_file(directory: Path, patterns: tuple[str, ...]) -> Path:
    matches = [path for pattern in patterns for path in directory.glob(pattern)]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one of {patterns} below {directory}, found {matches}"
        )
    return matches[0]


def compare_weights(
    left: Path,
    right: Path,
    cosine_min: float,
    *,
    allow_nondeterministic: bool = False,
    nondeterminism_reason: str | None = None,
) -> dict[str, Any]:
    left_hash, right_hash = sha256(left), sha256(right)
    if left_hash == right_hash:
        return {
            "pass": True,
            "bitwise": True,
            "file_bitwise": True,
            "left_sha256": left_hash,
            "right_sha256": right_hash,
            "minimum_tensor_cosine": 1.0,
            "max_abs_delta": 0.0,
        }

    max_abs = 0.0
    minimum_cosine = 1.0
    tensor_metrics = {}
    all_tensors_bitwise = True
    with safe_open(left, framework="pt", device="cpu") as left_file, safe_open(
        right, framework="pt", device="cpu"
    ) as right_file:
        left_keys, right_keys = set(left_file.keys()), set(right_file.keys())
        if left_keys != right_keys:
            return {"pass": False, "bitwise": False, "keys_equal": False}
        for key in sorted(left_keys):
            left_tensor, right_tensor = (
                left_file.get_tensor(key),
                right_file.get_tensor(key),
            )
            if (
                left_tensor.shape != right_tensor.shape
                or left_tensor.dtype != right_tensor.dtype
            ):
                return {
                    "pass": False,
                    "bitwise": False,
                    "keys_equal": True,
                    "mismatch": key,
                }
            left_float, right_float = (
                left_tensor.float().flatten(),
                right_tensor.float().flatten(),
            )
            tensor_bitwise = torch.equal(
                left_tensor.contiguous().view(torch.uint8),
                right_tensor.contiguous().view(torch.uint8),
            )
            all_tensors_bitwise &= tensor_bitwise
            if left_float.numel():
                delta = left_float - right_float
                if not (
                    torch.isfinite(left_float).all()
                    and torch.isfinite(right_float).all()
                    and torch.isfinite(delta).all()
                ):
                    return {
                        "pass": False,
                        "bitwise": False,
                        "keys_equal": True,
                        "nonfinite_tensor": key,
                    }
                tensor_max_abs = float(delta.abs().max())
                if torch.equal(left_float, right_float):
                    cosine = 1.0
                else:
                    left_norm = float(torch.linalg.vector_norm(left_float))
                    right_norm = float(torch.linalg.vector_norm(right_float))
                    cosine = (
                        float(torch.dot(left_float, right_float))
                        / (left_norm * right_norm)
                        if left_norm and right_norm
                        else 0.0
                    )
                tensor_metrics[key] = {
                    "bitwise": tensor_bitwise,
                    "cosine": cosine,
                    "max_abs_delta": tensor_max_abs,
                }
                max_abs = max(max_abs, tensor_max_abs)
                minimum_cosine = min(minimum_cosine, cosine)
    fallback_allowed = bool(allow_nondeterministic and nondeterminism_reason)
    return {
        "pass": all_tensors_bitwise
        or (fallback_allowed and minimum_cosine >= cosine_min),
        "bitwise": all_tensors_bitwise,
        "file_bitwise": False,
        "keys_equal": True,
        "left_sha256": left_hash,
        "right_sha256": right_hash,
        "minimum_tensor_cosine": minimum_cosine,
        "max_abs_delta": max_abs,
        "fallback_allowed": fallback_allowed,
        "nondeterminism_reason": nondeterminism_reason,
        "failing_tensors": {
            key: value
            for key, value in tensor_metrics.items()
            if value["cosine"] < cosine_min
        },
    }


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {prefix: value}
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            result.update(flatten(child, f"{prefix}.{key}" if prefix else str(key)))
        return result
    if isinstance(value, (list, tuple)):
        result = {}
        for index, child in enumerate(value):
            result.update(flatten(child, f"{prefix}[{index}]"))
        return result
    if (
        value.__class__.__module__.startswith("deepspeed.runtime.fp16.loss_scaler")
        and hasattr(value, "__dict__")
    ):
        type_name = (
            f"{value.__class__.__module__}.{value.__class__.__qualname__}"
        )
        result = {f"{prefix}.__type__": type_name}
        result.update(flatten(vars(value), prefix))
        return result
    return {prefix: value}


def scalar_equal(left: Any, right: Any) -> bool:
    try:
        result = left == right
        return bool(result) if isinstance(result, bool) else bool(result.all())
    except (AttributeError, TypeError, ValueError):
        return repr(left) == repr(right)


def compare_state(left: Any, right: Any, max_abs_allowed: float) -> dict[str, Any]:
    left_values, right_values = flatten(left), flatten(right)
    if left_values.keys() != right_values.keys():
        return {"pass": False, "keys_equal": False, "tensor_shapes_equal": False}
    max_abs, mismatches, shapes_equal = 0.0, [], True
    for key in left_values:
        left_value, right_value = left_values[key], right_values[key]
        if isinstance(left_value, torch.Tensor):
            if (
                not isinstance(right_value, torch.Tensor)
                or left_value.shape != right_value.shape
            ):
                shapes_equal = False
                mismatches.append(key)
            elif left_value.dtype != right_value.dtype:
                mismatches.append(key)
            elif left_value.is_floating_point() or left_value.is_complex():
                if left_value.numel():
                    delta = left_value - right_value
                    if not (
                        torch.isfinite(left_value).all()
                        and torch.isfinite(right_value).all()
                        and torch.isfinite(delta).all()
                    ):
                        mismatches.append(f"{key}:nonfinite")
                    else:
                        max_abs = max(max_abs, float(delta.abs().max()))
            elif not torch.equal(left_value, right_value):
                mismatches.append(key)
        elif not scalar_equal(left_value, right_value):
            mismatches.append(key)
    return {
        "pass": not mismatches and max_abs <= max_abs_allowed,
        "keys_equal": True,
        "tensor_shapes_equal": shapes_equal,
        "max_abs_delta": max_abs,
        "nonfloating_or_shape_mismatches": mismatches[:100],
    }


def compare_trainer_state(
    left: Path, right: Path, expected_step: int
) -> dict[str, Any]:
    left_path, right_path = left / "trainer_state.json", right / "trainer_state.json"
    if not left_path.is_file() or not right_path.is_file():
        return {"pass": False, "reason": "trainer_state.json is missing"}
    left_step = json.loads(left_path.read_text(encoding="utf-8")).get("global_step")
    right_step = json.loads(right_path.read_text(encoding="utf-8")).get("global_step")
    return {
        "pass": left_step == right_step == expected_step,
        "expected_global_step": expected_step,
        "continuous_global_step": left_step,
        "resumed_global_step": right_step,
        "global_steps_equal": left_step == right_step,
    }


def compare_outputs(
    continuous_output: Path,
    resumed_output: Path,
    *,
    max_abs_allowed: float = 1e-4,
    cosine_min: float = 0.999999,
    allow_nondeterministic_weights: bool = False,
    nondeterminism_reason: str | None = None,
) -> dict[str, Any]:
    left_checkpoint, right_checkpoint = (
        latest_checkpoint(continuous_output),
        latest_checkpoint(resumed_output),
    )
    trainer_state = compare_trainer_state(
        left_checkpoint, right_checkpoint, expected_step=100
    )
    left_weights = one_file(
        left_checkpoint, ("model.safetensors", "adapter_model.safetensors")
    )
    right_weights = one_file(
        right_checkpoint, ("model.safetensors", "adapter_model.safetensors")
    )
    weights = compare_weights(
        left_weights,
        right_weights,
        cosine_min,
        allow_nondeterministic=allow_nondeterministic_weights,
        nondeterminism_reason=nondeterminism_reason,
    )

    # DeepSpeed stores rank-local ZeRO state below checkpoint-N/global_stepN/.
    left_optimizer = one_file(left_checkpoint, ("**/*optim_states.pt",))
    right_optimizer = one_file(right_checkpoint, ("**/*optim_states.pt",))
    optimizer = compare_state(
        torch.load(left_optimizer, map_location="cpu", weights_only=False),
        torch.load(right_optimizer, map_location="cpu", weights_only=False),
        max_abs_allowed,
    )

    left_scheduler, right_scheduler = (
        left_checkpoint / "scheduler.pt",
        right_checkpoint / "scheduler.pt",
    )
    if left_scheduler.is_file() and right_scheduler.is_file():
        scheduler = compare_state(
            torch.load(left_scheduler, map_location="cpu", weights_only=False),
            torch.load(right_scheduler, map_location="cpu", weights_only=False),
            max_abs_allowed,
        )
    else:
        left_model_state = torch.load(
            one_file(left_checkpoint, ("**/*model_states.pt",)),
            map_location="cpu",
            weights_only=False,
        )
        right_model_state = torch.load(
            one_file(right_checkpoint, ("**/*model_states.pt",)),
            map_location="cpu",
            weights_only=False,
        )
        if (
            "lr_scheduler" not in left_model_state
            or "lr_scheduler" not in right_model_state
        ):
            scheduler = {"pass": False, "reason": "scheduler state is missing"}
        else:
            scheduler = compare_state(
                left_model_state["lr_scheduler"],
                right_model_state["lr_scheduler"],
                max_abs_allowed,
            )
    result = {
        "pass": trainer_state["pass"]
        and weights["pass"]
        and optimizer["pass"]
        and scheduler["pass"],
        "continuous_checkpoint": str(left_checkpoint),
        "resumed_checkpoint": str(right_checkpoint),
        "weights": weights,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "trainer_state": trainer_state,
        "optimizer_state_keys_equal": optimizer.get("keys_equal", False),
        "optimizer_tensor_shapes_equal": optimizer.get("tensor_shapes_equal", False),
        "scheduler_state_keys_equal": scheduler.get("keys_equal", False),
        "scheduler_tensor_shapes_equal": scheduler.get("tensor_shapes_equal", False),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("continuous_output", type=Path)
    parser.add_argument("resumed_output", type=Path)
    parser.add_argument("--allow-nondeterministic-weights", action="store_true")
    parser.add_argument("--nondeterminism-reason")
    args = parser.parse_args()
    if args.allow_nondeterministic_weights and not args.nondeterminism_reason:
        parser.error("--allow-nondeterministic-weights requires --nondeterminism-reason")
    result = compare_outputs(
        args.continuous_output,
        args.resumed_output,
        allow_nondeterministic_weights=args.allow_nondeterministic_weights,
        nondeterminism_reason=args.nondeterminism_reason,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
