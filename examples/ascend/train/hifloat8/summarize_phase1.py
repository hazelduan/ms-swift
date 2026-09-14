#!/usr/bin/env python3
"""Apply the predeclared Dense HiFloat8 acceptance gates to a phase-1 run tree."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from compare_checkpoints import compare_outputs


STEPS = 100
EVAL_STEPS = (25, 50, 75, 100)
THRESHOLDS = {
    "loss_pearson_min": 0.98,
    "loss_mean_abs_delta_max": 0.03,
    "loss_p95_abs_delta_max": 0.05,
    "eval_loss_abs_delta_max": 0.05,
    "eval_acc_abs_delta_max": 0.01,
    "resume_next_loss_abs_delta_max": 1e-5,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def history(output: Path) -> list[dict[str, Any]]:
    records = read_jsonl(output / "logging.jsonl")
    summaries = [
        record["log_history"]
        for record in records
        if isinstance(record.get("log_history"), list)
    ]
    if summaries:
        return summaries[-1]
    normalized = []
    for record in records:
        item = dict(record)
        if step_text := item.pop("global_step/max_steps", None):
            item["step"] = int(step_text.split("/", 1)[0])
        normalized.append(item)
    return normalized


def exitcode(run: Path) -> int | None:
    path = run / "exitcode"
    return int(path.read_text().strip()) if path.is_file() else None


def indexed(records: list[dict[str, Any]], key: str) -> dict[int, dict[str, Any]]:
    return {
        int(item["step"]): item for item in records if key in item and "step" in item
    }


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    covariance = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left)
        * sum((value - right_mean) ** 2 for value in right)
    )
    return covariance / denominator if denominator else None


def precision(run_root: Path, tuner: str) -> dict[str, Any]:
    baseline = run_root / f"accuracy/{tuner}_bf16"
    candidate = run_root / f"accuracy/{tuner}_hifloat8"
    baseline_history, candidate_history = (
        history(baseline / "output"),
        history(candidate / "output"),
    )
    baseline_train, candidate_train = (
        indexed(baseline_history, "loss"),
        indexed(candidate_history, "loss"),
    )
    steps = sorted(set(baseline_train) & set(candidate_train))
    baseline_loss = [float(baseline_train[step]["loss"]) for step in steps]
    candidate_loss = [float(candidate_train[step]["loss"]) for step in steps]
    deltas = [abs(left - right) for left, right in zip(baseline_loss, candidate_loss)]
    correlation = pearson(baseline_loss, candidate_loss)
    mean_delta = statistics.fmean(deltas) if deltas else None
    p95_delta = percentile(deltas, 0.95)
    train_finite = all(
        all(
            isinstance(row.get(key), (int, float)) and math.isfinite(float(row[key]))
            for key in ("loss", "grad_norm", "token_acc", "learning_rate")
        )
        for records in (baseline_train, candidate_train)
        for row in records.values()
    )

    baseline_eval, candidate_eval = (
        indexed(baseline_history, "eval_loss"),
        indexed(candidate_history, "eval_loss"),
    )
    evaluations, eval_pass = [], True
    for step in EVAL_STEPS:
        left, right = baseline_eval.get(step), candidate_eval.get(step)
        if (
            left is None
            or right is None
            or left.get("eval_token_acc") is None
            or right.get("eval_token_acc") is None
        ):
            evaluations.append(
                {
                    "step": step,
                    "pass": False,
                    "reason": "missing eval loss/token accuracy",
                }
            )
            eval_pass = False
            continue
        loss_delta = abs(float(left["eval_loss"]) - float(right["eval_loss"]))
        acc_delta = abs(float(left["eval_token_acc"]) - float(right["eval_token_acc"]))
        passed = (
            loss_delta <= THRESHOLDS["eval_loss_abs_delta_max"]
            and acc_delta <= THRESHOLDS["eval_acc_abs_delta_max"]
        )
        evaluations.append(
            {
                "step": step,
                "loss_abs_delta": loss_delta,
                "acc_abs_delta": acc_delta,
                "pass": passed,
            }
        )
        eval_pass &= passed

    passed = (
        exitcode(baseline) == exitcode(candidate) == 0
        and steps == list(range(1, STEPS + 1))
        and train_finite
        and correlation is not None
        and correlation >= THRESHOLDS["loss_pearson_min"]
        and mean_delta is not None
        and mean_delta <= THRESHOLDS["loss_mean_abs_delta_max"]
        and p95_delta is not None
        and p95_delta <= THRESHOLDS["loss_p95_abs_delta_max"]
        and eval_pass
    )
    return {
        "pass": passed,
        "steps": steps,
        "train_finite": train_finite,
        "loss_pearson": correlation,
        "loss_mean_abs_delta": mean_delta,
        "loss_p95_abs_delta": p95_delta,
        "evaluations": evaluations,
    }


def module_census(
    run_root: Path, tuner: str, phase: str = "accuracy"
) -> dict[str, Any]:
    log = run_root / f"{phase}/{tuner}_hifloat8/stdout.log"
    text = log.read_text(encoding="utf-8", errors="replace") if log.is_file() else ""
    matches = re.findall(
        r"HiFloat8 training converted (\d+) Linear modules \((\d+) matrix elements\): (\[[^\n]+\])",
        text,
    )
    if not matches:
        return {"pass": False, "reason": "conversion census not found", "log": str(log)}
    count, elements, names_text = matches[-1]
    names = ast.literal_eval(names_text)
    suffixes = [
        f".mlp.{projection}" for projection in ("gate_proj", "up_proj", "down_proj")
    ]
    if tuner == "lora":
        suffixes = [f"{suffix}.base_layer" for suffix in suffixes]
    suffix_counts = {
        suffix: sum(name.endswith(suffix) for name in names) for suffix in suffixes
    }
    forbidden = [
        name
        for name in names
        if any(part in name for part in ("self_attn", "lm_head", "embed", "lora_"))
    ]
    passed = (
        int(count) == len(names) == 84 and int(elements) == 264241152 and not forbidden
    )
    passed &= all(value == 28 for value in suffix_counts.values())
    return {
        "pass": passed,
        "count": int(count),
        "matrix_elements": int(elements),
        "suffix_counts": suffix_counts,
        "forbidden": forbidden,
        "names": names,
        "log": str(log),
    }


def operator_counts(
    run_root: Path,
    tuner: str,
    phase: str = "accuracy",
    expected_steps: list[int] | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(
        run_root / f"{phase}/{tuner}_hifloat8/output/step_metrics.jsonl"
    )
    expected_steps = expected_steps or list(range(1, STEPS + 1))
    totals: defaultdict[str, int] = defaultdict(int)
    failures = []
    expected_rows = []
    for row in rows:
        observed = row.get("hifloat8_ops", {})
        expected = row.get("hifloat8_expected_ops", {})
        expected_rows.append(expected)
        for key, value in observed.items():
            totals[key] += int(value)
        for key in ("forward", "backward_dx", "backward_dw"):
            value = expected.get(key)
            if value is None:
                failures.append(
                    {
                        "step": row.get("step"),
                        "operator": key,
                        "reason": "independent expected count is missing",
                    }
                )
                continue
            if int(observed.get(key, 0)) != value:
                failures.append(
                    {
                        "step": row.get("step"),
                        "operator": key,
                        "expected": value,
                        "actual": observed.get(key, 0),
                    }
                )
        if row.get("hifloat8_op_contract", {}).get("module_count") != 84:
            failures.append(
                {
                    "step": row.get("step"),
                    "reason": "HiFloat8 module count is not 84",
                }
            )
    expected_forward = {item.get("forward") for item in expected_rows}
    expected_dx = {item.get("backward_dx") for item in expected_rows}
    expected_dw = {item.get("backward_dw") for item in expected_rows}
    expected_no_dx_suffixes = (
        set()
        if tuner == "full"
        else {
            ".layers.0.mlp.gate_proj.base_layer",
            ".layers.0.mlp.up_proj.base_layer",
        }
    )
    no_dx_pass = all(
        len(row.get("hifloat8_no_dx_modules", [])) == len(expected_no_dx_suffixes)
        and {
            suffix
            for suffix in expected_no_dx_suffixes
            if any(
                name.endswith(suffix)
                for name in row.get("hifloat8_no_dx_modules", [])
            )
        }
        == expected_no_dx_suffixes
        for row in rows
    )
    invariant_pass = (
        expected_forward == {84}
        and expected_dx == ({84} if tuner == "full" else {82})
        and expected_dw == ({84} if tuner == "full" else {0})
        and no_dx_pass
    )
    return {
        "pass": (
            [int(row.get("step", -1)) for row in rows] == expected_steps
            and invariant_pass
            and not failures
        ),
        "steps": len(rows),
        "expected_per_step_values": {
            "forward": sorted(value for value in expected_forward if value is not None),
            "backward_dx": sorted(value for value in expected_dx if value is not None),
            "backward_dw": sorted(value for value in expected_dw if value is not None),
        },
        "totals": dict(totals),
        "no_dx_modules_match_fixed_graph": no_dx_pass,
        "failures": failures[:100],
    }


def lora_invariants(run_root: Path) -> dict[str, Any]:
    runs = {}
    for phase, expected_steps in (
        ("accuracy", list(range(1, STEPS + 1))),
        ("resume", list(range(51, STEPS + 1))),
    ):
        for precision_name in ("bf16", "hifloat8"):
            name = f"{phase}/lora_{precision_name}"
            run = run_root / name
            rows = read_jsonl(run / "output/step_metrics.jsonl")
            static_path = run / "output/lora_invariants.json"
            static = (
                json.loads(static_path.read_text(encoding="utf-8"))
                if static_path.is_file()
                else {}
            )
            static_pass = (
                bool(rows)
                and static.get("pass") is True
                and static.get("base_parameter_count") == 84
                and static.get("base_all_frozen") is True
                and static.get("base_dtypes") == ["torch.bfloat16"]
                and static.get("adapter_parameter_count") == 168
                and static.get("adapter_all_trainable") is True
                and static.get("adapter_dtypes") == ["torch.bfloat16"]
                and all(row.get("lora_static") == static for row in rows)
            )
            gradients_pass = bool(rows) and all(
                row.get("lora_adapter_gradients")
                == {"expected": 168, "present": 168, "all_finite": True}
                for row in rows
            )
            observed_steps = [int(row["step"]) for row in rows if "step" in row]
            base_dw_zero = None
            if precision_name == "hifloat8":
                base_dw_zero = bool(rows) and all(
                    int(row.get("hifloat8_ops", {}).get("forward", 0)) == 84
                    and int(row.get("hifloat8_ops", {}).get("backward_dx", 0))
                    == int(
                        row.get("hifloat8_expected_ops", {}).get(
                            "backward_dx", -1
                        )
                    )
                    and int(
                        row.get("hifloat8_expected_ops", {}).get(
                            "backward_dx", 0
                        )
                    )
                    > 0
                    and int(row.get("hifloat8_ops", {}).get("backward_dw", 0)) == 0
                    for row in rows
                )
            passed = (
                exitcode(run) == 0
                and observed_steps == expected_steps
                and static_pass
                and gradients_pass
                and base_dw_zero is not False
            )
            runs[name] = {
                "pass": passed,
                "static_pass": static_pass,
                "base_all_frozen": static.get("base_all_frozen"),
                "base_parameter_count": static.get("base_parameter_count"),
                "base_dtypes": static.get("base_dtypes"),
                "adapter_all_trainable": static.get("adapter_all_trainable"),
                "adapter_parameter_count": static.get("adapter_parameter_count"),
                "adapter_dtypes": static.get("adapter_dtypes"),
                "adapter_gradients_finite_every_step": gradients_pass,
                "base_hifloat8_dw_zero_every_step": base_dw_zero,
                "steps": observed_steps,
            }
    return {"pass": all(value["pass"] for value in runs.values()), "runs": runs}


def resume(
    run_root: Path,
    tuner: str,
    precision_name: str,
    *,
    allow_nondeterministic_weights: bool = False,
    nondeterminism_reason: str | None = None,
) -> dict[str, Any]:
    continuous = run_root / f"accuracy/{tuner}_{precision_name}"
    resumed = run_root / f"resume/{tuner}_{precision_name}"
    continuous_train = indexed(history(continuous / "output"), "loss")
    resumed_train = indexed(history(resumed / "output"), "loss")
    executed = {
        int(row["step"]) for row in read_jsonl(resumed / "output/step_metrics.jsonl")
    }
    steps = sorted(set(continuous_train) & set(resumed_train) & executed)
    deltas = [
        abs(float(continuous_train[step]["loss"]) - float(resumed_train[step]["loss"]))
        for step in steps
    ]
    checkpoint = compare_outputs(
        continuous / "output",
        resumed / "output",
        allow_nondeterministic_weights=allow_nondeterministic_weights,
        nondeterminism_reason=nondeterminism_reason,
    )
    passed = (
        exitcode(continuous) == exitcode(resumed) == 0
        and steps == list(range(51, STEPS + 1))
        and deltas
        and deltas[0] <= THRESHOLDS["resume_next_loss_abs_delta_max"]
        and checkpoint["pass"]
    )
    return {
        "pass": bool(passed),
        "steps": steps,
        "next_loss_abs_delta": deltas[0] if deltas else None,
        "max_loss_abs_delta": max(deltas) if deltas else None,
        "checkpoint": checkpoint,
    }


def performance(run_root: Path, tuner: str, precision_name: str) -> dict[str, Any]:
    repeats = []
    for repeat in (1, 2, 3):
        run = run_root / f"perf/{tuner}_{precision_name}_r{repeat}"
        rows = [
            row
            for row in read_jsonl(run / "output/step_metrics.jsonl")
            if int(row["step"]) > 10
        ]
        compute = [float(row["compute_s"]) for row in rows]
        e2e = [float(row["e2e_s"]) for row in rows if row.get("e2e_s") is not None]
        tokens = sum(int(row.get("delta_input_tokens", 0)) for row in rows)
        timing_finite = (
            len(compute) == len(e2e) == 40
            and all(math.isfinite(value) and value > 0 for value in compute + e2e)
            and tokens > 0
        )
        repeats.append(
            {
                "repeat": repeat,
                "pass": exitcode(run) == 0 and timing_finite,
                "steps": len(rows),
                "timing_finite_positive": timing_finite,
                "compute_p50_s": percentile(compute, 0.50),
                "compute_p95_s": percentile(compute, 0.95),
                "e2e_p50_s": percentile(e2e, 0.50),
                "e2e_p95_s": percentile(e2e, 0.95),
                "tokens_per_compute_s": tokens / sum(compute)
                if compute and sum(compute)
                else None,
                "tokens_per_e2e_s": tokens / sum(e2e)
                if e2e and sum(e2e)
                else None,
                "peak_npu_allocated_gib": max(
                    (row.get("npu_peak_allocated_gib", 0.0) for row in rows),
                    default=None,
                ),
                "peak_npu_reserved_gib": max(
                    (row.get("npu_peak_reserved_gib", 0.0) for row in rows),
                    default=None,
                ),
                "peak_host_rss_gib": max(
                    (row.get("host_max_rss_gib", 0.0) for row in rows), default=None
                ),
            }
        )
    fields = (
        "compute_p50_s",
        "compute_p95_s",
        "e2e_p50_s",
        "e2e_p95_s",
        "tokens_per_compute_s",
        "tokens_per_e2e_s",
        "peak_npu_allocated_gib",
        "peak_npu_reserved_gib",
        "peak_host_rss_gib",
    )
    means = {
        field: statistics.fmean(row[field] for row in repeats if row[field] is not None)
        if any(row[field] is not None for row in repeats)
        else None
        for field in fields
    }
    return {
        "pass": all(row["pass"] for row in repeats),
        "repeats": repeats,
        "mean_of_repeats": means,
    }


def performance_delta(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, float | None]:
    baseline, candidate = baseline["mean_of_repeats"], candidate["mean_of_repeats"]

    def reduction(field: str) -> float | None:
        left, right = baseline[field], candidate[field]
        return 100 * (left - right) / left if left and right is not None else None

    def gain(field: str) -> float | None:
        left, right = baseline[field], candidate[field]
        return 100 * (right - left) / left if left and right is not None else None

    return {
        "e2e_p50_reduction_pct": reduction("e2e_p50_s"),
        "e2e_p95_reduction_pct": reduction("e2e_p95_s"),
        "token_throughput_gain_pct": gain("tokens_per_compute_s"),
        "e2e_token_throughput_gain_pct": gain("tokens_per_e2e_s"),
        "peak_npu_allocated_reduction_pct": reduction("peak_npu_allocated_gib"),
        "peak_host_rss_reduction_pct": reduction("peak_host_rss_gib"),
    }


def primitives(root: Path) -> dict[str, Any]:
    cases = {}
    for rows in (1, 127, 128, 2560):
        result = root / "primitives" / f"m{rows}.json"
        code = root / "primitives" / f"m{rows}.exitcode"
        payload = json.loads(result.read_text()) if result.is_file() else {}
        status = int(code.read_text().strip()) if code.is_file() else None
        cases[str(rows)] = {
            "pass": status == 0 and payload.get("status") == "pass",
            "result": payload,
        }
    return {"pass": all(case["pass"] for case in cases.values()), "cases": cases}


def profiler(run_root: Path) -> dict[str, Any]:
    result = {}
    patterns = {
        "quantize": re.compile(
            r"^aclnn(?:Dynamic)?Quantize_.*(?:AiCore|AIVEC|Quantize)", re.I
        ),
        "quant_matmul": re.compile(
            r"^aclnnQuantMatmulV[0-9]+_.*QuantBatchMatmul", re.I
        ),
    }
    for tuner in ("full", "lora"):
        run = run_root / f"profile/{tuner}_hifloat8"
        directory = run / "output/profiler"
        profiled_steps = [
            int(row.get("step", -1))
            for row in read_jsonl(run / "output/step_metrics.jsonl")
        ]
        hits = {name: [] for name in patterns}
        csv_files = (
            sorted(directory.rglob("ASCEND_PROFILER_OUTPUT/kernel_details.csv"))
            if directory.is_dir()
            else []
        )
        completed = []
        for path in csv_files:
            if not (path.parent / "analyse.done").is_file():
                continue
            completed.append(str(path))
            with path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    kernel_name = row.get("Name", "")
                    try:
                        physical_device = int(row.get("Device_id", ""))
                        duration = float(row.get("Duration(us)", ""))
                    except ValueError:
                        continue
                    if physical_device != 4 or duration <= 0:
                        continue
                    for name, pattern in patterns.items():
                        if pattern.search(kernel_name):
                            hits[name].append(
                                {
                                    "file": str(path),
                                    "device_id": physical_device,
                                    "kernel": kernel_name,
                                    "duration_us": duration,
                                    "step_id": row.get("Step Id"),
                                }
                            )
        result[tuner] = {
            "pass": (
                exitcode(run) == 0
                and profiled_steps == [1, 2, 3, 4]
                and bool(completed)
                and all(hits.values())
            ),
            "exitcode": exitcode(run),
            "training_steps": profiled_steps,
            "hits": hits,
            "completed_kernel_exports": completed,
            "directory": str(directory),
        }
    result["pass"] = result["full"]["pass"] and result["lora"]["pass"]
    return result


def report_markdown(result: dict[str, Any]) -> str:
    overall = "PASS" if result["pass"] else "FAIL / INCOMPLETE"
    lines = ["# Qwen3-0.6B Dense HiFloat8 phase-1", "", f"Overall: **{overall}**", ""]
    lines += ["| Gate | Result |", "| --- | --- |"]
    for name in result["hard_gates"]:
        lines.append(
            f"| {name} | {'PASS' if result[name]['pass'] else 'FAIL / INCOMPLETE'} |"
        )
    lines += [
        "",
        "## Performance",
        "",
        "| Mode | Compute P50/P95 (s) | E2E P50/P95 (s) | Tokens/s compute/e2e | Peak NPU alloc/reserved (GiB) | Peak host RSS (GiB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, value in result["performance"].items():
        mean = value["mean_of_repeats"]
        lines.append(
            f"| {name} | {mean['compute_p50_s']} / {mean['compute_p95_s']} | "
            f"{mean['e2e_p50_s']} / {mean['e2e_p95_s']} | "
            f"{mean['tokens_per_compute_s']} / {mean['tokens_per_e2e_s']} | "
            f"{mean['peak_npu_allocated_gib']} / {mean['peak_npu_reserved_gib']} | "
            f"{mean['peak_host_rss_gib']} |"
        )
    lines += [
        "",
        "Performance is informational and never changes the correctness result. Relative deltas and all three raw "
        "repeats are in `training_results.json`.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-nondeterministic-weights", action="store_true")
    parser.add_argument("--nondeterminism-reason")
    args = parser.parse_args()
    if args.allow_nondeterministic_weights and not args.nondeterminism_reason:
        parser.error("--allow-nondeterministic-weights requires --nondeterminism-reason")
    run_root = args.experiment_root / "runs"
    performance_results = {
        f"{tuner}_{precision_name}": performance(run_root, tuner, precision_name)
        for tuner in ("full", "lora")
        for precision_name in ("bf16", "hifloat8")
    }
    result = {
        "thresholds": THRESHOLDS,
        "weight_reproducibility_policy": {
            "require_tensor_bitwise": not args.allow_nondeterministic_weights,
            "allow_nondeterministic_weights": args.allow_nondeterministic_weights,
            "nondeterminism_reason": args.nondeterminism_reason,
            "fallback_minimum_tensor_cosine": 0.999999,
        },
        "primitives": primitives(args.experiment_root),
        "full_precision": precision(run_root, "full"),
        "lora_precision": precision(run_root, "lora"),
        "full_module_census": module_census(run_root, "full"),
        "lora_module_census": module_census(run_root, "lora"),
        "full_operator_counts": operator_counts(run_root, "full"),
        "lora_operator_counts": operator_counts(run_root, "lora"),
        "full_resume_module_census": module_census(run_root, "full", "resume"),
        "lora_resume_module_census": module_census(run_root, "lora", "resume"),
        "full_resume_operator_counts": operator_counts(
            run_root, "full", "resume", list(range(51, STEPS + 1))
        ),
        "lora_resume_operator_counts": operator_counts(
            run_root, "lora", "resume", list(range(51, STEPS + 1))
        ),
        "lora_invariants": lora_invariants(run_root),
        "full_hifloat8_resume": resume(
            run_root,
            "full",
            "hifloat8",
            allow_nondeterministic_weights=args.allow_nondeterministic_weights,
            nondeterminism_reason=args.nondeterminism_reason,
        ),
        "lora_hifloat8_resume": resume(
            run_root,
            "lora",
            "hifloat8",
            allow_nondeterministic_weights=args.allow_nondeterministic_weights,
            nondeterminism_reason=args.nondeterminism_reason,
        ),
        "full_bf16_resume": resume(
            run_root,
            "full",
            "bf16",
            allow_nondeterministic_weights=args.allow_nondeterministic_weights,
            nondeterminism_reason=args.nondeterminism_reason,
        ),
        "lora_bf16_resume": resume(
            run_root,
            "lora",
            "bf16",
            allow_nondeterministic_weights=args.allow_nondeterministic_weights,
            nondeterminism_reason=args.nondeterminism_reason,
        ),
        "profiler": profiler(run_root),
        "performance": performance_results,
        "performance_complete": {
            "pass": all(value["pass"] for value in performance_results.values())
        },
        "performance_comparisons": {
            "full_hifloat8_vs_bf16": performance_delta(
                performance_results["full_bf16"], performance_results["full_hifloat8"]
            ),
            "lora_hifloat8_vs_bf16": performance_delta(
                performance_results["lora_bf16"], performance_results["lora_hifloat8"]
            ),
            "bf16_lora_vs_full": performance_delta(
                performance_results["full_bf16"], performance_results["lora_bf16"]
            ),
            "hifloat8_lora_vs_full": performance_delta(
                performance_results["full_hifloat8"],
                performance_results["lora_hifloat8"],
            ),
        },
    }
    result["hard_gates"] = [
        "primitives",
        "full_precision",
        "lora_precision",
        "full_module_census",
        "lora_module_census",
        "full_operator_counts",
        "lora_operator_counts",
        "full_resume_module_census",
        "lora_resume_module_census",
        "full_resume_operator_counts",
        "lora_resume_operator_counts",
        "lora_invariants",
        "full_hifloat8_resume",
        "lora_hifloat8_resume",
        "full_bf16_resume",
        "lora_bf16_resume",
        "profiler",
        "performance_complete",
    ]
    result["pass"] = all(result[name]["pass"] for name in result["hard_gates"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "training_results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "experiment_report.md").write_text(
        report_markdown(result), encoding="utf-8"
    )
    with (args.output_dir / "validation_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(("gate", "pass"))
        writer.writerows((name, result[name]["pass"]) for name in result["hard_gates"])
    print(args.output_dir / "training_results.json")
    raise SystemExit(0 if result["pass"] else 1)


if __name__ == "__main__":
    main()
