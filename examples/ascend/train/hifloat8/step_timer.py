"""Synchronized per-step timing and memory callback for phase-1 experiments."""

import json
import os
import resource
import time

import torch

from swift import TrainerCallback, callbacks_map


class HiFloat8StepTimer(TrainerCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        self.path = os.path.join(args.output_dir, "step_metrics.jsonl")
        self.previous_end = None
        self.previous_tokens = int(getattr(state, "num_input_tokens_seen", 0) or 0)
        if state.is_world_process_zero and state.global_step == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            open(self.path, "w", encoding="utf-8").close()
        self.get_hifloat8_op_counts = None
        try:
            from torch_npu.utils.hifloat8_train.hifloat8_linear import (
                get_hifloat8_op_counts,
                reset_hifloat8_op_counts,
            )

            reset_hifloat8_op_counts()
            self.get_hifloat8_op_counts = get_hifloat8_op_counts
        except (ImportError, RuntimeError):
            pass

    @staticmethod
    def _synchronize():
        if hasattr(torch, "npu") and torch.npu.is_available():
            torch.npu.synchronize()

    def on_step_begin(self, args, state, control, **kwargs):
        self._synchronize()
        self.step_start = time.perf_counter()
        self.step_start_op_counts = self.get_hifloat8_op_counts() if self.get_hifloat8_op_counts else {}

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
            "host_max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2,
        }
        if self.get_hifloat8_op_counts:
            current_op_counts = self.get_hifloat8_op_counts()
            record["hifloat8_ops"] = {
                name: count - self.step_start_op_counts.get(name, 0)
                for name, count in current_op_counts.items()
                if count != self.step_start_op_counts.get(name, 0)
            }
        if hasattr(torch, "npu") and torch.npu.is_available():
            record.update(
                {
                    "npu_allocated_gib": torch.npu.memory_allocated() / 1024**3,
                    "npu_reserved_gib": torch.npu.memory_reserved() / 1024**3,
                    "npu_peak_allocated_gib": torch.npu.max_memory_allocated() / 1024**3,
                    "npu_peak_reserved_gib": torch.npu.max_memory_reserved() / 1024**3,
                }
            )
        if state.is_world_process_zero:
            os.makedirs(args.output_dir, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        self.previous_end = now
        self.previous_tokens = tokens


callbacks_map["hifloat8_step_timer"] = HiFloat8StepTimer
