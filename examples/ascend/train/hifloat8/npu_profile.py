"""Collect a short native NPU trace for the HiFloat8 phase-1 experiment."""

from pathlib import Path

import torch_npu

from swift import TrainerCallback, callbacks_map


class HiFloat8Profiler(TrainerCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        output = Path(args.output_dir) / 'profiler'
        output.mkdir(parents=True, exist_ok=True)
        self.profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output)),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_modules=True,
        )
        self.profiler.start()

    def on_step_end(self, args, state, control, **kwargs):
        self.profiler.step()

    def on_train_end(self, args, state, control, **kwargs):
        self.profiler.stop()


callbacks_map['hifloat8_profiler'] = HiFloat8Profiler
