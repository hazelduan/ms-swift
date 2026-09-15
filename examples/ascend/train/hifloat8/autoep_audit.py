"""Per-rank AutoEP correctness evidence; use only in explicit validation runs."""
import hashlib
import json
import os
import resource
import sys
import time
import torch
import torch_npu
from pathlib import Path

from swift import TrainerCallback, callbacks_map


class AutoEPHiFloat8Audit(TrainerCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        from deepspeed.module_inject.auto_ep_layer import AutoEPMoELayer
        from torch_npu.utils.hifloat8_train import get_hifloat8_op_counts, reset_hifloat8_op_counts

        self.engine = self.trainer.model_wrapped
        self.model = self.engine.module
        self.rank = args.process_index
        self.root = Path(args.output_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.get_counts = get_hifloat8_op_counts
        reset_hifloat8_op_counts()
        self.model.enable_input_require_grads()
        layers = [(n, m) for n, m in self.model.named_modules() if isinstance(m, AutoEPMoELayer)]
        assert len(layers) == 40 and all(m.num_local_experts == 64 for _, m in layers)
        self.trainable = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        assert len(self.trainable) == 12, list(self.trainable)
        assert all('.experts.' in n or '.shared_experts.' in n for n in self.trainable)
        self.initial = {n: p.detach().cpu().clone() for n, p in self.trainable.items()}
        initial_digest = hashlib.sha256()
        for name, value in self.initial.items():
            initial_digest.update(name.encode())
            initial_digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
        self.checks = []
        self.hooks = [p.register_hook(self.capture_gradient(n)) for n, p in self.trainable.items()]
        self.previous_end = None
        self.original_compute_loss = self.trainer.compute_loss

        def audited_loss(model, inputs, *extra_args, **extra_kwargs):
            digest = hashlib.sha256()
            for key in ('input_ids', 'labels', 'attention_mask'):
                tensor = inputs.get(key)
                if isinstance(tensor, torch.Tensor):
                    digest.update(key.encode())
                    digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
            result = self.original_compute_loss(model, inputs, *extra_args, **extra_kwargs)
            if model.training:
                loss = result[0] if isinstance(result, tuple) else result
                self.batches.append({'sha256': digest.hexdigest(), 'loss': float(loss.detach())})
            return result

        self.trainer.compute_loss = audited_loss
        self.batches = []
        manifest = {
            'rank': self.rank,
            'layers': len(layers),
            'local_experts': 64,
            'trainable': {
                n: {
                    'shape': list(p.shape),
                    'dtype': str(p.dtype)
                }
                for n, p in self.trainable.items()
            },
            'hifloat8': self.engine.hifloat8_enabled(),
            'initial_step': state.global_step,
            'initial_trainable_sha256': initial_digest.hexdigest()
        }
        (self.root / f'autoep_manifest_rank{self.rank}.json').write_text(json.dumps(manifest, indent=2))
        self.profiler = None
        if os.environ.get('HIF8_PROFILE') == '1':
            self.install_profile_scopes()
            self.profiler = torch_npu.profiler.profile(
                activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=2, repeat=1),
                on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(self.root / 'profiler')),
                experimental_config=torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1),
                record_shapes=True,
                profile_memory=True)
            self.profiler.start()

    def install_profile_scopes(self):
        import deepspeed.comm as comm
        import torch.nn.functional as functional
        from functools import wraps

        self.profile_originals = []

        def wrap(owner, name):
            original = getattr(owner, name)
            self.profile_originals.append((owner, name, original))

            @wraps(original)
            def measured(*args, **kwargs):
                tensors = []
                for value in args[:2]:
                    values = value if isinstance(value, (list, tuple)) else [value]
                    tensors.extend(str(t.dtype) for t in values if isinstance(t, torch.Tensor))
                metadata = {
                    key: str(kwargs[key])
                    for key in ('group_type', 'x_dtype', 'weight_dtype', 'x1_dtype', 'x2_dtype') if key in kwargs
                }
                if name in ('npu_grouped_matmul', 'softmax'):
                    frame = sys._getframe(1)
                    while frame is not None:
                        caller = frame.f_code.co_qualname
                        if caller.startswith(('_GroupedMatmulWithHiFloat8.', '_NPUGroupedMatmul.',
                                              'TokenChoiceTopKRouter.')) and caller.endswith(('forward', 'backward')):
                            metadata['caller'] = caller
                            break
                        frame = frame.f_back
                    del frame
                label = f'autoep_audit::{name}::{json.dumps([tensors, metadata], sort_keys=True)}'
                with torch.profiler.record_function(label):
                    return original(*args, **kwargs)

            setattr(owner, name, measured)

        for name in ('npu_grouped_matmul', 'npu_quant_matmul', 'npu_dynamic_quant', 'npu_quantize'):
            wrap(torch_npu, name)
        wrap(functional, 'softmax')
        wrap(comm, 'all_to_all_single')

    def capture_gradient(self, name):

        def capture(grad):
            self.checks.append((name, torch.isfinite(grad).all()))
            return grad

        return capture

    def on_step_begin(self, args, state, control, **kwargs):
        torch_npu.npu.synchronize()
        self.begin = time.perf_counter()
        self.checks = []
        self.batches = []

    def on_step_end(self, args, state, control, **kwargs):
        torch_npu.npu.synchronize()
        now = time.perf_counter()
        assert {n for n, _ in self.checks} == set(self.trainable)
        assert bool(torch.stack([v for _, v in self.checks]).all())
        row = {
            'step': state.global_step,
            'rank': self.rank,
            'compute_s': now - self.begin,
            'e2e_s': None if self.previous_end is None else now - self.previous_end,
            'grad_norm': float(self.engine.get_global_grad_norm()),
            'gradients_finite': True,
            'batches': self.batches,
            'counts': self.get_counts(),
            'peak_npu_bytes': torch_npu.npu.max_memory_allocated(),
            'host_peak_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        }
        with (self.root / f'autoep_steps_rank{self.rank}.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        self.previous_end = now
        if self.profiler is not None:
            self.profiler.step()

    def on_train_end(self, args, state, control, **kwargs):
        if self.profiler is not None:
            self.profiler.stop()
            for owner, name, original in self.profile_originals:
                setattr(owner, name, original)
        final = {n: p.detach().cpu() for n, p in self.trainable.items()}
        updates = {n: not torch.equal(value, self.initial[n]) for n, value in final.items()}
        assert all(updates.values()), updates
        for hook in self.hooks:
            hook.remove()
        self.trainer.compute_loss = self.original_compute_loss
        torch.save(final, self.root / f'autoep_final_rank{self.rank}.pt')
        (self.root / f'autoep_updates_rank{self.rank}.json').write_text(json.dumps(updates, indent=2))


callbacks_map['autoep_hifloat8_audit'] = AutoEPHiFloat8Audit
