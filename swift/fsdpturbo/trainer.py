# Copyright (c) ModelScope Contributors. All rights reserved.
import math
from functools import partial
from typing import Dict, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from fsdp_turbo.training.trainer import BaseTrainer

from swift.pipelines.train.tuner import TunerMixin
from swift.utils import get_logger, seed_worker
from .model_specs import FSDPTurboModelSpec, get_model_spec


logger = get_logger()


def _validate_dtype(args) -> None:
    if not (args.bf16 or args.fp16 or args.torch_dtype in (torch.bfloat16, torch.float16)):
        raise ValueError('FSDPTurbo currently requires bf16 or fp16 training.')


def _save_steps(train_args) -> int:
    if str(train_args.save_strategy).lower().endswith('no'):
        return 2**31 - 1
    if isinstance(train_args.save_steps, float) and train_args.save_steps < 1:
        return max(1, math.ceil(train_args.max_steps * train_args.save_steps))
    return max(1, int(train_args.save_steps))


def _scheduler_name(train_args) -> str:
    return getattr(train_args.lr_scheduler_type, 'value', train_args.lr_scheduler_type)


def build_fsdpturbo_config(args, spec: FSDPTurboModelSpec):
    from fsdp_turbo.fsdp_turbo_config import (CheckpointConfig, CPPlanConfig, DataConfig, DistributedConfig, EPPlanConfig,
                                               FSDPPlanConfig, FSDPTurboConfig, MemoryConfig, ModelConfig,
                                               OptimizerConfig, TPPlanConfig, TrainRunConfig)

    train_args = args.training_args
    _validate_dtype(train_args)
    cp_size = args.cp_size
    if cp_size > 1 and not spec.cp_function_patches:
        raise ValueError(f'FSDPTurbo CP is not implemented for {spec.model_type}. Use cp_size=1.')
    module_patches = spec.module_patches + (spec.cp_module_patches if cp_size > 1 else ())
    return FSDPTurboConfig(
        module_patches=[{'target': target, 'replacement': replacement} for target, replacement in module_patches],
        model=ModelConfig(
            model_name_or_path=args.model,
            tokenizer_name_or_path=args.model,
            torch_dtype=args.torch_dtype,
        ),
        optimizer=OptimizerConfig(
            optimizer_type='AdamW',
            weight_decay=train_args.weight_decay,
            adam_beta1=train_args.adam_beta1,
            adam_beta2=train_args.adam_beta2,
            adam_epsilon=train_args.adam_epsilon,
            lr=train_args.learning_rate,
            warmup_ratio=train_args.warmup_ratio,
            lr_scheduler_type=_scheduler_name(train_args),
            clip_grad=train_args.max_grad_norm,
        ),
        data=DataConfig(
            dataset_path=','.join(args.dataset or args.cached_dataset),
            batch_size=train_args.per_device_train_batch_size,
            max_seq_length=args.max_length,
            num_workers=train_args.dataloader_num_workers,
            pin_memory=train_args.dataloader_pin_memory,
            shuffle=train_args.train_dataloader_shuffle,
        ),
        run=TrainRunConfig(
            seed=train_args.seed,
            max_steps=train_args.max_steps,
            num_train_epochs=math.ceil(train_args.num_train_epochs),
            gradient_accumulation_steps=train_args.gradient_accumulation_steps,
            logging_steps=train_args.logging_steps,
        ),
        checkpoint=CheckpointConfig(
            output_dir=args.output_dir,
            resume_from_checkpoint=args.resume_from_checkpoint,
            save_steps=_save_steps(train_args),
            save_optim=not train_args.save_only_model,
            load_optim=not args.resume_only_model,
        ),
        distributed=DistributedConfig(
            fully_shard_parallel_size=args.fsdp_size,
            tensor_parallel_size=args.tp_size,
            ulysses_parallel_size=cp_size,
            cp_plan=CPPlanConfig(
                ulysses_function_patches=[{'target_functions': [target], 'type': patch_type}
                                          for target, patch_type in spec.cp_function_patches] if cp_size > 1 else [],
                loss_function_patches=[{'target_functions': ['transformers.loss.loss_utils.ForCausalLMLoss'],
                                        'type': 'causal_lm_loss'}] if cp_size > 1 else [],
            ),
            expert_parallel_size=args.ep_size,
            expert_fully_shard_parallel_size=args.efsdp_size,
            fsdp_plan=FSDPPlanConfig(
                ignored_modules=[],
                apply_modules={pattern: {} for pattern in spec.fsdp_modules},
                # Swift's loader already materializes ordinary weights in the
                # requested dtype while preserving numerically sensitive
                # checkpoint parameters (for example Qwen3.5 A_log) in fp32.
                # A global FSDP param cast would erase that mixed-dtype policy.
                param_dtype=None,
                reduce_dtype='fp32',
                # Preserve HF's fp32 CE/auxiliary loss; forcing the entire
                # ModelOutput to bf16 quantizes scalar losses and metrics.
                output_dtype=None,
                num_to_forward_prefetch=args.forward_prefetch,
                num_to_backward_prefetch=args.backward_prefetch,
                hook_modules=list(spec.fsdp_hook_modules),
                cpu_offload=args.offload_params,
                pin_memory=args.pin_memory,
                fsdp_implementation=args.fsdp_implementation,
            ),
            tp_plan=TPPlanConfig(
                colwise_parallel=list(spec.tp_colwise_modules),
                rowwise_parallel=list(spec.tp_rowwise_modules),
                sequence_parallel=[],
            ),
            ep_plan=EPPlanConfig(
                apply_modules=list(spec.ep_modules),
                apply_efsdp_modules=list(spec.efsdp_modules),
                dispatcher=(spec.eager_dispatcher or args.ep_dispatcher)
                if args.ep_dispatcher == 'eager' else args.ep_dispatcher,
            ),
        ),
        memory=MemoryConfig(
            recompute=args.gradient_checkpointing,
            recompute_plan=list(spec.recompute_modules),
        ),
    )


class FSDPTurboTrainer(BaseTrainer):
    """A narrow Swift data/model adapter around FSDPTurbo's training lifecycle."""

    def __init__(self, args, template, processor, train_dataset):
        self.args = args
        self.train_args = args.training_args
        self.template = template
        self.train_dataset = train_dataset
        self.spec = get_model_spec(args.model_type)
        if self.spec.native_model_factory:
            if args.gradient_checkpointing:
                raise ValueError('Native V4.1 CSA2 shared runtime does not support recompute yet.')
            if args.ep_size < 2 or args.ep_dispatcher != 'eager' or args.resume_from_checkpoint:
                raise ValueError('Native V4.1 currently requires EP > 1, eager dispatch, and no checkpoint resume.')
            if args.router_aux_loss_coef:
                raise ValueError('Native V4.1 router auxiliary loss is not implemented.')
        if args.tp_size > 1 and not (self.spec.tp_colwise_modules and self.spec.tp_rowwise_modules):
            raise ValueError(f'FSDPTurbo TP is not implemented for {args.model_type}. Use tp_size=1.')
        self._replicated_parameters = []
        super().__init__(config=build_fsdpturbo_config(args, self.spec))
        self.processor = processor

    def setup(self):
        super().setup()
        self._initialize_loss_group()
        steps_per_epoch = len(self.dataloader)
        if steps_per_epoch < 1:
            raise ValueError('FSDPTurbo requires at least one local training batch per epoch.')
        required_epochs = math.ceil(self.train_args.max_steps / steps_per_epoch)
        self.config.run.num_train_epochs = max(self.config.run.num_train_epochs, required_epochs)

    def _initialize_loss_group(self):
        # HCCL allocates communicator buffers lazily. Initialize the separate
        # metric-reduction group before backward fills the allocator cache.
        parallel_state = self.model.parallel_state
        if parallel_state.get_data_group_size() > 1:
            device = self.train_args.device
            probe = torch.zeros((), device=device)
            dist.all_reduce(probe, group=parallel_state.get_data_group())
            probe.item()  # Complete communicator allocation before training.

    def _init_distributed(self):
        from fsdp_turbo.utils.log import set_log_level

        # Swift may already own the process group, so BaseTrainer can return
        # before configuring the backend topology and per-step metric logs.
        set_log_level("INFO")
        super()._init_distributed()
        self.args.validate_fsdpturbo(dist.get_world_size())

    def build_tokenizer(self):
        return self.template.tokenizer

    def build_processor(self):
        return self.processor

    def build_model(self):
        from .loading import create_meta_model, load_rank0_model, materialize_resume_model, materialize_sharded_model

        source_model, dtypes = load_rank0_model(self.args)
        from accelerate import init_empty_weights
        with init_empty_weights(include_buffers=False):
            model = create_meta_model(self.args)
        for name, param in model.named_parameters():
            param.data = param.data.to(dtypes[name])
        model = TunerMixin.prepare_model(self.args, model, template=self.template, train_dataset=self.train_dataset)
        for names in self.spec.matching_modules(model, self.spec.frozen_modules).values():
            for name in names:
                model.get_submodule(name).requires_grad_(False)
        self.spec.validate_model(
            model,
            require_tp=self.args.tp_size > 1,
            require_ep=self.args.ep_size > 1,
            require_recompute=self.args.gradient_checkpointing,
        )
        parameter_names = {name for name, param in model.named_parameters()
                           if param.dtype == torch.float32 and self.args.torch_dtype != torch.float32}
        self.config.distributed.fsdp_plan.ignored_params = sorted(parameter_names)
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
        if self.args.router_aux_loss_coef:
            text_config = getattr(model.config, 'text_config', model.config)
            text_config.output_router_logits = True
            text_config.router_aux_loss_coef = self.args.router_aux_loss_coef
            if hasattr(model, 'router_aux_loss_coef'):
                model.router_aux_loss_coef = self.args.router_aux_loss_coef
        self.template.model = model

        from fsdp_turbo.fsdp_turbo import FSDPTurbo

        logger.info('Applying the independent FSDPTurbo parallel backend.')
        model = FSDPTurbo(self.config, model)
        if self.args.resume_from_checkpoint:
            device = 'cpu' if self.args.offload_params else self.train_args.device
            materialize_resume_model(model.model, device=device)
        else:
            materialize_sharded_model(model.model, source_model, offload=self.args.offload_params)
        # DCP cpu_offload includes persistent buffers. FSDP only offloads
        # parameters; routing lookup tables and rotary buffers stay on device.
        for module in model.model.modules():
            for name, buffer in module.named_buffers(recurse=False):
                setattr(module, name, buffer.to(self.train_args.device))
        self._replicated_parameters = [param for name, param in model.model.named_parameters()
                                      if name in parameter_names]
        for param in self._replicated_parameters:
            param.data = param.data.to(self.train_args.device)
        self.template.model = model.model
        return model

    def _sync_replicated_gradients(self):
        if not self._replicated_parameters or not dist.is_initialized():
            return
        world_size = dist.get_world_size()
        if world_size == 1:
            return
        params = [param for param in self._replicated_parameters if param.requires_grad]
        if not params:
            return
        # Every rank issues the same collectives even when a compressed branch
        # is unused for its local sequence. Globally unused parameters keep
        # grad=None so AdamW does not decay them or create optimizer state.
        present = torch.tensor([param.grad is not None for param in params], device=params[0].device, dtype=torch.int32)
        gradients = torch.cat([
            (param.grad.detach() if param.grad is not None else torch.zeros_like(param)).reshape(-1)
            for param in params
        ])
        dist.all_reduce(gradients, op=dist.ReduceOp.SUM)
        dist.all_reduce(present, op=dist.ReduceOp.SUM)
        gradients.div_(world_size)
        for param, gradient, active in zip(params, gradients.split([param.numel() for param in params]), present.tolist()):
            param.grad = gradient.view_as(param) if active else None

    def _on_optimizer_step(self):
        self._sync_replicated_gradients()
        return super()._on_optimizer_step()

    def build_dataloader(self):
        if self.train_dataset is None:
            raise ValueError('FSDPTurbo requires a training dataset.')
        parallel_state = self.model.parallel_state
        sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=parallel_state.get_data_group_size(),
            rank=parallel_state.get_data_rank(),
            shuffle=self.train_args.train_dataloader_shuffle,
            seed=self.args.data_seed,
            drop_last=self.train_args.dataloader_drop_last,
        )
        workers = self.train_args.dataloader_num_workers
        kwargs = {
            'dataset': self.train_dataset,
            'batch_size': self.train_args.per_device_train_batch_size,
            'sampler': sampler,
            'collate_fn': partial(self.template.data_collator, padding_to=None),
            'num_workers': workers,
            'pin_memory': self.train_args.dataloader_pin_memory,
            'drop_last': self.train_args.dataloader_drop_last,
            'worker_init_fn': partial(
                seed_worker, num_workers=workers, rank=parallel_state.get_data_rank()),
        }
        if workers > 0:
            kwargs.update({
                'persistent_workers': self.train_args.dataloader_persistent_workers,
                'prefetch_factor': self.train_args.dataloader_prefetch_factor,
                'multiprocessing_context': self.train_args.dataloader_multiprocessing_context,
            })
            kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return DataLoader(**kwargs)

    def build_scheduler(self):
        if self.optimizer is None:
            return None
        from transformers import get_scheduler

        num_training_steps = self._resolve_max_steps()
        num_warmup_steps = self.train_args.get_warmup_steps(num_training_steps)
        factory = partial(
            get_scheduler,
            _scheduler_name(self.train_args),
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=self.train_args.lr_scheduler_kwargs or {},
        )
        builder = getattr(self.optimizer, 'build_scheduler', None)
        return builder(factory) if callable(builder) else factory(optimizer=self.optimizer)

    def build_optimizer(self):
        builder = getattr(self.model.model, 'build_optimizer', None)
        return builder(self.config.optimizer) if callable(builder) else super().build_optimizer()

    def train_step(self, batch):
        loss_scale = batch.pop('loss_scale', None)
        batch.pop('channel', None)
        batch.pop('text_position_ids', None)
        batch.pop('_extra_kwargs', None)
        if loss_scale is not None and 'labels' in batch:
            active_scale = loss_scale.masked_select(batch['labels'].ne(-100))
            if active_scale.numel() and not torch.all(active_scale.eq(1)):
                raise NotImplementedError('Non-uniform Swift loss scaling is not supported by FSDPTurbo yet.')
        batch['use_cache'] = False
        if self.args.router_aux_loss_coef:
            batch['output_router_logits'] = True
        outputs = self.model(**batch)
        if outputs.loss is None:
            raise RuntimeError('Model did not return a loss; ensure the batch contains labels.')
        loss = outputs.loss / self.config.run.gradient_accumulation_steps
        loss.backward()
        return loss, getattr(outputs, 'aux_loss', None)

    def sync_loss(self, accumulated_losses: Dict[str, list]) -> Dict[str, Optional[torch.Tensor]]:
        group = self.model.parallel_state.get_data_group()
        group_size = self.model.parallel_state.get_data_group_size()
        result = {}
        for name, losses in accumulated_losses.items():
            if not losses:
                result[name] = None
                continue
            value = torch.stack(losses).sum()
            if group_size > 1:
                dist.all_reduce(value, op=dist.ReduceOp.SUM, group=group)
                value.div_(group_size)
            result[name] = value
        return result

    @property
    def result(self):
        history = []
        if self._monitor is not None:
            history = [{
                'step': step,
                'loss': loss,
                'step_time': step_time,
                'grad_norm': float(grad_norm) if grad_norm is not None else None,
            } for step, loss, step_time, grad_norm in self._monitor.loss_history]
        return {'global_step': self._global_step, 'log_history': history}
