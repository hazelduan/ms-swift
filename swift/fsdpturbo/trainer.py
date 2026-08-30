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


def _dtype_name(args) -> str:
    if args.bf16 or args.torch_dtype == torch.bfloat16:
        return 'bf16'
    if args.fp16 or args.torch_dtype == torch.float16:
        return 'fp16'
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
    from fsdp_turbo.fsdp_turbo_config import (CheckpointConfig, DataConfig, DistributedConfig, EPPlanConfig,
                                               FSDPPlanConfig, FSDPTurboConfig, MemoryConfig, ModelConfig,
                                               OptimizerConfig, TPPlanConfig, TrainRunConfig)

    train_args = args.training_args
    dtype = _dtype_name(train_args)
    return FSDPTurboConfig(
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
            expert_parallel_size=args.ep_size,
            expert_fully_shard_parallel_size=args.efsdp_size,
            fsdp_plan=FSDPPlanConfig(
                ignored_modules=[],
                apply_modules={pattern: {} for pattern in spec.fsdp_modules},
                param_dtype=dtype,
                reduce_dtype='fp32',
                output_dtype=dtype,
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
                dispatcher=args.ep_dispatcher,
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
        super().__init__(config=build_fsdpturbo_config(args, self.spec))
        self.processor = processor

    def setup(self):
        super().setup()
        steps_per_epoch = len(self.dataloader)
        if steps_per_epoch < 1:
            raise ValueError('FSDPTurbo requires at least one local training batch per epoch.')
        required_epochs = math.ceil(self.train_args.max_steps / steps_per_epoch)
        self.config.run.num_train_epochs = max(self.config.run.num_train_epochs, required_epochs)

    def _init_distributed(self):
        super()._init_distributed()
        self.args.validate_fsdpturbo(dist.get_world_size())

    def build_tokenizer(self):
        return self.template.tokenizer

    def build_processor(self):
        return self.processor

    def build_model(self):
        logger.info('Loading the Swift model on CPU before FSDPTurbo sharding.')
        model, processor = self.args.get_model_processor(device_map='cpu')
        if processor is not None:
            self.processor = processor
        model = TunerMixin.prepare_model(self.args, model, template=self.template, train_dataset=self.train_dataset)
        from fsdp_turbo.utils.model import convert_model_dtype
        convert_model_dtype(model, self.config.model.torch_dtype)
        self.spec.validate_model(
            model,
            require_tp=self.args.tp_size > 1,
            require_ep=self.args.ep_size > 1,
            require_recompute=self.args.gradient_checkpointing,
        )
        if hasattr(model.config, 'use_cache'):
            model.config.use_cache = False
        self.template.model = model

        from fsdp_turbo.fsdp_turbo import FSDPTurbo

        logger.info('Applying the independent FSDPTurbo parallel backend.')
        return FSDPTurbo(self.config, model)

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
        return get_scheduler(
            _scheduler_name(self.train_args),
            optimizer=self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            scheduler_specific_kwargs=self.train_args.lr_scheduler_kwargs or {},
        )

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
