# Copyright (c) ModelScope Contributors. All rights reserved.
import os
from dataclasses import dataclass
from typing import Literal, Optional

from swift.arguments import SftArguments


@dataclass
class FSDPTurboArguments:
    """Topology owned by the independent FSDPTurbo training backend."""

    fsdp_size: int = 1
    tp_size: int = 1
    ep_size: int = 1
    efsdp_size: int = 1
    pp_size: int = 1
    ep_dispatcher: Literal['eager', 'fused', 'mc2', 'domino'] = 'eager'
    fsdp_implementation: Literal['native', 'custom'] = 'native'
    offload_params: bool = False
    pin_memory: bool = True
    forward_prefetch: int = 1
    backward_prefetch: int = 1

    def validate_fsdpturbo(self, world_size: Optional[int] = None) -> None:
        sizes = {
            'fsdp_size': self.fsdp_size,
            'tp_size': self.tp_size,
            'ep_size': self.ep_size,
            'efsdp_size': self.efsdp_size,
            'pp_size': self.pp_size,
        }
        for name, size in sizes.items():
            if size < 1:
                raise ValueError(f'{name} must be positive, got {size}.')
        if self.pp_size != 1:
            raise ValueError(
                'FSDPTurbo main does not implement pipeline parallelism; pp_size must remain 1. '
                'Adding PP outside the backend would build FSDP/EP meshes across pipeline stages.')
        if self.efsdp_size > 1 and self.ep_size == 1:
            raise ValueError('efsdp_size > 1 requires ep_size > 1 in FSDPTurbo.')
        if self.forward_prefetch < 0 or self.backward_prefetch < 0:
            raise ValueError('FSDPTurbo prefetch counts must be non-negative.')

        if world_size is None and 'WORLD_SIZE' in os.environ:
            world_size = int(os.environ['WORLD_SIZE'])
        if world_size is None:
            return
        if world_size < 1:
            raise ValueError(f'world_size must be positive, got {world_size}.')

        dense_product = self.fsdp_size * self.tp_size
        expert_product = self.efsdp_size * self.ep_size
        if world_size % dense_product:
            raise ValueError(
                f'world_size ({world_size}) must be divisible by fsdp_size * tp_size ({dense_product}).')
        if world_size % expert_product:
            raise ValueError(
                f'world_size ({world_size}) must be divisible by efsdp_size * ep_size ({expert_product}).')


@dataclass
class FSDPTurboSftArguments(FSDPTurboArguments, SftArguments):
    tuner_type: Literal['full'] = 'full'
    add_version: bool = False

    def __post_init__(self) -> None:
        self.validate_fsdpturbo()
        if self.enable_npu_model_patch:
            raise ValueError('FSDPTurbo requires `--enable_npu_model_patch false` before Swift model imports.')
        if self.add_version:
            raise ValueError('FSDPTurbo requires a shared explicit output_dir; `--add_version` must be false.')
        if self.resume_only_model:
            raise ValueError('The initial FSDPTurbo backend does not support checkpoint resume yet.')
        if self.resume_from_checkpoint:
            raise ValueError('The initial FSDPTurbo backend does not support checkpoint resume yet.')
        if self.fsdp:
            raise ValueError('`--fsdp` selects the HF/Accelerate backend and cannot be combined with FSDPTurbo.')
        if self.deepspeed:
            raise ValueError('DeepSpeed cannot be combined with the independent FSDPTurbo backend.')
        if self.tuner_type != 'full':
            raise ValueError('The initial FSDPTurbo backend supports full SFT only.')
        if self.sequence_parallel_size > 1:
            raise ValueError('Use `--tp_size` for FSDPTurbo; Swift sequence parallel is a different backend.')
        if self.padding_free or self.packing:
            raise ValueError('FSDPTurbo padding-free and packing paths have not been validated yet.')
        super().__post_init__()
        self.validate_fsdpturbo()
        if self.task_type != 'causal_lm':
            raise ValueError('The initial FSDPTurbo backend supports causal-LM SFT only.')
        if self.max_steps <= 0:
            raise ValueError('The initial FSDPTurbo backend requires `--max_steps` to be a positive integer.')
        if self.gradient_accumulation_steps != 1:
            raise ValueError('The initial FSDPTurbo backend requires `--gradient_accumulation_steps 1`.')
        if self.streaming:
            raise ValueError('The initial FSDPTurbo backend requires a map-style dataset, not streaming data.')
        save_strategy = getattr(self.training_args.save_strategy, 'value', self.training_args.save_strategy)
        if save_strategy != 'no':
            raise ValueError('The initial FSDPTurbo backend requires `--save_strategy no`.')
        if self.optimizer is not None or self.use_galore:
            raise ValueError('The initial FSDPTurbo backend supports its native AdamW optimizer only.')
        optim = getattr(self.training_args.optim, 'value', self.training_args.optim)
        if not str(optim).startswith('adamw'):
            raise ValueError(f'FSDPTurbo supports AdamW only, got optim={optim!r}.')
        if self.loss_type is not None or self.enable_dft_loss or self.enable_channel_loss:
            raise ValueError('Swift custom loss plugins are not supported by the initial FSDPTurbo backend.')
        if self.label_smoothing_factor:
            raise ValueError('FSDPTurbo label smoothing has not been validated yet.')
        if self.router_aux_loss_coef:
            raise ValueError('FSDPTurbo router auxiliary loss integration has not been validated yet.')
        if self.use_flash_ckpt:
            raise ValueError('DLRover flash checkpoints cannot be combined with FSDPTurbo checkpoints.')
        if self.quant_method is not None:
            raise ValueError('The initial FSDPTurbo backend does not support Swift model quantization.')
        if self.experts_impl is not None:
            raise ValueError('FSDPTurbo owns the expert implementation; leave `--experts_impl` unset.')
        if self.group_by_length:
            raise ValueError('FSDPTurbo group-by-length sampling has not been implemented yet.')
        if self.vit_gradient_checkpointing != self.gradient_checkpointing:
            raise ValueError('FSDPTurbo currently uses one recompute setting for both language and vision blocks.')
        if self.attn_impl is None:
            self.attn_impl = 'eager'
        elif self.attn_impl != 'eager':
            raise ValueError('The initial Qwen3.5 FSDPTurbo model plan requires `--attn_impl eager`.')
