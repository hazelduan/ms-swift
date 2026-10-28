# Copyright (c) ModelScope Contributors. All rights reserved.
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Union

import torch.nn as nn


@dataclass(frozen=True)
class FSDPTurboModelSpec:
    model_type: str
    fsdp_modules: Tuple[str, ...]
    fsdp_hook_modules: Tuple[str, ...]
    tp_colwise_modules: Tuple[str, ...]
    tp_rowwise_modules: Tuple[str, ...]
    ep_modules: Tuple[str, ...]
    efsdp_modules: Tuple[str, ...]
    recompute_modules: Tuple[str, ...]

    @staticmethod
    def matching_modules(model: nn.Module, patterns: Iterable[str]) -> Dict[str, Tuple[str, ...]]:
        from fsdp_turbo.utils.str_match import module_name_match

        names = tuple(name for name, _ in model.named_modules())
        return {pattern: tuple(name for name in names if module_name_match(pattern, name)) for pattern in patterns}

    def validate_model(self,
                       model: nn.Module,
                       *,
                       require_tp: bool = False,
                       require_ep: bool = False,
                       require_recompute: bool = False) -> Dict[str, Dict[str, Tuple[str, ...]]]:
        groups = {
            'fsdp_modules': self.fsdp_modules,
            'fsdp_hook_modules': self.fsdp_hook_modules,
        }
        if require_tp:
            groups.update({
                'tp_colwise_modules': self.tp_colwise_modules,
                'tp_rowwise_modules': self.tp_rowwise_modules,
            })
        if require_ep:
            groups.update({
                'ep_modules': self.ep_modules,
                'efsdp_modules': self.efsdp_modules,
            })
        if require_recompute:
            groups['recompute_modules'] = self.recompute_modules

        matches = {name: self.matching_modules(model, patterns) for name, patterns in groups.items()}
        missing = [f'{group}:{pattern}' for group, result in matches.items() for pattern, names in result.items()
                   if not names]
        if missing:
            raise RuntimeError(f'FSDPTurbo model spec did not match declared modules: {missing}.')
        return matches


_MODEL_SPECS = {
    'qwen3_5_moe':
    FSDPTurboModelSpec(
        model_type='qwen3_5_moe',
        fsdp_modules=(
            'model.visual.blocks.{*}',
            'model.visual.patch_embed',
            'model.visual.pos_embed',
            'model.language_model.layers.{*}',
            'model.language_model.embed_tokens',
            'lm_head',
        ),
        fsdp_hook_modules=('model.language_model.layers.{*}', ),
        tp_colwise_modules=('*.q_proj', '*.k_proj', '*.v_proj'),
        tp_rowwise_modules=('*.o_proj', ),
        ep_modules=('model.language_model.layers.{*}.mlp.experts', ),
        efsdp_modules=('model.language_model.layers.{*}.mlp.experts', ),
        recompute_modules=('model.language_model.layers.{*}', 'model.visual.blocks.{*}'),
    ),
}


def get_model_spec(model_or_type: Union[nn.Module, str]) -> FSDPTurboModelSpec:
    if isinstance(model_or_type, str):
        model_type = model_or_type
    else:
        model_type = getattr(getattr(model_or_type, 'config', None), 'model_type', None)
    try:
        return _MODEL_SPECS[model_type]
    except KeyError as error:
        raise ValueError(f'No FSDPTurbo model spec is registered for model_type={model_type!r}.') from error
