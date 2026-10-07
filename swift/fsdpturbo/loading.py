# Copyright (c) ModelScope Contributors. All rights reserved.
"""HF checkpoint conversion on rank 0, followed by FSDP/TP/EP shard loading."""
from contextlib import contextmanager
from unittest.mock import patch

import torch.distributed as dist


@contextmanager
def _checked_checkpoint_read():
    """A partially loaded checkpoint must fail before rank-0 broadcasting."""
    from transformers import PreTrainedModel

    original = PreTrainedModel.from_pretrained.__func__

    def load(cls, *args, **kwargs):
        kwargs['output_loading_info'] = True
        model, info = original(cls, *args, **kwargs)
        failures = {key: info[key] for key in ('missing_keys', 'mismatched_keys', 'error_msgs') if info.get(key)}
        if failures:
            raise ValueError(f'Incomplete pretrained checkpoint: {failures}')
        return model

    with patch.object(PreTrainedModel, 'from_pretrained', classmethod(load)):
        yield


@contextmanager
def _fp8_checkpoint_conversion(dtype=None):
    """Repair anchored single-source expert conversion in Transformers 5.9.

    HF's quantizer only adds dequantization to sources ending in ``.weight``;
    the V4 down-projection source has already become ``.weight$``. Retain HF's
    conversion operations and let the quantizer add the final anchors itself.
    The override exists only during this rank-0 checkpoint read.
    """
    from transformers.core_model_loading import WeightConverter
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    from transformers.quantizers.quantizer_finegrained_fp8 import FineGrainedFP8HfQuantizer

    original = FineGrainedFP8HfQuantizer.update_weight_conversions
    original_dequantize = Fp8Dequantize._dequantize_one

    def dequantize(self, quantized, scales):
        result = original_dequantize(self, quantized, scales)
        return result.to(dtype) if dtype is not None else result

    def update(self, conversions):
        fixed = []
        for conversion in conversions:
            if isinstance(conversion, WeightConverter) and any(
                    source.endswith('.weight$') for source in conversion.source_patterns):
                replacement = WeightConverter(
                    source_patterns=[source.removesuffix('$') for source in conversion.source_patterns],
                    target_patterns=[target.removesuffix('$') for target in conversion._original_target_patterns],
                    operations=conversion.operations,
                )
                replacement.scope_prefix = conversion.scope_prefix
                conversion = replacement
            fixed.append(conversion)
        return original(self, fixed)

    with patch.object(FineGrainedFP8HfQuantizer, 'update_weight_conversions', update), \
            patch.object(Fp8Dequantize, '_dequantize_one', dequantize):
        yield


def load_rank0_model(args):
    """Read weights once globally; propagate loading errors before mesh setup."""
    source_model = None
    payload = [None]
    if dist.get_rank() == 0:
        try:
            if args.resume_from_checkpoint:
                from torch.distributed.checkpoint import FileSystemReader
                metadata = FileSystemReader(args.resume_from_checkpoint).read_metadata()
                prefix = 'app.model.model.'  # app state -> FSDPTurbo wrapper -> HF model
                dtypes = {name.removeprefix(prefix): value.properties.dtype
                          for name, value in metadata.state_dict_metadata.items() if name.startswith(prefix)}
                if not dtypes:
                    raise ValueError('Checkpoint has no FSDPTurbo model tensor metadata.')
                payload[0] = (dtypes, None)
            else:
                source_model = _read_pretrained_model(args)
                payload[0] = ({name: param.dtype for name, param in source_model.named_parameters()}, None)
        except Exception as error:
            payload[0] = (None, f'{type(error).__name__}: {error}')
    dist.broadcast_object_list(payload, src=0)
    dtypes, error = payload[0]
    if error is not None:
        raise RuntimeError(f'FSDPTurbo rank-0 checkpoint loading failed: {error}')
    return source_model, dtypes


def _read_pretrained_model(args):
    kwargs = {'device_map': 'cpu'}
    if args.model_info.quant_method == 'fp8':
        from transformers import FineGrainedFP8Config
        kwargs['quantization_config'] = FineGrainedFP8Config(dequantize=True)
        with _checked_checkpoint_read(), _fp8_checkpoint_conversion(args.torch_dtype):
            source_model, _ = args.get_model_processor(**kwargs)
        for name, param in source_model.named_parameters():
            if not param.is_floating_point() or param.element_size() < 2:
                raise ValueError(f'Checkpoint parameter {name} was not dequantized: {param.dtype}')
    else:
        with _checked_checkpoint_read():
            source_model, _ = args.get_model_processor(**kwargs)
    return source_model


def materialize_resume_model(model, *, device):
    """Allocate only local shards; DCP restores them after optimizer creation."""
    buffers = [(module, name, buffer) for module in model.modules()
               for name, buffer in module.named_buffers(recurse=False)
               if name in module._non_persistent_buffers_set]
    model.to_empty(device=device)
    for module, name, buffer in buffers:
        setattr(module, name, buffer.to(device))


def materialize_sharded_model(model, source_model, *, offload=False):
    """Broadcast each tensor into rank-local shards using PyTorch DCP.

    Meta construction preserves non-persistent buffers (for example rotary
    frequencies). All parameters and persistent buffers come from HF's loaded
    state, including its expert-layout and checkpoint dequantization mappings.
    """
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    state_dict = source_model.state_dict() if source_model is not None else {}
    set_model_state_dict(
        model,
        state_dict,
        options=StateDictOptions(full_state_dict=True, broadcast_from_rank0=True, cpu_offload=offload),
    )
    missing = [name for name, tensor in list(model.named_parameters()) + list(model.named_buffers()) if tensor.is_meta]
    if missing:
        raise RuntimeError(f'FSDPTurbo checkpoint left meta tensors: {missing}')
