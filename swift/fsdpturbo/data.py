# Copyright (c) ModelScope Contributors. All rights reserved.
import torch


def _normalize_native_text_batch(batch):
    """Validate Swift text metadata on CPU before passing native model inputs."""
    for name in ('images', 'pixel_values', 'pixel_values_videos', 'image_grid_thw', 'video_grid_thw'):
        if batch.get(name) is not None:
            raise ValueError(f'Native V4.1 Swift training is text-only; {name} is not supported.')
    input_ids = batch['input_ids']
    if input_ids.ndim != 2 or input_ids.device.type != 'cpu':
        raise ValueError('Native text collation requires CPU input_ids with shape [batch, sequence].')
    for name in ('attention_mask', 'position_ids', 'image_token_types', 'token_types'):
        value = batch.get(name)
        if value is None:
            continue
        if value.device.type != 'cpu':
            raise ValueError(f'{name} must be validated on CPU before device transfer.')
        if value.shape != input_ids.shape:
            raise ValueError(f'{name} shape must match input_ids.')
    for name in ('image_token_types', 'token_types'):
        value = batch.get(name)
        if value is not None and not torch.all(value.eq(-1)):
            raise ValueError(f'Native V4.1 Swift text batches require {name} to contain only TEXT=-1.')
    mask = batch.get('attention_mask')
    if mask is None:
        active = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        if not torch.all(mask.eq(0) | mask.eq(1)):
            raise ValueError('Native text attention_mask must be binary.')
        active = mask.bool()
        if torch.any(~active[:, :-1] & active[:, 1:]):
            raise ValueError('Native V4.1 Swift text batches must be right-padded.')
    positions = batch.get('position_ids')
    if positions is not None:
        expected = torch.arange(input_ids.shape[1], device=input_ids.device).expand_as(input_ids)
        if not torch.equal(positions[active], expected[active]):
            raise ValueError('Native V4.1 Swift text batches require contiguous active position_ids.')
        batch.pop('position_ids')
    image_types = batch.pop('image_token_types', None)
    if image_types is not None:
        batch['token_types'] = image_types
    return batch


def _native_text_collator(data_collator, rows):
    return _normalize_native_text_batch(data_collator(rows, padding_to=None))
