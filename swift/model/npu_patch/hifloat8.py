# Copyright (c) ModelScope Contributors. All rights reserved.
"""Narrow, lazy bridges to torch-npu HiFloat8 training operators."""

from __future__ import annotations

import torch


def _grouped_mm(input: torch.Tensor, weight: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
    """Call torch-npu's grouped HiFloat8 operator without owning its semantics."""
    from torch_npu.utils.hifloat8_train import hifloat8_grouped_mm
    return hifloat8_grouped_mm(input, weight, group_list)
