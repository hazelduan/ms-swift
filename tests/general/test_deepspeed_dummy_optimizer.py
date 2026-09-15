# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
from accelerate.utils import DummyOptim
from copy import copy
from types import SimpleNamespace

from swift.optimizers.base import OptimizerCallback
from swift.trainers.mixin import SwiftMixin


def test_deepspeed_owned_dummy_optimizer_survives_trainer_creation():
    # DeepSpeed creates the real optimizer later; DummyOptim has no param_groups.
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = DummyOptim([parameter], lr=1e-5)
    calls = []
    trainer = SimpleNamespace(
        optimizer_callback=SimpleNamespace(create_optimizer=lambda model: optimizer),
        _disable_foreach_for_deepspeed=lambda: calls.append(True))
    assert SwiftMixin.create_optimizer(trainer) is optimizer
    assert trainer._optimizer_ori is optimizer
    assert not calls


def test_torch_optimizer_empty_groups_are_still_removed():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([{'params': [parameter]}, {'params': []}], lr=1e-5)
    calls = []
    trainer = SimpleNamespace(
        optimizer_callback=SimpleNamespace(create_optimizer=lambda model: optimizer),
        _disable_foreach_for_deepspeed=lambda: calls.append(True))
    assert SwiftMixin.create_optimizer(trainer) is optimizer
    assert len(optimizer.param_groups) == 1
    assert calls == [True]


def test_deferred_scheduler_uses_copied_trainer_not_original_dummy():
    from accelerate.utils import DummyScheduler

    parameter = torch.nn.Parameter(torch.ones(2))
    dummy = DummyScheduler(DummyOptim([parameter]))
    args = SimpleNamespace(lr_scheduler_type='cosine', lr_scheduler_kwargs={}, get_warmup_steps=lambda steps: 0)
    trainer = SimpleNamespace(args=args, lr_scheduler=dummy)
    trainer.optimizer_callback = OptimizerCallback(args, trainer)
    trainer_copy = copy(trainer)
    trainer_copy.lr_scheduler = None
    optimizer = torch.optim.AdamW([parameter], lr=1e-5)
    scheduler = SwiftMixin.create_scheduler(trainer_copy, 20, optimizer)
    assert scheduler.optimizer is optimizer
    assert trainer.lr_scheduler is dummy
    assert trainer.optimizer_callback.trainer is trainer
    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr()[0] < 1e-5
