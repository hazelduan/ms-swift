import pytest
import torch
from torch import nn
from types import SimpleNamespace
from unittest.mock import patch

from swift.arguments import SftArguments
from swift.fsdpturbo.trainer import FSDPTurboTrainer
from tests.fsdpturbo import test_arguments as argument_tests


def test_bias_and_norm_keep_their_values_under_weight_decay():
    model = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))
    model.register_parameter('frozen', nn.Parameter(torch.ones(2), requires_grad=False))
    trainer = object.__new__(FSDPTurboTrainer)
    trainer.model = model
    trainer.spec = SimpleNamespace(model_type='qwen3_5_moe')
    trainer.config = SimpleNamespace(
        optimizer=SimpleNamespace(
            optimizer_type='AdamW', lr=0.1, weight_decay=0.2, adam_beta1=0.9, adam_beta2=0.95, adam_epsilon=1e-8))
    optimizer = trainer.build_optimizer()
    original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    torch.testing.assert_close(model[0].weight, original['0.weight'] * 0.98)
    for name in ('0.bias', '1.weight', '1.bias', 'frozen'):
        torch.testing.assert_close(dict(model.named_parameters())[name], original[name], rtol=0, atol=0)
    assert all(parameter is not model.frozen for group in optimizer.param_groups for parameter in group['params'])


@pytest.mark.parametrize('optim', ['adamw_torch_fused', 'adamw_bnb_8bit'])
def test_optimizer_variants_are_rejected_instead_of_silently_replaced(optim):
    args = argument_tests.TestFSDPTurboArguments._bare_sft_args(
        training_args=SimpleNamespace(save_strategy='no', optim=optim), vit_gradient_checkpointing=True)
    with patch.object(SftArguments, '__post_init__'), pytest.raises(ValueError, match='AdamW'):
        args.__post_init__()
