import importlib.util
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn


@unittest.skipUnless(importlib.util.find_spec('fsdp_turbo'), 'requires the optional FSDPTurbo package')
class TestFSDPTurboDtypePolicy(unittest.TestCase):

    def test_build_model_preserves_loader_selected_parameter_dtypes(self):
        from swift.fsdpturbo.trainer import FSDPTurboTrainer
        from swift.pipelines.train.tuner import TunerMixin

        class ToyModel(nn.Module):

            def __init__(self):
                super().__init__()
                self.regular = nn.Parameter(torch.ones(2, dtype=torch.bfloat16))
                self.sensitive = nn.Parameter(torch.ones(2, dtype=torch.float32))
                self.register_buffer('indices', torch.arange(2, dtype=torch.int64))
                self.config = SimpleNamespace(use_cache=True)

        model = ToyModel()
        trainer = object.__new__(FSDPTurboTrainer)
        trainer.args = SimpleNamespace(
            get_model_processor=mock.Mock(return_value=(model, None)),
            tp_size=1,
            ep_size=1,
            gradient_checkpointing=False,
        )
        trainer.template = SimpleNamespace(model=None)
        trainer.train_dataset = mock.sentinel.dataset
        trainer.processor = mock.sentinel.processor
        trainer.train_args = SimpleNamespace(device='cpu')
        trainer.spec = SimpleNamespace(
            replicated_params=('sensitive', ),
            validate_model=mock.Mock(),
            matching_parameters=mock.Mock(return_value={'sensitive': ('sensitive', )}),
        )
        trainer.config = SimpleNamespace(model=SimpleNamespace(torch_dtype=torch.bfloat16))

        with mock.patch.object(TunerMixin, 'prepare_model', return_value=model), \
                mock.patch('fsdp_turbo.fsdp_turbo.FSDPTurbo', side_effect=lambda _config, wrapped: wrapped):
            result = trainer.build_model()

        self.assertIs(result, model)
        self.assertEqual(model.regular.dtype, torch.bfloat16)
        self.assertEqual(model.sensitive.dtype, torch.float32)
        self.assertEqual(model.indices.dtype, torch.int64)
        self.assertFalse(model.config.use_cache)

    def test_replicated_fp32_gradients_are_world_averaged_before_step(self):
        from swift.fsdpturbo.trainer import FSDPTurboTrainer

        trainer = object.__new__(FSDPTurboTrainer)
        param = nn.Parameter(torch.ones(2, dtype=torch.float32))
        param.grad = torch.full_like(param, 3.0)
        trainer._replicated_parameters = [param]

        def all_reduce(tensor, **_kwargs):
            tensor.mul_(2)

        with mock.patch('swift.fsdpturbo.trainer.dist.is_initialized', return_value=True), \
                mock.patch('swift.fsdpturbo.trainer.dist.get_world_size', return_value=2), \
                mock.patch('swift.fsdpturbo.trainer.dist.all_reduce', side_effect=all_reduce) as reduce:
            trainer._sync_replicated_gradients()

        reduce.assert_called_once()
        torch.testing.assert_close(param.grad, torch.full_like(param, 3.0))


if __name__ == '__main__':
    unittest.main()
