import importlib.util
import torch
import unittest
from torch import nn
from types import SimpleNamespace
from unittest import mock


@unittest.skipUnless(importlib.util.find_spec('fsdp_turbo'), 'requires the optional FSDPTurbo package')
class TestFSDPTurboDtypePolicy(unittest.TestCase):

    def test_minimal_backend_rejects_mixed_dtypes_without_new_turbo_config_fields(self):
        from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig

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
            torch_dtype=torch.bfloat16,
            offload_params=False,
            resume_from_checkpoint=None,
            router_aux_loss_coef=0,
        )
        trainer.template = SimpleNamespace(model=None)
        trainer.train_dataset = mock.sentinel.dataset
        trainer.processor = mock.sentinel.processor
        trainer.train_args = SimpleNamespace(device='cpu')
        trainer.spec = SimpleNamespace(
            native_model_factory=None,
            frozen_modules=(),
            validate_model=mock.Mock(),
            matching_modules=mock.Mock(return_value={}),
        )
        trainer.config = FSDPTurboConfig()
        original_config = trainer.config

        dtypes = {name: param.dtype for name, param in model.named_parameters()}
        with mock.patch.object(TunerMixin, 'prepare_model', return_value=model), \
                mock.patch('swift.fsdpturbo.loading.load_rank0_model', return_value=(model, dtypes)), \
                self.assertRaisesRegex(ValueError, 'uniform parameter dtype'):
            trainer.build_model()
        self.assertIs(trainer.config, original_config)
        self.assertFalse(hasattr(trainer.config.distributed.fsdp_plan, 'ignored_params'))


if __name__ == '__main__':
    unittest.main()
