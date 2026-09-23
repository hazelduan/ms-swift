# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch


@unittest.skipUnless(importlib.util.find_spec('fsdp_turbo'), 'requires the optional FSDPTurbo package')
class TestLossGroupInitialization(unittest.TestCase):

    def make_trainer(self, size):
        from swift.fsdpturbo.trainer import FSDPTurboTrainer

        trainer = object.__new__(FSDPTurboTrainer)
        trainer.train_args = SimpleNamespace(device=torch.device('cpu'))
        trainer.model = SimpleNamespace(
            parallel_state=SimpleNamespace(get_data_group_size=lambda: size, get_data_group=lambda: 'loss-group'))
        return trainer

    def test_collective_initializes_the_loss_group_before_training(self):
        trainer = self.make_trainer(8)
        with patch('torch.distributed.all_reduce') as reduce:
            trainer._initialize_loss_group()
        reduce.assert_called_once()
        self.assertEqual(reduce.call_args.kwargs['group'], 'loss-group')
        self.assertEqual(reduce.call_args.args[0].item(), 0)

    def test_single_data_rank_needs_no_collective(self):
        trainer = self.make_trainer(1)
        with patch('torch.distributed.all_reduce') as reduce:
            trainer._initialize_loss_group()
        reduce.assert_not_called()


if __name__ == '__main__':
    unittest.main()
