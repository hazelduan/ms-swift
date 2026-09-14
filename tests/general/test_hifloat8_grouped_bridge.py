import sys
import unittest
from types import ModuleType
from unittest.mock import patch

import torch

from swift.model.npu_patch.hifloat8 import _grouped_mm


class TestHiFloat8GroupedBridge(unittest.TestCase):

    def test_bridge_is_lazy_and_forwards_arguments_unchanged(self):
        input_value = torch.randn(3, 4)
        weight = torch.randn(2, 4, 5)
        group_list = torch.tensor([1, 3], dtype=torch.int32)
        expected = torch.randn(3, 5)
        observed = []

        def grouped_mm(input_arg, weight_arg, group_list_arg):
            observed.append((input_arg, weight_arg, group_list_arg))
            return expected

        torch_npu = ModuleType('torch_npu')
        utils = ModuleType('torch_npu.utils')
        hifloat8_train = ModuleType('torch_npu.utils.hifloat8_train')
        hifloat8_train.hifloat8_grouped_mm = grouped_mm
        torch_npu.utils = utils
        utils.hifloat8_train = hifloat8_train
        modules = {
            'torch_npu': torch_npu,
            'torch_npu.utils': utils,
            'torch_npu.utils.hifloat8_train': hifloat8_train,
        }

        with patch.dict(sys.modules, modules):
            actual = _grouped_mm(input_value, weight, group_list)

        self.assertIs(actual, expected)
        self.assertEqual(len(observed), 1)
        self.assertIs(observed[0][0], input_value)
        self.assertIs(observed[0][1], weight)
        self.assertIs(observed[0][2], group_list)


if __name__ == '__main__':
    unittest.main()
