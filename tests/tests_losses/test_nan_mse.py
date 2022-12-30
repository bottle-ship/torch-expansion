__author__ = ["Byungseon Choi"]

import unittest

import numpy as np
import torch
from numpy.testing import assert_array_equal
from torch.nn import MSELoss

from torch_expansion.losses import NanMSELoss


class TestNanMSELoss(unittest.TestCase):

    def test_reduction(self):
        size = (3, 5)
        for reduction in ("none", "mean", "sum"):
            input = torch.randn(size).to(dtype=torch.float32)  # noqa
            target = torch.randn(size).to(dtype=torch.float32)
            target_nan = torch.randint(low=0, high=2, size=size).to(dtype=torch.float32)
            target_nan = torch.where(target_nan == 1.0, torch.nan, target_nan)
            target_nan += target

            criterion = MSELoss(reduction=reduction)
            criterion_nan = NanMSELoss(reduction=reduction)

            mse = criterion(input, target).detach().cpu().numpy()
            mse_nan = criterion_nan(input, target_nan).detach().cpu().numpy()

            if reduction == "none":
                mse_nan = np.where(np.isnan(mse_nan), mse, mse_nan)
                assert_array_equal(mse, mse_nan)
            else:
                self.assertTupleEqual(mse.shape, mse_nan.shape)


if __name__ == "__main__":
    unittest.main()
