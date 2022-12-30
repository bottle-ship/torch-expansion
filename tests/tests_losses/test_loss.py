__author__ = ["Byungseon Choi"]

import unittest

from sklearn.utils._param_validation import InvalidParameterError  # noqa

from torch_expansion.losses import Loss


class TestLoss(unittest.TestCase):

    def test_validate_parameters(self):
        with self.assertRaises(InvalidParameterError):
            Loss(reduction="test")


if __name__ == "__main__":
    unittest.main()
