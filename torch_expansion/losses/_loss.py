__author__ = ["Byungseon Choi"]

import torch.nn as nn

from sklearn.utils._param_validation import (  # noqa
    StrOptions,
    validate_parameter_constraints
)

__all__ = ["Loss"]


class Loss(nn.Module):
    reduction: str
    _parameter_constraints: dict = {
        "reduction": [StrOptions({"none", "mean", "sum"})],
    }

    def __init__(self, reduction: str = "mean"):
        super(Loss, self).__init__()
        self.reduction = reduction

        self._validate_parameters()

    ####################################################################################################################
    # Protected Methods
    ####################################################################################################################

    def _validate_parameters(self):
        params = dict()
        for key in self._parameter_constraints.keys():
            params[key] = getattr(self, key)
        validate_parameter_constraints(
            parameter_constraints=self._parameter_constraints,
            params=params,
            caller_name=self.__class__.__name__
        )
