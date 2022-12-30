__author__ = ["Byungseon Choi"]

import torch

from ._loss import Loss

__all__ = ["NanMSELoss"]


class NanMSELoss(Loss):
    r"""Nan mean squared error loss.

    Creates a criterion that measures the mean squared error (squared L2 norm) between
    each element in the input :math:`x` and target :math:`y`, treating Not a Numbers (NaNs) as zero.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = \left( x_n - y_n \right)^2,

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), &  \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  &  \text{if reduction} = \text{`sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The mean operation still operates over all the elements, and divides by :math:`n`.

    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.

    Parameters
    ----------
    reduction : {'none', 'mean', 'sum'}, default='mean'
        Specifies the reduction to apply to the output.

        - ``'none'``: no reduction will be applied.
        - ``'mean'``: the sum of the output will be divided by the number of elements in the output.
        - ``'sum'``: the output will be summed.

    """

    def __init__(self, reduction: str = "mean"):
        super(NanMSELoss, self).__init__(reduction=reduction)

    ####################################################################################################################
    # Public Methods
    ####################################################################################################################

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:  # noqa
        r"""Compute mean squared error, treating Not a Numbers (NaNs) as zero.

        Parameters
        ----------
        input : torch.Tensor of shape(*)
            The input :math:`x`.

        target : torch.Tensor of shape(*)
            The target :math:`y`.

        Returns
        -------
        loss : torch.Tensor
            A non-negative floating point value (the best value is 0.0), NaNs or
            an array of floating point values and NaNs.

        """
        residual = target - input
        squared_residual = torch.square(residual)

        if self.reduction == "sum":
            return torch.nansum(squared_residual)
        elif self.reduction == "mean":
            return torch.nanmean(squared_residual)
        else:
            return squared_residual
