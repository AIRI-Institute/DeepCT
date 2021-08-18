import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss


class FocalLoss(nn.Module):
    """
    Focal loss (https://arxiv.org/abs/1708.02002) implementation.
    """

    def __init__(self, pos_weight=1, gamma=2, logits=True, reduction=True, weight=None):
        super(FocalLoss, self).__init__()
        self.pos_weight = pos_weight
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs,
                targets,
                reduction="none",
                weight=self.weight,
                pos_weight=self.pos_weight,
            )
            sig_x = torch.sigmoid(inputs)
        else:
            BCE_loss = F.binary_cross_entropy(
                inputs,
                targets,
                reduction="none",
                weight=self.weight,
                pos_weight=self.pos_weight,
            )
            sig_x = inputs

        # This won't work when pos_weight is not None:
        # pt = torch.exp(-BCE_loss)

        pt = sig_x * targets + (1 - sig_x) * (1 - targets)
        F_loss = (1 - pt) ** self.gamma * BCE_loss

        if self.reduction:
            return torch.mean(F_loss)
        else:
            return F_loss

class WeightedMSELoss(MSELoss):
    r"""Creates a criterion that measures the mean squared error between
    `n` elements in the input `x` and target `y` weighted by weight `c`.
    Note that loss is averaged, i.e. loss = mean( (x[i] - y[i])^2 * c[i] )

    The loss can be described as:

    .. math::
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

    pos_weight: str or array(float)
        either vector of weights or path to file where i-th line
        contains value of i-th element of weights vector
    device: device to store weights tensor (str)
        i.e. 'cpu', 'cuda:0'
    Examples::

        >>> loss = nn.MSELoss(pos_weight=[1,2,3,4,5])
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, pos_weight, device="cpu"):
        super(WeightedMSELoss, self).__init__(
            size_average=None, reduce=None, reduction="elementwise_mean"
        )
        # construct weights tensor
        try:
            self.weights = torch.tensor(pos_weight).to(device)
        except TypeError:
            if not os.path.isfile(pos_weight):
                raise ValueError(
                    "Provided pos_weight could not be neither "
                    + "converted to tensor nor opened as file:\n"
                    + str(pos_weight)
                )
            with open(pos_weight) as f:
                pos_weight = list(map(float, f.readlines()))
                self.weights = torch.tensor(pos_weight).to(device)

        self.F = lambda a, b, c: ((a - b) ** 2) * c

    def forward(self, input, target):
        return torch.mean(self.F(input, target, self.weights))
