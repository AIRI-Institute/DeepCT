import os

import torch
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

    weight: str or array(float) or None
        either tensor of weights or path to file where i-th line
        contains value of i-th element of weights vector
        if weight is None no error will be raised but one would need
        to update it's value before computing loss
    Examples::

        >>> loss = nn.MSELoss(weight=[1,2,3,4,5])
        >>> input = torch.randn(3, 5, requires_grad=True)
        >>> target = torch.randn(3, 5)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, weight=None):
        super(WeightedMSELoss, self).__init__(
            size_average=None, reduce=None, reduction="elementwise_mean"
        )
        # construct weights tensor
        if weight is None:  # In this case weights will be provided later
            self.register_buffer("weight", weight)
        else:
            try:
                weight = torch.tensor(weight)
            except TypeError:
                if not os.path.isfile(weight):
                    raise ValueError(
                        "Provided pos_weight could not be neither "
                        + "converted to tensor nor opened as file:\n"
                        + str(weight)
                    )
                with open(weight) as f:
                    weight = list(map(float, f.readlines()))
                    weight = torch.tensor(weight)
            self.register_buffer("weight", weight)

        self.F = lambda a, b, c: ((a - b) ** 2) * c

    def forward(self, input, target):
        return torch.sum(self.F(input, target, self.weight)) / torch.sum(self.weight)


class WeightedMSELossWithMPI(WeightedMSELoss):
    r"""This loss combines two MSE measurements:
    1.) the MSE of predicted mean (accross cell types) feature value and
    2.) the MSE of deviation of cell-type specific feature value from the mean feature value
    The alpha parameter specifies ballance between 1.) and 2.)
    i.e. let N is number of cell types, K is number of features,
    A[n,k] (n=1...N, k=1...K) is target feature value, and
    mean_predicted[k] is predicted mean of feature k across cell types,
    deviation[n,k] is predicted deviation from mean_predicted[k] of feature k in cell type n
    then loss is
    alpha * mean ( (nanmean(A[:,k]) - mean_predicted[k] )**2 ) +
    (1-alpha) * mean (A[k,n]/nanmean(A[:,k] - deviation_predicted[k,n])

    .. math::
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input

    weight: str or array(float) or None
        either tensor of weights or path to file where i-th line
        contains value of i-th element of weights vector
        if weight is None no error will be raised but one would need
        to update it's value before computing loss
    alpha: float
        importance of mean feature signal over cell type-specific deviation. When set
        to 1 meausures only mean feature signal loss, when set to 0 measures only
        cell type-specific information.
    """

    def __init__(self, alpha, weight=None):
        super(WeightedMSELossWithMPI, self).__init__(weight=weight)
        self.alpha = alpha
        self.F = lambda a, b: ((a - b) ** 2)

    def forward(self, input, target):
        # input and target are expected to be shaped like
        # input:  (torch.Size([batch_size,1,n_features]),
        #            torch.Size([batch_size, n_cell_types, n_features])
        #         )
        # target: torch.Size([batch_size, n_cell_types, n_features])

        _mean_feature_value = torch.sum(
            target * self.weight, 1, keepdim=True
        ) / torch.sum(self.weight, 1, keepdim=True)
        # _mean_feature_value expected to have shape ([batch_size, 1, n_features])
        # then it could be broadcasted to the shape of input repeating values across dim 1
        _deviation = target - _mean_feature_value
        _predicted_mean, _predicted_deviation = input[:, -1:, :], input[:, :-1, :]
        _MSE_mean = torch.mean(self.F(_mean_feature_value, _predicted_mean))
        _MSE_dev = torch.sum(
            self.F(_deviation, _predicted_deviation) * self.weight
        ) / torch.sum(self.weight)
        return self.alpha * _MSE_mean + (1 - self.alpha) * _MSE_dev
