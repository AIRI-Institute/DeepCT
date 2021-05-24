import torch
import torch.nn as nn
import torch.nn.functional as F


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
