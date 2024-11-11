"""
Custom loss functions for model training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling imbalanced datasets.
    """

    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, weights):
        """
        Calculate weighted cross entropy loss.

        Args:
            inputs: Model predictions
            targets: True labels
            weights: Sample weights

        Returns:
            Weighted loss value
        """
        log_probs = F.log_softmax(inputs, dim=1)
        weighted_loss = -weights * log_probs[range(len(targets)), targets]
        return weighted_loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling imbalanced datasets.
    """

    def __init__(self, gamma=2., alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Calculate focal loss.

        Args:
            inputs: Model predictions
            targets: True labels

        Returns:
            Focal loss value
        """
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()
