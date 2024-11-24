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


def contrastive_loss(embeddings, labels, temperature=0.07, device='cuda'):
    """
    Compute contrastive loss to separate classes better

    Args:
        embeddings: tensor of shape [batch_size, embedding_dim]
        labels: tensor of shape [batch_size]
        temperature: scaling factor (smaller value = harder contrast)
        device: device to put tensors on
    """
    # Normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Compute similarity matrix
    similarity_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    # Create labels matrix
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    # For numerical stability
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()

    # Compute positive pairs
    pos_mask = mask
    pos_logits = logits * pos_mask

    # Compute negative pairs
    neg_mask = 1 - mask
    neg_logits = logits * neg_mask

    # Compute log_prob
    exp_logits = torch.exp(logits) * (1 - torch.eye(embeddings.shape[0]).to(device))
    log_prob = pos_logits - torch.log(exp_logits.sum(1, keepdim=True))

    # Compute mean of log-likelihood over positive pairs
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

    # Loss
    loss = -mean_log_prob_pos.mean()

    return loss


# First create base loss class
class BaseLoss:
    def __init__(self, criterion):
        self.criterion = criterion



# Contrastive + reconstruction loss
class ContrastiveAutoencoderLoss(BaseLoss):
    def __init__(self, criterion, lambda_contrast=5.0, temperature=0.1, base_temperature=0.07):
        super().__init__(criterion)
        self.lambda_contrast = lambda_contrast
        self.temperature = temperature
        self.base_temperature = base_temperature

    def contrastive_loss(self, encoder_output, labels, device):
        """
        Supervised normalized temperature-scaled cross entropy loss.
        Using encoder outputs directly.
        """
        outputs, hidden = encoder_output  # Unpack the encoder output

        # Use the outputs sequence
        embeddings = outputs

        # Get a single vector per sequence using mean pooling
        if len(embeddings.shape) == 3:  # [batch_size, seq_len, features]
            # embeddings = torch.mean(embeddings, dim=1)  # [batch_size, features]
            embeddings = torch.max(embeddings, dim=1)[0]  # Use max pooling instead of mean

        batch_size = embeddings.shape[0]

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        anchor_dot_contrast = torch.matmul(embeddings, embeddings.T) / self.temperature

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create mask for positive pairs
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Mask out self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask

        # Compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute mean of log-likelihood over positive pairs
        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-8)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def calculate_loss(self, reconstructed, inputs, encoder_output, labels):
        recon_loss = self.criterion(reconstructed, inputs)
        contra_loss = self.contrastive_loss(encoder_output, labels, inputs.device)
        total_loss = recon_loss + self.lambda_contrast * contra_loss
        return total_loss, recon_loss, contra_loss