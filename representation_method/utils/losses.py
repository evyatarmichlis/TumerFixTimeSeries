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


class ContrastiveAutoencoderLoss(nn.Module):
    def __init__(self, criterion, lambda_contrast=5.0, temperature=0.1,
                 base_temperature=0.07, center_momentum=0.9):
        super().__init__()
        self.criterion = criterion
        self.lambda_contrast = lambda_contrast
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.center_momentum = center_momentum
        self.register_buffer('class_centers', None)
        self.register_buffer('class_weights', None)

    def update_centers(self, embeddings, labels):
        """Update class centers with momentum and handle class imbalance"""
        unique_labels = torch.unique(labels)

        # Initialize centers if not already done
        if self.class_centers is None:
            self.class_centers = torch.zeros((len(unique_labels), embeddings.shape[1])).to(embeddings.device)

            # Initialize class weights based on frequency
            label_counts = torch.bincount(labels)
            total_samples = len(labels)
            self.class_weights = (total_samples - label_counts) / total_samples
            self.class_weights = self.class_weights / self.class_weights.sum()

        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                center = embeddings[mask].mean(0)
                if self.class_centers[label].sum() == 0:  # First update
                    self.class_centers[label] = center
                else:
                    self.class_centers[label] = (
                            self.center_momentum * self.class_centers[label] +
                            (1 - self.center_momentum) * center
                    )

    def clustering_loss(self, embeddings, labels):
        """Compute weighted clustering loss for imbalanced data"""
        if self.class_centers is None:
            return torch.tensor(0.0).to(embeddings.device)

        loss = 0
        for label in torch.unique(labels):
            mask = labels == label
            if mask.sum() > 0:
                cluster_samples = embeddings[mask]
                center = self.class_centers[label]
                weight = self.class_weights[label]

                # Weighted attraction loss - give more weight to minority class
                attraction_loss = weight * torch.norm(
                    cluster_samples - center.unsqueeze(0),
                    dim=1
                ).mean()

                # Push away from other centers
                other_centers = torch.cat([
                    self.class_centers[i].unsqueeze(0)
                    for i in range(len(self.class_centers))
                    if i != label
                ])

                if len(other_centers) > 0:
                    repulsion_loss = weight * torch.exp(-torch.norm(
                        cluster_samples.unsqueeze(1) - other_centers.unsqueeze(0),
                        dim=2
                    )).mean()

                    loss += attraction_loss + repulsion_loss

        return loss / len(torch.unique(labels))

    def contrastive_loss(self, encoder_output, labels, device):
        """Compute weighted contrastive loss for imbalanced data"""
        outputs, _ = encoder_output
        embeddings = torch.max(outputs, dim=1)[0]  # Use max pooling
        batch_size = embeddings.shape[0]

        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Update centers and compute clustering loss
        self.update_centers(embeddings.detach(), labels)
        cluster_loss = self.clustering_loss(embeddings, labels)

        # Compute similarity matrix
        anchor_dot_contrast = torch.matmul(embeddings, embeddings.T) / self.temperature

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Create mask for positive pairs
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # Add class weights to the mask
        label_weights = self.class_weights[labels.squeeze()]
        weight_matrix = torch.matmul(
            label_weights.unsqueeze(1),
            label_weights.unsqueeze(0)
        )
        weighted_mask = mask * weight_matrix

        # Mask out self-contrast
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        weighted_mask = weighted_mask * logits_mask

        # Compute weighted log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # Compute weighted mean of log-likelihood over positive pairs
        mask_sum = weighted_mask.sum(1)
        mean_log_prob_pos = (weighted_mask * log_prob).sum(1) / (mask_sum + 1e-8)

        # Final weighted contrastive loss
        contrast_loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos

        return contrast_loss.mean() + cluster_loss

    def calculate_loss(self, reconstructed, inputs, encoder_output, labels):
        recon_loss = self.criterion(reconstructed, inputs)
        contra_loss = self.contrastive_loss(encoder_output, labels, inputs.device)
        total_loss = recon_loss + self.lambda_contrast * contra_loss
        return total_loss, recon_loss, contra_loss


import torch
import torch.nn as nn
import torch.nn.functional as F


class ImbalancedTripletContrastiveLoss(nn.Module):
    def __init__(self, criterion, lambda_contrast=5.0, lambda_triplet=2.0,
                 temperature=0.1, base_temperature=0.07, margin=1.0,
                 center_momentum=0.9):
        super().__init__()
        self.criterion = criterion
        self.lambda_contrast = lambda_contrast
        self.lambda_triplet = lambda_triplet
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.margin = margin
        self.center_momentum = center_momentum
        self.register_buffer('class_centers', None)
        self.register_buffer('class_weights', None)

    def get_triplets(self, embeddings, labels):
        """
        Generate balanced triplets from a batch
        Returns: anchors, positives, negatives tensors of same size
        """
        device = embeddings.device
        minority_mask = labels == 1
        majority_mask = labels == 0

        # Get minority and majority samples
        minority_indices = torch.where(minority_mask)[0]
        majority_indices = torch.where(majority_mask)[0]

        if len(minority_indices) == 0 or len(majority_indices) == 0:
            return None, None, None

        triplets = []

        # Generate triplets for minority class (higher priority)
        for anchor_idx in minority_indices:
            # Find positive pairs (same class)
            pos_indices = [idx for idx in minority_indices if idx != anchor_idx]
            if not pos_indices:
                continue

            # Select random positive
            pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,)).item()]

            # Select random negative from majority class
            neg_idx = majority_indices[torch.randint(0, len(majority_indices), (1,)).item()]

            triplets.append((anchor_idx, pos_idx, neg_idx))

        # Generate some triplets for majority class (with lower frequency)
        majority_sample_size = min(len(minority_indices), len(majority_indices) // 4)
        sampled_majority = majority_indices[torch.randperm(len(majority_indices))[:majority_sample_size]]

        for anchor_idx in sampled_majority:
            # Find positive pairs (same class)
            pos_indices = [idx for idx in majority_indices if idx != anchor_idx]
            if not pos_indices:
                continue

            # Select random positive
            pos_idx = pos_indices[torch.randint(0, len(pos_indices), (1,)).item()]

            # Select random negative from minority class
            neg_idx = minority_indices[torch.randint(0, len(minority_indices), (1,)).item()]

            triplets.append((anchor_idx, pos_idx, neg_idx))

        if not triplets:
            return None, None, None

        # Convert triplet indices to tensors
        triplets = torch.LongTensor(triplets).to(device)
        anchors = embeddings[triplets[:, 0]]
        positives = embeddings[triplets[:, 1]]
        negatives = embeddings[triplets[:, 2]]

        return anchors, positives, negatives

    def triplet_loss(self, embeddings, labels):
        """
        Compute weighted triplet loss with emphasis on minority class
        """
        anchors, positives, negatives = self.get_triplets(embeddings, labels)

        if anchors is None:
            return torch.tensor(0.0).to(embeddings.device)

        # Get the triplets indexes
        triplets_idx = []
        for i in range(len(anchors)):
            anchor_idx = torch.where(torch.all(embeddings == anchors[i], dim=1))[0][0]
            pos_idx = torch.where(torch.all(embeddings == positives[i], dim=1))[0][0]
            neg_idx = torch.where(torch.all(embeddings == negatives[i], dim=1))[0][0]
            triplets_idx.append([anchor_idx, pos_idx, neg_idx])

        triplets = torch.LongTensor(triplets_idx).to(embeddings.device)

        # Compute distances
        pos_dist = torch.sum((anchors - positives) ** 2, dim=1)
        neg_dist = torch.sum((anchors - negatives) ** 2, dim=1)

        # Basic triplet loss
        losses = torch.relu(pos_dist - neg_dist + self.margin)

        # Weight the losses based on anchor class
        anchor_labels = labels[triplets[:, 0]]
        weights = torch.where(anchor_labels == 1,
                              torch.tensor(2.0).to(embeddings.device),  # Higher weight for minority class
                              torch.tensor(1.0).to(embeddings.device))  # Lower weight for majority class

        weighted_losses = weights * losses

        return weighted_losses.mean()

    def contrastive_loss(self, encoder_output, labels, device):
        outputs, _ = encoder_output
        embeddings = torch.max(outputs, dim=1)[0]  # Use max pooling
        batch_size = embeddings.shape[0]
        embeddings = F.normalize(embeddings, dim=1)

        # Update centers and compute clustering loss
        self.update_centers(embeddings.detach(), labels)
        cluster_loss = self.clustering_loss(embeddings, labels)

        # Compute triplet loss
        trip_loss = self.triplet_loss(embeddings, labels)

        # Standard contrastive loss computation
        anchor_dot_contrast = torch.matmul(embeddings, embeddings.T) / self.temperature
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        labels_reshaped = labels.view(-1, 1)
        mask = torch.eq(labels_reshaped, labels_reshaped.T).float().to(device)
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size).to(device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mask_sum = mask.sum(1)
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask_sum + 1e-8)
        contrast_loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos

        # Combine all losses
        total_contrastive_loss = (contrast_loss.mean() +
                                  cluster_loss +
                                  self.lambda_triplet * trip_loss)

        return total_contrastive_loss

    def calculate_loss(self, reconstructed, inputs, encoder_output, labels):
        recon_loss = self.criterion(reconstructed, inputs)
        contra_loss = self.contrastive_loss(encoder_output, labels, inputs.device)
        total_loss = recon_loss + self.lambda_contrast * contra_loss
        return total_loss, recon_loss, contra_loss

    def update_centers(self, embeddings, labels):
        """Update class centers with momentum"""
        unique_labels = torch.unique(labels)
        if self.class_centers is None:
            self.class_centers = torch.zeros((len(unique_labels), embeddings.shape[1])).to(embeddings.device)

            # Initialize class weights based on frequency
            label_counts = torch.bincount(labels)
            total_samples = len(labels)
            self.class_weights = (total_samples - label_counts) / total_samples
            self.class_weights = self.class_weights / self.class_weights.sum()

        for label in unique_labels:
            mask = labels == label
            if mask.sum() > 0:
                center = embeddings[mask].mean(0)
                if self.class_centers[label].sum() == 0:  # First update
                    self.class_centers[label] = center
                else:
                    self.class_centers[label] = (
                            self.center_momentum * self.class_centers[label] +
                            (1 - self.center_momentum) * center
                    )

    def clustering_loss(self, embeddings, labels):
        """Compute weighted clustering loss for imbalanced data"""
        if self.class_centers is None:
            return torch.tensor(0.0).to(embeddings.device)

        loss = 0
        for label in torch.unique(labels):
            mask = labels == label
            if mask.sum() > 0:
                cluster_samples = embeddings[mask]
                center = self.class_centers[label]
                weight = self.class_weights[label]

                # Weighted attraction loss - give more weight to minority class
                attraction_loss = weight * torch.norm(
                    cluster_samples - center.unsqueeze(0),
                    dim=1
                ).mean()

                # Push away from other centers
                other_centers = torch.cat([
                    self.class_centers[i].unsqueeze(0)
                    for i in range(len(self.class_centers))
                    if i != label
                ])

                if len(other_centers) > 0:
                    repulsion_loss = weight * torch.exp(-torch.norm(
                        cluster_samples.unsqueeze(1) - other_centers.unsqueeze(0),
                        dim=2
                    )).mean()

                    loss += attraction_loss + repulsion_loss

        return loss / len(torch.unique(labels))