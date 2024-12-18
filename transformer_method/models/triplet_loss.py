import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class TripletLoss(nn.Module):
    """
    Triplet loss with online mining of hard triplets
    """

    def __init__(self, margin: float = 2.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss with online mining

        Args:
            embeddings: Normalized embeddings of shape [batch_size, embedding_dim]
            labels: Labels of shape [batch_size]

        Returns:
            loss: Triplet loss value
        """
        pairwise_dist = torch.cdist(embeddings, embeddings)

        # Visualize pairwise distances (optional)
        # self.plot_pairwise_distances(pairwise_dist)

        # Dynamically adjust margin based on pairwise distance statistics
        dynamic_margin = self.compute_dynamic_margin(pairwise_dist)
        self.margin = max(self.margin, dynamic_margin)

        # For each anchor, find the hardest positive and negative
        mask_anchor_positive = self._get_anchor_positive_mask(labels)
        mask_anchor_negative = self._get_anchor_negative_mask(labels)

        # Get hardest positive and negative distances
        hardest_positive_dist = (pairwise_dist * mask_anchor_positive.float()).max(dim=1)[0]
        hardest_negative_dist = (pairwise_dist * mask_anchor_negative.float()).min(dim=1)[0]

        # Check if the triplet is hard


        hard_triplet_mask = self.check_hard_triplet(hardest_positive_dist, hardest_negative_dist, self.margin)
        # Compute triplet loss
        if not hard_triplet_mask.any():
            print("Warning: No hard triplets found!")
            return torch.tensor(0.0).to(embeddings.device)
        loss = torch.clamp(hardest_positive_dist - hardest_negative_dist + self.margin, min=0.0)

        # Only consider non-zero loss terms
        num_valid_triplets = (loss > 1e-16).float().sum()
        if num_valid_triplets > 0:
            loss = loss.mean()
        else:
            loss = torch.tensor(0.0).to(loss.device)

        return loss

    @staticmethod
    def _get_anchor_positive_mask(labels: torch.Tensor) -> torch.Tensor:
        """Get boolean mask for valid anchor-positive pairs"""
        return (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~torch.eye(
            labels.size(0), device=labels.device, dtype=torch.bool
        )

    @staticmethod
    def _get_anchor_negative_mask(labels: torch.Tensor) -> torch.Tensor:
        """Get boolean mask for valid anchor-negative pairs"""
        return labels.unsqueeze(0) != labels.unsqueeze(1)

    def plot_pairwise_distances(self, pairwise_dist):
        """Function to plot the pairwise distances as a heatmap for visualization."""
        plt.figure(figsize=(8, 6))
        plt.imshow(pairwise_dist.cpu().detach().numpy(), cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title("Pairwise Distance Heatmap")
        plt.show()

    def compute_dynamic_margin(self, pairwise_dist):
        """Dynamically compute a margin based on the pairwise distance statistics."""
        mean_dist = pairwise_dist.mean().item()
        std_dist = pairwise_dist.std().item()
        margin = mean_dist + std_dist  # This will ensure some gap between the positive and negative distances
        return margin

    def check_hard_triplet(self, hardest_positive_dist, hardest_negative_dist, margin=1.0):
        """Check if the hardest negative and positive are sufficiently distinct."""
        # Compute a mask for hard triplets: where the negative distance is larger than the positive distance by at least margin
        hard_triplet_mask = (hardest_negative_dist - hardest_positive_dist) >= margin

        # Check if there are any valid hard triplets in the batch
        return hard_triplet_mask