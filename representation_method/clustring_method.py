import os.path

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster._hdbscan import hdbscan
from sklearn.decomposition import PCA
import matplotlib
from sklearn.svm import SVC

matplotlib.use('Agg')  # or Qt5Agg, depends on what's installed

import matplotlib.pyplot as plt
from matplotlib import colormaps
cmap = colormaps['tab10']
from representation_method.utils.data_loader import DataConfig, load_eye_tracking_data
from representation_method.utils.data_utils import create_dynamic_time_series, split_train_test_for_time_series
from sklearn.cluster import KMeans

from ruptures.detection import Binseg

def segment_and_cluster(
    X_train: np.ndarray,
    target_location_train: np.ndarray,
    n_segments: int = 5,
    n_clusters: int = 6,
    segmentation_method: str = 'eq',
    clustering_method: str = 'kmeans',
    dbscan_eps: float = 1,
    dbscan_min_samples: int = 5,
    normalized = True
):
    """
    1) Segments each window in X_train into subsegments using dynamic or equal-length segmentation.
    2) Extracts simple features per segment (mean, std, min, max for each feature).
    3) Checks if the segment contains at least one target index (using target_location_train).
    4) Clusters all segment feature-vectors using K-Means (k = n_clusters).
    5) Calculates:
       - Average segment size
       - Average number of targets per segment
       - Average number of targets per cluster

    Parameters:
       - segmentation_method: 'changepoint' for dynamic segmentation or 'equal' for fixed-length segmentation.

    Returns:
       - segment_features_all (N*n_segments, 4*F)
       - cluster_labels (N*n_segments,)
       - kmeans_model (fitted KMeans)
       - segment_indices: list of (window_idx, segment_idx) for each row in segment_features_all
       - has_target_segment: bool array (N*n_segments,) indicating whether the segment contains a target
       - segment_target_counts: list of target counts for each segment
    """
    N, W, F = X_train.shape

    segment_features_all = []
    segment_indices = []
    has_target_segment = []
    segment_sizes = []
    segment_target_counts = []

    for i in range(N):
        window_data = X_train[i]                # shape (W, F)
        target_idxs = target_location_train[i]  # array of indices where target=1 in this window

        if segmentation_method == 'changepoint':
            # Use change-point detection to determine segments dynamically
            signal = np.mean(window_data, axis=1)  # Use the mean of features as the signal
            algo = Binseg(model="l2", jump=1)
            breakpoints = algo.fit(signal).predict(n_bkps=n_segments - 1)
            segments = [(0 if j == 0 else breakpoints[j - 1], breakpoints[j]) for j in range(len(breakpoints))]
        else:
            # Default to equal-length segmentation
            seg_len = W // n_segments
            segments = [(j * seg_len, (j + 1) * seg_len) for j in range(n_segments)]
            if segments[-1][1] != W:
                segments[-1] = (segments[-1][0], W)  # Adjust the last segment to include any remainder

        for s, (start_idx, end_idx) in enumerate(segments):
            # Extract the segment
            segment_data = window_data[start_idx:end_idx]
            segment_sizes.append(end_idx - start_idx)  # Track segment size

            # Simple features: mean, std, min, max
            means = np.mean(segment_data, axis=0)
            stds = np.std(segment_data, axis=0)
            mins = np.min(segment_data, axis=0)
            maxs = np.max(segment_data, axis=0)

            feat_vector = np.concatenate([means, stds, mins, maxs], axis=0)

            segment_features_all.append(feat_vector)
            segment_indices.append((i, s))

            # Check if this segment contains any targets
            segment_tgts = [t for t in target_idxs if (start_idx <= t < end_idx)]
            has_target_segment.append(len(segment_tgts) > 0)
            segment_target_counts.append(len(segment_tgts))  # Track target count

    segment_features_all = np.array(segment_features_all)        # (N*n_segments, 4*F)
    has_target_segment = np.array(has_target_segment, dtype=bool)
    from sklearn.preprocessing import StandardScaler
    if normalized:
        scaler = StandardScaler()
        segment_features_all = scaler.fit_transform(segment_features_all)
    # Cluster the segment-level features
    if clustering_method == 'kmeans':
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = clustering_model.fit_predict(segment_features_all)
    elif clustering_method == 'dbscan':
        clustering_model = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        cluster_labels = clustering_model.fit_predict(segment_features_all)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

    elif clustering_method == 'hdbscan':
         clustering_model = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5)
         cluster_labels = clustering_model.fit_predict(segment_features_all)
         n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
         print(n_clusters)
         return
    else:
        raise ValueError("Unsupported clustering method. Choose 'kmeans' or 'dbscan'.")
    if hasattr(clustering_model, 'cluster_centers_'):  # For KMeans
        cluster_centers = clustering_model.cluster_centers_
    else:  # Approximate cluster centers for DBSCAN
        unique_labels = set(cluster_labels) - {-1}  # Exclude noise points (-1)
        cluster_centers = np.array([
            segment_features_all[cluster_labels == label].mean(axis=0)
            for label in unique_labels
        ])
    # Cluster centers shape: (n_clusters, n_features)
    print("Cluster Centers Shape:", cluster_centers.shape)
    # Calculate variance of each feature across cluster centers


    # Calculate statistics
    avg_segment_size = np.mean(segment_sizes)
    avg_targets_per_segment = np.mean(segment_target_counts)

    print(f"Average segment size: {avg_segment_size:.2f}")
    print(f"Average number of targets per segment: {avg_targets_per_segment:.2f}")

    # Average targets per cluster
    cluster_target_counts = np.zeros(n_clusters)
    cluster_sizes = np.zeros(n_clusters)
    cluster_segment_sizes = [[] for _ in range(n_clusters)]

    for cluster_id in range(n_clusters):
        cluster_mask = (cluster_labels == cluster_id)
        cluster_sizes[cluster_id] = np.sum(cluster_mask)
        cluster_target_counts[cluster_id] = np.sum(
            np.array(segment_target_counts)[cluster_mask]
        )
        cluster_segment_sizes[cluster_id] = np.array(segment_sizes)[cluster_mask]

    avg_targets_per_cluster = cluster_target_counts / cluster_sizes
    avg_segment_size_per_cluster = [
        np.mean(cluster_segment_sizes[cluster_id]) if len(cluster_segment_sizes[cluster_id]) > 0 else 0
        for cluster_id in range(n_clusters)
    ]

    for cluster_id in range(n_clusters):
        print(f"Cluster {cluster_id}: Average targets per segment = {avg_targets_per_cluster[cluster_id]:.2f}")
        print(f"Cluster {cluster_id}: Average segment size = {avg_segment_size_per_cluster[cluster_id]:.2f}")
        print(f"Cluster {cluster_id}: targets size = {cluster_target_counts[cluster_id]:.2f}")

    return (segment_features_all,
            cluster_labels,
            clustering_model,
            segment_indices,
            has_target_segment,
            segment_target_counts)




def plot_clusters_with_targets(
            segment_features: np.ndarray,
            cluster_labels: np.ndarray,
            has_target_segment: np.ndarray,
            n_clusters: int = 6,
            split='train',
            clustering_method='kmeans',
            n_segments=10,
            seed=0
    ):
        """
        Reduce segment features to 2D with PCA and plot:
          - Different color for each cluster
          - Use a different marker or overlay for segments that contain a target
          - Plot cluster centers with standard deviation ellipses
          - Plot global center with two standard deviation lines.
        """

        from matplotlib.patches import Ellipse

        def plot_std_ellipse(ax, center, std_dev, color, label=None):
            """Plot an ellipse representing 1 standard deviation."""
            ellipse = Ellipse(
                xy=center,
                width=2 * std_dev[0],
                height=2 * std_dev[1],
                edgecolor=color,
                facecolor='none',
                linestyle='--',
                alpha=0.6,
                label=label
            )
            ax.add_patch(ellipse)

        pca = PCA(n_components=2, random_state=42)
        segment_features_2d = pca.fit_transform(segment_features)

        plt.figure(figsize=(10, 8))
        ax = plt.gca()

        # Global statistics
        global_center = segment_features_2d.mean(axis=0)
        global_std = segment_features_2d.std(axis=0)

        # Plot global center
        plt.scatter(global_center[0], global_center[1],
                    color='red', edgecolors='black', marker='o', s=300, label="Global Center")

        # Plot two standard deviation lines for the global distribution
        plot_std_ellipse(ax, global_center, global_std, color='red', label="1 Std Dev")
        plot_std_ellipse(ax, global_center, global_std * 2, color='orange', label="2 Std Dev")

        # Color map for clusters
        colors = plt.colormaps.get_cmap('tab10')
        for cluster_id in range(n_clusters):
            # Indices for segments in this cluster
            idx_cluster = (cluster_labels == cluster_id)

            idx_cluster_target = idx_cluster & (has_target_segment == True)
            idx_cluster_nontarget = idx_cluster & (has_target_segment == False)

            # Calculate cluster center and standard deviation
            cluster_points = segment_features_2d[idx_cluster]
            if len(cluster_points) > 0:
                cluster_center = cluster_points.mean(axis=0)
                cluster_std = cluster_points.std(axis=0)

                # Plot cluster center
                plt.scatter(cluster_center[0], cluster_center[1],
                            color=colors(cluster_id),
                            edgecolors='black',
                            marker='X',
                            s=200,
                            label=f'Cluster {cluster_id} Center')

                # Plot standard deviation ellipse for the cluster
                plot_std_ellipse(ax, cluster_center, cluster_std, colors(cluster_id))

            # Plot the segments that do NOT contain a target
            plt.scatter(segment_features_2d[idx_cluster_nontarget, 0],
                        segment_features_2d[idx_cluster_nontarget, 1],
                        color=colors(cluster_id),
                        alpha=0.4,
                        label=f'Cluster {cluster_id} (no target)' if np.sum(idx_cluster_nontarget) > 0 else None)

            # Overlay the segments that DO contain a target
            if np.any(idx_cluster_target):
                plt.scatter(segment_features_2d[idx_cluster_target, 0],
                            segment_features_2d[idx_cluster_target, 1],
                            color=colors(cluster_id),
                            edgecolors='black',
                            marker='*',
                            s=120,
                            label=f'Cluster {cluster_id} (has target)')

        plt.title("Segment Clusters (2D PCA) with Target Highlight, Cluster Centers, and Global Distribution")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(
            f"cluster_results/{split}_{clustering_method}_{n_clusters}_{n_segments}_seed_{seed}_clusters_plot_with_global_center.png")
        plt.show()


def compute_avg_distances_per_cluster(segment_features_test, test_cluster_labels, kmeans_model):
    train_cluster_centers = kmeans_model.cluster_centers_
    n_clusters = len(train_cluster_centers)

    # Compute distances between test points and train cluster centers
    test_distances = cdist(segment_features_test, train_cluster_centers, metric='euclidean')

    # Initialize variables to calculate average distances
    cluster_distances = {cluster_id: [] for cluster_id in range(n_clusters)}

    for i, label in enumerate(test_cluster_labels):
        cluster_distances[label].append(test_distances[i, label])

    # Compute average distances per cluster
    for cluster_id in range(n_clusters):
        avg_distance = np.mean(cluster_distances[cluster_id]) if cluster_distances[cluster_id] else 0
        print(f"Cluster {cluster_id}: Average distance of test points = {avg_distance:.2f}")




def plot_train_test_centroids(train_centroids, test_centroids, train_labels=None, test_labels=None):
    """
    Visualize train and test centroids in a 2D PCA space with cluster numbers and distinct colors.

    Parameters:
    - train_centroids: Array of train cluster centroids.
    - test_centroids: Array of test cluster centroids.
    - train_labels: Optional cluster IDs for train centroids.
    - test_labels: Optional cluster IDs for test centroids.
    """
    # Perform PCA on the centroids
    pca = PCA(n_components=2)
    all_centroids = np.vstack([train_centroids, test_centroids])
    centroids_2d = pca.fit_transform(all_centroids)

    # Assign colors for clusters
    n_train_clusters = len(train_centroids)
    n_test_clusters = len(test_centroids)
    total_clusters = max(n_train_clusters, n_test_clusters)
    cmap = plt.cm.get_cmap('tab10', total_clusters)

    plt.figure(figsize=(10, 8))

    # Plot train centroids
    for i, point in enumerate(centroids_2d[:n_train_clusters]):
        label = f"Train Cluster {i}" if train_labels is None else f"Train Cluster {train_labels[i]}"
        plt.scatter(point[0], point[1], label=label, color=cmap(i), marker='o', edgecolor='black', s=150)

    # Plot test centroids
    for i, point in enumerate(centroids_2d[n_train_clusters:]):
        label = f"Test Cluster {i}" if test_labels is None else f"Test Cluster {test_labels[i]}"
        plt.scatter(point[0], point[1], label=label, color=cmap(i), marker='x', s=150)

    # Add legend and title
    plt.title("Comparison of Train and Test Cluster Centroids")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"cluster_results/comparison of train and test cluster centroids.png")




def filter_clusters_by_distance_and_targets(
    train_centroids,
    test_centroids,
    test_cluster_labels,
    has_target_segment_test,
    train_cluster_labels,
    filter_method ='max_distance'

):
    """
    Filter test clusters based on distances from train centroids and target-to-non-target ratio.

    Parameters:
    - train_centroids: Centroids of training clusters (n_train_clusters, n_features).
    - test_centroids: Centroids of test clusters (n_test_clusters, n_features).
    - test_cluster_labels: Cluster labels for test segments.
    - has_target_segment_test: Boolean array indicating if a test segment contains a target.
    - max_distance: Maximum allowed distance between train and test centroids to retain clusters.
    - min_target_ratio: Minimum target-to-non-target ratio to retain a cluster.

    Returns:
    - retained_indices: Boolean array indicating which test segments are retained.
    - filtered_out_counts: Dictionary with counts of filtered-out targets and non-targets.
    - retained_clusters: List of retained cluster IDs.
    """
    # Compute distances between test centroids and all train centroids
    distance_matrix = euclidean_distances(test_centroids, train_centroids)
    closest_distances = np.min(distance_matrix, axis=1)

    # Initialize arrays to store counts
    n_test_clusters = len(test_centroids)
    target_counts = np.zeros(n_test_clusters)
    non_target_counts = np.zeros(n_test_clusters)

    # Compute target and non-target counts for each test cluster
    for cluster_id in range(n_test_clusters):
        cluster_mask = train_cluster_labels == cluster_id
        target_counts[cluster_id] = np.sum(has_target_segment_train[cluster_mask])
        non_target_counts[cluster_id] = np.sum(~has_target_segment_train[cluster_mask])

    # Compute target-to-non-target ratio for each cluster
    target_ratios = target_counts / (non_target_counts + 1e-6)
    max_distance = sorted(closest_distances)[-4]
    min_ratio = min(ratio for ratio in target_ratios if ratio > 0)
    # Filter clusters based on distance and target ratio
    if filter_method == "min_ratio":
        retained_clusters = [
            cluster_id for cluster_id in range(n_test_clusters)
            if target_ratios[cluster_id] <= min_ratio
        ]
    elif filter_method == "max_distance":
        retained_clusters = [
            cluster_id for cluster_id in range(n_test_clusters)
            if closest_distances[cluster_id] <= max_distance
            # if target_ratios[cluster_id] > min_ratio
        ]

    # Create a mask for retained segments
    retained_indices = np.isin(test_cluster_labels, retained_clusters)

    # Compute counts of filtered-out targets and non-targets
    filtered_out_targets = np.sum(has_target_segment_test[~retained_indices])
    filtered_out_non_targets = np.sum(~has_target_segment_test[~retained_indices])

    filtered_out_counts = {
        "targets_filtered": filtered_out_targets,
        "non_targets_filtered": filtered_out_non_targets
    }

    # Print stats
    print(f"Total test clusters: {n_test_clusters}")
    print(f"Retained clusters: {len(retained_clusters)}")
    print(f"Filtered-out targets: {filtered_out_targets}")
    print(f"Filtered-out non-targets: {filtered_out_non_targets}")

    return retained_indices, filtered_out_counts, retained_clusters
def plot_pca_with_centroids(
    centroids,
    segment_features,
    cluster_labels,
    title="PCA of Clusters with Centroids",
    split="train"
):
    """
    Plot PCA of clusters with their centroids.

    Parameters:
    - centroids: Centroids of clusters (n_clusters, n_features).
    - segment_features: Features of segments.
    - cluster_labels: Cluster labels for segments.
    - title: Title of the plot.
    - split: "train" or "test" to differentiate the split.
    """
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(segment_features)
    centroids_pca = pca.transform(centroids)

    # Plot clusters
    plt.figure(figsize=(10, 8))
    n_clusters = len(centroids)
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        plt.scatter(
            pca_result[cluster_mask, 0],
            pca_result[cluster_mask, 1],
            label=f"Cluster {cluster_id}",
            alpha=0.5
        )

    # Plot centroids
    plt.scatter(
        centroids_pca[:, 0],
        centroids_pca[:, 1],
        c='black',
        marker='x',
        s=150,
        label="Centroids"
    )

    # Add labels and legend
    plt.title(f"{title} ({split})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"cluster_results/{split}_PCA_of_Clusters_with_Centroids.png")
    plt.show()
def plot_cluster_counts(
    target_counts,
    non_target_counts,
    retained_clusters=None,
    total_targets=None,
    total_non_targets=None,
    title="Cluster Counts",
    split="train"
):
    """
    Plot bar chart showing target and non-target counts for each cluster.

    Parameters:
    - target_counts: Array of target counts for each cluster.
    - non_target_counts: Array of non-target counts for each cluster.
    - retained_clusters: List of retained clusters (highlighted in plot).
    - total_targets: Total number of target segments in the split.
    - total_non_targets: Total number of non-target segments in the split.
    - title: Title of the plot.
    - split: "train" or "test" to differentiate the split.
    """
    n_clusters = len(target_counts)
    indices = np.arange(n_clusters)

    plt.figure(figsize=(12, 8))
    bar_width = 0.4

    # Plot bars
    plt.bar(
        indices, target_counts, bar_width, label="Targets", color="blue", alpha=0.7
    )
    plt.bar(
        indices, non_target_counts, bar_width, label="Non-Targets", color="red", alpha=0.7, bottom=target_counts
    )

    # Highlight retained clusters
    if retained_clusters is not None:
        for cluster_id in retained_clusters:
            plt.text(
                cluster_id, target_counts[cluster_id] + non_target_counts[cluster_id] + 1,
                "Retained", ha="center", color="green", fontsize=10, fontweight="bold"
            )

    # Add values on bars
    for i in range(n_clusters):
        plt.text(i, target_counts[i] / 2, str(int(target_counts[i])), ha='center', color='white')
        plt.text(i, target_counts[i] + non_target_counts[i] / 2, str(int(non_target_counts[i])), ha='center', color='white')

    # Add total targets and non-targets
    if total_targets is not None and total_non_targets is not None:
        plt.text(
            n_clusters - 1, max(target_counts + non_target_counts) * 1.1,
            f"Total Targets: {int(total_targets)}\nTotal Non-Targets: {int(total_non_targets)}",
            ha="center", fontsize=12, fontweight="bold"
        )

    # Add labels and legend
    plt.title(f"{title} ({split})")
    plt.xlabel("Cluster ID")
    plt.ylabel("Count")
    plt.xticks(indices)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"cluster_results/{split}_Cluster_Counts.png")
    plt.show()







# Print results
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np


# --------------------------------------------------
# 1) Create a PyTorch Dataset for segments + labels
# --------------------------------------------------

class SegmentDataset(Dataset):
    def __init__(self, segment_features, has_target_segment):
        """
        segment_features: np.array of shape (N, D)
        has_target_segment: np.array of bool of length N
        """
        self.segments = torch.from_numpy(segment_features).float()
        self.labels = torch.from_numpy(has_target_segment.astype(int))  # 1=target, 0=no target
        self.num_samples = len(self.segments)

        # Precompute indices for target vs. non-target for easy sampling
        self.target_indices = torch.where(self.labels == 1)[0]
        self.nontarget_indices = torch.where(self.labels == 0)[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        By default, we just return the single item in normal usage.
        We'll handle triplet sampling with a custom collate_fn or separate logic.
        """
        return self.segments[idx], self.labels[idx]

    def get_triplet(self, batch_data, batch_labels):
        """
        Generate a triplet (anchor, positive, negative) from the current batch:
        - Anchor and positive are from the same class.
        - Negative is from the opposite class.
        """
        # Separate indices by class
        target_indices = torch.where(batch_labels == 1)[0]
        non_target_indices = torch.where(batch_labels == 0)[0]

        # Handle cases where one of the groups is empty
        if len(target_indices) < 2 or len(non_target_indices) == 0:
            # Fallback: Sample random triplet from the entire batch
            indices = torch.arange(len(batch_data))
            anchor_idx, positive_idx, negative_idx = torch.randint(len(indices), (3,))
        else:
            # Sample anchor and positive from the same class
            if torch.rand(1).item() > 0.5:
                anchor_idx = target_indices[torch.randint(len(target_indices), (1,))].item()
                positive_idx = target_indices[torch.randint(len(target_indices), (1,))].item()
                while positive_idx == anchor_idx:
                    positive_idx = target_indices[torch.randint(len(target_indices), (1,))].item()
                negative_idx = non_target_indices[torch.randint(len(non_target_indices), (1,))].item()
            else:
                anchor_idx = non_target_indices[torch.randint(len(non_target_indices), (1,))].item()
                positive_idx = non_target_indices[torch.randint(len(non_target_indices), (1,))].item()
                while positive_idx == anchor_idx:
                    positive_idx = non_target_indices[torch.randint(len(non_target_indices), (1,))].item()
                negative_idx = target_indices[torch.randint(len(target_indices), (1,))].item()

        # Extract corresponding data
        anchor = batch_data[anchor_idx]
        positive = batch_data[positive_idx]
        negative = batch_data[negative_idx]

        return anchor, positive, negative


# --------------------------------------------------
# 2) Define an Autoencoder with an Embedding
# --------------------------------------------------

# class TripletAutoencoder(nn.Module):
#     def __init__(self, input_dim=40, latent_dim=8):
#         super().__init__()
#         # Encoder
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, latent_dim)
#         )
#         # Decoder
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, input_dim)
#         )
#
#     def forward(self, x):
#         z = self.encoder(x)  # (batch, latent_dim)
#         x_recon = self.decoder(z)  # (batch, input_dim)
#         return x_recon, z


class TripletAutoencoder(nn.Module):
    def __init__(self, input_dim=40, latent_dim=8, dropout_rate=0.5, activation='ReLU', normalize_latent=True):
        """
        Enhanced Triplet Autoencoder with residual connections, flexible activation, and latent space normalization.

        Args:
            input_dim (int): Input feature dimension.
            latent_dim (int): Latent space dimension.
            dropout_rate (float): Dropout rate for regularization.
            activation (str): Activation function ('ReLU', 'LeakyReLU', 'GELU').
            normalize_latent (bool): Whether to normalize the latent space embeddings.
        """
        super().__init__()
        self.normalize_latent = normalize_latent

        # Select activation function
        activation_fn = {
            'ReLU': nn.ReLU,
            'LeakyReLU': nn.LeakyReLU,
            'GELU': nn.GELU
        }[activation]

        # Encoder with residual connections
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, latent_dim)
        )

        # Decoder with residual connections
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, input_dim)
        )

        # Classifier for target vs. non-target
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 16),
            activation_fn(),
            nn.Dropout(dropout_rate),
            nn.Linear(16, 2)
        )

    def forward(self, x):
        # Add configurable noise during training for regularization
        if self.training:
            noise_scale = 0.1
            x = x + torch.randn_like(x) * noise_scale

        # Forward pass
        z = self.encoder(x)  # Latent embedding
        if len(z.shape) == 1:  # If z is a 1D tensor
            z = z.unsqueeze(0)
        if self.normalize_latent:
            z = F.normalize(z, p=2, dim=1)  # Normalize latent vectors to unit length

        x_recon = self.decoder(z)  # Reconstructed input
        logits = self.classifier(z)  # Classification logits

        return x_recon, z, logits

    def encode(self, x):
        z = self.encoder(x)
        return F.normalize(z, p=2, dim=1) if self.normalize_latent else z

    def decode(self, z):
        return self.decoder(z)


# --------------------------------------------------
# 3) Training routine with triplet + recon losses
# --------------------------------------------------


import torch
import torch.nn.functional as F

def hard_negative_triplet_loss(
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    z_neg_candidates: torch.Tensor,
    margin: float = 1.0,
    p: int = 2
) -> torch.Tensor:
    """
    Compute triplet loss with Hard Negative Mining:
      - z_anchor: Embeddings of anchors  [B, D]
      - z_positive: Embeddings of positives [B, D]
      - z_neg_candidates: Candidate negative embeddings [M, D] (M >= B)
      - margin: Margin for triplet loss
      - p: Norm degree (2 for Euclidean distance)

    Returns: A scalar tensor (mean triplet loss across the batch).
    """
    # 1) Compute pairwise distances between each anchor and each negative candidate
    #    dists_anchor_neg shape: (B, M)
    dists_anchor_neg = torch.cdist(z_anchor, z_neg_candidates, p=p)  # cdist is pairwise distance

    # 2) For each anchor, pick the "hardest negative" => the negative with the MIN distance to that anchor
    hardest_neg_idxs = dists_anchor_neg.argmin(dim=1)  # shape: [B]

    # 3) Gather those negative embeddings
    z_negative = z_neg_candidates[hardest_neg_idxs, :]  # shape: [B, D]

    # 4) Compute distances for anchor-positive and anchor-negative
    dist_anchor_pos = torch.norm(z_anchor - z_positive, p=p, dim=1)   # shape: [B]
    dist_anchor_neg = torch.norm(z_anchor - z_negative, p=p, dim=1)   # shape: [B]

    # 5) Standard margin-based hinge loss
    losses = F.relu(dist_anchor_pos - dist_anchor_neg + margin)       # shape: [B]
    return losses.mean()  # scalar


def combined_loss_fn(
    x, x_recon, z_anchor, z_positive, z_negative, logits_anchor, target_anchor,
    reconstruction_weight=1.0,
    triplet_weight=1.0,
    clf_weight=1.0,
    margin=1.0
):
    # 1) Reconstruction Loss
    recon_loss = F.mse_loss(x_recon, x)

    # 2) Triplet Loss
    dist_ap = torch.norm(z_anchor - z_positive, p=2, dim=1)
    dist_an = torch.norm(z_anchor - z_negative, p=2, dim=1)
    trip_loss = F.relu(dist_ap - dist_an + margin).mean()

    # 3) Classification Loss
    clf_loss = F.cross_entropy(logits_anchor, target_anchor)

    total_loss = (
        reconstruction_weight * recon_loss +
        triplet_weight * trip_loss +
        clf_weight * clf_loss
    )
    return total_loss, recon_loss, trip_loss, clf_loss
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def evaluate_model(predictions, true_labels):
    """
    Evaluate the performance of the classifier.

    Args:
        predictions (np.ndarray): Predicted class labels.
        true_labels (np.ndarray): Ground truth labels.

    Returns:
        None
    """
    print("Classification Report:")
    print(classification_report(true_labels, predictions))


def predict_with_classifier(model, data_loader, device):
    """
    Use the classifier head of the trained model to predict target vs. non-target.

    Args:
        model (nn.Module): Trained TripletAutoencoder model.
        data_loader (DataLoader): DataLoader for inference data.
        device (str): 'cpu' or 'cuda'.

    Returns:
        predictions (np.ndarray): Predicted class labels (0 or 1).
        probabilities (np.ndarray): Predicted probabilities for each class.
        true_labels (np.ndarray): Ground truth labels.
    """
    model.eval()
    predictions = []
    probabilities = []
    true_labels = []

    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)
            logits = model(batch_data)[2]  # Obtain logits from the classifier
            probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities
            preds = torch.argmax(probs, dim=1)  # Get predicted class labels

            predictions.extend(preds.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            true_labels.extend(batch_labels.numpy())

    return np.array(predictions), np.array(probabilities), np.array(true_labels)

def train_triplet_ae(
    train_dataset,
    val_dataset,
    input_dim,
    latent_dim=8,
    margin=1.0,
    lambda_recon=1.0,
    lambda_triplet=1.0,
    lambda_clf = 2.0,
    batch_size=64,
    epochs=10,
    lr=1e-3,
    device="cpu"
):
    """
    train_dataset: Training dataset instance of SegmentDataset
    val_dataset: Validation dataset instance of SegmentDataset
    input_dim: Dimension of segment features
    latent_dim: Size of the latent space
    margin: Margin for triplet loss
    """
    # Create DataLoaders
    labels = train_dataset.labels.numpy()  # Assuming `train_dataset.labels` contains class labels
    class_counts = np.bincount(labels)  # Count occurrences of each class
    class_weights = 1.0 / class_counts  # Inverse of class frequency

    # Step 2: Assign weights to each sample
    sample_weights = class_weights[labels]

    # Step 3: Create WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Step 4: Use sampler in DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use sampler instead of shuffle=True
        drop_last=False  # Ensure all samples are seen
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss functions
    model = TripletAutoencoder(input_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse_loss_fn = nn.MSELoss()
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    best_model = model
    # Training loop
    best_val_loss = float('inf')  # Track the best validation loss
    for epoch in range(epochs):
        # Training
        model.train()
        train_recon_loss = 0.0
        train_triplet_loss = 0.0
        num_train_batches = 0
        train_clf_loss = 0.0
        for batch_data, batch_labels in train_loader:
            batch_data = batch_data.to(device)  # Input features
            batch_labels = batch_labels.to(device)  # Target vs. non-target labels
            x_recon, z_batch, logits_batch = model(batch_data)
            clf_loss = F.cross_entropy(logits_batch, batch_labels)

            # Generate triplets from the current batch
            anchor, positive, negative = train_dataset.get_triplet(batch_data, batch_labels)
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            # Forward pass
            x_recon, z_anchor, logits_anchor = model(anchor)
            _, z_positive, _ = model(positive)
            _, z_negative, _ = model(negative)
            trip_loss = triplet_loss_fn(z_anchor, z_positive, z_negative)
            recon_loss = mse_loss_fn(x_recon, batch_data)
            loss =lambda_recon * recon_loss +lambda_triplet * trip_loss +lambda_clf * clf_loss


            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track losses (for logging purposes)

            train_recon_loss += recon_loss.item()
            train_triplet_loss += trip_loss.item()
            train_clf_loss+=clf_loss.item()
            num_train_batches += 1

        avg_train_recon_loss = train_recon_loss / num_train_batches
        avg_train_triplet_loss = train_triplet_loss / num_train_batches
        avg_train_clf_loss = train_clf_loss / num_train_batches

        # Validation
        model.eval()
        val_recon_loss = 0.0
        val_triplet_loss = 0.0
        val_clf_loss = 0.0  # If using classification
        num_val_batches = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_loader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                x_recon, z_batch, logits_batch = model(batch_data)
                clf_loss = F.cross_entropy(logits_batch, batch_labels)

                # Forward pass: reconstruction and embeddings
                recon_loss = mse_loss_fn(x_recon, batch_data)  # Reconstruction loss
                val_recon_loss += recon_loss.item()

                # Generate triplets from the current batch
                anchor, positive, negative = val_dataset.get_triplet(batch_data, batch_labels)
                anchor = anchor.to(device)
                positive = positive.to(device)
                negative = negative.to(device)

                # Compute embeddings for triplets
                _, z_anchor, logits_anchor = model(anchor)
                _, z_positive, _ = model(positive)
                _, z_negative, _ = model(negative)

                # Triplet loss
                trip_loss = triplet_loss_fn(z_anchor, z_positive, z_negative)
                val_triplet_loss += trip_loss.item()

                # Optional: Classification loss
                val_clf_loss += clf_loss.item()
                num_val_batches += 1

        # Calculate average validation losses
        avg_val_recon_loss = val_recon_loss / num_val_batches
        avg_val_triplet_loss = val_triplet_loss / num_val_batches
        avg_val_clf_loss = val_clf_loss / num_val_batches if num_val_batches > 0 else 0

        # Combined validation loss
        avg_val_loss = (
                lambda_recon * avg_val_recon_loss +
                lambda_triplet * avg_val_triplet_loss +
                lambda_clf * avg_val_clf_loss
        )



        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model
            torch.save(model.state_dict(), "best_triplet_ae_model.pt")

        # Print epoch metrics
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Recon: {avg_train_recon_loss:.3f}, Train Triplet: {avg_train_triplet_loss:.3f} ,Train clf: {avg_train_clf_loss:.3f}  "
            f"Val Recon: {avg_val_recon_loss:.3f}, Val Triplet: {avg_val_triplet_loss:.3f}, Val Loss: {avg_val_loss:.3f} val clf: {avg_val_clf_loss:.3f} "
        )

    print("Training complete. Best validation loss:", best_val_loss)
    return best_model




if __name__ == "__main__":
    feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'relative_x', 'relative_y',
                       'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT', 'gaze_velocity']
    # ------------------------------------------------------------
    # Example usage within your pipeline:
    config = DataConfig(
        data_path='data/Categorized_Fixation_Data_1_18.csv',
        approach_num=8,
        normalize=True,
        per_slice_target=False,
        participant_id=1
    )

    # Load legacy data
    df = load_eye_tracking_data(
        data_path=config.data_path,
        approach_num=config.approach_num,
        participant_id=config.participant_id,
        data_format="legacy"
    )

    os.makedirs('cluster_results', exist_ok=True)
    window_size = 500
    seed = 2024
    print(f"seed{seed}")

    n_segments = 10
    n_clusters = 3
    clustering_method = 'kmeans'
    segmentation_method = 'eq'
    print(f"window_size is {window_size}")
    train_df, test_df = split_train_test_for_time_series(df, test_size=0.2, random_state=seed)
    train_df, val_df = split_train_test_for_time_series(train_df, test_size=0.2, random_state=seed)

    print("Original class distribution in test set:")
    print(test_df["target"].value_counts())
    # Create time series
    # 1) Get your data from create_dynamic_time_series
    X_train, Y_train, target_location_train = create_dynamic_time_series(
        train_df,
        feature_columns=None,
        participant_id=config.participant_id,
        load_existing=False,
        split_type='train',
        window_size=window_size
    )

    X_test, Y_test, target_location_test = create_dynamic_time_series(
        test_df,
        feature_columns=None,
        participant_id=config.participant_id,
        load_existing=False,
        split_type='test',
        window_size=window_size
    )

    X_val, Y_val, target_location_val = create_dynamic_time_series(
        val_df,
        feature_columns=None,
        participant_id=config.participant_id,
        load_existing=False,
        split_type='test',  # or 'val' if you prefer
        window_size=window_size
    )


    # Create dataset

    # 2) Segment and cluster on TRAIN

    (segment_features_train,
     cluster_labels_train,
     clustering_model,
     segment_indices,
     has_target_segment_train,
     segment_target_counts) = segment_and_cluster(
        X_train,
        target_location_train,
        n_segments=n_segments,
        n_clusters=n_clusters, dbscan_eps=0.5, clustering_method=clustering_method,
        segmentation_method=segmentation_method
    )
    (segment_features_val,
     cluster_labels_val,
     _,
     _,
     has_target_segment_val,
     _) = segment_and_cluster(
        X_val,
        target_location_val,
        n_segments=n_segments,
        n_clusters=n_clusters, dbscan_eps=0.5, clustering_method=clustering_method,
        segmentation_method=segmentation_method
    )

    # 3) Plot the clusters, highlighting segments that contain a target
    # plot_clusters_with_targets(
    #     segment_features_train,
    #     cluster_labels_train,
    #     has_target_segment_train,
    #     n_clusters=n_clusters,
    #     split='train'
    # )

    segment_features_test, cluster_labels_test, clustering_model_test, segment_indices_test, has_target_segment_test, segment_target_test = segment_and_cluster(
        X_test,
        target_location_test,
        n_segments=n_segments,
        n_clusters=n_clusters, segmentation_method=segmentation_method, clustering_method=clustering_method
        )
    print(np.sum(has_target_segment_test))
    train_dataset = SegmentDataset(segment_features_train, has_target_segment_train)
    val_dataset = SegmentDataset(segment_features_val, has_target_segment_val)
    test_dataset = SegmentDataset(segment_features_test, has_target_segment_test)
    # Train the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = train_triplet_ae(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        input_dim=segment_features_train.shape[1],
        latent_dim=8,
        margin=0.5,
        lambda_recon=0.1,
        lambda_triplet=3,
        lambda_clf=3,
        batch_size=64,
        epochs=500,
        lr=1e-4,
        device=device
    )

    # Ensure the model is in evaluation mode
    model.eval()

    # Encode the test set into the latent space
    with torch.no_grad():
        latent_test = model.encoder(torch.from_numpy(segment_features_test).float().to(device)).cpu().numpy()
        latent_train = model.encoder(torch.from_numpy(segment_features_train).float().to(device)).cpu().numpy()

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Get predictions
    predictions, probabilities, true_labels = predict_with_classifier(model, test_loader, device)

    # Evaluate
    evaluate_model(predictions, true_labels)

    # Perform clustering in the latent space
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels_test = kmeans.fit_predict(latent_test)
    cluster_labels_train = kmeans.fit_predict(latent_train)

    # cluster_labels_test now contains the predicted cluster IDs for each test sample

    plot_clusters_with_targets(
        latent_test,
        cluster_labels_test,
        has_target_segment_test,
        n_clusters=n_clusters,
        split="test_latent",n_segments = n_segments, seed = seed
    )
    plot_clusters_with_targets(
        latent_train,
        cluster_labels_train,
        has_target_segment_train,
        n_clusters=n_clusters,
        split="train_latent",n_segments = n_segments, seed = seed
    )
    # Train the SVM Classifier in High-Dimensional Latent Space

    # Compute the center of the test latent space
    test_center = latent_test.mean(axis=0)
    train_center =  latent_train.mean(axis=0)
    # Compute distances from the test center
    distances_train_to_train_center = np.linalg.norm(latent_train - test_center, axis=1)
    distances_test_to_test_center = np.linalg.norm(latent_test - test_center, axis=1)

    # Set a threshold (e.g., 1 standard deviation of test distances)
    threshold_test = distances_test_to_test_center.std()
    threshold_train = distances_train_to_train_center.std()

    # Keep far-away points (distances above the threshold)
    far_indices_train = distances_train_to_train_center >= threshold_train
    far_indices_test = distances_test_to_test_center >= threshold_test

    # Filter latent spaces and labels
    far_latent_train = latent_train
    far_labels_train = has_target_segment_train

    far_latent_test = latent_test[far_indices_test]
    far_labels_test = has_target_segment_test[far_indices_test]

    svm_far = SVC(kernel='rbf', C=0.5, gamma=1, probability=True, random_state=42, class_weight='balanced')
    svm_far.fit(far_latent_train, far_labels_train)

    # Predict on Far-Away Test Set
    train_predictions = svm_far.predict(far_latent_train)
    test_predictions = svm_far.predict(far_latent_test)
    #
    # svm = SVC(kernel='rbf', C=0.5, gamma='scale', probability=True, random_state=42,class_weight='balanced')
    #
    # svm.fit(latent_train, has_target_segment_train)
    #
    # # Predict on Train and Test Sets in High-Dimensional Space
    # train_predictions = svm.predict(latent_train)
    # test_predictions = svm.predict(latent_test)

    # Reduce Latent Space and SVM Decision Boundary to 2D Using PCA
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

    # Train SVM on Far-Away Data
    svm_far = SVC(
        kernel='rbf',
        C=0.1,  # Adjust for stricter regularization
        gamma=1.0,  # Adjust for stricter boundary
        probability=True,
        random_state=42,
        class_weight='balanced'
    )
    svm_far.fit(far_latent_train, far_labels_train)

    # Predict on Far-Away Test Set
    far_train_predictions = svm_far.predict(far_latent_train)
    far_test_predictions = svm_far.predict(far_latent_test)

    # Evaluate the SVM Performance
    print("Far-Away Train Set Performance:")
    print(classification_report(far_labels_train, far_train_predictions))
    print(f"Train Accuracy: {accuracy_score(far_labels_train, far_train_predictions):.4f}")

    print("\nFar-Away Test Set Performance:")
    print(classification_report(far_labels_test, far_test_predictions))
    print(f"Test Accuracy: {accuracy_score(far_labels_test, far_test_predictions):.4f}")

    # Confusion Matrix for Train Set
    cm_train = confusion_matrix(far_labels_train, far_train_predictions)
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=["Class 0", "Class 1"])
    disp_train.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Train Set)")
    plt.savefig(f"cluster_results/confusion_matrix_train_seed_{seed}.png")
    plt.show()

    # Confusion Matrix for Test Set
    cm_test = confusion_matrix(far_labels_test, far_test_predictions)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=["Class 0", "Class 1"])
    disp_test.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Test Set)")
    plt.savefig(f"cluster_results/confusion_matrix_test_seed_{seed}.png")
    plt.show()

    pca = PCA(n_components=2, random_state=42)
    latent_train_2d = pca.fit_transform(far_latent_train)
    latent_test_2d = pca.transform(far_latent_test)

    # Project SVM Decision Boundary to 2D
    xx_min, xx_max = latent_train_2d[:, 0].min() - 1, latent_train_2d[:, 0].max() + 1
    yy_min, yy_max = latent_train_2d[:, 1].min() - 1, latent_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(xx_min, xx_max, 500), np.linspace(yy_min, yy_max, 500))
    grid_2d = np.c_[xx.ravel(), yy.ravel()]
    grid_high_dim = pca.inverse_transform(grid_2d)  # Map back to high-dimensional space
    Z = svm_far.predict(grid_high_dim).reshape(xx.shape)  # Predict in high-dimensional space and reshape to 2D


    # Step 4: Plot Decision Boundary and Clusters
    def plot_svm_high_dim(latent_2d, true_labels, predictions, decision_boundary, split,seed = 0):
        """
        Visualize SVM decision boundary (trained in high-dimensional space) in 2D.

        Parameters:
            latent_2d (ndarray): PCA-reduced 2D latent space.
            true_labels (ndarray): Ground truth target labels.
            predictions (ndarray): SVM predictions.
            decision_boundary (ndarray): 2D SVM decision boundary grid.
            split (str): 'train' or 'test' for plot title.
        """
        plt.figure(figsize=(10, 8))

        # Plot decision boundary
        plt.contourf(xx, yy, decision_boundary, alpha=0.3, cmap='coolwarm')

        # Plot true labels (always shown under predictions)
        plt.scatter(
            latent_2d[true_labels == 0, 0],
            latent_2d[true_labels == 0, 1],
            color='blue',
            label="True Label 0",
            alpha=0.8,
            marker='X',
            s=80
        )
        plt.scatter(
            latent_2d[true_labels == 1, 0],
            latent_2d[true_labels == 1, 1],
            color='red',
            label="True Label 1",
            alpha=0.8,
            marker='X',
            s=80
        )

        # Highlight SVM predictions
        plt.scatter(
            latent_2d[predictions == 1, 0],
            latent_2d[predictions == 1, 1],
            facecolor='none',
            marker='o',
            s=120,
            linewidth=1.5,
            alpha=0.2,

            label="SVM Positive Predictions"
        )
        plt.scatter(
            latent_2d[predictions == 0, 0],
            latent_2d[predictions == 0, 1],
            facecolor='none',
            marker='o',
            s=120,
            linewidth=1.5,
            alpha=0.2,

            label="SVM Negative Predictions"
        )

        plt.title(f"SVM Decision Boundary (High Dim to 2D) ({split})")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(f"cluster_results/svm_high_dim_{split}_decision_boundary_seed_{seed}.png")
        plt.show()


    # Plot for Train Set
    plot_svm_high_dim(
        latent_train_2d,
        true_labels=far_labels_train,
        predictions=train_predictions,
        decision_boundary=Z,
        split="train",seed=seed
    )

    # Plot for Test Set
    plot_svm_high_dim(
        latent_test_2d,
        true_labels=far_labels_test,
        predictions=test_predictions,
        decision_boundary=Z,
        split="test",seed=seed
    )

    quit()

    # plot_clusters_with_targets(
    #     segment_features_test,
    #     cluster_labels_test,
    #     has_target_segment_test,
    #     n_clusters=n_clusters,
    #     split="test", clustering_method=clustering_method
    # )
    # Assume segment_features_test contains the test features
    test_cluster_labels = clustering_model.predict(segment_features_test)
    from scipy.spatial.distance import cdist

    # Compute distances between test points and their assigned cluster centers
    test_distances = cdist(segment_features_test, clustering_model.cluster_centers_, metric='euclidean')
    test_assigned_distances = [test_distances[i, label] for i, label in enumerate(test_cluster_labels)]

    # Print average distance for test points
    avg_test_distance = np.mean(test_assigned_distances)
    print(f"Average distance of test points to assigned cluster centers: {avg_test_distance:.2f}")
    # Compute the Euclidean distance between train cluster centers and test data points
    train_cluster_centers = clustering_model.cluster_centers_

    # Compute distances between test points and train cluster centers
    test_distances = cdist(segment_features_test, train_cluster_centers, metric='euclidean')
    train_distances = cdist(segment_features_train, clustering_model.cluster_centers_, metric='euclidean')
    avg_train_distance = np.mean(np.min(train_distances, axis=1))
    print(f"Average distance of train points to assigned cluster centers: {avg_train_distance:.2f}")

    # Calculate the average distance of test points to their assigned cluster centers
    test_assigned_distances = [test_distances[i, label] for i, label in enumerate(test_cluster_labels)]
    avg_test_distance = np.mean(test_assigned_distances)
    print(f"Average distance of test points to their assigned cluster centers: {avg_test_distance:.2f}")
    compute_avg_distances_per_cluster(segment_features_test, test_cluster_labels, clustering_model)

    from sklearn.metrics import silhouette_score

    # silhouette = silhouette_score(segment_features_test, test_cluster_labels)
    # print(f"Silhouette score for test data: {silhouette:.2f}")
    test_inertia = np.sum(np.min(test_distances, axis=1))
    print(f"Inertia for test data: {test_inertia:.2f}")

    # 4) Apply same segmentation + feature extraction + cluster assignment to X_val/X_test
    #    so you can build the same cluster-hist representation. Then do clf.predict(...)
    #    to classify your windows.
    from sklearn.metrics.pairwise import euclidean_distances
    import numpy as np

    # Step 1: Cluster the test set using the same model or re-cluster
    test_cluster_labels = clustering_model.predict(segment_features_test)

    # Step 2: Compute centroids for train and test clusters
    train_centroids = clustering_model.cluster_centers_  # Shape: (n_clusters, n_features)

    # Compute centroids for test clusters
    test_centroids = []
    for cluster_id in range(clustering_model.n_clusters):
        cluster_points = segment_features_test[test_cluster_labels == cluster_id]
        if len(cluster_points) > 0:
            test_centroids.append(np.mean(cluster_points, axis=0))
        else:
            test_centroids.append(np.zeros_like(train_centroids[0]))  # Handle empty clusters
    test_centroids = np.array(test_centroids)

    # Step 3: Compare train and test centroids
    distance_matrix = euclidean_distances(train_centroids, test_centroids)  # Shape: (n_clusters_train, n_clusters_test)

    # Assign each test cluster to the closest train cluster
    closest_train_clusters = np.argmin(distance_matrix, axis=0)
    plot_train_test_centroids(train_centroids, test_centroids)

    retained_indices, filtered_out_counts, retained_clusters = filter_clusters_by_distance_and_targets(
        train_centroids=clustering_model.cluster_centers_,
        test_centroids=test_centroids,
        test_cluster_labels=test_cluster_labels,
        has_target_segment_test=has_target_segment_test,
        train_cluster_labels=cluster_labels_train
    )

    # Plot train PCA
    plot_pca_with_centroids(
        centroids=clustering_model.cluster_centers_,
        segment_features=segment_features_train,
        cluster_labels=cluster_labels_train,
        title="PCA of Train Clusters with Centroids",
        split="train"
    )

    # Plot test PCA
    plot_pca_with_centroids(
        centroids=test_centroids,
        segment_features=segment_features_test,
        cluster_labels=test_cluster_labels,
        title="PCA of Test Clusters with Centroids",
        split="test"
    )

    # Compute counts for train and test clusters
    train_target_counts = np.bincount(cluster_labels_train, weights=has_target_segment_train,
                                      minlength=len(clustering_model.cluster_centers_))
    train_non_target_counts = np.bincount(cluster_labels_train, weights=~has_target_segment_train,
                                          minlength=len(clustering_model.cluster_centers_))

    test_target_counts = np.bincount(test_cluster_labels, weights=has_target_segment_test,
                                     minlength=len(test_centroids))
    test_non_target_counts = np.bincount(test_cluster_labels, weights=~has_target_segment_test,
                                         minlength=len(test_centroids))
    print("total targets in train")
    print(np.sum(train_target_counts))
    # Plot cluster counts for train and test
    plot_cluster_counts(
        target_counts=train_target_counts,
        non_target_counts=train_non_target_counts,
        retained_clusters=None,
        total_targets=np.sum(train_target_counts),
        total_non_targets=np.sum(train_non_target_counts),
        title="Cluster Counts for Train",
        split="train"
    )
    print("total targets in test")
    print(np.sum(test_target_counts))
    plot_cluster_counts(
        target_counts=test_target_counts,
        non_target_counts=test_non_target_counts,
        retained_clusters=retained_clusters,
        total_targets=np.sum(test_target_counts),
        total_non_targets=np.sum(test_non_target_counts),
        title="Cluster Counts for Test",
        split="test"
    )