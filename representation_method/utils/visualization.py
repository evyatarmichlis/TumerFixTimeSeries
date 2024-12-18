"""
Visualization utilities for model training and evaluation.
"""

import os
import matplotlib
import torch
from scipy.stats import gaussian_kde
from torch import pdist
from tqdm import tqdm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, silhouette_score, davies_bouldin_score
import umap

import signal
from contextlib import contextmanager
import time


@contextmanager
def timeout(seconds):
    """Simple timeout context manager using signal"""

    def signal_handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds} seconds")

    # Set the signal handler
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)  # Start the timer

    try:
        yield
    finally:
        signal.alarm(0)  # Disable the alarm


def analyze_embeddings_from_loader(encoder, loader, device, method_dir, loader_name="data", sample_ratio=0.1):
    """
    Analyze embeddings with sampling of majority class.
    sample_ratio: proportion of class 0 samples to keep (e.g., 0.1 for 10%)
    """
    print(f"\nAnalyzing {loader_name} embeddings...")

    embedding_dir = os.path.join(method_dir, f'embedding_analysis_{loader_name}')
    os.makedirs(embedding_dir, exist_ok=True)

    encoder.eval()
    all_embeddings = []
    all_labels = []

    # First pass: collect all samples
    with torch.no_grad():
        for batch, labels in loader:
            batch = batch.to(device)
            encoded_outputs, hidden = encoder(batch)
            if isinstance(hidden, tuple):
                embeddings = hidden[0][-1]
            else:
                embeddings = hidden[-1]

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    # Combine data
    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    # Sample class 0
    class_0_indices = np.where(labels == 0)[0]
    class_1_indices = np.where(labels == 1)[0]

    # Randomly sample from class 0
    num_samples = int(len(class_0_indices) * sample_ratio)
    sampled_class_0_indices = np.random.choice(class_0_indices, num_samples, replace=False)

    # Combine sampled indices
    selected_indices = np.concatenate([sampled_class_0_indices, class_1_indices])

    # Get sampled data
    embeddings = embeddings[selected_indices]
    labels = labels[selected_indices]

    print(f"Original class 0 samples: {len(class_0_indices)}")
    print(f"Sampled class 0 samples: {len(sampled_class_0_indices)}")
    print(f"Class 1 samples: {len(class_1_indices)}")

    # Rest of the analysis remains the same...
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    # Calculate metrics
    print("Computing clustering metrics...")
    silhouette_avg = silhouette_score(embeddings_2d, labels)
    davies_bouldin = davies_bouldin_score(embeddings_2d, labels)
    total_samples = len(embeddings)

    # Save analysis results
    with open(os.path.join(embedding_dir, 'embedding_analysis.txt'), 'w') as f:
        f.write(f"{loader_name} Embedding Analysis Results\n")
        f.write(f"========================\n")
        f.write(f"Total samples: {total_samples}\n")
        f.write(f"Embedding dimension: {embeddings.shape[1]}\n\n")

        f.write(f"Clustering Metrics:\n")
        f.write(f"Silhouette Score: {silhouette_avg:.3f}\n")
        f.write(f"Davies-Bouldin Score: {davies_bouldin:.3f}\n\n")

        # Class distribution
        unique, counts = np.unique(labels, return_counts=True)
        f.write(f"Class Distribution:\n")
        for label, count in zip(unique, counts):
            percentage = count / len(labels) * 100
            f.write(f"Class {label}: {count} samples ({percentage:.1f}%)\n")

    # Create and save visualizations
    print("Creating visualizations...")
    plt.figure(figsize=(15, 5))

    # Plot 1: Scatter plot of embeddings
    plt.subplot(131)
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=labels, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f"{loader_name} t-SNE Embedding Distribution")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Plot 2: Class densities
    plt.subplot(132)
    for label in unique:
        mask = labels == label
        if np.sum(mask) > 1:  # Need at least 2 points for KDE
            kde = gaussian_kde(embeddings_2d[mask].T)
            x_range = np.linspace(embeddings_2d[:, 0].min(), embeddings_2d[:, 0].max(), 100)
            y_range = np.linspace(embeddings_2d[:, 1].min(), embeddings_2d[:, 1].max(), 100)
            x_mesh, y_mesh = np.meshgrid(x_range, y_range)
            positions = np.vstack([x_mesh.ravel(), y_mesh.ravel()])
            z = kde(positions).reshape(x_mesh.shape)
            plt.contour(x_mesh, y_mesh, z, levels=5, alpha=0.3,
                        colors=['red' if label == 1 else 'blue'])
    plt.title("Class Density Distribution")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Plot 3: Combined visualization
    plt.subplot(133)
    for label in unique:
        mask = labels == label
        plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1],
                    alpha=0.6, label=f'Class {label}')
    plt.legend()
    plt.title("Class Distribution")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    plt.tight_layout()
    plt.savefig(os.path.join(embedding_dir, 'embedding_visualization.png'))
    plt.close()

    print(f"{loader_name} embedding analysis saved to {embedding_dir}")
    print(f"Total samples: {total_samples}")
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")

    return embeddings, embeddings_2d, labels


# Usage example:
# For train loader:
# embeddings_train, embeddings_2d_train, labels_train = analyze_embeddings_from_loader(
#     encoder, train_loader, device, method_dir, "train")

# For test loader:
# embeddings_test, embeddings_2d_test, labels_test = analyze_embeddings_from_loader(
#     encoder, test_loader, device, method_dir, "test")

# For validation loader:
# embeddings_val, embeddings_2d_val, labels_val = analyze_embeddings_from_loader(
#     encoder, val_loader, device, method_dir, "validation")


def plot_training_curves(train_losses, val_losses, train_accs=None, val_accs=None, save_path=None):
    """Plot training and validation curves."""
    if train_accs is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Loss plot
        ax1.plot(train_losses, label='Training Loss')
        ax1.plot(val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()

        # Accuracy plot
        ax2.plot(train_accs, label='Training Accuracy')
        ax2.plot(val_accs, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
    else:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def visualize_synthetic_data(original_data, synthetic_data, save_dir):
    """Visualize synthetic data quality."""
    os.makedirs(save_dir, exist_ok=True)

    # Feature distributions
    for i in range(original_data.shape[2]):
        plt.figure(figsize=(10, 5))
        plt.hist(original_data[:, :, i].flatten(), bins=50, alpha=0.5,
                 density=True, label='Original', color='blue')
        plt.hist(synthetic_data[:, :, i].flatten(), bins=50, alpha=0.5,
                 density=True, label='Synthetic', color='red')
        plt.title(f'Feature {i} Distribution')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'feature_{i}_dist.png'))
        plt.close()


def plot_pca_tsne_visualization(features, labels, save_dir, prefix=''):
    """Create PCA and t-SNE visualizations."""
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)

    plt.figure(figsize=(10, 5))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis')
    plt.title('PCA Visualization')
    plt.savefig(os.path.join(save_dir, f'{prefix}pca.png'))
    plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(features)

    plt.figure(figsize=(10, 5))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis')
    plt.title('t-SNE Visualization')
    plt.savefig(os.path.join(save_dir, f'{prefix}tsne.png'))
    plt.close()


def analyze_vae_embeddings_from_loader(vae_model, loader, device, method_dir, loader_name="data", n_iterations=5,
                                       sample_ratio=0.1, perplexity=30):
    """
    Analyze embeddings with multiple iterations of sampling and averaged metrics.

    Args:
        vae_model: The VAE model
        loader: DataLoader
        device: torch device
        method_dir: Output directory
        loader_name: Name for the analysis
        n_iterations: Number of sampling iterations
        sample_ratio: Ratio of majority class to sample
        perplexity: t-SNE perplexity parameter
    """
    print(f"\nAnalyzing {loader_name} embeddings...")
    embedding_dir = os.path.join(method_dir, f'embedding_analysis_{loader_name}')
    os.makedirs(embedding_dir, exist_ok=True)

    # Collect all embeddings first
    vae_model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch, labels in loader:
            batch = batch.to(device)
            mu, _ = vae_model.encode(batch)
            all_embeddings.append(mu.cpu())
            all_labels.append(labels)

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    # Initialize arrays to store metrics
    silhouette_scores = []
    davies_bouldin_scores = []
    tsne_results = []

    # Get indices for each class
    class_0_indices = np.where(labels == 0)[0]
    class_1_indices = np.where(labels == 1)[0]
    num_samples = int(len(class_0_indices) * sample_ratio)

    print(f"Running {n_iterations} iterations with {num_samples} samples from majority class...")

    for i in tqdm(range(n_iterations)):
        # Sample from majority class
        sampled_class_0_indices = np.random.choice(class_0_indices, num_samples, replace=False)
        selected_indices = np.concatenate([sampled_class_0_indices, class_1_indices])

        iter_embeddings = embeddings[selected_indices]
        iter_labels = labels[selected_indices]

        # Compute t-SNE with optimized parameters
        tsne = TSNE(
            n_components=2,
            method='barnes_hut',  # Faster approximation method
        )
        embeddings_2d = tsne.fit_transform(iter_embeddings)

        # Calculate metrics
        sil_score = silhouette_score(embeddings_2d, iter_labels)
        db_score = davies_bouldin_score(embeddings_2d, iter_labels)

        silhouette_scores.append(sil_score)
        davies_bouldin_scores.append(db_score)
        tsne_results.append((embeddings_2d, iter_labels))

    # Calculate statistics
    avg_silhouette = np.mean(silhouette_scores)
    std_silhouette = np.std(silhouette_scores)
    avg_davies = np.mean(davies_bouldin_scores)
    std_davies = np.std(davies_bouldin_scores)

    # Find median result based on silhouette score
    median_idx = np.argsort(silhouette_scores)[len(silhouette_scores) // 2]
    median_embeddings_2d, median_labels = tsne_results[median_idx]

    # Save analysis results
    with open(os.path.join(embedding_dir, 'embedding_analysis.txt'), 'w') as f:
        f.write(f"{loader_name} Embedding Analysis Results\n")
        f.write(f"========================\n")
        f.write(f"Analysis Parameters:\n")
        f.write(f"Number of iterations: {n_iterations}\n")
        f.write(f"Sample ratio: {sample_ratio}\n")
        f.write(f"Perplexity: {perplexity}\n\n")
        f.write(f"Clustering Metrics (mean ± std):\n")
        f.write(f"Silhouette Score: {avg_silhouette:.3f} ± {std_silhouette:.3f}\n")
        f.write(f"Davies-Bouldin Score: {avg_davies:.3f} ± {std_davies:.3f}\n\n")

        # Class distribution
        f.write(f"Class Distribution:\n")
        f.write(f"Original majority class (0): {len(class_0_indices)}\n")
        f.write(f"Sampled majority class (0): {num_samples}\n")
        f.write(f"Minority class (1): {len(class_1_indices)}\n")

    # Create visualization with median result
    plt.figure(figsize=(15, 5))

    # Plot 1: Scatter plot of median embeddings
    plt.subplot(131)
    scatter = plt.scatter(median_embeddings_2d[:, 0], median_embeddings_2d[:, 1],
                          c=median_labels, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f"Median t-SNE Result\nSilhouette: {silhouette_scores[median_idx]:.3f}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")

    # Plot 2: Metrics Distribution
    plt.subplot(132)
    plt.boxplot([silhouette_scores, davies_bouldin_scores],
                labels=['Silhouette', 'Davies-Bouldin'])
    plt.title("Metrics Distribution")

    # Plot 3: Latent Space Norms
    plt.subplot(133)
    for label in [0, 1]:
        mask = labels == label
        norms = np.linalg.norm(embeddings[mask], axis=1)
        plt.hist(norms, bins=30, alpha=0.5, label=f'Class {label}',
                 density=True)
    plt.title("Latent Space Norms")
    plt.xlabel("L2 Norm")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(embedding_dir, 'vae_embedding_visualization.png'))
    plt.close()

    print(f"\nResults Summary:")
    print(f"Silhouette Score: {avg_silhouette:.3f} ± {std_silhouette:.3f}")
    print(f"Davies-Bouldin Score: {avg_davies:.3f} ± {std_davies:.3f}")

    return embeddings, median_embeddings_2d, median_labels, {
        'silhouette_scores': silhouette_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'avg_silhouette': avg_silhouette,
        'avg_davies': avg_davies
    }


def analyze_vae_embeddings_with_umap(encoder_model, loader, device, method_dir, loader_name="data",
                                     n_iterations=10, sample_ratio=0.2, n_neighbors=15, min_dist=0.1):
    """
    Analyze embeddings using UMAP with multiple iterations of sampling and averaged metrics.

    Args:
        vae_model: The VAE model
        loader: DataLoader
        device: torch device
        method_dir: Output directory
        loader_name: Name for the analysis
        n_iterations: Number of sampling iterations
        sample_ratio: Ratio of majority class to sample
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
    """
    print(f"\nAnalyzing {loader_name} embeddings with UMAP...")
    embedding_dir = os.path.join(method_dir, f'embedding_analysis_umap_{loader_name}')
    os.makedirs(embedding_dir, exist_ok=True)

    # Collect all embeddings
    encoder_model.eval()
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for batch, labels in loader:
            batch = batch.to(device)
            mu, _ = encoder_model.encode(batch)
            all_embeddings.append(mu.cpu())
            all_labels.append(labels)

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    # Initialize arrays for metrics
    silhouette_scores = []
    davies_bouldin_scores = []
    umap_results = []

    # Get indices for each class
    class_0_indices = np.where(labels == 0)[0]
    class_1_indices = np.where(labels == 1)[0]
    num_samples = int(len(class_0_indices) * sample_ratio)

    print(f"Running {n_iterations} iterations with {num_samples} samples from majority class...")

    for i in tqdm(range(n_iterations)):
        # Sample from majority class
        sampled_class_0_indices = np.random.choice(class_0_indices, num_samples, replace=False)
        selected_indices = np.concatenate([sampled_class_0_indices, class_1_indices])

        iter_embeddings = embeddings[selected_indices]
        iter_labels = labels[selected_indices]

        # Compute UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            # random_state=42 + i
        )
        try:
            with timeout(60):
                embeddings_2d = reducer.fit_transform(iter_embeddings)
        except TimeoutError:
            print(f"\nWarning: Iteration {i + 1} timed out after {300} seconds")
            continue
        # Calculate metrics
        sil_score = silhouette_score(embeddings_2d, iter_labels)
        db_score = davies_bouldin_score(embeddings_2d, iter_labels)

        silhouette_scores.append(sil_score)
        davies_bouldin_scores.append(db_score)
        umap_results.append((embeddings_2d, iter_labels))

    # Calculate statistics
    avg_silhouette = np.mean(silhouette_scores)
    std_silhouette = np.std(silhouette_scores)
    avg_davies = np.mean(davies_bouldin_scores)
    std_davies = np.std(davies_bouldin_scores)

    # Find median result based on silhouette score
    median_idx = np.argsort(silhouette_scores)[len(silhouette_scores) // 2]
    median_embeddings_2d, median_labels = umap_results[median_idx]

    # Save analysis results
    with open(os.path.join(embedding_dir, 'umap_analysis.txt'), 'w') as f:
        f.write(f"{loader_name} UMAP Analysis Results\n")
        f.write(f"========================\n")
        f.write(f"Analysis Parameters:\n")
        f.write(f"Number of iterations: {n_iterations}\n")
        f.write(f"Sample ratio: {sample_ratio}\n")
        f.write(f"UMAP n_neighbors: {n_neighbors}\n")
        f.write(f"UMAP min_dist: {min_dist}\n\n")
        f.write(f"Clustering Metrics (mean ± std):\n")
        f.write(f"Silhouette Score: {avg_silhouette:.3f} ± {std_silhouette:.3f}\n")
        f.write(f"Davies-Bouldin Score: {avg_davies:.3f} ± {std_davies:.3f}\n\n")

        # Save all iterations' scores
        f.write("\nDetailed Scores per Iteration:\n")
        for i in range(n_iterations):
            f.write(f"Iteration {i + 1}:\n")
            f.write(f"  Silhouette: {silhouette_scores[i]:.3f}\n")
            f.write(f"  Davies-Bouldin: {davies_bouldin_scores[i]:.3f}\n")

    # Create visualization
    plt.figure(figsize=(15, 5))

    # Plot 1: Scatter plot of median embeddings
    plt.subplot(131)
    scatter = plt.scatter(median_embeddings_2d[:, 0], median_embeddings_2d[:, 1],
                          c=median_labels, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f"Median UMAP Result\nSilhouette: {silhouette_scores[median_idx]:.3f}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    # Plot 2: Metrics Distribution
    plt.subplot(132)
    plt.boxplot([silhouette_scores, davies_bouldin_scores],
                labels=['Silhouette', 'Davies-Bouldin'])
    plt.title("Metrics Distribution")

    # Plot 3: Score Progression
    plt.subplot(133)
    plt.plot(silhouette_scores, label='Silhouette', marker='o')
    plt.plot(davies_bouldin_scores, label='Davies-Bouldin', marker='s')
    plt.title("Scores Across Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(embedding_dir, 'umap_visualization.png'))
    plt.close()

    print(f"\nResults Summary:")
    print(f"Silhouette Score: {avg_silhouette:.3f} ± {std_silhouette:.3f}")
    print(f"Davies-Bouldin Score: {avg_davies:.3f} ± {std_davies:.3f}")

    return embeddings, median_embeddings_2d, median_labels, {
        'silhouette_scores': silhouette_scores,
        'davies_bouldin_scores': davies_bouldin_scores,
        'avg_silhouette': avg_silhouette,
        'std_silhouette': std_silhouette,
        'avg_davies': avg_davies,
        'std_davies': std_davies
    }


def analyze_encoder_embeddings_with_umap_2(encoder_model, loader, device, method_dir, loader_name="data",
                                         n_iterations=10, sample_ratio=0.2, n_neighbors=15, min_dist=0.1):
    """
    Analyze embeddings from the triplet-based encoder using UMAP with multiple iterations.

    Args:
        encoder_model: The trained encoder model
        loader: DataLoader
        device: torch device
        method_dir: Output directory
        loader_name: Name for the analysis
        n_iterations: Number of sampling iterations
        sample_ratio: Ratio of majority class to sample
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
    """
    print(f"\nAnalyzing {loader_name} embeddings with UMAP...")
    embedding_dir = os.path.join(method_dir, f'embedding_analysis_umap_{loader_name}')
    os.makedirs(embedding_dir, exist_ok=True)

    # Collect all embeddings
    encoder_model.eval()
    all_embeddings = []
    all_labels = []
    triplet_distances = []  # To store triplet distances for analysis

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            # Get normalized embeddings
            embeddings = encoder_model.get_normalized_embeddings(inputs)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

            # Calculate triplet distances for analysis
            for i in range(len(labels)):
                anchor_emb = embeddings[i:i + 1]
                pos_mask = labels == labels[i]
                pos_mask[i] = False  # Exclude self
                neg_mask = labels != labels[i]

                if torch.any(pos_mask) and torch.any(neg_mask):
                    pos_dists = torch.cdist(anchor_emb, embeddings[pos_mask])
                    neg_dists = torch.cdist(anchor_emb, embeddings[neg_mask])
                    triplet_distances.append({
                        'positive': pos_dists.mean().item(),
                        'negative': neg_dists.min().item(),
                        'class': labels[i].item()
                    })

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    # Initialize arrays for metrics
    silhouette_scores = []
    davies_bouldin_scores = []
    umap_results = []
    intraclass_distances = {0: [], 1: []}
    interclass_distances = []

    # Get indices for each class
    class_0_indices = np.where(labels == 0)[0]
    class_1_indices = np.where(labels == 1)[0]
    num_samples = int(len(class_0_indices) * sample_ratio)

    print(f"Running {n_iterations} iterations with {num_samples} samples from majority class...")

    for i in tqdm(range(n_iterations)):
        # Sample from majority class
        sampled_class_0_indices = np.random.choice(class_0_indices, num_samples, replace=False)
        selected_indices = np.concatenate([sampled_class_0_indices, class_1_indices])

        iter_embeddings = embeddings[selected_indices]
        iter_labels = labels[selected_indices]

        # Compute UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42 + i
        )
        try:
            embeddings_2d = reducer.fit_transform(iter_embeddings)

            # Calculate metrics
            sil_score = silhouette_score(embeddings_2d, iter_labels)
            db_score = davies_bouldin_score(embeddings_2d, iter_labels)

            silhouette_scores.append(sil_score)
            davies_bouldin_scores.append(db_score)
            umap_results.append((embeddings_2d, iter_labels))


        except TimeoutError:
            print(f"\nWarning: Iteration {i + 1} timed out")
            continue

    # Calculate statistics
    avg_silhouette = np.mean(silhouette_scores)
    std_silhouette = np.std(silhouette_scores)
    avg_davies = np.mean(davies_bouldin_scores)
    std_davies = np.std(davies_bouldin_scores)

    # Find median result based on silhouette score
    median_idx = np.argsort(silhouette_scores)[len(silhouette_scores) // 2]
    median_embeddings_2d, median_labels = umap_results[median_idx]

    # Analyze triplet distances
    triplet_stats = {
        'minority': {
            'pos_dist_mean': np.mean([t['positive'] for t in triplet_distances if t['class'] == 1]),
            'neg_dist_mean': np.mean([t['negative'] for t in triplet_distances if t['class'] == 1])
        },
        'majority': {
            'pos_dist_mean': np.mean([t['positive'] for t in triplet_distances if t['class'] == 0]),
            'neg_dist_mean': np.mean([t['negative'] for t in triplet_distances if t['class'] == 0])
        }
    }

    # Save analysis results
    with open(os.path.join(embedding_dir, 'umap_analysis.txt'), 'w') as f:
        f.write(f"{loader_name} UMAP Analysis Results\n")
        f.write(f"========================\n\n")

        f.write(f"Analysis Parameters:\n")
        f.write(f"Number of iterations: {n_iterations}\n")
        f.write(f"Sample ratio: {sample_ratio}\n")
        f.write(f"UMAP n_neighbors: {n_neighbors}\n")
        f.write(f"UMAP min_dist: {min_dist}\n\n")

        f.write(f"Clustering Metrics (mean ± std):\n")
        f.write(f"Silhouette Score: {avg_silhouette:.3f} ± {std_silhouette:.3f}\n")
        f.write(f"Davies-Bouldin Score: {avg_davies:.3f} ± {std_davies:.3f}\n\n")

        f.write(f"Triplet Distance Analysis:\n")
        f.write("Minority Class:\n")
        f.write(f"  Mean Positive Distance: {triplet_stats['minority']['pos_dist_mean']:.3f}\n")
        f.write(f"  Mean Negative Distance: {triplet_stats['minority']['neg_dist_mean']:.3f}\n")
        f.write("Majority Class:\n")
        f.write(f"  Mean Positive Distance: {triplet_stats['majority']['pos_dist_mean']:.3f}\n")
        f.write(f"  Mean Negative Distance: {triplet_stats['majority']['neg_dist_mean']:.3f}\n\n")

        f.write("Class Separation Analysis:\n")
        f.write(f"Mean Intraclass Distance (Class 0): {np.mean(intraclass_distances[0]):.3f}\n")
        f.write(f"Mean Intraclass Distance (Class 1): {np.mean(intraclass_distances[1]):.3f}\n")
        f.write(f"Mean Interclass Distance: {np.mean(interclass_distances):.3f}\n")

    # Create visualizations
    plt.figure(figsize=(20, 5))

    # Plot 1: UMAP scatter plot
    plt.subplot(141)
    scatter = plt.scatter(median_embeddings_2d[:, 0], median_embeddings_2d[:, 1],
                          c=median_labels, cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter)
    plt.title(f"Median UMAP Result\nSilhouette: {silhouette_scores[median_idx]:.3f}")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    # Plot 2: Metrics Distribution
    plt.subplot(142)
    plt.boxplot([silhouette_scores, davies_bouldin_scores],
                labels=['Silhouette', 'Davies-Bouldin'])
    plt.title("Metrics Distribution")

    # Plot 3: Triplet Distances
    plt.subplot(143)
    x = np.arange(2)
    width = 0.35
    plt.bar(x - width / 2,
            [triplet_stats['majority']['pos_dist_mean'],
             triplet_stats['minority']['pos_dist_mean']],
            width, label='Positive')
    plt.bar(x + width / 2,
            [triplet_stats['majority']['neg_dist_mean'],
             triplet_stats['minority']['neg_dist_mean']],
            width, label='Negative')
    plt.xticks(x, ['Majority', 'Minority'])
    plt.title("Triplet Distances")
    plt.legend()

    # Plot 4: Class Separation
    plt.subplot(144)
    plt.boxplot([intraclass_distances[0], intraclass_distances[1], interclass_distances],
                labels=['Intra-Maj', 'Intra-Min', 'Inter'])
    plt.title("Class Separation")

    plt.tight_layout()
    plt.savefig(os.path.join(embedding_dir, 'embedding_analysis.png'))
    plt.close()

    # Save embeddings and labels for potential further analysis
    np.save(os.path.join(embedding_dir, 'embeddings.npy'), embeddings)
    np.save(os.path.join(embedding_dir, 'labels.npy'), labels)

    print(f"\nResults Summary:")
    print(f"Silhouette Score: {avg_silhouette:.3f} ± {std_silhouette:.3f}")
    print(f"Davies-Bouldin Score: {avg_davies:.3f} ± {std_davies:.3f}")
    print(f"Interclass Distance: {np.mean(interclass_distances):.3f}")
    print(
        f"Class Separation Ratio: {np.mean(interclass_distances) / np.mean(list(intraclass_distances[0]) + list(intraclass_distances[1])):.3f}")

    return {
        'embeddings': embeddings,
        'umap_embeddings': median_embeddings_2d,
        'labels': labels,
        'metrics': {
            'silhouette_scores': silhouette_scores,
            'davies_bouldin_scores': davies_bouldin_scores,
            'avg_silhouette': avg_silhouette,
            'std_silhouette': std_silhouette,
            'avg_davies': avg_davies,
            'std_davies': std_davies,
            'triplet_stats': triplet_stats,
            'intraclass_distances': intraclass_distances,
            'interclass_distances': interclass_distances
        }
    }