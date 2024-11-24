"""
Visualization utilities for model training and evaluation.
"""

import os
import matplotlib
import torch
from scipy.stats import gaussian_kde

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, silhouette_score, davies_bouldin_score


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


