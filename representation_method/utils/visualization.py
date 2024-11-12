"""
Visualization utilities for model training and evaluation.
"""

import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


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


