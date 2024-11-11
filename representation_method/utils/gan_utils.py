"""
Utilities for TimeGAN training and data generation.
"""

import os
import numpy as np
import scipy

from TimeGan.timegan import timegan
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt #3.4.0
import seaborn as sns
def validate_synthetic_data(original_data, synthetic_data, save_dir):
    """
    Comprehensive validation of synthetic data quality through various metrics and visualizations.

    Args:
        original_data: Original minority class data (numpy array of shape [n_samples, seq_len, n_features])
        synthetic_data: Generated synthetic data (same shape as original_data)
        save_dir: Directory to save validation results and plots
    """
    os.makedirs(save_dir, exist_ok=True)

    # Open a file to write statistical results
    with open(os.path.join(save_dir, 'validation_metrics.txt'), 'w') as f:
        f.write("Synthetic Data Validation Metrics\n")
        f.write("================================\n\n")

        # Basic shape information
        f.write(f"Data Shapes:\n")
        f.write(f"Original data: {original_data.shape}\n")
        f.write(f"Synthetic data: {synthetic_data.shape}\n\n")

        # Statistical moments for each feature
        f.write("Statistical Moments per Feature:\n")
        for feature_idx in range(original_data.shape[2]):
            orig_feat = original_data[:, :, feature_idx].flatten()
            syn_feat = synthetic_data[:, :, feature_idx].flatten()

            f.write(f"\nFeature {feature_idx}:\n")
            f.write(f"Mean - Original: {np.mean(orig_feat):.4f}, Synthetic: {np.mean(syn_feat):.4f}\n")
            f.write(f"Std  - Original: {np.std(orig_feat):.4f}, Synthetic: {np.std(syn_feat):.4f}\n")
            f.write(
                f"Skew - Original: {scipy.stats.skew(orig_feat):.4f}, Synthetic: {scipy.stats.skew(syn_feat):.4f}\n")
            f.write(
                f"Kurt - Original: {scipy.stats.kurtosis(orig_feat):.4f}, Synthetic: {scipy.stats.kurtosis(syn_feat):.4f}\n")

    # Create visualizations
    _plot_feature_distributions(original_data, synthetic_data, save_dir)
    _plot_temporal_patterns(original_data, synthetic_data, save_dir)
    _plot_correlation_matrices(original_data, synthetic_data, save_dir)
    _plot_pca_analysis(original_data, synthetic_data, save_dir)



def _plot_feature_distributions(original_data, synthetic_data, save_dir):
    """Plot distribution comparisons for each feature."""
    n_features = original_data.shape[2]
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols

    plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    for i in range(n_features):
        plt.subplot(n_rows, n_cols, i + 1)

        # Plot histograms
        plt.hist(original_data[:, :, i].flatten(), bins=50, alpha=0.5,
                 density=True, label='Original', color='blue')
        plt.hist(synthetic_data[:, :, i].flatten(), bins=50, alpha=0.5,
                 density=True, label='Synthetic', color='red')

        # Add KDE plots
        sns.kdeplot(data=original_data[:, :, i].flatten(), color='blue', linewidth=2)
        sns.kdeplot(data=synthetic_data[:, :, i].flatten(), color='red', linewidth=2)

        plt.title(f'Feature {i} Distribution')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_distributions.png'))
    plt.close()


def _plot_temporal_patterns(original_data, synthetic_data, save_dir):
    """Plot temporal patterns and autocorrelations."""
    n_features = original_data.shape[2]
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols * 2  # *2 for time series and autocorr

    plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    for i in range(n_features):
        # Time series plot
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(original_data[0, :, i], label='Original', color='blue', alpha=0.7)
        plt.plot(synthetic_data[0, :, i], label='Synthetic', color='red', alpha=0.7)
        plt.title(f'Feature {i} Time Series Example')
        plt.legend()

        # Autocorrelation plot
        plt.subplot(n_rows, n_cols, i + 1 + n_features)
        orig_autocorr = np.correlate(original_data[0, :, i],
                                     original_data[0, :, i], mode='full')
        syn_autocorr = np.correlate(synthetic_data[0, :, i],
                                    synthetic_data[0, :, i], mode='full')

        max_lag = min(50, len(orig_autocorr) // 2)
        lags = range(max_lag)
        plt.plot(lags, orig_autocorr[len(orig_autocorr) // 2:len(orig_autocorr) // 2 + max_lag],
                 label='Original', color='blue')
        plt.plot(lags, syn_autocorr[len(syn_autocorr) // 2:len(syn_autocorr) // 2 + max_lag],
                 label='Synthetic', color='red')
        plt.title(f'Feature {i} Autocorrelation')
        plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'temporal_patterns.png'))
    plt.close()


def _plot_correlation_matrices(original_data, synthetic_data, save_dir):
    """Plot and compare correlation matrices."""
    # Calculate correlation matrices
    orig_corr = np.corrcoef(original_data.reshape(-1, original_data.shape[2]).T)
    syn_corr = np.corrcoef(synthetic_data.reshape(-1, synthetic_data.shape[2]).T)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(orig_corr, ax=ax1, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                annot=True, fmt='.2f', square=True)
    ax1.set_title('Original Data Correlation Matrix')

    sns.heatmap(syn_corr, ax=ax2, cmap='coolwarm', vmin=-1, vmax=1, center=0,
                annot=True, fmt='.2f', square=True)
    ax2.set_title('Synthetic Data Correlation Matrix')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'correlation_matrices.png'))
    plt.close()


def _plot_pca_analysis(original_data, synthetic_data, save_dir):
    """Perform and visualize PCA analysis."""
    # Reshape data for PCA
    orig_reshaped = original_data.reshape(-1, original_data.shape[2])
    syn_reshaped = synthetic_data.reshape(-1, synthetic_data.shape[2])

    # Combine data for PCA
    combined_data = np.vstack([orig_reshaped, syn_reshaped])

    # Perform PCA
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined_data)

    # Split back into original and synthetic
    n_orig = orig_reshaped.shape[0]
    orig_pca = combined_pca[:n_orig]
    syn_pca = combined_pca[n_orig:]

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(orig_pca[:, 0], orig_pca[:, 1], alpha=0.5, label='Original', color='blue')
    plt.scatter(syn_pca[:, 0], syn_pca[:, 1], alpha=0.5, label='Synthetic', color='red')
    plt.title('PCA Analysis of Original vs Synthetic Data')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'pca_analysis.png'))
    plt.close()

def train_time_gan(X_minority, device, method_dir, params):
    """
    Train TimeGAN and generate synthetic data.

    Args:
        X_minority: Minority class samples
        device: PyTorch device
        method_dir: Directory to save results
        params: GAN parameters

    Returns:
        Generated synthetic data
    """
    print("Training TimeGAN...")

    # Prepare data
    ori_data = np.asarray(X_minority)

    # Set up parameters
    parameters = {
        'module': params.get('module', 'gru'),
        'hidden_dim': params.get('hidden_dim', 24),
        'num_layer': params.get('num_layer', 3),
        'iterations': params.get('iterations', 1000),
        'batch_size': params.get('batch_size', 128),
        'metric_iterations': params.get('metric_iterations', 10)
    }

    # Save parameters
    with open(os.path.join(method_dir, 'timegan_params.txt'), 'w') as f:
        for key, value in parameters.items():
            f.write(f"{key}: {value}\n")

    print("Original data shape:", ori_data.shape)
    generated_data = timegan(ori_data, parameters)

    return generated_data


def generate_balanced_data_with_gan(X_train_scaled, Y_train, window_weight_train, method_dir, device):
    """
    Generate synthetic data using TimeGAN to balance the dataset.

    Args:
        X_train_scaled: Scaled training data
        Y_train: Training labels
        window_weight_train: Window weights
        method_dir: Directory to save results
        device: PyTorch device

    Returns:
        Tuple of (balanced_X, balanced_Y, balanced_weights)
    """
    print("Starting TimeGAN data generation process...")

    # Find minority class samples
    minority_indices = np.where(Y_train == 1)[0]
    majority_indices = np.where(Y_train == 0)[0]
    X_minority = X_train_scaled[minority_indices]

    # Calculate needed synthetic samples
    num_synthetic_needed = len(majority_indices) - len(minority_indices)
    print(f"Need to generate {num_synthetic_needed} synthetic samples")

    # GAN parameters
    gan_params = {
        'module': 'gru',
        'hidden_dim': 24,
        'num_layer': 3,
        'iterations': 1000,
        'batch_size': 128,
        'metric_iterations': 10
    }

    # Generate synthetic data
    synthetic_data = train_time_gan(X_minority, device, method_dir, gan_params)

    # Trim if necessary
    if len(synthetic_data) > num_synthetic_needed:
        synthetic_data = synthetic_data[:num_synthetic_needed]
    elif len(synthetic_data) < num_synthetic_needed:
        print(f"Warning: Generated only {len(synthetic_data)} samples out of {num_synthetic_needed} needed")

    # Combine data
    X_balanced = np.concatenate([X_train_scaled, synthetic_data])
    Y_balanced = np.concatenate([Y_train, np.ones(len(synthetic_data))])
    weights_balanced = np.concatenate([window_weight_train, np.ones(len(synthetic_data))])

    # Validate synthetic data
    gan_validation_dir = os.path.join(method_dir, 'gan_validation')
    os.makedirs(gan_validation_dir, exist_ok=True)
    validate_synthetic_data(X_minority, synthetic_data, gan_validation_dir)

    return X_balanced, Y_balanced, weights_balanced


