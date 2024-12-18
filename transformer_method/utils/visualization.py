import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_confusion_matrix(conf_matrix, save_path):
    plt.figure(figsize=(10, 8))

    # Calculate percentages
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations
    annotations = np.asarray([
        [f'{count}\n({percent:.1f}%)'
         for count, percent in zip(row_counts, row_percentages)]
        for row_counts, row_percentages in zip(conf_matrix, conf_matrix_percent)
    ])

    sns.heatmap(
        conf_matrix,
        annot=annotations,
        fmt='',
        cmap='Blues',
        square=True,
        cbar=True,
        cbar_kws={'label': 'Count'}
    )

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.xticks([0.5, 1.5], ['Negative (0)', 'Positive (1)'])
    plt.yticks([0.5, 1.5], ['Negative (0)', 'Positive (1)'])

    # Add metrics text
    metrics = calculate_metrics_from_confusion_matrix(conf_matrix)
    metrics_text = '\n'.join([f'{k}: {v:.3f}' for k, v in metrics.items()])
    plt.text(2.5, 0.5, metrics_text, bbox=dict(facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def plot_training_history(train_history, val_history, save_dir):
    save_dir = Path(save_dir)

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot([x['loss'] for x in train_history], label='Train')
    if val_history:
        plt.plot([x['loss'] for x in val_history], label='Validation')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_dir / 'loss_history.png')
    plt.close()

    # Plot metrics
    metrics = ['acc', 'precision', 'recall', 'f1']
    for metric in metrics:
        plt.figure(figsize=(10, 5))
        plt.plot([x[metric] for x in train_history], label=f'Train {metric}')
        if val_history:
            plt.plot([x[metric] for x in val_history], label=f'Val {metric}')
        plt.title(f'{metric.capitalize()} History')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.savefig(save_dir / f'{metric}_history.png')
        plt.close()