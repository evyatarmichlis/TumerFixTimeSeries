import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def calculate_metrics(outputs, targets):
    """Calculate all classification metrics"""
    preds = np.argmax(outputs, axis=1)
    conf_matrix = confusion_matrix(targets, preds)

    metrics = {
        'acc': (preds == targets).mean(),
        'precision': precision_score(targets, preds, zero_division=0),
        'recall': recall_score(targets, preds, zero_division=0),
        'f1': f1_score(targets, preds, zero_division=0),
        'confusion_matrix': conf_matrix
    }

    # Add detailed metrics from confusion matrix
    cm_metrics = calculate_metrics_from_confusion_matrix(conf_matrix)
    metrics.update(cm_metrics)

    return metrics


def calculate_metrics_from_confusion_matrix(conf_matrix):
    """Calculate detailed metrics from confusion matrix"""
    total = conf_matrix.sum()
    tn, fp, fn, tp = conf_matrix.ravel()

    metrics = {
        'accuracy': (tp + tn) / total,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
    }

    metrics['f1'] = 2 * (metrics['precision'] * metrics['sensitivity']) / \
                    (metrics['precision'] + metrics['sensitivity']) \
        if (metrics['precision'] + metrics['sensitivity']) > 0 else 0

    return metrics