"""
General utility functions.
"""

import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def seed_everything(seed):
    """
    Set seeds for reproducibility.

    Args:
        seed: Seed value for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)




def reconstruct_and_evaluate_efficiency(
        stage2_results,
        test_df,
        test_metadata,
        strategy='majority',
        default_window_size: int = 3,
):
    """
    Reconstructs predictions and evaluates efficiency with selectable voting strategy.

    strategy ∈ {
        'majority'           : row positive if >= 50% of covering windows voted for it (unweighted)
        'any'                : row positive if ≥ 1 window voted for it (OR)
        'all'                : row positive only if *every* covering window voted for it
        'majority_weighted'  : like 'majority', but each window contributes a weight equal to its
                               per-window frequency feature if available (else 1)
    }
    """
    if not stage2_results:
        print("Warning: No Stage 2 results to process. Returning empty DataFrame.")
        return pd.DataFrame(), {}

    valid_strategies = {'majority', 'any', 'all', 'majority_weighted'}
    if strategy not in valid_strategies:
        raise ValueError(f"Strategy must be one of {sorted(valid_strategies)}.")

    print("\n" + "=" * 60)
    print(f"Reconstructing Final Predictions using '{strategy.upper()}' STRATEGY")
    print("=" * 60)

    # Choose a weight column if present (supports both the new and old name)
    weight_col = None
    if 'WINDOW_MAX_SLICE_FREQ' in test_df.columns:
        weight_col = 'WINDOW_MAX_SLICE_FREQ'
    elif 'WINDOW_UNIQUE_SLICES' in test_df.columns:
        weight_col = 'WINDOW_UNIQUE_SLICES'

    # Canvases
    max_index = int(test_df.index.max()) + 1
    prediction_canvas = np.zeros(max_index, dtype=float)   # yes-votes (possibly weighted)
    voter_canvas = np.zeros(max_index, dtype=int)          # number of covering windows
    voter_weight_canvas = np.zeros(max_index, dtype=float) # sum of window weights (for weighted majority)

    # ----- accumulate votes -----
    for result in stage2_results:
        w_idx = result['window_index']
        pred_rel = int(result['predicted_row'])

        wm = test_metadata[w_idx]
        wlen = int(wm.get('window_size', wm.get('window_length', default_window_size)))

        # rows from this trial
        trial_df = test_df[
            (test_df['RECORDING_SESSION_LABEL'] == wm['participant_id']) &
            (test_df['TRIAL_INDEX'] == wm['trial_id'])
        ]

        start_abs = int(wm['window_start_idx'])
        abs_idx_in_window = trial_df.index[start_abs: start_abs + wlen]
        if len(abs_idx_in_window) == 0:
            continue

        # window weight (used only in majority_weighted)
        if strategy == 'majority_weighted' and weight_col is not None:
            try:
                win_weight = float(test_df.loc[abs_idx_in_window[-1], weight_col])
            except Exception:
                win_weight = 1.0
        else:
            win_weight = 1.0

        # every covered row gets an “opportunity”
        voter_canvas[abs_idx_in_window] += 1
        if strategy == 'majority_weighted':
            voter_weight_canvas[abs_idx_in_window] += win_weight

        # positive vote for the predicted row
        if 0 <= pred_rel < len(abs_idx_in_window):
            pred_abs = abs_idx_in_window[pred_rel]
            if strategy == 'majority_weighted':
                prediction_canvas[pred_abs] += win_weight
            else:
                prediction_canvas[pred_abs] += 1.0

    # ----- decide -----
    if strategy == 'majority':
        final_predictions = np.zeros(max_index, dtype=int)
        relevant = np.nonzero(voter_canvas > 0)[0]
        half = voter_canvas / 2.0
        final_predictions[relevant] = (prediction_canvas[relevant] >= half[relevant]).astype(int)
    elif strategy == 'majority_weighted':
        final_predictions = np.zeros(max_index, dtype=int)
        relevant = np.nonzero(voter_weight_canvas > 0)[0]
        half_weight = voter_weight_canvas / 2.0
        final_predictions[relevant] = (prediction_canvas[relevant] >= half_weight[relevant]).astype(int)
    elif strategy == 'any':
        final_predictions = (prediction_canvas > 0).astype(int)
    else:  # 'all'
        final_predictions = np.zeros(max_index, dtype=int)
        relevant = np.nonzero(voter_canvas > 0)[0]
        final_predictions[relevant] = (prediction_canvas[relevant] == voter_canvas[relevant]).astype(int)

    # ----- reconstruct & metrics -----
    reconstructed_df = test_df.copy()
    reconstructed_df['pipeline_prediction'] = final_predictions[reconstructed_df.index]
    final_predicted_df = reconstructed_df[reconstructed_df['pipeline_prediction'] == 1].copy()

    original_rows = len(test_df)
    predicted_rows = len(final_predicted_df)
    reduction_percentage = (1 - (predicted_rows / original_rows)) * 100 if original_rows > 0 else 0
    compression_ratio = (original_rows / predicted_rows) if predicted_rows > 0 else float('inf')

    true_positives = float(final_predicted_df['target'].sum()) if predicted_rows > 0 else 0.0
    precision = (true_positives / predicted_rows) if predicted_rows > 0 else 0.0
    total_actual_positives = float(test_df['target'].sum()) if original_rows > 0 else 0.0
    recall = (true_positives / total_actual_positives) if total_actual_positives > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print(f"Original test data had {original_rows:,} rows.")
    print(f"Pipeline flagged {predicted_rows:,} specific rows as targets.")
    print(f"Data Reduction: {reduction_percentage:.2f}%")
    print(f"Compression Ratio: {compression_ratio:.2f}x")

    print(f"\n--- Final End-to-End Performance ('{strategy.upper()}' Strategy) ---")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    metrics = {
        'strategy': strategy,
        'original_rows': int(original_rows),
        'predicted_rows': int(predicted_rows),
        'data_reduction_percent': float(reduction_percentage),
        'compression_ratio': float(compression_ratio),
        'final_precision': float(precision),
        'final_recall': float(recall),
        'final_f1_score': float(f1),
    }

    return final_predicted_df, metrics



def evaluate_reconstructed_predictions(y_true, y_pred, strategy_name):
    """
    Calculates and prints performance metrics for the reconstructed row-level predictions.
    This evaluates the end-to-end pipeline performance after Stage 2.

    Args:
        y_true (pd.Series or np.array): Ground truth labels (0 or 1).
        y_pred (pd.Series or np.array): Predicted labels (0 or 1).
        strategy_name (str): The name of the reconstruction strategy (e.g., 'Majority', 'Any').
    """
    # Ensure a 2x2 confusion matrix by specifying labels, handles cases with no positives.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n--- Final Row-Level Performance (Strategy: {strategy_name}) ---")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f} (Of all rows flagged as targets, how many were correct?)")
    print(f"  Recall:    {recall:.4f} (Of all true target rows, how many were found?)")
    print(f"  F1 Score:  {f1:.4f}")

    print("\n  Confusion Matrix:")
    print("             Predicted 0   Predicted 1")
    print(f"  Actual 0   {cm[0, 0]:<13d} {cm[0, 1]:<13d}")
    print(f"  Actual 1   {cm[1, 0]:<13d} {cm[1, 1]:<13d}")
    print("-" * 60)
