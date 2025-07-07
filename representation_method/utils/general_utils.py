"""
General utility functions.
"""

import random
import numpy as np
import pandas as pd
import torch


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
        strategy='majority'
):
    """
    Reconstructs predictions and evaluates efficiency, allowing the user to choose
    the voting strategy for overlapping windows.

    Args:
        stage2_results (list of dicts): Results from the Stage 2 loop.
        test_df (pd.DataFrame): The original, full test DataFrame.
        test_metadata (list of dicts): Metadata for all test set windows.
        strategy (str): The voting strategy to use.
                        'majority': A row is positive if it gets at least 50% of the possible votes.
                        'any': A row is positive if it gets at least one vote (OR logic).

    Returns:
        pd.DataFrame: A new DataFrame with the final predictions.
        dict: A dictionary containing efficiency and performance metrics.
    """
    if not stage2_results:
        print("Warning: No Stage 2 results to process. Returning empty DataFrame.")
        return pd.DataFrame(), {}

    if strategy not in ['majority', 'any']:
        raise ValueError("Strategy must be either 'majority' or 'any'.")

    print("\n" + "=" * 60)
    print(f"Reconstructing Final Predictions using '{strategy.upper()}' STRATEGY")
    print("=" * 60)

    # --- SETUP: Canvases for voting ---
    max_index = test_df.index.max() + 1
    prediction_canvas = np.zeros(max_index, dtype=int)  # Counts "yes" votes

    # Voter canvas is only needed for the 'majority' strategy
    if strategy == 'majority':
        voter_canvas = np.zeros(max_index, dtype=int)

    # --- VOTE ACCUMULATION ---
    for result in stage2_results:
        window_meta_index = result['window_index']
        predicted_row_relative = result['predicted_row']

        window_meta = test_metadata[window_meta_index]
        trial_df = test_df[
            (test_df['RECORDING_SESSION_LABEL'] == window_meta['participant_id']) &
            (test_df['TRIAL_INDEX'] == window_meta['trial_id'])
            ]

        window_start_abs_trial = window_meta['window_start_idx']
        window_size = 3  # Assuming window size is 3, make dynamic if needed
        absolute_indices_in_window = trial_df.index[window_start_abs_trial: window_start_abs_trial + window_size]

        # Update "yes" vote canvas (used by both strategies)
        if predicted_row_relative < len(absolute_indices_in_window):
            predicted_absolute_index = absolute_indices_in_window[predicted_row_relative]
            prediction_canvas[predicted_absolute_index] += 1

        # Update voter canvas (only for 'majority' strategy)
        if strategy == 'majority':
            voter_canvas[absolute_indices_in_window] += 1

    # --- DECISION MAKING BASED ON STRATEGY ---
    if strategy == 'majority':
        final_predictions = np.zeros(max_index, dtype=int)
        # Find all rows that were part of at least one positive window
        relevant_indices = np.where(voter_canvas > 0)[0]
        for i in relevant_indices:
            # Majority vote: "yes" votes must be >= 50% of opportunities
            if prediction_canvas[i] >= (voter_canvas[i] / 2.0):
                final_predictions[i] = 1
    else:  # 'any' strategy
        # OR logic: at least one "yes" vote is enough
        final_predictions = (prediction_canvas > 0).astype(int)

    # --- Reconstruct DataFrame and Calculate Metrics (same for both strategies) ---
    reconstructed_df = test_df.copy()
    reconstructed_df['pipeline_prediction'] = final_predictions[reconstructed_df.index]
    final_predicted_df = reconstructed_df[reconstructed_df['pipeline_prediction'] == 1].copy()

    # ... (the rest of the function for calculating metrics is identical) ...
    original_rows = len(test_df)
    predicted_rows = len(final_predicted_df)
    reduction_percentage = (1 - (predicted_rows / original_rows)) * 100 if original_rows > 0 else 0
    compression_ratio = original_rows / predicted_rows if predicted_rows > 0 else float('inf')

    true_positives = final_predicted_df['target'].sum()
    precision = true_positives / predicted_rows if predicted_rows > 0 else 0
    total_actual_positives = test_df['target'].sum()
    recall = true_positives / total_actual_positives if total_actual_positives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

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
        'original_rows': original_rows,
        'predicted_rows': predicted_rows,
        'data_reduction_percent': reduction_percentage,
        'compression_ratio': compression_ratio,
        'final_precision': precision,
        'final_recall': recall,
        'final_f1_score': f1,
    }

    return final_predicted_df, metrics