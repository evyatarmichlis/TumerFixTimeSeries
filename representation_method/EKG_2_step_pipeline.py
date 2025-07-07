import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import math
from pathlib import Path

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from representation_method.models.classifier import GradientLocalizer
from representation_method.models.self_supervised_transformer import TargetLocalizer
from representation_method.utils.data_utils import split_train_test_for_time_series
from representation_method.utils.trainers import EnsembleTrainer
from transformer_method.utils.metrics import calculate_metrics
from representation_method.utils.general_utils import seed_everything, reconstruct_and_evaluate_efficiency
import pandas as pd
import numpy as np






class CNN1DModel(nn.Module):
    def __init__(self, input_dim, window_size, output_classes=2):
        """
        A robust 1D CNN model that uses Adaptive Pooling to handle
        various window sizes without crashing.
        """
        super(CNN1DModel, self).__init__()

        # We don't need window_size for the architecture anymore, but it's good practice to keep it
        self.window_size = window_size

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),  # BatchNorm helps stabilize training
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=16)  # Guarantees output length is 16
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=8)  # Guarantees output length is 8
        )

        # The flattened size is now predictable and fixed, regardless of window_size
        flattened_size = 64 * 8

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased dropout for more regularization
            nn.Linear(128, output_classes)
        )

    def forward(self, x):
        # Input x has shape (batch_size, input_dim, window_size)
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out = self.classifier(out)
        return out



def create_dynamic_time_series_for_ekg_with_ailment(df, feature_columns, window_size=100, window_step=1):
        windows, labels, metadata, ailment_locations = [], [], [], []

        # Group by session and trial to process each time series independently
        for (participant_id, trial_id), trial_df in df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']):
            trial_length = len(trial_df)
            trial_df = trial_df.reset_index(drop=True)

            for start_idx in range(0, trial_length - window_size + 1, window_step):
                end_idx = start_idx + window_size
                window = trial_df.iloc[start_idx:end_idx]

                if len(window) < window_size: continue

                target_array = window['target'].values
                window_target_positions = np.where(target_array == 1)[0]
                window_features = window[feature_columns].values

                # Ensure 'AILMENT_NUMBER' column exists, default to -1 if not
                ailment_numbers = window['AILMENT_NUMBER'].values if 'AILMENT_NUMBER' in window.columns else np.full(
                    window_size, -1)
                window_ailment_positions = np.where(ailment_numbers != -1)[0]
                valid_ailments = ailment_numbers[ailment_numbers != -1]
                unique_ailments = list(set(valid_ailments))

                windows.append(window_features)
                labels.append(int(len(window_target_positions) > 0))
                ailment_locations.append(window_ailment_positions)

                window_meta = {
                    'participant_id': participant_id, 'trial_id': trial_id,
                    'window_start_idx': start_idx, 'window_end_idx': end_idx,
                    'target_positions': window_target_positions.tolist(),
                    'has_target': int(len(window_target_positions) > 0),
                    'ailment_positions': window_ailment_positions.tolist(),
                    'ailment_numbers': unique_ailments,
                'has_valid_ailment': len(unique_ailments) > 0,
                'ailment_details': [{'relative_position': pos, 'ailment_number': str(ailment)}
                                    for pos, ailment in enumerate(ailment_numbers) if ailment != -1]
                }
                metadata.append(window_meta)

        return np.array(windows), np.array(labels), metadata, np.array(ailment_locations, dtype=object)

# --- [UNCHANGED] Evaluation Functions ---
# These functions are for post-processing and do not need to be changed.
# track_ailments_in_predictions, modified_stage2_evaluation_with_ailment_tracking, etc.
# (Your original evaluation functions go here)
# ... (omitted for brevity, but they should be included in your final script)
def track_ailments_in_predictions(predicted_positions, window_metadata, tolerance=1):
    """
    Track which ailments are found by the model's predictions within a window

    Args:
        predicted_positions: List of positions where model predicts targets
        window_metadata: Metadata containing ailment information for this window
        tolerance: Tolerance for matching predicted positions to ailment positions

    Returns:
        Dictionary with ailment tracking results
    """
    ailment_results = {
        'ailments_in_window': set(),
        'ailments_found_by_model': set(),
        'ailment_positions_found': [],
        'total_ailment_positions_in_window': 0,
        'ailment_positions_matched': 0,
        'ailment_detection_success': False
    }

    # Extract ailment information from metadata
    if not window_metadata.get('has_valid_ailment', False):
        return ailment_results

    ailment_positions = window_metadata.get('ailment_positions', [])
    ailment_numbers = window_metadata.get('ailment_numbers', [])
    ailment_details = window_metadata.get('ailment_details', [])

    # Track all ailments present in this window
    ailment_results['ailments_in_window'] = set(ailment_numbers)
    ailment_results['total_ailment_positions_in_window'] = len(ailment_positions)

    # Check which ailment positions are matched by model predictions
    matched_ailment_positions = set()
    found_ailments = set()

    for pred_pos in predicted_positions:
        for i, ailment_pos in enumerate(ailment_positions):
            if abs(pred_pos - ailment_pos) <= tolerance:
                matched_ailment_positions.add(ailment_pos)
                # Find which ailment number this position corresponds to
                for detail in ailment_details:
                    if detail['relative_position'] == ailment_pos:
                        found_ailments.add(detail['ailment_number'])
                        break

    ailment_results['ailment_positions_matched'] = len(matched_ailment_positions)
    ailment_results['ailments_found_by_model'] = found_ailments
    ailment_results['ailment_positions_found'] = list(matched_ailment_positions)
    ailment_results['ailment_detection_success'] = len(found_ailments) > 0

    return ailment_results

def modified_stage2_evaluation_with_ailment_tracking(transformer_model, tokenizer, feature_columns,
                                                     eval_windows, eval_metadata, device):
    """
    Modified Stage 2 evaluation that tracks ailments found by the model
    """
    localizer = TargetLocalizer(transformer_model, device)

    print("Evaluating target localization with ailment tracking...")
    all_results = []

    # Track overall ailment statistics
    total_ailments_in_eval_windows = set()
    ailments_found_by_stage2 = set()
    windows_with_successful_ailment_detection = 0
    total_ailment_positions_in_eval = 0
    ailment_positions_found_by_stage2 = 0

    for window, meta in zip(eval_windows, eval_metadata):
        if meta.get('has_valid_ailment', False):
            total_ailments_in_eval_windows.update(meta.get('ailment_numbers', []))
            total_ailment_positions_in_eval += len(meta.get('ailment_positions', []))

        # Tokenize and run model
        window_df = pd.DataFrame(window, columns=feature_columns)
        tokenized_window = tokenizer.tokenize(window_df, feature_columns)
        window_tensor = torch.tensor(tokenized_window, dtype=torch.long).unsqueeze(0).to(device)

        # Get standard localize_targets results first
        results = localizer.localize_targets(
            window_tensor,
            meta['target_positions'],  # Ground truth target positions
            0,
            window_metadata=meta,
            supervised=True
        )

        ailment_results = track_ailments_in_predictions(
            results.get('top_k_positions', []), meta, tolerance=1
        )

        # Merge results
        results.update(ailment_results)

        # Track ailments found
        found_ailments = results.get('ailments_found_by_model', set())
        ailments_found_by_stage2.update(found_ailments)

        if results.get('ailment_detection_success', False):
            windows_with_successful_ailment_detection += 1

        ailment_positions_found_by_stage2 += results.get('ailment_positions_matched', 0)

        # Add metadata for tracking
        results.update({
            'participant_id': meta['participant_id'],
            'trial_id': meta['trial_id'],
            'window_has_ailments': meta.get('has_valid_ailment', False),
            'ailments_in_window': list(meta.get('ailment_numbers', [])),
        })

        all_results.append(results)

    # Calculate Stage 2 ailment detection metrics
    stage2_ailment_detection_rate = len(ailments_found_by_stage2) / len(
        total_ailments_in_eval_windows) if total_ailments_in_eval_windows else 0
    stage2_ailment_position_precision = ailment_positions_found_by_stage2 / total_ailment_positions_in_eval if total_ailment_positions_in_eval > 0 else 0

    print(f"\nStage 2 Ailment Detection Results:")
    print(f"  Total unique ailments in eval windows: {len(total_ailments_in_eval_windows)}")
    print(f"  Ailments found by Stage 2: {len(ailments_found_by_stage2)}")
    print(f"  Found ailments: {sorted(list(ailments_found_by_stage2))}")

    # Calculate traditional target detection metrics
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        target_metrics = calculate_metrics(results_df)
    else:
        target_metrics = {'precision': 0, 'recall': 0, 'f1': 0}

    return {
        'target_metrics': target_metrics,
        'ailment_metrics': {
            'total_ailments_in_eval': len(total_ailments_in_eval_windows),
            'ailments_found': len(ailments_found_by_stage2),
            'detection_rate': stage2_ailment_detection_rate,
            'found_ailments': sorted(list(ailments_found_by_stage2)),
            'successful_windows': windows_with_successful_ailment_detection,
            'ailment_position_precision': stage2_ailment_position_precision,
        },
        'detailed_results': all_results
    }

# --- [NEW] Main Pipeline for EKG Data ---
def ekg_two_step_pipeline(participant_id, window_size=50, seed=42):
    """
    Two-step pipeline adapted for EKG data with AILMENT_NUMBER tracking.
    """
    print(f"Running EKG Two-Step Pipeline for Participant {participant_id}")
    print(f"Window size: {window_size}, Seed: {seed}")
    print("=" * 60)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    feature_columns = [
        'Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_INDEX',
        'CURRENT_FIX_COMPONENT_COUNT',  'rolling_mean_10',
    'rolling_std_10',
    'signal_derivative'
    ]
    input_columns = ['CURRENT_FIX_INDEX', 'Pupil_Size','CURRENT_FIX_DURATION','AILMENT_NUMBER',    'rolling_mean_10',
    'rolling_std_10',
    'signal_derivative']

    print(f"Training features being used: {feature_columns}")

    # 1. --- MODIFIED: Load and preprocess EKG data ---
    csv_path = Path(__file__).parent.parent / "EKG data" / "ML_ECG_Data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found at: {csv_path}")

    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path, engine='python', on_bad_lines='skip')

    # Define the target variable based on 'LOCATION_TYPE'
    df['target'] = np.where(df['LOCATION_TYPE'] == 'MI_HIT', 1, 0)

    # Ensure AILMENT_NUMBER column exists, fill with -1 if not
    print("Cleaning 'AILMENT_NUMBER' column...")
    if 'AILMENT_NUMBER' in df.columns:
        # Step 1: Force non-numeric values (like '.') into NaN (Not a Number).
        df['AILMENT_NUMBER'] = pd.to_numeric(df['AILMENT_NUMBER'], errors='coerce')

        # Step 2: Fill all NaN values (which now include the original '.' and any others) with -1.
        df['AILMENT_NUMBER'] = df['AILMENT_NUMBER'].fillna(-1)

        # Step 3: Convert the column to a clean integer type.
        df['AILMENT_NUMBER'] = df['AILMENT_NUMBER'].astype(int)
        print("Cleaning complete. '.' and other non-numeric ailments are now marked as -1.")
    else:
        print("Warning: 'AILMENT_NUMBER' column not found. Ailment tracking will be disabled.")
        df['AILMENT_NUMBER'] = -1

    # df = df[df['RECORDING_SESSION_LABEL'] == participant_id].copy()
    if df.empty:
        print(f"No data found for participant {participant_id}. Exiting.")
        return

    SIGNAL_COL = 'Pupil_Size'

    # Calculate rolling statistics
    df['rolling_mean_10'] = df[SIGNAL_COL].rolling(window=10, min_periods=1).mean()
    df['rolling_std_10'] = df[SIGNAL_COL].rolling(window=10, min_periods=1).std()

    # Calculate rate of change (derivative)
    df['signal_derivative'] = df[SIGNAL_COL].diff().fillna(0)

    # Fill any potential NaN values created by rolling std
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    train_df, test_df = split_train_test_for_time_series(df, test_size=0.2, random_state=seed,input_columns=input_columns)
    train_df, val_df = split_train_test_for_time_series(train_df, test_size=0.2, random_state=seed,input_columns=input_columns)
    print(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    test_ailments = test_df[test_df['AILMENT_NUMBER'] != -1]['AILMENT_NUMBER'].unique()
    total_unique_ailments_in_test = len(test_ailments)
    print(f"Total unique ailments in test set: {total_unique_ailments_in_test} -> {sorted(test_ailments)}")
    window_step = 3
    # 2. Create windows using the general function
    X_train, Y_train, train_metadata, _ = create_dynamic_time_series_for_ekg_with_ailment(
        train_df, feature_columns, window_size=window_size,window_step=window_step
    )
    X_val, Y_val, val_metadata, _ = create_dynamic_time_series_for_ekg_with_ailment(
        val_df, feature_columns, window_size=window_size,window_step = window_step
    )
    X_test, Y_test, test_metadata, test_ailment_locations = create_dynamic_time_series_for_ekg_with_ailment(
        test_df, feature_columns, window_size=window_size,window_step = window_step
    )
    print(f"Windows created - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create save directory
    save_dir = f'results/ekg_ailment_tracking_participant_{participant_id}'
    os.makedirs(save_dir, exist_ok=True)

    # ===== STEP 1: WINDOW-LEVEL PREDICTION =====
    print("\nSTEP 1: Window-Level Prediction")
    print("-" * 40)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).permute(0, 2, 1)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).permute(0, 2, 1)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    test_loader = DataLoader(TensorDataset(X_test_tensor, torch.tensor(Y_test, dtype=torch.long)), batch_size=32,
                             shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # --- MODIFIED: Instantiate the new dynamic model ---
    ensemble_trainer = EnsembleTrainer(
        base_model_class=CNN1DModel,  # <-- CHANGE THIS
        model_params={'input_dim': X_train_tensor.shape[1], 'window_size': window_size, 'output_classes': 2},
        n_models=10,
        device=device,
        save_path=os.path.join(save_dir, 'ensemble_models')
    )
    class_counts = np.bincount(Y_train)
    print(np.bincount(Y_test))
    weights = 1.0 / class_counts
    # Train ensemble
    print("Training ensemble for Stage 1...")
    ensemble_trainer.train_ensemble(
        train_dataset=train_dataset,
        val_loader=val_loader,
        batch_size=32,
        epochs=300,
        criterion=nn.CrossEntropyLoss(),
        optimizer_class=optim.Adam,
        optimizer_params={'lr': 0.001},
        majority_weight=weights[0] if len(weights) > 1 else 0.1
    )


    # Evaluate Step 1
    best_threshold = ensemble_trainer.find_best_threshold(val_loader)

    test_predictions = ensemble_trainer.predict(test_loader, threshold=best_threshold)
    step1_accuracy = accuracy_score(Y_test, test_predictions)
    step1_precision = precision_score(Y_test, test_predictions, zero_division=0)
    step1_recall = recall_score(Y_test, test_predictions, zero_division=0)
    step1_cm = confusion_matrix(Y_test, test_predictions)
    # Calculate Step 1 metrics
    step1_f1 = f1_score(Y_test, test_predictions, zero_division=0)
    print(f"\nStep 1 F1 Score: {step1_f1:.4f}")
    print(f"\nStep 1 Results:")
    print(f"  Accuracy: {step1_accuracy:.4f}")
    print(f"  Precision: {step1_precision:.4f}")
    print(f"  Recall: {step1_recall:.4f}")
    print(f"  F1 Score: {step1_f1:.4f}")
    print(f"\nConfusion Matrix:")
    print("   Predicted 0  Predicted 1")
    print(f"Actual 0   {step1_cm[0, 0]:<10} {step1_cm[0, 1]:<10}")
    print(f"Actual 1   {step1_cm[1, 0]:<10} {step1_cm[1, 1]:<10}")
    # Track ailments found by Stage 1
    step1_positive_indices = np.where(test_predictions == 1)[0]
    step1_ailments_found = set()
    for idx in step1_positive_indices:
        if test_metadata[idx]['has_valid_ailment']:
            step1_ailments_found.update(test_metadata[idx]['ailment_numbers'])

    print(f"Stage 1 found {len(step1_ailments_found)} unique ailments out of {total_unique_ailments_in_test}")

    positive_indices = np.where(test_predictions == 1)[0]
    if len(positive_indices) == 0:
        print("No positive windows for Stage 2.")
    else:
        # Use the first model from the trained ensemble for localization
        stage1_model_for_loc = ensemble_trainer.models[0]
        localizer = GradientLocalizer(stage1_model_for_loc, device)

        all_stage2_results = []

        for idx in positive_indices:
            # Get the original window data and metadata
            window_metadata = test_metadata[idx]
            # Get the scaled tensor that was fed to the model
            window_tensor = X_test_tensor[idx].unsqueeze(0)

            # Use the localizer to predict the row index
            predicted_row = localizer.localize_target(window_tensor.clone())  # Use clone to avoid grad issues

            # --- Evaluate the prediction ---
            true_target_positions = window_metadata.get('target_positions', [])

            # Check if the predicted row is one of the actual targets
            is_correct = 1 if predicted_row in true_target_positions else 0

            # Track which ailment (if any) was at the predicted location
            ailment_found = 'None'
            for detail in window_metadata.get('ailment_details', []):
                if detail['relative_position'] == predicted_row:
                    ailment_found = detail['ailment_number']
                    break

            all_stage2_results.append({
                'window_index': idx,
                'predicted_row': predicted_row,
                'true_targets': true_target_positions,
                'is_correct': is_correct,
                'ailment_found': ailment_found,
                'ailments_in_window': window_metadata.get('ailment_numbers', [])
            })

        # --- Calculate Stage 2 Metrics ---
        results_df = pd.DataFrame(all_stage2_results)
        true_positive_windows_df = results_df[results_df['true_targets'].apply(len) > 0]
        if not true_positive_windows_df.empty:
            stage2_accuracy = true_positive_windows_df['is_correct'].mean()
            print(f"Stage 2 Pinpointing Accuracy (on TP windows): {stage2_accuracy:.4f}")

        found_ailments = set(results_df[results_df['ailment_found'] != 'None']['ailment_found'])
        print(f"Stage 2 found ailments: {sorted(list(found_ailments))}")
        if all_stage2_results:
            # Call the new function to get the final results
            final_df, efficiency_metrics = reconstruct_and_evaluate_efficiency(
                all_stage2_results,
                test_df,
                test_metadata ,
            strategy = 'majority'

            )

            print("\n--- Sample of Final Predicted DataFrame ---")
            print(final_df.head())


if __name__ == "__main__":
    # --- MODIFIED: Set parameters for the EKG data ---
    # Use a smaller window size appropriate for EKG signals
    window_size = 3
    seed = 42

    # Use the participant ID format from the new dataset
    participant_id = None

    try:
        # Call the new pipeline function
        ekg_two_step_pipeline(participant_id, window_size, seed)
        print("\nSUCCESS: EKG pipeline completed!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()