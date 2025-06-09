import os
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from utils.general_utils import seed_everything
from utils.data_loader import load_eye_tracking_data, DataConfig
from models.classifier import CombinedModel
from utils.trainers import EnsembleTrainer


def create_fold_assignments(df, n_folds=5, seed=42):
    """
    Create fold assignments for all trials in the dataframe.

    Args:
        df: DataFrame with eye tracking data
        n_folds: Number of cross-validation folds
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping (participant_id, trial_id) to fold assignment
    """
    # Get unique trials
    trials = df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).size().reset_index()
    trials = list(zip(trials['RECORDING_SESSION_LABEL'], trials['TRIAL_INDEX']))

    # Create fold assignments using KFold
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_assignments = {}

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(trials)):
        # Assign test trials to this fold
        for idx in test_idx:
            trial = trials[idx]
            fold_assignments[trial] = fold_idx

    return fold_assignments


def process_participant(participant_id, n_folds=5, window_size=100, seed=42):
    """
    Process a single participant's data with cross-validation.
    Preserves all original data points with their test/train status, predictions, and AILMENT_NUMBER.

    Args:
        participant_id: ID of the participant to process
        n_folds: Number of cross-validation folds
        window_size: Size of the sliding window
        seed: Random seed for reproducibility
    """
    seed_everything(seed)
    print(f"\n=== Processing Participant {participant_id} ===")

    # Create output directory
    output_dir = f"outputs/participant_{participant_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Load the participant's data
    csv_path = Path(__file__).parent.parent / "fwd_data" / 'Nodule_Categorized_Fixation_Data_1_18.csv'

    config = DataConfig(
        data_path=str(csv_path),
        approach_num=15,
        normalize=True,
        per_slice_target=True,
        participant_id=participant_id
    )

    df = load_eye_tracking_data(
        data_path=config.data_path,
        approach_num=config.approach_num,
        participant_id=config.participant_id,
        data_format="legacy"
    )

    # Add a unique index to identify each point
    df = df.reset_index(drop=True)
    df['point_id'] = df.index

    # Create fold assignments for trials
    fold_assignments = create_fold_assignments(df, n_folds=n_folds, seed=seed)

    # Create a dictionary to store results for each fold
    fold_results = {fold: pd.DataFrame() for fold in range(n_folds)}
    fold_metrics = []

    # Store the original dataframe columns for later reconstruction
    original_columns = df.columns.tolist()

    # Process each fold
    for fold in range(n_folds):
        print(f"\nProcessing fold {fold + 1}/{n_folds}")

        # Create train/test split based on fold assignments
        train_mask = pd.Series(False, index=df.index)
        test_mask = pd.Series(False, index=df.index)

        # Assign data points to train or test based on their trial's fold assignment
        for idx, row in df.iterrows():
            participant = row['RECORDING_SESSION_LABEL']
            trial = row['TRIAL_INDEX']
            trial_key = (participant, trial)

            if trial_key in fold_assignments and fold_assignments[trial_key] == fold:
                test_mask.loc[idx] = True
            else:
                train_mask.loc[idx] = True

        # Create train and test dataframes
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        print(f"Training data: {len(train_df)} points, Test data: {len(test_df)} points")

        train_windows, train_labels, train_point_indices = create_windows_with_indices(
            train_df, window_size=window_size)
        test_windows, test_labels, test_point_indices = create_windows_with_indices(
            test_df, window_size=window_size)

        if len(train_windows) == 0 or len(test_windows) == 0:
            print(f"Warning: Insufficient data for fold {fold}, skipping.")
            continue

        # Scale features
        scaler = StandardScaler()
        train_windows_scaled = scaler.fit_transform(
            train_windows.reshape(-1, train_windows.shape[-1])).reshape(train_windows.shape)
        test_windows_scaled = scaler.transform(
            test_windows.reshape(-1, test_windows.shape[-1])).reshape(test_windows.shape)

        # Convert to tensors
        X_train_tensor = torch.tensor(train_windows_scaled, dtype=torch.float32).permute(0, 2, 1)
        X_test_tensor = torch.tensor(test_windows_scaled, dtype=torch.float32).permute(0, 2, 1)
        y_train_tensor = torch.tensor(train_labels, dtype=torch.long)
        y_test_tensor = torch.tensor(test_labels, dtype=torch.long)

        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # Create loaders
        test_loader = DataLoader(test_dataset, batch_size=32)

        # Calculate class weights
        class_counts = np.bincount(train_labels)
        weights = 1.0 / class_counts

        # Set up ensemble trainer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_dim = X_train_tensor.shape[1]

        ensemble_save_path = os.path.join(output_dir, f'fold_{fold}_models')
        os.makedirs(ensemble_save_path, exist_ok=True)

        ensemble_trainer = EnsembleTrainer(
            base_model_class=CombinedModel,
            model_params={'input_dim': input_dim, 'output_classes': 2},
            n_models=5,
            device=device,
            save_path=ensemble_save_path
        )

        # Train ensemble
        try:
            ensemble_trainer.train_ensemble(
                train_dataset=train_dataset,
                val_loader=test_loader,
                batch_size=32,
                epochs=20,
                criterion=nn.CrossEntropyLoss(),
                optimizer_class=optim.Adam,
                optimizer_params={'lr': 0.001},
                majority_weight=weights[0] if len(weights) > 1 else 0.1
            )

            # Get predictions
            test_preds = ensemble_trainer.predict(test_loader)
            test_probs = ensemble_trainer.predict_proba(test_loader)[:, 1]  # Class 1 probabilities

            # Calculate metrics
            accuracy = accuracy_score(test_labels, test_preds)
            f1 = f1_score(test_labels, test_preds, zero_division=0)
            precision = precision_score(test_labels, test_preds, zero_division=0)
            recall = recall_score(test_labels, test_preds, zero_division=0)

            print(f"Fold {fold} Metrics:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            cm = confusion_matrix(test_labels, test_preds)
            print(f"Confusion Matrix: {cm}")
            # Store metrics
            fold_metrics.append({
                'fold': fold,
                'accuracy': float(accuracy),
                'f1': float(f1),
                'precision': float(precision),
                'recall': float(recall)
            })

            # Map window predictions back to individual points
            # First mark all test points as "test"
            test_df['set'] = 'test'

            # Initialize prediction and probability columns
            test_df['prediction'] = np.nan
            test_df['probability'] = np.nan

            # Map predictions to points
            for window_idx, point_indices in enumerate(test_point_indices):
                if window_idx < len(test_preds):
                    # Get the prediction and probability for this window
                    pred = int(test_preds[window_idx])
                    prob = float(test_probs[window_idx])

                    # Assign to all points in the window
                    for point_idx in point_indices:
                        if point_idx in test_df.index:
                            test_df.loc[point_idx, 'prediction'] = pred
                            test_df.loc[point_idx, 'probability'] = prob

            # Mark all training points as "train"
            train_df['set'] = 'train'
            train_df['prediction'] = np.nan
            train_df['probability'] = np.nan

            # Store the fold assignment
            test_df['fold'] = fold
            train_df['fold'] = fold

            # Combine results - this preserves all original columns including AILMENT_NUMBER
            fold_df = pd.concat([train_df, test_df])

            # Store in the results dictionary
            fold_results[fold] = fold_df

            # Save to CSV - this will include AILMENT_NUMBER
            fold_df.to_csv(f"{output_dir}/fold_{fold}_results.csv", index=False)

        except Exception as e:
            print(f"Error in fold {fold}: {str(e)}")
            continue

    # Combine all fold results
    all_results = pd.concat([fold_results[fold] for fold in range(n_folds) if not fold_results[fold].empty])

    # Create a "combined" dataframe that includes all points with their fold assignments
    # This will automatically include AILMENT_NUMBER since it's in the original columns
    combined_df = all_results[original_columns + ['fold', 'set', 'prediction', 'probability']]
    combined_df.to_csv(f"{output_dir}/all_folds_results.csv", index=False)

    # Calculate average metrics
    if fold_metrics:
        metrics_df = pd.DataFrame(fold_metrics)
        avg_metrics = {
            'participant_id': participant_id,
            'accuracy_mean': metrics_df['accuracy'].mean(),
            'accuracy_std': metrics_df['accuracy'].std(),
            'f1_mean': metrics_df['f1'].mean(),
            'f1_std': metrics_df['f1'].std(),
            'precision_mean': metrics_df['precision'].mean(),
            'precision_std': metrics_df['precision'].std(),
            'recall_mean': metrics_df['recall'].mean(),
            'recall_std': metrics_df['recall'].std()
        }

        print(f"\nParticipant {participant_id} Average Metrics:")
        print(f"  Accuracy: {avg_metrics['accuracy_mean']:.4f} ± {avg_metrics['accuracy_std']:.4f}")
        print(f"  F1 Score: {avg_metrics['f1_mean']:.4f} ± {avg_metrics['f1_std']:.4f}")
        print(f"  Precision: {avg_metrics['precision_mean']:.4f} ± {avg_metrics['precision_std']:.4f}")
        print(f"  Recall: {avg_metrics['recall_mean']:.4f} ± {avg_metrics['recall_std']:.4f}")

        with open(f"{output_dir}/avg_metrics.json", 'w') as f:
            json.dump(avg_metrics, f, indent=4)

        return avg_metrics

    return None


def create_windows_with_indices(df, feature_columns=None, window_size=100):
    """
    Create windows from dataframe while keeping track of original data point indices.
    AILMENT_NUMBER is not included in the features but is preserved in the original dataframe.

    Args:
        df: DataFrame with eye tracking data (includes AILMENT_NUMBER)
        feature_columns: List of feature columns to use for training (excludes AILMENT_NUMBER)
        window_size: Size of the sliding window

    Returns:
        windows: Array of shape [n_windows, window_size, n_features] (only training features)
        labels: Array of shape [n_windows]
        point_indices: List of lists containing the original point indices for each window
    """
    if feature_columns is None:
        # Define feature columns for training - explicitly exclude AILMENT_NUMBER
        feature_columns = [
            'Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X',
            'CURRENT_FIX_IA_Y', 'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT'
        ]

    windows = []
    labels = []
    point_indices = []

    # Group by trial
    for (participant_id, trial_id), trial_df in df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']):
        trial_length = len(trial_df)

        # Skip if trial is shorter than window_size
        if trial_length < window_size:
            continue

        # Reset index to access the original indices
        trial_df = trial_df.reset_index()

        for start_idx in range(0, trial_length - window_size + 1):
            end_idx = start_idx + window_size
            window = trial_df.iloc[start_idx:end_idx]

            # Extract ONLY the training features for this window (excludes AILMENT_NUMBER)
            window_features = window[feature_columns].values

            # Check if this window contains a target
            has_target = (window['target'] == 1).any()

            # Store the window, label, and original point indices
            windows.append(window_features)
            labels.append(int(has_target))
            point_indices.append(window['index'].tolist())

    if not windows:
        return np.array([]), np.array([]), []

    return np.array(windows), np.array(labels), point_indices


def main():
    """Run cross-validation for all participants"""
    os.makedirs("outputs", exist_ok=True)

    all_participant_metrics = []

    # Process each participant
    for participant_id in range(1, 2):
        try:
            avg_metrics = process_participant(
                participant_id=participant_id,
                n_folds=5,
                window_size=100,
                seed=0
            )

            if avg_metrics:
                all_participant_metrics.append(avg_metrics)

        except Exception as e:
            print(f"Error processing participant {participant_id}: {str(e)}")
            continue

    # Calculate overall metrics
    if all_participant_metrics:
        metrics_df = pd.DataFrame(all_participant_metrics)

        overall_metrics = {
            'accuracy_mean': metrics_df['accuracy_mean'].mean(),
            'accuracy_std': metrics_df['accuracy_mean'].std(),
            'f1_mean': metrics_df['f1_mean'].mean(),
            'f1_std': metrics_df['f1_mean'].std(),
            'precision_mean': metrics_df['precision_mean'].mean(),
            'precision_std': metrics_df['precision_mean'].std(),
            'recall_mean': metrics_df['recall_mean'].mean(),
            'recall_std': metrics_df['recall_mean'].std()
        }

        print("\n=== Overall Average Metrics Across All Participants ===")
        print(f"Accuracy: {overall_metrics['accuracy_mean']:.4f} ± {overall_metrics['accuracy_std']:.4f}")
        print(f"F1 Score: {overall_metrics['f1_mean']:.4f} ± {overall_metrics['f1_std']:.4f}")
        print(f"Precision: {overall_metrics['precision_mean']:.4f} ± {overall_metrics['precision_std']:.4f}")
        print(f"Recall: {overall_metrics['recall_mean']:.4f} ± {overall_metrics['recall_std']:.4f}")

        with open("outputs/overall_avg_metrics.json", 'w') as f:
            json.dump(overall_metrics, f, indent=4)

        # Save all participant metrics
        metrics_df.to_csv("outputs/all_participant_metrics.csv", index=False)


if __name__ == "__main__":
    main()

    # Approach 15
    # Participant 1 Average Metrics:
    #   Accuracy: 0.3571 ± 0.0423
    #   F1 Score: 0.2680 ± 0.0541
    #   Precision: 0.1644 ± 0.0427
    #   Recall: 0.7919 ± 0.1176