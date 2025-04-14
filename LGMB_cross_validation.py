import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import lightgbm as lgbm

from representation_method.utils.general_utils import seed_everything
from representation_method.utils.data_loader import load_eye_tracking_data, DataConfig


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


def process_participant(participant_id, n_folds=5, seed=42):
    """
    Process a single participant's data with cross-validation using LightGBM.
    Each data point is treated as an individual sample.

    Args:
        participant_id: ID of the participant to process
        n_folds: Number of cross-validation folds
        seed: Random seed for reproducibility
    """
    seed_everything(seed)
    print(f"\n=== Processing Participant {participant_id} ===")

    # Create output directory
    output_dir = f"lgbm_group_outputs/participant_{participant_id}"
    os.makedirs(output_dir, exist_ok=True)

    # Load the participant's data
    config = DataConfig(
        data_path='data/Categorized_Fixation_Data_1_18.csv',
        approach_num=6,
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

    # Define feature columns
    feature_columns = [
        'Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X',
        'CURRENT_FIX_IA_Y', 'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT'
    ]

    # Check which columns actually exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]

    print(f"Using features: {available_features}")

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

        # Extract features and labels
        X_train = train_df[available_features].values
        y_train = train_df['target'].values

        X_test = test_df[available_features].values
        y_test = test_df['target'].values

        if len(X_train) == 0 or len(X_test) == 0:
            print(f"Warning: Insufficient data for fold {fold}, skipping.")
            continue

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Create and train LightGBM model with specified parameters
        lgbm_model = lgbm.LGBMClassifier(
            num_leaves=300,
            learning_rate=0.1,
            min_child_samples=44,
            n_estimators=100,
            random_state=seed,
            class_weight='balanced',
            verbose=-1
        )

        try:
            # Train the model
            lgbm_model.fit(
                X_train_scaled, y_train,
                eval_set=[(X_test_scaled, y_test)],
                eval_metric='auc'
            )

            # Get predictions and probabilities
            test_preds = lgbm_model.predict(X_test_scaled)
            test_probs = lgbm_model.predict_proba(X_test_scaled)[:, 1]

            # Calculate metrics
            accuracy = accuracy_score(y_test, test_preds)
            precision = precision_score(y_test, test_preds, zero_division=0)
            recall = recall_score(y_test, test_preds, zero_division=0)
            f1 = f1_score(y_test, test_preds, zero_division=0)
            cm = confusion_matrix(y_test, test_preds)

            print(f"Fold {fold} Metrics:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            print(f"  Confusion Matrix:\n{cm}")

            # Feature importance analysis
            feature_importance = lgbm_model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': available_features,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)

            importance_df.to_csv(f"{output_dir}/fold_{fold}_feature_importance.csv", index=False)

            # Store metrics
            fold_metrics.append({
                'fold': fold,
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'confusion_matrix': cm.tolist()
            })

            # Add predictions to test data
            test_df['set'] = 'test'
            test_df['prediction'] = test_preds
            test_df['probability'] = test_probs

            # Mark all training points as "train"
            train_df['set'] = 'train'
            train_df['prediction'] = np.nan
            train_df['probability'] = np.nan

            # Store the fold assignment
            test_df['fold'] = fold
            train_df['fold'] = fold

            # Combine results
            fold_df = pd.concat([train_df, test_df])

            # Store in the results dictionary
            fold_results[fold] = fold_df

            # Save to CSV
            fold_df.to_csv(f"{output_dir}/fold_{fold}_results.csv", index=False)

        except Exception as e:
            print(f"Error in fold {fold}: {str(e)}")
            continue

    # Combine all fold results
    all_results = pd.concat([fold_results[fold] for fold in range(n_folds) if not fold_results[fold].empty])

    # Create a "combined" dataframe that includes all points with their fold assignments
    combined_df = all_results[original_columns + ['fold', 'set', 'prediction', 'probability']]
    combined_df.to_csv(f"{output_dir}/all_folds_results.csv", index=False)

    # Calculate average metrics
    if fold_metrics:
        metrics_df = pd.DataFrame(fold_metrics)
        avg_metrics = {
            'participant_id': participant_id,
            'accuracy_mean': metrics_df['accuracy'].mean(),
            'accuracy_std': metrics_df['accuracy'].std(),
            'precision_mean': metrics_df['precision'].mean(),
            'precision_std': metrics_df['precision'].std(),
            'recall_mean': metrics_df['recall'].mean(),
            'recall_std': metrics_df['recall'].std(),
            'f1_mean': metrics_df['f1'].mean(),
            'f1_std': metrics_df['f1'].std()
        }

        print(f"\nParticipant {participant_id} Average Metrics:")
        print(f"  Accuracy: {avg_metrics['accuracy_mean']:.4f} ± {avg_metrics['accuracy_std']:.4f}")
        print(f"  Precision: {avg_metrics['precision_mean']:.4f} ± {avg_metrics['precision_std']:.4f}")
        print(f"  Recall: {avg_metrics['recall_mean']:.4f} ± {avg_metrics['recall_std']:.4f}")
        print(f"  F1 Score: {avg_metrics['f1_mean']:.4f} ± {avg_metrics['f1_std']:.4f}")

        with open(f"{output_dir}/avg_metrics.json", 'w') as f:
            json.dump(avg_metrics, f, indent=4)

        return avg_metrics

    return None


def main():
    """Run cross-validation for all participants"""
    os.makedirs("lgbm_group_outputs", exist_ok=True)

    all_participant_metrics = []

    # Process each participant
    for participant_id in range(1, 40):
        try:
            avg_metrics = process_participant(
                participant_id=participant_id,
                n_folds=5,
                seed=42
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
            'precision_mean': metrics_df['precision_mean'].mean(),
            'precision_std': metrics_df['precision_mean'].std(),
            'recall_mean': metrics_df['recall_mean'].mean(),
            'recall_std': metrics_df['recall_mean'].std(),
            'f1_mean': metrics_df['f1_mean'].mean(),
            'f1_std': metrics_df['f1_mean'].std()
        }

        print("\n=== Overall Average Metrics Across All Participants ===")
        print(f"Accuracy: {overall_metrics['accuracy_mean']:.4f} ± {overall_metrics['accuracy_std']:.4f}")
        print(f"Precision: {overall_metrics['precision_mean']:.4f} ± {overall_metrics['precision_std']:.4f}")
        print(f"Recall: {overall_metrics['recall_mean']:.4f} ± {overall_metrics['recall_std']:.4f}")
        print(f"F1 Score: {overall_metrics['f1_mean']:.4f} ± {overall_metrics['f1_std']:.4f}")

        with open("lgbm_group_outputs/overall_avg_metrics.json", 'w') as f:
            json.dump(overall_metrics, f, indent=4)

        # Save all participant metrics
        metrics_df.to_csv("lgbm_group_outputs/all_participant_metrics.csv", index=False)


if __name__ == "__main__":
    main()