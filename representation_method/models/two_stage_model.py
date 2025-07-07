import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

from representation_method.EKG_2_step_pipeline import reconstruct_and_evaluate_efficiency
# Import your existing modules
from representation_method.utils.general_utils import seed_everything
from representation_method.utils.data_loader import load_eye_tracking_data, DataConfig
from representation_method.utils.data_utils import create_dynamic_time_series, split_train_test_for_time_series, \
    create_dynamic_time_series_with_ailment
from representation_method.utils.trainers import EnsembleTrainer
from representation_method.models.classifier import CombinedModel, CNN1DModel, GradientLocalizer

# Import supervised transformer components
from representation_method.models.supervised_transformer import (
    IntegratedEyeTrackingTransformer,
    EyeTrackingTokenizer,
    TargetLocalizer,
    create_dataset,
    custom_collate,
    train_multi_task
)
from representation_method.models.self_supervised_transformer import calculate_metrics


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
        window_tensor = torch.tensor(tokenized_window, dtype=torch.long).unsqueeze(0)

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
    print(
        f"  Stage 2 ailment detection rate: {stage2_ailment_detection_rate:.4f} ({stage2_ailment_detection_rate * 100:.2f}%)")
    print(f"  Found ailments: {sorted(list(ailments_found_by_stage2))}")
    print(
        f"  Windows with successful ailment detection: {windows_with_successful_ailment_detection}/{len(eval_windows)}")
    print(f"  Ailment positions found: {ailment_positions_found_by_stage2}/{total_ailment_positions_in_eval}")
    print(f"  Ailment position precision: {stage2_ailment_position_precision:.4f}")

    # Calculate traditional target detection metrics
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        target_metrics = calculate_metrics(results_df)

        print(f"\nStage 2 Target Detection Results:")
        print(f"  Target Precision: {target_metrics['precision']:.4f}")
        print(f"  Target Recall: {target_metrics['recall']:.4f}")
        print(f"  Target F1 Score: {target_metrics['f1']:.4f}")
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
            'total_ailment_positions': total_ailment_positions_in_eval,
            'ailment_positions_found': ailment_positions_found_by_stage2
        },
        'detailed_results': all_results
    }




def stage2_evaluation_for_fp_windows(transformer_model, tokenizer, feature_columns,
                                                     eval_windows, eval_metadata, device):
    """
    Modified Stage 2 evaluation that tracks ailments found by the model
    """
    localizer = TargetLocalizer(transformer_model, device)

    print("Evaluating target localization with ailment tracking...")
    all_results = []
    fp_positions = 0
    for window, meta in zip(eval_windows, eval_metadata):
        # Tokenize and run model
        window_df = pd.DataFrame(window, columns=feature_columns)
        tokenized_window = tokenizer.tokenize(window_df, feature_columns)
        window_tensor = torch.tensor(tokenized_window, dtype=torch.long).unsqueeze(0)

        results = localizer.localize_targets(
            window_tensor,
            meta['target_positions'],  # Ground truth target positions
            0,
            window_metadata=meta,
            supervised=True
        )
        fp_positions+= results['num_predictions']
        results.update({
            'participant_id': meta['participant_id'],
            'trial_id': meta['trial_id'],
        })

        all_results.append(results)

    print(f"\nStage 2 FP rows number is  {fp_positions}:")

def calculate_overall_pipeline_ailment_detection(stage1_ailments_found, stage2_ailments_found,
                                                 total_unique_ailments_in_test):
    """
    Calculate overall pipeline performance for ailment detection
    """
    # Union of ailments found in both stages (since Stage 2 operates on Stage 1 positives)
    # In practice, Stage 2 ailments should be a subset of Stage 1 ailments
    overall_ailments_found = stage1_ailments_found.union(stage2_ailments_found)

    # Calculate detection rates
    stage1_detection_rate = len(
        stage1_ailments_found) / total_unique_ailments_in_test if total_unique_ailments_in_test > 0 else 0
    stage2_detection_rate = len(stage2_ailments_found) / len(stage1_ailments_found) if stage1_ailments_found else 0
    overall_detection_rate = len(
        overall_ailments_found) / total_unique_ailments_in_test if total_unique_ailments_in_test > 0 else 0

    # Calculate how much Stage 2 improves upon Stage 1
    stage2_improvement = len(stage2_ailments_found) / len(stage1_ailments_found) if stage1_ailments_found else 0

    print(f"\nOverall Pipeline Ailment Detection:")
    print(
        f"  Stage 1 found: {len(stage1_ailments_found)}/{total_unique_ailments_in_test} ({stage1_detection_rate:.4f})")
    print(f"  Stage 2 confirmed: {len(stage2_ailments_found)}/{len(stage1_ailments_found)} ({stage2_improvement:.4f})")
    print(
        f"  Overall pipeline: {len(overall_ailments_found)}/{total_unique_ailments_in_test} ({overall_detection_rate:.4f})")

    # Show which ailments were lost between stages
    lost_ailments = stage1_ailments_found - stage2_ailments_found
    if lost_ailments:
        print(f"  Ailments lost in Stage 2: {sorted(list(lost_ailments))}")

    return {
        'stage1_detection_rate': stage1_detection_rate,
        'stage2_improvement_rate': stage2_improvement,
        'overall_detection_rate': overall_detection_rate,
        'stage1_ailments': sorted(list(stage1_ailments_found)),
        'stage2_ailments': sorted(list(stage2_ailments_found)),
        'overall_ailments': sorted(list(overall_ailments_found)),
        'lost_ailments': sorted(list(lost_ailments))
    }


def two_step_pipeline(participant_id, window_size=100, seed=42):
    """
    Two-step pipeline with AILMENT_NUMBER tracking
    """
    print(f"Running Two-Step Pipeline with AILMENT_NUMBER Tracking for Participant {participant_id}")
    print(f"Window size: {window_size}, Seed: {seed}")
    print("=" * 60)
    print("NOTE: AILMENT_NUMBER is used ONLY for post-processing evaluation")
    print("      It is NOT used during training - only for measuring performance")
    print("=" * 60)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Feature columns (ONLY these are used for training - NO AILMENT_NUMBER)
    feature_columns = [
        'Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X',
        'CURRENT_FIX_IA_Y', 'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT',
    ]

    print(f"Training features (AILMENT_NUMBER excluded): {feature_columns}")

    # 1. Load and split data
    csv_path = str(Path(__file__).parent.parent.parent / "fwd_data" / 'Nodule_Categorized_Fixation_Data_1_18.csv')

    config = DataConfig(
        data_path=csv_path,
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


    SIGNAL_COL = 'Pupil_Size'
    df['rolling_mean_10'] = df[SIGNAL_COL].rolling(window=10, min_periods=1).mean()
    df['rolling_std_10'] = df[SIGNAL_COL].rolling(window=10, min_periods=1).std()

    # Calculate rate of change (derivative)
    df['signal_derivative'] = df[SIGNAL_COL].diff().fillna(0)

    # Fill any potential NaN values created by rolling std
    df.fillna(method='bfill', inplace=True)
    df.fillna(method='ffill', inplace=True)

    # Split data
    train_df, test_df = split_train_test_for_time_series(df, test_size=0.2, random_state=seed)
    train_df, val_df = split_train_test_for_time_series(train_df, test_size=0.2, random_state=seed)

    print(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Count total ailments in test set for baseline (POST-PROCESSING EVALUATION ONLY)
    test_ailments = test_df[test_df['AILMENT_NUMBER'] != -1]['AILMENT_NUMBER'].unique()
    total_unique_ailments_in_test = len(test_ailments)
    total_ailment_instances_in_test = len(test_df[test_df['AILMENT_NUMBER'] != -1])

    print(f"\nPOST-PROCESSING: AILMENT_NUMBER Statistics in Test Set (for evaluation only):")
    print(f"  Total unique ailments: {total_unique_ailments_in_test}")
    print(f"  Total ailment instances (rows): {total_ailment_instances_in_test}")
    print(f"  Unique ailment numbers: {sorted(test_ailments)}")
    print(f"  (These are NOT used for training - only for measuring final performance)")

    # 2. Create windows with ailment tracking

    X_train, Y_train, train_metadata, train_ailment_locations = create_dynamic_time_series_with_ailment(
        train_df, feature_columns, window_size=window_size
    )
    X_val, Y_val, val_metadata, val_ailment_locations = create_dynamic_time_series_with_ailment(
        val_df, feature_columns, window_size=window_size
    )
    X_test, Y_test, test_metadata, test_ailment_locations = create_dynamic_time_series_with_ailment(
        test_df, feature_columns, window_size=window_size
    )

    print(f"\nWindows created - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Count windows with ailments
    test_windows_with_ailments = sum(1 for meta in test_metadata if meta['has_valid_ailment'])
    test_total_ailment_positions = sum(len(positions) for positions in test_ailment_locations)

    print(f"Test windows with valid ailments: {test_windows_with_ailments}/{len(test_metadata)}")
    print(f"Total ailment positions in test windows: {test_total_ailment_positions}")

    # Show some example ailment locations for verification
    if len(test_metadata) > 0:
        example_with_ailments = [meta for meta in test_metadata[:5] if meta['has_valid_ailment']]
        if example_with_ailments:
            print(f"\nExample window with ailments:")
            example = example_with_ailments[0]
            print(f"  Window: Participant {example['participant_id']}, Trial {example['trial_id']}")
            print(f"  Ailment positions: {example['ailment_positions']}")
            print(f"  Ailment numbers: {example['ailment_numbers']}")
            print(f"  Ailment details: {example['ailment_details'][:3]}...")  # Show first 3

    # Create save directory
    save_dir = f'results/ailment_tracking_participant_{participant_id}'
    os.makedirs(save_dir, exist_ok=True)

    # ===== STEP 1: WINDOW-LEVEL PREDICTION =====
    print("\nSTEP 1: Window-Level Prediction (using only eye-tracking features)")
    print("-" * 40)
    print("Training features used:", feature_columns)
    print("AILMENT_NUMBER: NOT used in training")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).permute(0, 2, 1)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).permute(0, 2, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).permute(0, 2, 1)

    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.long)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, Y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train ensemble
    ensemble_save_path = os.path.join(save_dir, 'ensemble_models')
    os.makedirs(ensemble_save_path, exist_ok=True)

    ensemble_trainer = EnsembleTrainer(
        base_model_class=CNN1DModel,
        model_params={'input_dim': X_train_tensor.shape[1], 'window_size': window_size, 'output_classes': 2},
        n_models=10,
        device=device,
        save_path=os.path.join(save_dir, 'ensemble_models')
    )

    # Calculate class weights
    class_counts = np.bincount(Y_train)
    weights = 1.0 / class_counts

    print("Training ensemble...")
    ensemble_trainer.train_ensemble(
        train_dataset=train_dataset,
        val_loader=val_loader,
        batch_size=32,
        epochs=50,  # Reduced for testing
        criterion=nn.CrossEntropyLoss(),
        optimizer_class=optim.Adam,
        optimizer_params={'lr': 0.001},
        majority_weight=weights[0] if len(weights) > 1 else 0.1
    )

    # Evaluate Step 1
    test_predictions = ensemble_trainer.predict(test_loader, minority_weight=1, threshold=0.9)

    # DEBUG: Check prediction counts
    print(f"\nDEBUG - Step 1 Predictions:")
    print(f"  Length of test_predictions: {len(test_predictions)}")
    print(f"  Length of Y_test: {len(Y_test)}")
    print(f"  Length of test_metadata: {len(test_metadata)}")
    print(f"  Predicted positive (1): {np.sum(test_predictions == 1)}")
    print(f"  Predicted negative (0): {np.sum(test_predictions == 0)}")

    # Calculate Step 1 metrics properly
    step1_accuracy = accuracy_score(Y_test, test_predictions)
    step1_precision = precision_score(Y_test, test_predictions, zero_division=0)
    step1_recall = recall_score(Y_test, test_predictions, zero_division=0)
    step1_f1 = f1_score(Y_test, test_predictions, zero_division=0)
    step1_cm = confusion_matrix(Y_test, test_predictions)

    print(f"\nStep 1 Results:")
    print(f"  Accuracy: {step1_accuracy:.4f}")
    print(f"  Precision: {step1_precision:.4f}")
    print(f"  Recall: {step1_recall:.4f}")
    print(f"  F1 Score: {step1_f1:.4f}")
    print(f"\nConfusion Matrix:")
    print("   Predicted 0  Predicted 1")
    print(f"Actual 0   {step1_cm[0, 0]:<10} {step1_cm[0, 1]:<10}")
    print(f"Actual 1   {step1_cm[1, 0]:<10} {step1_cm[1, 1]:<10}")

    step1_positive_indices = np.where(test_predictions == 1)[0]
    step_1_negative_indices =  np.where(test_predictions == 0)[0]
    print(f"\nDEBUG - Positive Windows:")
    print(f"  step1_positive_indices length: {len(step1_positive_indices)}")
    print(f"  Should match predicted positive count: {np.sum(test_predictions == 1)}")

    step1_ailments_found = set()
    step1_windows_with_ailments = 0
    step_1_true_pos_trials_ailments = set()
    step_1_pos_trials_ailments = set()
    for idx in step1_positive_indices:
        if idx < len(test_metadata):
            meta = test_metadata[idx]
            step_1_pos_trials_ailments.update(str(meta['trial_id']))
            if meta['has_valid_ailment']:
                step_1_true_pos_trials_ailments.update(str(meta['trial_id']))
                step1_windows_with_ailments += 1
                step1_ailments_found.update(meta['ailment_numbers'])
        else:
            print(f"WARNING: Index {idx} out of range for test_metadata (length: {len(test_metadata)})")

    step_1_neg_trials_ailments = set()
    for idx in step_1_negative_indices:
        if idx < len(test_metadata):
            meta = test_metadata[idx]
            step_1_neg_trials_ailments.update(str(meta['trial_id']))
        else:
            print(f"WARNING: Index {idx} out of range for test_metadata (length: {len(test_metadata)})")
    print(f"  step1_true_positive_trials_with_ailments: {step_1_true_pos_trials_ailments}")
    print(f"  step1_positive_trials_with_ailments: {step_1_pos_trials_ailments}")
    print(f"  step1_neg_trials_with_ailments: {step_1_neg_trials_ailments}")

    step1_ailment_coverage = len(
        step1_ailments_found) / total_unique_ailments_in_test if total_unique_ailments_in_test > 0 else 0

    print(f"\nStep 1 - POST-PROCESSING Ailment Tracking:")
    print(f"  Predicted positive windows: {len(step1_positive_indices)}")
    print(f"  Positive windows with ailments: {step1_windows_with_ailments}")
    print(f"  Unique ailments found: {len(step1_ailments_found)}/{total_unique_ailments_in_test}")
    print(f"  Ailment coverage: {step1_ailment_coverage:.4f}")
    print(f"  Found ailments: {sorted(step1_ailments_found)}")

    print("\nSTEP 2: Target Localization within Predicted Positive Windows")
    print("-" * 60)
    positive_indices = np.where(test_predictions == 1)[0]
    if len(positive_indices) == 0:
        print("No positive windows for Stage 2.")
    else:
        # Use the first model from the trained ensemble for localization
        stage1_model_for_loc = ensemble_trainer.models[0]
        localizer = GradientLocalizer(stage1_model_for_loc, device)
        all_stage2_results = []

        for idx in positive_indices:
            window_metadata = test_metadata[idx]
            window_tensor = X_test_tensor[idx].unsqueeze(0)
            predicted_row = localizer.localize_target(window_tensor.clone())  # Use clone to avoid grad issues
            true_target_positions = window_metadata.get('target_positions', [])
            is_correct = 1 if predicted_row in true_target_positions else 0
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
        print(f"Stage 2 found ailments:{len(found_ailments)}/{total_unique_ailments_in_test}")
        if all_stage2_results:
            # Call the new function to get the final results
            final_df, efficiency_metrics = reconstruct_and_evaluate_efficiency(
                all_stage2_results,
                test_df,
                test_metadata ,
                strategy = 'majority'
            )

            print("\n--- Sample of Final Predicted DataFrame ---")
            print(len(final_df[final_df["AILMENT_NUMBER"]!=-1])//len(final_df))
            tumors_found_by_pipeline = test_df[test_df['AILMENT_NUMBER'] != -1]['AILMENT_NUMBER'].unique()
            print(
                f"Stage 2 after reconstruct found ailments:{len(tumors_found_by_pipeline)}/{total_unique_ailments_in_test}")
            print(final_df.head())
            pipeline_precision = final_df['target'].sum() / len(final_df) if len(
                final_df) > 0 else 0

            print("\n" + "=" * 60)
            print("           PIPELINE PERFORMANCE")
            print("=" * 60)
            print(f"Total unique tumors found by pipeline: {tumors_found_by_pipeline}")
            print(f"Tumor Coverage: {len(tumors_found_by_pipeline)/total_unique_ailments_in_test*100:.2f}%")
            print(f"Precision of flagged slices: {pipeline_precision:.4f}")
            print("=" * 60)

            total_unique_tumors = test_df[test_df['target'] == 1]['AILMENT_NUMBER'].nunique()

            # The total number of individual CT slices that show a tumor
            # This is the workload for the "Before" (naive) approach.
            baseline_rows_to_check = len(test_df)

            print("\n" + "=" * 60)
            print("           BASELINE (Examining all tumor slices)")
            print("=" * 60)
            print(f"Total unique tumors in test set: {total_unique_tumors}")
            print(f"Total slices to review in baseline: {baseline_rows_to_check:,}")
            print("=" * 60)

    # quit(2)
    #
    # # Filter windows predicted as positive
    # positive_indices = np.where(test_predictions == 1)[0]
    # positive_windows = X_test[positive_indices]
    # positive_metadata = [test_metadata[i] for i in positive_indices]
    # positive_ailment_locations = test_ailment_locations[positive_indices]
    #
    # print(f"Step 1 predicted {len(positive_windows)} positive windows out of {len(X_test)} total")
    #
    # if len(positive_windows) == 0:
    #     print("No positive windows predicted - cannot run Step 2")
    #     step2_precision = step2_recall = step2_f1 = 0.0
    #     final_ailments_found = set()
    #     final_ailment_coverage = 0
    # else:
    #     # Filter for windows that actually have targets (for evaluation)
    #     tp_eval_windows = []
    #     tp_eval_metadata = []
    #     eval_ailment_locations = []
    #
    #
    #     fp_eval_windows = []
    #     fp_eval_metadata = []
    #     for i, meta in enumerate(positive_metadata):
    #         if meta['has_target']:
    #             tp_eval_windows.append(positive_windows[i])
    #             tp_eval_metadata.append(meta)
    #             eval_ailment_locations.append(positive_ailment_locations[i])
    #         else:
    #             fp_eval_windows.append(positive_windows[i])
    #             fp_eval_metadata.append(meta)
    #
    #     print(f"Found {len(tp_eval_windows)} positive windows with ground truth targets for evaluation")
    #
    #     if len(tp_eval_windows) == 0:
    #         print("No positive windows with ground truth targets - cannot evaluate Step 2")
    #         step2_precision = step2_recall = step2_f1 = 0.0
    #         final_ailments_found = step1_ailments_found  # Use Step 1 results
    #         final_ailment_coverage = step1_ailment_coverage
    #     else:
    #         # Create tokenizer
    #         tokenizer = EyeTrackingTokenizer()
    #
    #         # Use all positive windows for tokenizer fitting
    #         all_windows_flat = positive_windows.reshape(-1, positive_windows.shape[-1])
    #         tokenizer_df = pd.DataFrame(all_windows_flat, columns=feature_columns)
    #         tokenizer.fit(tokenizer_df, feature_columns)
    #
    #         train_positive_indices = np.where(Y_train == 1)[0]
    #         if len(train_positive_indices) > 0:
    #             train_positive_windows = X_train[train_positive_indices[:min(50, len(train_positive_indices))]]
    #             train_positive_metadata = [train_metadata[i] for i in
    #                                        train_positive_indices[:min(50, len(train_positive_indices))]]
    #
    #             # Create target vectors for training
    #             def create_target_vector(window_size, target_positions):
    #                 target_vector = [0] * window_size
    #                 for pos in target_positions:
    #                     if 0 <= pos < window_size:
    #                         target_vector[pos] = 1
    #                 return target_vector
    #
    #             train_targets = [create_target_vector(window_size, meta['target_positions']) for meta in
    #                              train_positive_metadata]
    #
    #             # Create transformer dataset
    #             train_dataset_transformer = create_dataset(
    #                 train_positive_windows,
    #                 np.ones(len(train_positive_windows)),
    #                 train_targets,
    #                 tokenizer,
    #                 feature_columns
    #             )
    #
    #             # Create minimal validation set
    #             val_size = min(10, len(train_positive_windows) // 4)
    #             if val_size > 0:
    #                 val_dataset_transformer = create_dataset(
    #                     train_positive_windows[:val_size],
    #                     np.ones(val_size),
    #                     train_targets[:val_size],
    #                     tokenizer,
    #                     feature_columns
    #                 )
    #             else:
    #                 val_dataset_transformer = train_dataset_transformer
    #
    #             # Create data loaders
    #             train_loader_transformer = DataLoader(
    #                 train_dataset_transformer, batch_size=16, shuffle=True, collate_fn=custom_collate
    #             )
    #             val_loader_transformer = DataLoader(
    #                 val_dataset_transformer, batch_size=16, shuffle=False, collate_fn=custom_collate
    #             )
    #
    #             # Create and train transformer
    #             transformer_model = IntegratedEyeTrackingTransformer(
    #                 vocab_size=tokenizer.vocab_size,
    #                 n_features=len(feature_columns),
    #                 d_model=256,
    #                 max_len=window_size
    #             )
    #
    #             # Try to load pretrained model
    #             pretrained_path = f'results/self_supervised_from_cv_participant_{participant_id}_fold_{0}_window_size_{window_size}/model.pth'
    #
    #             if os.path.exists(pretrained_path):
    #                 print(f"Loading pretrained weights from: {pretrained_path}")
    #                 try:
    #                     pretrained_state = torch.load(pretrained_path)['model_state_dict']
    #                     model_state = transformer_model.state_dict()
    #
    #                     # Filter compatible weights
    #                     filtered_state = {k: v for k, v in pretrained_state.items()
    #                                       if k in model_state and model_state[k].shape == v.shape}
    #                     model_state.update(filtered_state)
    #                     transformer_model.load_state_dict(model_state)
    #                     print(f"Loaded {len(filtered_state)} compatible layers from pretrained model")
    #                 except Exception as e:
    #                     print(f"Error loading pretrained model: {e}")
    #                     print("Training from scratch")
    #             else:
    #                 print(f"Pretrained model not found at {pretrained_path}, training from scratch")
    #
    #             print("Training supervised transformer...")
    #             train_multi_task(
    #                 model=transformer_model,
    #                 train_loader=train_loader_transformer,
    #                 val_loader=val_loader_transformer,
    #                 epochs=50,  # Reduced for speed
    #                 learning_rate=1e-4,
    #                 device=device,
    #                 alpha=0.1,
    #                 beta=0.1,
    #                 gamma=15.0,
    #                 patience=10
    #             )
    #
    #             transformer_model = transformer_model.to(device)
    #
    #             # NOW call the modified evaluation function
    #             stage2_results = modified_stage2_evaluation_with_ailment_tracking(
    #                 transformer_model, tokenizer, feature_columns,
    #                 tp_eval_windows, tp_eval_metadata, device
    #             )
    #             stage2_evaluation_for_fp_windows(
    #                 transformer_model, tokenizer, feature_columns,
    #                 fp_eval_windows, fp_eval_metadata, device
    #             )                # Extract results
    #             step2_metrics = stage2_results['target_metrics']
    #             step2_ailment_metrics = stage2_results['ailment_metrics']
    #
    #             step2_precision = step2_metrics['precision']
    #             step2_recall = step2_metrics['recall']
    #             step2_f1 = step2_metrics['f1']
    #
    #             # Get Stage 2 ailments found
    #             stage2_ailments_found = set(step2_ailment_metrics['found_ailments'])
    #
    #             print(f"Stage 2 Results:")
    #             print(f"  Target - Precision: {step2_precision:.4f}, Recall: {step2_recall:.4f}, F1: {step2_f1:.4f}")
    #             print(f"  Ailment Detection Rate: {step2_ailment_metrics['detection_rate']:.4f}")
    #             print(f"  Ailments Found: {step2_ailment_metrics['found_ailments']}")
    #             print(f"  Windows with successful ailment detection: {step2_ailment_metrics['successful_windows']}")
    #             print(f"  Ailment position precision: {step2_ailment_metrics['ailment_position_precision']:.4f}")
    #
    #             # Calculate final ailment coverage
    #             final_ailment_coverage = len(
    #                 stage2_ailments_found) / total_unique_ailments_in_test if total_unique_ailments_in_test > 0 else 0
    #
    #         else:
    #             step2_precision = step2_recall = step2_f1 = 0.0
    #             stage2_ailments_found = set()
    #             final_ailment_coverage = step1_ailment_coverage
    #             print("No positive training windows available for Step 2")
    #
    #     # Calculate overall pipeline performance
    #     overall_precision = step1_precision * step2_precision
    #     overall_recall = step1_recall * step2_recall
    #     if overall_precision + overall_recall > 0:
    #         overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
    #     else:
    #         overall_f1 = 0.0
    #
    #     print(f"\n" + "=" * 60)
    #     print(f"EFFICIENCY ANALYSIS: Rows Needed to Find Ailments")
    #     print(f"=" * 60)
    #
    #     # Calculate baseline efficiency first (what would happen if we examined ALL test rows)
    #     total_test_rows = len(test_df)
    #     total_ailment_rows = len(test_df[test_df['AILMENT_NUMBER'] != -1])
    #     baseline_efficiency = total_ailment_rows / total_test_rows if total_test_rows > 0 else 0
    #
    #     # Step 1: Count rows in predicted positive windows
    #     step1_total_rows_examined = len(step1_positive_indices) * window_size
    #     step1_rows_with_ailments = 0
    #
    #     for idx in step1_positive_indices:
    #         if idx < len(test_metadata):
    #             meta = test_metadata[idx]
    #             step1_rows_with_ailments += len(meta.get('ailment_positions', []))
    #
    #     step1_efficiency = step1_rows_with_ailments / step1_total_rows_examined if step1_total_rows_examined > 0 else 0
    #
    #     print(f"STEP 1 EFFICIENCY:")
    #     print(f"  Total rows examined: {step1_total_rows_examined:,}")
    #     print(f"  Rows with ailments found: {step1_rows_with_ailments:,}")
    #     print(f"  Efficiency (ailment rows / total examined): {step1_efficiency:.4f} ({step1_efficiency * 100:.2f}%)")
    #     print(f"  Precision: {step1_rows_with_ailments}/{step1_total_rows_examined} = {step1_efficiency:.4f}")
    #     print(f"\nROWS PER AILMENT TYPE:")
    #     improvement_factor = step1_efficiency / baseline_efficiency if baseline_efficiency > 0 else 0
    #     ailment_row_counts = {}
    #     for idx in step1_positive_indices:
    #         if idx < len(test_metadata):
    #             meta = test_metadata[idx]
    #             for detail in meta.get('ailment_details', []):
    #                 ailment_num = detail['ailment_number']
    #                 if ailment_num not in ailment_row_counts:
    #                     ailment_row_counts[ailment_num] = 0
    #                 ailment_row_counts[ailment_num] += 1
    #
    #     for ailment_num in sorted(ailment_row_counts.keys()):
    #         count = ailment_row_counts[ailment_num]
    #         print(f"  Ailment {ailment_num}: {count} rows found")
    #
    #     print(f"\nBASELINE (examining all test data):")
    #     print(f"  Total test rows: {total_test_rows:,}")
    #     print(f"  Rows with ailments: {total_ailment_rows:,}")
    #     print(f"  Baseline efficiency: {baseline_efficiency:.4f} ({baseline_efficiency * 100:.2f}%)")
    #
    #     print(f"\nSTEP 1 IMPROVEMENT:")
    #     print(f"  Efficiency improvement: {improvement_factor:.2f}x")
    #     if improvement_factor > 1:
    #         print(f"  ✓ Model is {improvement_factor:.2f}x more efficient than random scanning")
    #     else:
    #         print(f"  ⚠ Model is less efficient than random scanning")
    #
    #     # Initialize Step 2 efficiency variables (will be updated if Step 2 runs)
    #     step2_total_targeted_rows = 0
    #     step2_rows_with_ailments = 0
    #     step2_efficiency = 0
    #     step2_improvement_over_step1 = 0
    #     end_to_end_efficiency = step1_efficiency
    #     end_to_end_improvement = improvement_factor
    #
    #     print(f"=" * 60)
    #     results = {
    #         'participant_id': participant_id,
    #         'window_size': window_size,
    #         'seed': seed,
    #         'step1': {
    #             'accuracy': step1_accuracy,
    #             'precision': step1_precision,
    #             'recall': step1_recall,
    #             'f1': step1_f1,
    #             'confusion_matrix': step1_cm.tolist()
    #         },
    #         'step2': {
    #             'precision': step2_precision,
    #             'recall': step2_recall,
    #             'f1': step2_f1
    #         },
    #         'overall': {
    #             'precision': overall_precision,
    #             'recall': overall_recall,
    #             'f1': overall_f1
    #         },
    #         'degradation': {
    #             'step1_to_step2': step1_f1 - step2_f1,
    #             'step1_to_overall': step1_f1 - overall_f1
    #         },
    #         'ailment_tracking': {
    #             'total_unique_ailments_in_test': total_unique_ailments_in_test,
    #             'total_ailment_instances_in_test': total_ailment_instances_in_test,
    #             'total_ailment_positions_in_windows': test_total_ailment_positions,
    #             'test_ailments': sorted(test_ailments.tolist()),
    #             'step1_ailments_found': sorted(list(step1_ailments_found)),
    #             'step1_ailment_coverage': step1_ailment_coverage,
    #             'final_ailments_found': sorted(list(stage2_ailments_found)),  # Will update this
    #             'final_ailment_coverage': final_ailment_coverage,
    #             'detection_rate': len(
    #                 stage2_ailments_found) / total_unique_ailments_in_test if total_unique_ailments_in_test > 0 else 0
    #         },
    #         # We'll add efficiency_analysis later after those variables are calculated
    #         'windows': {
    #             'total_test_windows': len(test_metadata),
    #             'test_windows_with_ailments': test_windows_with_ailments,
    #             'step1_positive_windows': len(step1_positive_indices),
    #             'step1_positive_with_ailments': step1_windows_with_ailments
    #         }
    #     }
    #
    #     # Calculate final metrics
    #     ailments_found_ratio = len(
    #         stage2_ailments_found) / total_unique_ailments_in_test if total_unique_ailments_in_test > 0 else 0
    #
    #     print(f"\nFinal Results - POST-PROCESSING Ailment Tracking:")
    #     print(f"  Stage 1 ailments found: {len(step1_ailments_found)}/{total_unique_ailments_in_test}")
    #     print(
    #         f"  Stage 2 ailments confirmed: {len(stage2_ailments_found)}/{len(step1_ailments_found) if step1_ailments_found else 0}")
    #     print(f"  Final ailment coverage: {final_ailment_coverage:.4f}")
    #     print(f"  Found ailments: {sorted(list(stage2_ailments_found))}")
    #
    #     # Calculate overall pipeline performance
    #     overall_ailment_results = calculate_overall_pipeline_ailment_detection(
    #         step1_ailments_found, stage2_ailments_found, total_unique_ailments_in_test
    #     )
    #
    #     # NOW add the new Stage 2 and overall results to the existing structure
    #     results['stage2_ailment_tracking'] = {
    #         'ailments_found': sorted(list(stage2_ailments_found)),
    #         'detection_rate': len(
    #             stage2_ailments_found) / total_unique_ailments_in_test if total_unique_ailments_in_test > 0 else 0,
    #         'improvement_over_stage1': len(stage2_ailments_found) / len(
    #             step1_ailments_found) if step1_ailments_found else 0,
    #         'confirmation_rate': len(stage2_ailments_found) / len(step1_ailments_found) if step1_ailments_found else 0,
    #     }
    #
    #     results['overall_pipeline_ailment_detection'] = overall_ailment_results
    #
    #     # Update the existing ailment_tracking section with Stage 2 results
    #     results['ailment_tracking']['final_ailments_found'] = sorted(list(stage2_ailments_found))
    #     results['ailment_tracking']['final_ailment_coverage'] = final_ailment_coverage
    #     results['ailment_tracking']['detection_rate'] = ailments_found_ratio
    #
    #     # NOW add efficiency analysis after all variables are calculated
    #     # (This should come after your existing efficiency analysis code in the original function)
    #     results['efficiency_analysis'] = {
    #         'step1_total_rows_examined': step1_total_rows_examined,
    #         'step1_rows_with_ailments': step1_rows_with_ailments,
    #         'step1_efficiency': step1_efficiency,
    #         'step2_total_targeted_rows': step2_total_targeted_rows if 'step2_total_targeted_rows' in locals() else 0,
    #         'step2_rows_with_ailments': step2_rows_with_ailments if 'step2_rows_with_ailments' in locals() else 0,
    #         'step2_efficiency': step2_efficiency if 'step2_efficiency' in locals() else 0,
    #         'step2_improvement_over_step1': step2_improvement_over_step1 if 'step2_improvement_over_step1' in locals() else 0,
    #         'end_to_end_efficiency': end_to_end_efficiency if 'end_to_end_efficiency' in locals() else step1_efficiency,
    #         'end_to_end_improvement': end_to_end_improvement if 'end_to_end_improvement' in locals() else improvement_factor if 'improvement_factor' in locals() else 0,
    #         'total_test_rows': total_test_rows,
    #         'total_ailment_rows': total_ailment_rows,
    #         'baseline_efficiency': baseline_efficiency if 'baseline_efficiency' in locals() else 0,
    #         'improvement_factor': improvement_factor if 'improvement_factor' in locals() else 0,
    #         'ailment_row_counts': ailment_row_counts if 'ailment_row_counts' in locals() else {},
    #         'data_reduction': {
    #             'original_rows': total_test_rows,
    #             'step1_rows': step1_total_rows_examined,
    #             'step2_rows': step2_total_targeted_rows if 'step2_total_targeted_rows' in locals() else step1_total_rows_examined,
    #             'total_reduction_ratio': (step2_total_targeted_rows / total_test_rows) if (
    #                         'step2_total_targeted_rows' in locals() and total_test_rows > 0) else (
    #                         step1_total_rows_examined / total_test_rows) if total_test_rows > 0 else 0,
    #             'compression_ratio': (total_test_rows / step2_total_targeted_rows) if (
    #                         'step2_total_targeted_rows' in locals() and step2_total_targeted_rows > 0) else (
    #                         total_test_rows / step1_total_rows_examined) if step1_total_rows_examined > 0 else 0
    #         }
    #     }
    #
    #     print(f"\n" + "=" * 60)
    #     print(f"KEY METRIC: STAGE 2 AILMENT DETECTION RATE")
    #     print(f"=" * 60)
    #     print(f"Stage 1 found: {len(step1_ailments_found)} ailments")
    #     print(f"Stage 2 confirmed: {len(stage2_ailments_found)} ailments")
    #     print(f"STAGE 2 DETECTION RATE: {ailments_found_ratio:.4f} ({ailments_found_ratio * 100:.2f}%)")
    #     print(
    #         f"STAGE 2 CONFIRMATION RATE: {len(stage2_ailments_found) / len(step1_ailments_found) if step1_ailments_found else 0:.4f}")
    #     print(f"Ailments Stage 2 confirmed: {sorted(list(stage2_ailments_found))}")
    #     if step1_ailments_found - stage2_ailments_found:
    #         print(f"Ailments lost in Stage 2: {sorted(list(step1_ailments_found - stage2_ailments_found))}")
    #     print(f"=" * 60)
    #     print(results)
    #
    #     # Save detailed results
    #     import json
    #     with open(os.path.join(save_dir, 'ailment_tracking_results.json'), 'w') as f:
    #         json.dump(results, f, indent=4)
    #
    #     print(f"\nDetailed results saved to: {save_dir}")
    #     return results



if __name__ == "__main__":
    # Test with a single participant
    window_size = 10
    seed = 0

    participant_id = 1
    try:
        two_step_pipeline(participant_id, window_size, seed)
        print("\n" + "=" * 60)
        print("SUCCESS - Pipeline completed with ailment tracking!")
        print("=" * 60)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

            #   Stage 2 ailment detection rate: 0.4545 (45.45%)