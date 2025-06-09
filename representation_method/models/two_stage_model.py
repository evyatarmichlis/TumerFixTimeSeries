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

# Import your existing modules
from representation_method.utils.general_utils import seed_everything
from representation_method.utils.data_loader import load_eye_tracking_data, DataConfig
from representation_method.utils.data_utils import create_dynamic_time_series, split_train_test_for_time_series
from representation_method.utils.trainers import EnsembleTrainer
from representation_method.models.classifier import CombinedModel

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


def simple_two_step_pipeline(participant_id, window_size=100, seed=42):
    """
    Simple two-step pipeline:
    1. Use existing ensemble method to predict window-level targets
    2. Use existing supervised transformer to predict target locations
    """
    print(f"Running Simple Two-Step Pipeline for Participant {participant_id}")
    print(f"Window size: {window_size}, Seed: {seed}")
    print("=" * 60)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Feature columns
    feature_columns = [
        'Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X',
        'CURRENT_FIX_IA_Y', 'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT'
    ]

    # 1. Load and split data (same as original)
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

    # Split data
    train_df, test_df = split_train_test_for_time_series(df, test_size=0.2, random_state=seed)
    train_df, val_df = split_train_test_for_time_series(train_df, test_size=0.2, random_state=seed)

    print(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 2. Create windows (same as original)
    X_train, Y_train, train_metadata = create_dynamic_time_series(
        train_df, feature_columns, window_size=window_size
    )
    X_val, Y_val, val_metadata = create_dynamic_time_series(
        val_df, feature_columns, window_size=window_size
    )
    X_test, Y_test, test_metadata = create_dynamic_time_series(
        test_df, feature_columns, window_size=window_size
    )

    print(f"Windows created - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Positive windows - Train: {np.sum(Y_train)}, Val: {np.sum(Y_val)}, Test: {np.sum(Y_test)}")

    # Create save directory
    save_dir = f'results/simple_two_step_participant_{participant_id}'
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

    # Train ensemble (same as original main_with_autoencoder when assemble=True)
    ensemble_save_path = os.path.join(save_dir, 'ensemble_models')
    os.makedirs(ensemble_save_path, exist_ok=True)

    ensemble_trainer = EnsembleTrainer(
        base_model_class=CombinedModel,
        model_params={'input_dim': X_train_tensor.shape[1], 'output_classes': 2},
        n_models=5,
        device=device,
        save_path=ensemble_save_path
    )

    # Calculate class weights
    class_counts = np.bincount(Y_train)
    weights = 1.0 / class_counts

    print("Training ensemble...")
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
    test_predictions = ensemble_trainer.predict(test_loader,minority_weight=2, threshold=0.8)

    step1_accuracy = accuracy_score(Y_test, test_predictions)
    step1_precision = precision_score(Y_test, test_predictions, zero_division=0)
    step1_recall = recall_score(Y_test, test_predictions, zero_division=0)
    step1_f1 = f1_score(Y_test, test_predictions, zero_division=0)
    step1_cm = confusion_matrix(Y_test, test_predictions)

    print(f"Step 1 Results:")
    print(f"  Accuracy: {step1_accuracy:.4f}")
    print(f"  Precision: {step1_precision:.4f}")
    print(f"  Recall: {step1_recall:.4f}")
    print(f"  F1 Score: {step1_f1:.4f}")
    print(f"  Confusion Matrix:\n{step1_cm}")

    # ===== STEP 2: TARGET LOCALIZATION =====
    print("\nSTEP 2: Target Localization within Predicted Positive Windows")
    print("-" * 60)

    # Filter windows predicted as positive
    positive_indices = np.where(test_predictions == 1)[0]
    positive_windows = X_test[positive_indices]
    positive_metadata = [test_metadata[i] for i in positive_indices]

    print(f"Step 1 predicted {len(positive_windows)} positive windows out of {len(X_test)} total")

    if len(positive_windows) == 0:
        print("No positive windows predicted - cannot run Step 2")
        step2_precision = step2_recall = step2_f1 = 0.0
    else:
        # Filter for windows that actually have targets (for evaluation)
        eval_windows = []
        eval_metadata = []

        for i, meta in enumerate(positive_metadata):
            if len(meta) > 0:  # Has ground truth targets
                eval_windows.append(positive_windows[i])
                eval_metadata.append(meta)

        print(f"Found {len(eval_windows)} positive windows with ground truth targets for evaluation")

        if len(eval_windows) == 0:
            print("No positive windows with ground truth targets - cannot evaluate Step 2")
            step2_precision = step2_recall = step2_f1 = 0.0
        else:
            # Create tokenizer
            tokenizer = EyeTrackingTokenizer()

            # Use all positive windows for tokenizer fitting
            all_windows_flat = positive_windows.reshape(-1, positive_windows.shape[-1])
            tokenizer_df = pd.DataFrame(all_windows_flat, columns=feature_columns)
            tokenizer.fit(tokenizer_df, feature_columns)

            # For training, use some windows with targets from the training set
            train_positive_indices = np.where(Y_train == 1)[0]
            if len(train_positive_indices) > 0:
                train_positive_windows = X_train[train_positive_indices[:min(50, len(train_positive_indices))]]
                train_positive_metadata = [train_metadata[i] for i in
                                           train_positive_indices[:min(50, len(train_positive_indices))]]

                # Create target vectors for training
                def create_target_vector(window_size, target_positions):
                    target_vector = [0] * window_size
                    for pos in target_positions:
                        if 0 <= pos < window_size:
                            target_vector[pos] = 1
                    return target_vector

                train_targets = [create_target_vector(window_size, meta) for meta in train_positive_metadata]

                # Create transformer dataset
                train_dataset_transformer = create_dataset(
                    train_positive_windows,
                    np.ones(len(train_positive_windows)),
                    train_targets,
                    tokenizer,
                    feature_columns
                )

                # Create minimal validation set
                val_size = min(10, len(train_positive_windows) // 4)
                if val_size > 0:
                    val_dataset_transformer = create_dataset(
                        train_positive_windows[:val_size],
                        np.ones(val_size),
                        train_targets[:val_size],
                        tokenizer,
                        feature_columns
                    )
                else:
                    val_dataset_transformer = train_dataset_transformer

                # Create data loaders
                train_loader_transformer = DataLoader(
                    train_dataset_transformer, batch_size=16, shuffle=True, collate_fn=custom_collate
                )
                val_loader_transformer = DataLoader(
                    val_dataset_transformer, batch_size=16, shuffle=False, collate_fn=custom_collate
                )

                # Create and train transformer
                transformer_model = IntegratedEyeTrackingTransformer(
                    vocab_size=tokenizer.vocab_size,
                    n_features=len(feature_columns),
                    d_model=256,
                    max_len=window_size
                )
                pretrained_path = f'results/self_supervised_from_cv_participant_{participant_id}_fold_{0}_window_size_{window_size}/model.pth'

                if os.path.exists(pretrained_path):
                    print(f"Loading pretrained weights from: {pretrained_path}")
                    try:
                        pretrained_state = torch.load(pretrained_path)['model_state_dict']
                        model_state = transformer_model.state_dict()

                        # Filter compatible weights
                        filtered_state = {k: v for k, v in pretrained_state.items()
                                          if k in model_state and model_state[k].shape == v.shape}
                        model_state.update(filtered_state)
                        transformer_model.load_state_dict(model_state)
                        print(f"Loaded {len(filtered_state)} compatible layers from pretrained model")
                    except Exception as e:
                        print(f"Error loading pretrained model: {e}")
                        print("Training from scratch")
                else:
                    print(f"Pretrained model not found at {pretrained_path}, training from scratch")
                print("Training supervised transformer...")
                train_multi_task(
                    model=transformer_model,
                    train_loader=train_loader_transformer,
                    val_loader=val_loader_transformer,
                    epochs=300,  # Reduced for speed
                    learning_rate=1e-4,
                    device=device,
                    alpha=0.1,
                    beta=0.1,
                    gamma=15.0,
                    patience=10
                )

                # Evaluate on positive windows
                transformer_model = transformer_model.to(device)
                localizer = TargetLocalizer(transformer_model, device)

                print("Evaluating target localization...")
                all_results = []

                for window, meta in zip(eval_windows, eval_metadata):
                    if len(meta) > 0:  # Only evaluate if there are ground truth targets
                        window_df = pd.DataFrame(window, columns=feature_columns)
                        tokenized_window = tokenizer.tokenize(window_df, feature_columns)
                        window_tensor = torch.tensor(tokenized_window, dtype=torch.long).unsqueeze(0)

                        results = localizer.localize_targets(
                            window_tensor,
                            meta,  # Ground truth target positions
                            0,  # Window start (relative)
                            supervised=True
                        )
                        all_results.append(results)

                if len(all_results) > 0:
                    results_df = pd.DataFrame(all_results)
                    step2_metrics = calculate_metrics(results_df)

                    step2_precision = step2_metrics['precision']
                    step2_recall = step2_metrics['recall']
                    step2_f1 = step2_metrics['f1']

                    print(f"Step 2 Results:")
                    print(f"  Precision: {step2_precision:.4f}")
                    print(f"  Recall: {step2_recall:.4f}")
                    print(f"  F1 Score: {step2_f1:.4f}")
                    print(f"  Windows evaluated: {len(all_results)}")
                else:
                    step2_precision = step2_recall = step2_f1 = 0.0
                    print("No windows could be evaluated for Step 2")
            else:
                step2_precision = step2_recall = step2_f1 = 0.0
                print("No positive training windows available for Step 2")

    # ===== OVERALL RESULTS =====
    print("\nOVERALL END-TO-END RESULTS")
    print("-" * 40)

    # Calculate overall performance (multiplicative)
    overall_precision = step1_precision * step2_precision
    overall_recall = step1_recall * step2_recall

    if overall_precision + overall_recall > 0:
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
    else:
        overall_f1 = 0.0

    print(f"Step 1 (Window): P={step1_precision:.4f}, R={step1_recall:.4f}, F1={step1_f1:.4f}")
    print(f"Step 2 (Target): P={step2_precision:.4f}, R={step2_recall:.4f}, F1={step2_f1:.4f}")
    print(f"Overall:         P={overall_precision:.4f}, R={overall_recall:.4f}, F1={overall_f1:.4f}")

    # Show degradation
    step1_to_step2_degradation = step1_f1 - step2_f1
    step1_to_overall_degradation = step1_f1 - overall_f1

    print(f"\nPerformance Degradation:")
    print(f"  Step1 → Step2: {step1_to_step2_degradation:.4f}")
    print(f"  Step1 → Overall: {step1_to_overall_degradation:.4f}")

    if step1_to_overall_degradation > 0:
        print("  ✓ Expected degradation observed (Step1 > Overall)")
    else:
        print("  ⚠ Unexpected: Overall performed better than Step1")

    # Save results
    results = {
        'participant_id': participant_id,
        'window_size': window_size,
        'seed': seed,
        'step1': {
            'accuracy': step1_accuracy,
            'precision': step1_precision,
            'recall': step1_recall,
            'f1': step1_f1,
            'confusion_matrix': step1_cm.tolist()
        },
        'step2': {
            'precision': step2_precision,
            'recall': step2_recall,
            'f1': step2_f1
        },
        'overall': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1': overall_f1
        },
        'degradation': {
            'step1_to_step2': step1_to_step2_degradation,
            'step1_to_overall': step1_to_overall_degradation
        }
    }

    import json
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\nResults saved to: {save_dir}")
    return results




if __name__ == "__main__":
    # Test with a single participant
    participant_id = 1
    window_size = 100
    seed = 0

    print("Running Simple Two-Step Pipeline")
    print("This will demonstrate the expected performance degradation")

    try:
        results = simple_two_step_pipeline(participant_id, window_size, seed)
        print("\n" + "=" * 60)
        print("SUCCESS - Pipeline completed!")
        print("=" * 60)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()