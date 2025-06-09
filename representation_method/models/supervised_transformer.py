import itertools
import os
from typing import List, Dict, Tuple, Optional

import matplotlib
import pandas as pd
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from transformers import XLNetModel, XLNetConfig
from torch.utils.data import Dataset

import torch
import numpy as np
from torch.utils.data import DataLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from representation_method.models.self_supervised_transformer import TargetLocalizer, calculate_metrics, \
    random_mask_mse_loss, windows_voting
from representation_method.utils.general_utils import seed_everything

TRAIN = False


class IntegratedEyeTrackingTransformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            n_features: int,
            d_model: int = 256,
            n_heads: int = 8,
            n_layers: int = 6,
            dropout: float = 0.1,
            max_len: int = 1700,
            pretrained_model: str = "xlnet-base-cased"
    ):
        super().__init__()

        d_model = (d_model // n_features // n_heads) * n_features * n_heads
        self.d_model = d_model
        self.feature_dim = d_model // n_features

        self.token_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, self.feature_dim)
            for _ in range(n_features)
        ])

        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

        config = XLNetConfig.from_pretrained(pretrained_model)
        config.num_attention_heads = n_heads
        config.hidden_size = d_model
        config.num_hidden_layers = n_layers
        config.dropout = dropout

        self.transformer = XLNetModel(config)
        self.attention_threshold = nn.Parameter(torch.tensor(0.1, requires_grad=True))
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_features)
        )

        self.classification_head = nn.Linear(d_model, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, output_attentions=False):
        batch_size, seq_len, n_features = x.shape

        embeddings = []
        for i, embedding_layer in enumerate(self.token_embeddings):
            feature_tokens = x[:, :, i].long()
            feature_embedding = embedding_layer(feature_tokens)
            embeddings.append(feature_embedding)

        x_emb = torch.cat(embeddings, dim=-1)
        x_emb = x_emb + self.pos_embedding[:, :seq_len, :]

        attention_mask = torch.ones((batch_size, seq_len), device=x_emb.device)

        outputs = self.transformer(
            inputs_embeds=x_emb,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True
        )

        hidden_states = outputs.last_hidden_state

        reconstructed = self.reconstruction_head(hidden_states)
        classification_logits = self.classification_head(self.dropout(hidden_states))
        attention_weights = outputs.attentions[-1]  # Last layer's attention weights

        if output_attentions:
            return reconstructed, classification_logits, self.attention_threshold, attention_weights
        else:
            return reconstructed, classification_logits


def target_focus_mse_loss(
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        target_locations: torch.Tensor,
        target_weight: float = 1.0
):
    """Efficiently compute MSE loss focused on target locations using vectorized operations."""
    mask = target_locations.unsqueeze(-1).expand_as(original) * target_weight
    squared_error = (reconstructed - original) ** 2
    masked_squared_error = squared_error * mask
    loss = masked_squared_error.sum() / (mask.sum() + 1e-8)
    return loss


def supervised_attention_loss(
        attention_weights: torch.Tensor,
        target_labels: torch.Tensor,
        seq_len: int,
        num_heads: int,
        device: str,
        threshold: torch.nn.Parameter
):
    """Compute supervised attention loss with a learned threshold for each head."""
    bce_loss_fn = nn.BCEWithLogitsLoss()
    target_mask = target_labels.float().to(device)
    total_loss = 0

    for head_idx in range(num_heads):
        head_attention = attention_weights[:, head_idx, :, :]
        avg_attention = head_attention.mean(dim=1)
        pred_attention = torch.sigmoid(avg_attention - threshold).view(-1)
        head_loss = bce_loss_fn(pred_attention, target_mask)
        total_loss += head_loss

    loss = total_loss / num_heads
    return loss


def train_multi_task(
        model: IntegratedEyeTrackingTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 300,
        learning_rate: float = 1e-5,
        device: str = 'cuda',
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 1.0,
        patience: int = 10,
        save_path: str = None
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.006)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for tokens, _, target_labels in train_loader:
            tokens = tokens.to(device)
            target_labels = target_labels.to(device)

            reconstructed, classification_logits, threshold, attention_weights = model(tokens, output_attentions=True)

            recon_loss = target_focus_mse_loss(reconstructed, tokens, target_locations=target_labels)
            classification_logits = classification_logits.view(-1, 2)
            target_labels_flat = target_labels.view(-1)

            class_loss = ce_loss_fn(classification_logits, target_labels_flat)
            seq_len = tokens.shape[1]
            num_heads = attention_weights.shape[1]
            attention_loss = supervised_attention_loss(
                attention_weights, target_labels_flat, seq_len, num_heads, device, threshold
            )

            # Total Loss
            loss = alpha * recon_loss + beta * class_loss + gamma * attention_loss

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for tokens, _, target_labels in val_loader:
                tokens = tokens.to(device)
                target_labels = target_labels.to(device)

                reconstructed, classification_logits, threshold, attention_weights = model(tokens,
                                                                                           output_attentions=True)
                recon_loss = target_focus_mse_loss(reconstructed, tokens, target_locations=target_labels)

                classification_logits = classification_logits.view(-1, 2)
                target_labels_flat = target_labels.view(-1)

                class_loss = ce_loss_fn(classification_logits, target_labels_flat)
                seq_len = tokens.shape[1]
                num_heads = attention_weights.shape[1]
                attention_loss = supervised_attention_loss(
                    attention_weights, target_labels_flat, seq_len, num_heads, device, threshold
                )

                # Total Loss
                loss = alpha * recon_loss + beta * class_loss + gamma * attention_loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # Check for improvement
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            print("Validation loss improved. Saving model...")

            if save_path:
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss
                }
                torch.save(save_dict, save_path)
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")

            # Early stopping
            if patience_counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break

    # Plot training curves
    if save_path:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig(save_path.replace('.pth', '_training_curves.png'))
        plt.close()

    return model


class EyeTrackingDataset(Dataset):
    def __init__(
            self,
            tokens: np.ndarray,
            reconstruction_labels: np.ndarray,
            target_labels: np.ndarray
    ):
        self.tokens = torch.LongTensor(tokens)
        self.reconstruction_labels = torch.FloatTensor(reconstruction_labels)
        self.target_labels = torch.LongTensor(target_labels)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return (
            self.tokens[idx],
            self.reconstruction_labels[idx],
            self.target_labels[idx]
        )


class EyeTrackingTokenizer:
    def __init__(self, n_bins: int = 20, strategy: str = 'quantile'):
        self.n_bins = n_bins
        self.strategy = strategy
        self.discretizers = {}
        self.vocab_size = None
        self.feature_offsets = {}

    def fit(self, df: pd.DataFrame, feature_columns: List[str]):
        current_offset = 0
        for feature in feature_columns:
            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode='ordinal',
                strategy=self.strategy
            )
            values = df[feature].values.reshape(-1, 1)
            discretizer.fit(values)

            self.discretizers[feature] = discretizer
            self.feature_offsets[feature] = current_offset
            current_offset += self.n_bins

        self.vocab_size = current_offset
        return self

    def tokenize(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        tokens_list = []
        for feature in feature_columns:
            values = df[feature].values.reshape(-1, 1)
            discretizer = self.discretizers[feature]
            offset = self.feature_offsets[feature]
            tokens = discretizer.transform(values) + offset
            tokens_list.append(tokens)
        return np.column_stack(tokens_list)


def custom_collate(batch):
    tokens, reconstruction_labels, target_labels = zip(*batch)
    tokens = torch.stack(tokens)
    reconstruction_labels = torch.stack(reconstruction_labels)
    target_labels = torch.stack(target_labels)
    return tokens, reconstruction_labels, target_labels


def create_dataset(windows, reconstruction_labels, target_labels, tokenizer, feature_columns):
    all_tokens = []
    for window in windows:
        tokens = tokenizer.tokenize(
            pd.DataFrame(window, columns=feature_columns),
            feature_columns
        )
        all_tokens.append(tokens)

    tokens_array = np.array(all_tokens)
    return EyeTrackingDataset(tokens_array, reconstruction_labels, target_labels)


def reconstruct_windows_from_cv_results(cv_results_df, target_fold, window_size=100):
    """
    Reconstruct windows from CV results for ensemble positive predictions.
    """
    print(f"Reconstructing windows from CV results for fold {target_fold}")

    # Filter for target fold and test set with positive predictions
    fold_data = cv_results_df[
        (cv_results_df['fold'] == target_fold) &
        (cv_results_df['set'] == 'test') &
        (cv_results_df['prediction'] == 1)  # FILTER BY ENSEMBLE PREDICTION
        ].copy()

    if len(fold_data) == 0:
        print(f"No positive predictions found in fold {target_fold}")
        return np.array([]), np.array([]), []

    print(f"Found {len(fold_data)} points with positive ensemble predictions in fold {target_fold}")

    # Feature columns
    feature_columns = [
        'Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X',
        'CURRENT_FIX_IA_Y', 'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT'
    ]

    windows = []
    labels = []
    metadata = []

    # Group by trial to reconstruct windows
    for (participant, trial), trial_data in fold_data.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']):
        trial_data = trial_data.sort_values('point_id')

        # Create sliding windows
        for start_idx in range(0, len(trial_data) - window_size + 1):
            end_idx = start_idx + window_size
            window_points = trial_data.iloc[start_idx:end_idx]

            # Verify this window was predicted as positive by ensemble
            if window_points['prediction'].iloc[0] == 1:
                # Extract features
                window_features = window_points[feature_columns].values

                # Get ground truth target positions for evaluation
                target_positions = []
                for i, (_, point) in enumerate(window_points.iterrows()):
                    if point['target'] == True:  # Ground truth target
                        target_positions.append(i)

                # Store window data
                windows.append(window_features)
                # Label: 1 if window contains any targets, 0 if no targets
                labels.append(1 if len(target_positions) > 0 else 0)

                metadata.append({
                    'participant_id': participant,
                    'trial_id': trial,
                    'window_start': window_points['point_id'].iloc[0],
                    'window_end': window_points['point_id'].iloc[-1],
                    'relative_target_positions': target_positions,
                    'ensemble_probability': window_points['probability'].iloc[0],
                    'has_ground_truth_targets': len(target_positions) > 0,
                    'window_label': 1 if len(target_positions) > 0 else 0  # Binary label for the window
                })

    X_windows = np.array(windows) if windows else np.array([])
    Y_windows = np.array(labels) if labels else np.array([])

    print(f"Reconstructed {len(X_windows)} windows from ensemble positive predictions")
    print(f"Windows with ground truth targets: {sum(1 for m in metadata if m['has_ground_truth_targets'])}")
    print(f"Windows without targets: {sum(1 for m in metadata if not m['has_ground_truth_targets'])}")

    return X_windows, Y_windows, metadata


def two_stage_target_localization_with_splits(
        cv_results_path: str,
        participant_id: int,
        target_fold: int = 0,
        window_size: int = 100,
        seed: int = 0,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
        epochs: int = 200,
        learning_rate: float = 1e-5,
        train_model: bool = True
):
    """
    Two-stage target localization with proper train/val/test splits on ensemble-predicted windows.

    Stage 1: Ensemble already predicted which windows contain targets
    Stage 2: For predicted windows, train model to localize exact target positions

    Args:
        cv_results_path: Path to CV results CSV
        participant_id: Participant ID
        target_fold: Which fold to use
        window_size: Size of the sliding window
        seed: Random seed
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        epochs: Number of training epochs
        learning_rate: Learning rate
        train_model: Whether to train or load existing model
    """
    seed_everything(seed)

    print(f"=== Two-Stage Target Localization with Train/Val/Test Splits ===")
    print(f"Participant: {participant_id}, Fold: {target_fold}")
    print(f"Window size: {window_size}")
    print(f"Train/Val/Test ratios: {train_ratio}/{val_ratio}/{test_ratio}")

    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Setup directories
    method_dir = os.path.join(
        'results',
        f'two_stage_supervised_participant_{participant_id}_fold_{target_fold}_window_{window_size}'
    )
    os.makedirs(method_dir, exist_ok=True)

    # Feature columns
    feature_columns = [
        'Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X',
        'CURRENT_FIX_IA_Y', 'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT'
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load CV results and reconstruct windows
    print(f"Loading CV results from: {cv_results_path}")
    cv_results_df = pd.read_csv(cv_results_path)

    X_all, Y_all, window_metadata = reconstruct_windows_from_cv_results(
        cv_results_df, target_fold, window_size
    )

    if len(X_all) == 0:
        print("No windows found with positive ensemble predictions!")
        return

    print(f"Total ensemble-predicted windows: {len(X_all)}")
    print(f"Windows with targets: {sum(Y_all)}")
    print(f"Windows without targets: {len(Y_all) - sum(Y_all)}")

    # Create train/val/test splits on ALL ensemble-predicted windows
    # First split: train vs (val + test)
    X_train, X_temp, Y_train, Y_temp, meta_train, meta_temp = train_test_split(
        X_all, Y_all, window_metadata,
        test_size=(val_ratio + test_ratio),
        random_state=seed,
        stratify=Y_all,  # Stratify to maintain class balance
        shuffle=True
    )

    # Second split: val vs test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    X_val, X_test, Y_val, Y_test, meta_val, meta_test = train_test_split(
        X_temp, Y_temp, meta_temp,
        test_size=(1 - val_test_ratio),
        random_state=seed,
        stratify=Y_temp,  # Stratify to maintain class balance
        shuffle=True
    )

    print(f"\nData splits:")
    print(f"Train: {len(X_train)} windows (targets: {sum(Y_train)}, no targets: {len(Y_train) - sum(Y_train)})")
    print(f"Val: {len(X_val)} windows (targets: {sum(Y_val)}, no targets: {len(Y_val) - sum(Y_val)})")
    print(f"Test: {len(X_test)} windows (targets: {sum(Y_test)}, no targets: {len(Y_test) - sum(Y_test)})")

    # Create and fit tokenizer on training data
    print("Creating tokenizer...")
    tokenizer = EyeTrackingTokenizer(n_bins=20, strategy='quantile')
    X_train_flat = X_train.reshape(-1, X_train.shape[-1])
    X_train_df = pd.DataFrame(X_train_flat, columns=feature_columns)
    tokenizer.fit(X_train_df, feature_columns)

    def create_target_vector(window_size, relative_positions):
        """Create binary target vector for positions in window"""
        target_vector = [0] * window_size
        for pos in relative_positions:
            if 0 <= pos < window_size:
                target_vector[pos] = 1
        return target_vector

    # Create datasets with target localization labels
    train_dataset = create_dataset(
        X_train,
        np.ones(len(X_train)),  # reconstruction labels (not used in loss)
        [create_target_vector(window_size, meta['relative_target_positions']) for meta in meta_train],
        tokenizer,
        feature_columns
    )

    val_dataset = create_dataset(
        X_val,
        np.ones(len(X_val)),  # reconstruction labels
        [create_target_vector(window_size, meta['relative_target_positions']) for meta in meta_val],
        tokenizer,
        feature_columns
    )

    test_dataset = create_dataset(
        X_test,
        np.ones(len(X_test)),  # reconstruction labels
        [create_target_vector(window_size, meta['relative_target_positions']) for meta in meta_test],
        tokenizer,
        feature_columns
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)

    # Create model
    model = IntegratedEyeTrackingTransformer(
        vocab_size=tokenizer.vocab_size,
        n_features=len(feature_columns),
        d_model=256,
        max_len=window_size
    )

    best_model_path = os.path.join(method_dir, "best_supervised_model.pth")

    if train_model:
        print("\n=== Training Supervised Model ===")

        # Try to load pretrained self-supervised model
        pretrained_path = f'results/self_supervised_from_cv_participant_{participant_id}_fold_{target_fold}_window_size_{window_size}/model.pth'

        if os.path.exists(pretrained_path):
            print(f"Loading pretrained weights from: {pretrained_path}")
            try:
                pretrained_state = torch.load(pretrained_path)['model_state_dict']
                model_state = model.state_dict()

                # Filter compatible weights
                filtered_state = {k: v for k, v in pretrained_state.items()
                                  if k in model_state and model_state[k].shape == v.shape}
                model_state.update(filtered_state)
                model.load_state_dict(model_state)
                print(f"Loaded {len(filtered_state)} compatible layers from pretrained model")
            except Exception as e:
                print(f"Error loading pretrained model: {e}")
                print("Training from scratch")
        else:
            print(f"Pretrained model not found at {pretrained_path}, training from scratch")

        # Train the model
        model = train_multi_task(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            alpha=0.1,  # Reconstruction loss weight
            beta=0.1,  # Classification loss weight
            gamma=15.0,  # Attention loss weight
            patience=15,
            save_path=best_model_path
        )

    else:
        print("Loading existing supervised model...")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {best_model_path}")
        else:
            print(f"Model not found at {best_model_path}!")
            return

    # Move model to device for evaluation
    model = model.to(device)
    localizer = TargetLocalizer(model, device)

    print("\n=== Evaluating Two-Stage Performance ===")

    # Stage 1: Evaluate ensemble predictions (window-level classification)
    print("\n--- Stage 1: Ensemble Window Prediction Evaluation ---")
    stage1_true_labels = Y_test  # 1 if window has targets, 0 if not
    stage1_pred_labels = [1] * len(Y_test)  # All test windows were predicted as positive by ensemble

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    stage1_accuracy = accuracy_score(stage1_true_labels, stage1_pred_labels)
    stage1_precision = precision_score(stage1_true_labels, stage1_pred_labels, zero_division=0)
    stage1_recall = recall_score(stage1_true_labels, stage1_pred_labels, zero_division=0)
    stage1_f1 = f1_score(stage1_true_labels, stage1_pred_labels, zero_division=0)
    stage1_cm = confusion_matrix(stage1_true_labels, stage1_pred_labels)

    print(f"Stage 1 - Ensemble Window Classification Results:")
    print(f"  Total test windows: {len(Y_test)}")
    print(f"  Windows with targets (ground truth): {sum(Y_test)}")
    print(f"  Windows without targets (ground truth): {len(Y_test) - sum(Y_test)}")
    print(f"  Windows predicted as positive (ensemble): {len(Y_test)}")  # All of them
    print(f"  Accuracy: {stage1_accuracy:.4f}")
    print(f"  Precision: {stage1_precision:.4f}")
    print(f"  Recall: {stage1_recall:.4f}")
    print(f"  F1 Score: {stage1_f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    TN: {stage1_cm[0, 0]}, FP: {stage1_cm[0, 1]}")
    print(f"    FN: {stage1_cm[1, 0]}, TP: {stage1_cm[1, 1]}")

    # Stage 2: Evaluate target localization (only on windows with actual targets)
    print("\n--- Stage 2: Target Localization Evaluation ---")
    stage2_results = []
    all_window_results = []  # For all windows, including those without targets

    for i, (window, meta) in enumerate(zip(X_test, meta_test)):
        if i % 50 == 0:
            print(f"Processing test window {i + 1}/{len(X_test)}")

        window_2d = window.squeeze()
        window_df = pd.DataFrame(window_2d, columns=feature_columns)
        tokenized_window = tokenizer.tokenize(window_df, feature_columns)
        window_tensor = torch.tensor(tokenized_window, dtype=torch.long).unsqueeze(0)

        target_positions = meta['relative_target_positions']
        has_targets = len(target_positions) > 0

        # Always run localization (for consistency), but handle empty targets
        if has_targets:
            results = localizer.localize_targets(
                window_tensor,
                target_positions,
                meta['window_start'],
                supervised=True
            )
        else:
            # For windows without targets, run localization but with empty target list
            results = localizer.localize_targets(
                window_tensor,
                [],  # Empty target list
                meta['window_start'],
                supervised=True
            )
            # Override some metrics since there are no targets to find
            results['similarity_score'] = 0.0
            results['total_targets'] = 0
            results['precision'] = 0.0
            results['recall'] = 0.0
            results['f1_score'] = 0.0

        # Add metadata
        results['participant_id'] = meta['participant_id']
        results['trial_id'] = meta['trial_id']
        results['ensemble_probability'] = meta['ensemble_probability']
        results['window_label'] = meta['window_label']
        results['has_targets'] = has_targets
        results['ground_truth_has_targets'] = has_targets
        results['ensemble_predicted_positive'] = True  # All test windows were predicted positive

        all_window_results.append(results)

        # Only add to stage2_results if window actually has targets
        if has_targets:
            stage2_results.append(results)

    # Save and analyze results
    print(f"\n=== Results Summary ===")

    # Save all window results (including those without targets)
    all_results_df = pd.DataFrame(all_window_results)
    all_results_df.to_csv(os.path.join(method_dir, 'all_test_results.csv'), index=False)

    # Save only results for windows with targets (traditional evaluation)
    if len(stage2_results) > 0:
        stage2_results_df = pd.DataFrame(stage2_results)
        stage2_results_df.to_csv(os.path.join(method_dir, 'stage2_target_localization_results.csv'), index=False)

        print(f"\nStage 2 - Target Localization Results (windows with targets only):")
        print(f"  Windows with targets evaluated: {len(stage2_results)}")
        print(f"  Average similarity score: {stage2_results_df['similarity_score'].mean():.4f}")

        # Calculate overall metrics for target localization
        stage2_metrics = calculate_metrics(stage2_results_df)
        for k, v in stage2_metrics.items():
            print(f"  {k}: {v:.4f}")

        print(f"\nDetailed Target Localization Results:")
        print(f"  Perfect matches (similarity = 1.0): {sum(stage2_results_df['similarity_score'] == 1.0)}")
        print(f"  Good matches (similarity >= 0.8): {sum(stage2_results_df['similarity_score'] >= 0.8)}")
        print(f"  Poor matches (similarity < 0.5): {sum(stage2_results_df['similarity_score'] < 0.5)}")

        # Analyze by ensemble probability
        if 'ensemble_probability' in stage2_results_df.columns:
            high_prob = stage2_results_df[stage2_results_df['ensemble_probability'] >= 0.8]
            low_prob = stage2_results_df[stage2_results_df['ensemble_probability'] < 0.8]

            if len(high_prob) > 0:
                print(f"\n  High ensemble confidence windows (>= 0.8): {len(high_prob)}")
                print(f"  Average similarity: {high_prob['similarity_score'].mean():.4f}")

            if len(low_prob) > 0:
                print(f"  Low ensemble confidence windows (< 0.8): {len(low_prob)}")
                print(f"  Average similarity: {low_prob['similarity_score'].mean():.4f}")

        # Run windows voting analysis
        windows_voting(stage2_results_df, method_dir, suffix="_stage2_targets")
    else:
        print("No windows with targets found for Stage 2 evaluation!")
        stage2_metrics = {'precision': 0, 'recall': 0, 'f1': 0}

    # Overall two-stage analysis
    print(f"\n=== Overall Two-Stage Analysis ===")
    print(f"Stage 1 (Ensemble Window Classification):")
    print(f"  - Accuracy: {stage1_accuracy:.4f}")
    print(f"  - Precision: {stage1_precision:.4f}")
    print(f"  - Recall: {stage1_recall:.4f}")
    print(f"  - F1 Score: {stage1_f1:.4f}")

    if len(stage2_results) > 0:
        print(f"Stage 2 (Target Localization within predicted windows):")
        print(f"  - Average similarity: {stage2_results_df['similarity_score'].mean():.4f}")
        print(f"  - Precision: {stage2_metrics['precision']:.4f}")
        print(f"  - Recall: {stage2_metrics['recall']:.4f}")
        print(f"  - F1 Score: {stage2_metrics['f1']:.4f}")

    # Calculate end-to-end performance
    # This would be: Stage1_Recall * Stage2_Performance for windows that have targets
    if len(stage2_results) > 0:
        end_to_end_recall = stage1_recall * stage2_metrics['recall']
        print(f"End-to-end effective recall: {end_to_end_recall:.4f}")
        print(f"  (Stage 1 recall × Stage 2 recall = {stage1_recall:.4f} × {stage2_metrics['recall']:.4f})")

    # Save comprehensive summary
    summary = {
        'participant_id': participant_id,
        'target_fold': target_fold,
        'window_size': window_size,

        # Stage 1 metrics
        'stage1_total_windows': len(Y_test),
        'stage1_windows_with_targets': sum(Y_test),
        'stage1_windows_without_targets': len(Y_test) - sum(Y_test),
        'stage1_accuracy': stage1_accuracy,
        'stage1_precision': stage1_precision,
        'stage1_recall': stage1_recall,
        'stage1_f1': stage1_f1,

        # Stage 2 metrics
        'stage2_windows_evaluated': len(stage2_results),
        'stage2_average_similarity': stage2_results_df['similarity_score'].mean() if len(stage2_results) > 0 else 0,
        'stage2_precision': stage2_metrics['precision'],
        'stage2_recall': stage2_metrics['recall'],
        'stage2_f1': stage2_metrics['f1'],
        'stage2_perfect_matches': sum(stage2_results_df['similarity_score'] == 1.0) if len(stage2_results) > 0 else 0,
        'stage2_good_matches': sum(stage2_results_df['similarity_score'] >= 0.8) if len(stage2_results) > 0 else 0,
        'stage2_poor_matches': sum(stage2_results_df['similarity_score'] < 0.5) if len(stage2_results) > 0 else 0,

        # End-to-end metrics
        'end_to_end_recall': stage1_recall * stage2_metrics['recall'] if len(stage2_results) > 0 else 0
    }

    with open(os.path.join(method_dir, 'test_summary.json'), 'w') as f:
        import json
        json.dump(summary, f, indent=4, default=str)

    print(f"\nResults saved to: {method_dir}")
    return summary


if __name__ == "__main__":
    participant_id = 1
    target_fold = 0  # Which fold to use for evaluation
    seed = 0

    try:
        # Read from CV results
        cv_results_path = Path(
            __file__).parent.parent / "outputs" / f"participant_{participant_id}" / "all_folds_results.csv"

        # Check if CV results file exists
        if not os.path.exists(cv_results_path):
            raise FileNotFoundError(f"CV results file not found: {cv_results_path}")

        print(f"Using CV results from: {cv_results_path}")

        # Run two-stage target localization with proper train/val/test splits
        summary = two_stage_target_localization_with_splits(
            cv_results_path=str(cv_results_path),
            participant_id=participant_id,
            target_fold=target_fold,
            window_size=100,
            seed=seed,
            train_ratio=0.7,  # 60% for training
            val_ratio=0.15,  # 20% for validation
            test_ratio=0.15,  # 20% for testing
            epochs=200,
            learning_rate=1e-5,
            train_model=TRAIN
        )

        print("\n=== FINAL SUMMARY ===")
        if summary:
            print(f"Stage 1 - Ensemble Window Classification:")
            print(f"  - Total windows: {summary['stage1_total_windows']}")
            print(f"  - Windows with targets: {summary['stage1_windows_with_targets']}")
            print(f"  - Windows without targets: {summary['stage1_windows_without_targets']}")
            print(f"  - Accuracy: {summary['stage1_accuracy']:.4f}")
            print(f"  - Precision: {summary['stage1_precision']:.4f}")
            print(f"  - Recall: {summary['stage1_recall']:.4f}")
            print(f"  - F1 Score: {summary['stage1_f1']:.4f}")

            print(f"\nStage 2 - Target Localization:")
            print(f"  - Windows evaluated: {summary['stage2_windows_evaluated']}")
            print(f"  - Average similarity score: {summary['stage2_average_similarity']:.4f}")
            print(f"  - Precision: {summary['stage2_precision']:.4f}")
            print(f"  - Recall: {summary['stage2_recall']:.4f}")
            print(f"  - F1 Score: {summary['stage2_f1']:.4f}")
            print(f"  - Perfect matches: {summary['stage2_perfect_matches']}")
            print(f"  - Good matches (>= 0.8): {summary['stage2_good_matches']}")
            print(f"  - Poor matches (< 0.5): {summary['stage2_poor_matches']}")

            print(f"\nEnd-to-End Performance:")
            print(f"  - Effective recall: {summary['end_to_end_recall']:.4f}")
            print(f"    (This represents the overall recall when both stages are combined)")

            print(f"\nInterpretation:")
            print(
                f"  - Stage 1 recall ({summary['stage1_recall']:.4f}) shows how well ensemble identified target windows")
            print(
                f"  - Stage 2 recall ({summary['stage2_recall']:.4f}) shows how well transformer localized targets within predicted windows")
            print(f"  - End-to-end recall ({summary['end_to_end_recall']:.4f}) shows overall system performance")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()