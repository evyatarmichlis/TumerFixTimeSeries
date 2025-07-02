import ast
import heapq
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import KBinsDiscretizer
from transformers import XLNetModel, XLNetConfig, BigBirdConfig, BigBirdModel
from torch.utils.data import Dataset
import seaborn as sns
from transformers import LongformerConfig, LongformerModel, LongformerTokenizer

import torch
import numpy as np
from torch.utils.data import DataLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from representation_method.utils.data_loader import DataConfig, load_eye_tracking_data
from representation_method.utils.data_utils import create_dynamic_time_series, split_train_test_for_time_series, \
    create_dynamic_time_series_with_indices
from representation_method.utils.general_utils import seed_everything

TRAIN = True


class EyeTrackingDataset(Dataset):
    def __init__(
            self,
            tokens: np.ndarray,
            labels: np.ndarray,
    ):
        self.tokens = torch.LongTensor(tokens)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return (
            self.tokens[idx],
            self.labels[idx]
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

        # Ensure d_model is divisible by both n_heads and n_features
        d_model = (d_model // n_features // n_heads) * n_features * n_heads
        self.d_model = d_model
        self.feature_dim = d_model // n_features

        # Token embeddings for each feature
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, self.feature_dim)
            for _ in range(n_features)
        ])

        # Position encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

        # Load pre-trained transformer config and adjust
        config = XLNetConfig.from_pretrained(pretrained_model)
        config.num_attention_heads = n_heads
        config.hidden_size = d_model
        config.num_hidden_layers = n_layers
        config.dropout = dropout

        # Initialize transformer backbone
        self.transformer = XLNetModel(config)

        # Self-supervised head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_features)
        )

    def forward(self, x, output_attentions=False):
        batch_size, seq_len, n_features = x.shape

        # Embed each feature
        embeddings = []
        for i, embedding_layer in enumerate(self.token_embeddings):
            feature_tokens = x[:, :, i].long()
            feature_embedding = embedding_layer(feature_tokens)
            embeddings.append(feature_embedding)

        x = torch.cat(embeddings, dim=-1)

        x = x + self.pos_embedding[:, :seq_len, :]
        attention_mask = torch.ones((batch_size, seq_len), device=x.device)

        outputs = self.transformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True
        )

        hidden_states = outputs.last_hidden_state
        reconstructed = self.reconstruction_head(hidden_states)

        if output_attentions:
            return reconstructed, outputs.attentions
        return reconstructed


def custom_collate(batch):
    """Custom collate function to handle variable-length target positions."""
    tokens, labels = zip(*batch)
    tokens = torch.stack(tokens)
    labels = torch.stack(labels)
    return tokens, labels


def random_mask_mse_loss(reconstructed, original, mask_ratio=0.15, mask_strategy='random'):
    """Create a loss with random masking"""
    batch_size, seq_len, n_features = original.shape
    mask = torch.zeros_like(original, dtype=torch.float32)

    for b in range(batch_size):
        if mask_strategy == 'random':
            mask_indices = torch.rand(seq_len) < mask_ratio
            mask[b, mask_indices, :] = 1.0
        elif mask_strategy == 'consecutive':
            num_masked_steps = int(seq_len * mask_ratio)
            start = random.randint(0, seq_len - num_masked_steps)
            mask[b, start:start + num_masked_steps, :] = 1.0
        elif mask_strategy == 'mixed':
            random_mask_indices = torch.rand(seq_len) < (mask_ratio / 2)
            mask[b, random_mask_indices, :] = 1.0
            num_consecutive_steps = int(seq_len * (mask_ratio / 2))
            start = random.randint(0, seq_len - num_consecutive_steps)
            mask[b, start:start + num_consecutive_steps, :] = 1.0

    mask = mask.to(reconstructed.device)
    squared_error = (reconstructed - original) ** 2
    masked_squared_error = squared_error * mask
    loss = masked_squared_error.sum() / (mask.sum() + 1e-8)
    return loss


def train_self_supervised(
        model: IntegratedEyeTrackingTransformer,
        train_loader: DataLoader,
        epochs: int = 50,
        learning_rate: float = 1e-5,
        device: str = 'cuda',
        patience: int = 5
):
    """Train the model in self-supervised mode"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            windows = batch[0].to(device)
            windows = windows.float()
            reconstructed = model(windows)
            loss = random_mask_mse_loss(
                reconstructed,
                windows,
                mask_ratio=0.3,
                mask_strategy='random'
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

    return model


class TargetLocalizer:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def analyze_attention(self, window_data: torch.Tensor, supervised=False) -> np.ndarray:
        """Analyze attention patterns to localize targets"""
        self.model.eval()
        with torch.no_grad():
            output = self.model(window_data.to(self.device), output_attentions=True)
            if supervised:
                attention_scores = output[-1]
            else:
                attention_scores = output[-1][-1]
        return attention_scores.cpu().numpy()

    def localize_targets(self, window_data: torch.Tensor, target_positions,
                                                        window_start, window_metadata, supervised=False):
        """Enhanced version that tracks ailment positions alongside target positions"""
        attentions = self.analyze_attention(window_data, supervised).squeeze()
        avg_attention = attentions.mean(axis=0)
        token_attentions = avg_attention.mean(axis=0)
        seq_len = token_attentions.shape[0]

        results = {
            "similarity_score": 0,
            "total_targets": len(target_positions),
            "window_size": seq_len,
            "top_k_positions": [],
            'target_locations': target_positions,
        }

        top_k_indices = dynamic_topk_by_threshold(token_attentions, 1.7)
        results["top_k_positions"] = top_k_indices
        results["abs_top_k_positions"] = [idx + window_start for idx in top_k_indices]
        results["abs_target_locations"] = [pos + window_start for pos in target_positions]

        # Original target matching logic
        matched_targets = set()
        matched_predictions = set()
        tolerance = 0

        for pred_pos in top_k_indices:
            for target_idx, target_pos in enumerate(target_positions):
                if target_idx not in matched_targets and abs(target_pos - pred_pos) <= tolerance:
                    matched_targets.add(target_idx)
                    matched_predictions.add(pred_pos)
                    break

        true_positives = len(matched_targets)
        precision = true_positives / len(top_k_indices) if any(top_k_indices) else 0.0
        recall = true_positives / len(target_positions) if any(target_positions) else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        # Calculate similarity score
        for target_pos in target_positions:
            min_distance = float('inf')
            for attention_argmax in top_k_indices:
                distance = abs(attention_argmax - target_pos)
                min_distance = min(min_distance, distance)

            similarity = 1 - (min_distance / seq_len)
            results["similarity_score"] += max(0, similarity)

        results["similarity_score"] /= len(target_positions) if len(target_positions) > 0 else 1

        # NEW: Ailment tracking logic
        ailment_results = self.track_ailments_in_predictions(
            top_k_indices, window_metadata, tolerance=tolerance
        )

        results.update({
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "num_predictions": len(top_k_indices),
            "num_targets": len(target_positions),
            # Add ailment tracking results
            **ailment_results
        })

        return results

    def track_ailments_in_predictions(self,predicted_positions, window_metadata, tolerance=1):
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
                    for detail in ailment_details:
                        if detail['relative_position'] == ailment_pos:
                            found_ailments.add(detail['ailment_number'])
                            break

        ailment_results['ailment_positions_matched'] = len(matched_ailment_positions)
        ailment_results['ailments_found_by_model'] = found_ailments
        ailment_results['ailment_positions_found'] = list(matched_ailment_positions)
        ailment_results['ailment_detection_success'] = len(found_ailments) > 0

        return ailment_results


def calculate_metrics(df):
    """Calculate confusion matrix metrics from DataFrame containing TP, predictions and targets"""
    df['FP'] = df['num_predictions'] - df['true_positives']
    df['FN'] = df['total_targets'] - df['true_positives']

    total_TP = df['true_positives'].sum()
    total_FP = df['FP'].sum()
    total_FN = df['FN'].sum()

    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {'precision': precision, 'recall': recall, 'f1': f1_score}


def dynamic_topk_by_threshold(attns: np.ndarray,
                              std_multiplier: float = 2.0,
                              fallback_k: int = 3) -> List[int]:
    """
    Return indices above mean+multiplier*std; if none, return top `fallback_k`.
    """
    mean_attn = np.mean(attns)
    std_attn  = np.std(attns)
    thresh    = mean_attn + std_multiplier * std_attn
    idxs      = np.where(attns > thresh)[0]
    if len(idxs) == 0:
         idxs = np.argsort(attns)[-fallback_k:]
    return idxs.tolist()


def create_dataset(windows, labels, tokenizer, feature_columns):
    all_tokens = []
    for window in windows:
        tokens = tokenizer.tokenize(
            pd.DataFrame(window, columns=feature_columns),
            feature_columns
        )
        all_tokens.append(tokens)

    tokens_array = np.array(all_tokens)
    print(f"Tokenized shape: {tokens_array.shape}")
    return EyeTrackingDataset(tokens_array, labels)


def z_score_detection(window_df, feature_columns, threshold=3.0):
    """Detect anomalies based on Z-Score in a given window."""
    anomalies = []
    for feature in feature_columns:
        mean = window_df[feature].mean()
        std = window_df[feature].std() + 1e-8
        z_scores = (window_df[feature] - mean) / std
        anomaly_indices = window_df[z_scores.abs() > threshold].index.tolist()
        anomalies.extend(anomaly_indices)
    return sorted(set(anomalies))


def reconstruct_windows_from_cv_results(cv_results_df, target_fold, window_size=100):
    """
    Reconstruct windows from CV results for ensemble positive predictions.

    Args:
        cv_results_df: DataFrame with CV results
        target_fold: Which fold to use for evaluation
        window_size: Size of the sliding window

    Returns:
        X_windows: Array of window features
        Y_windows: Array of window labels (ensemble predictions)
        window_metadata: List of metadata for each window
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

    # Feature columns (excluding AILMENT_NUMBER)
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
                labels.append(1)  # All these windows were predicted as positive

                metadata.append({
                    'participant_id': participant,
                    'trial_id': trial,
                    'window_start': window_points['point_id'].iloc[0],
                    'window_end': window_points['point_id'].iloc[-1],
                    'relative_target_positions': target_positions,
                    'ensemble_probability': window_points['probability'].iloc[0],
                    'has_ground_truth_targets': len(target_positions) > 0
                })

    X_windows = np.array(windows) if windows else np.array([])
    Y_windows = np.array(labels) if labels else np.array([])

    print(f"Reconstructed {len(X_windows)} windows from ensemble positive predictions")
    print(f"Windows with ground truth targets: {sum(1 for m in metadata if m['has_ground_truth_targets'])}")

    return X_windows, Y_windows, metadata


def find_attention_from_cv_results(cv_results_path, participant_id, target_fold=0, window_size=100, seed=0):
    """
    Modified version that reads from CV results instead of original CSV.
    Filters by ensemble predictions (pred == 1) instead of ground truth (target == 1).
    """
    seed_everything(seed)

    print(f"Loading CV results from: {cv_results_path}")
    cv_results_df = pd.read_csv(cv_results_path)

    # Feature columns (excluding AILMENT_NUMBER)
    feature_columns = [
        'Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X',
        'CURRENT_FIX_IA_Y', 'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT'
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method_dir = os.path.join('results',
                              f'self_supervised_from_cv_participant_{participant_id}_fold_{target_fold}_window_size_{window_size}')
    os.makedirs(method_dir, exist_ok=True)

    print(f"Window size: {window_size}")
    print(f"Target fold: {target_fold}")

    # Reconstruct windows from CV results (filtered by ensemble predictions)
    X_all, Y_all, window_metadata = reconstruct_windows_from_cv_results(
        cv_results_df, target_fold, window_size
    )

    if len(X_all) == 0:
        print("No windows found with positive ensemble predictions!")
        return

    # Add differential features if needed
    # feature_columns += ['diff_pupil', 'diff_fix_duration']

    # Create tokenizer and fit on the data
    tokenizer = EyeTrackingTokenizer()
    X_all_flat = X_all.reshape(-1, X_all.shape[-1])
    X_all_df = pd.DataFrame(X_all_flat, columns=feature_columns)
    tokenizer.fit(X_all_df, feature_columns)

    # Create dataset and dataloader
    all_dataset = create_dataset(X_all, Y_all, tokenizer, feature_columns)
    dataloader = DataLoader(all_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

    # Create model
    model = IntegratedEyeTrackingTransformer(
        vocab_size=tokenizer.vocab_size,
        n_features=len(feature_columns),
        d_model=256,
        max_len=window_size,
        pretrained_model='xlnet-base-cased'
    )

    best_model_path = os.path.join(method_dir, "model.pth")

    if TRAIN:
        print("Training self-supervised model on ensemble positive predictions...")
        model = train_self_supervised(
            model=model,
            train_loader=dataloader,
            epochs=300,
            learning_rate=1e-4,
            device='cuda'
        )
        save_dict = {
            'model_state_dict': model.state_dict(),
            'vocab_size': tokenizer.vocab_size,
            'n_features': len(feature_columns)
        }
        torch.save(save_dict, best_model_path)
    else:
        print("Loading existing model...")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    localizer = TargetLocalizer(model, device)

    # Evaluate target localization
    print("\nEvaluating target localization on ensemble positive windows...")
    all_results = []
    total_z_score_precision = 0
    total_z_score_recall = 0
    total_z_score_f1 = 0
    total_z_score_windows = 0

    for i, (window, meta) in enumerate(zip(X_all, window_metadata)):
        if i % 100 == 0:
            print(f"Processing window {i + 1}/{len(X_all)}")

        window_2d = window.squeeze()
        window_df = pd.DataFrame(window_2d, columns=feature_columns)
        tokenized_window = tokenizer.tokenize(window_df, feature_columns)
        window_tensor = torch.tensor(tokenized_window, dtype=torch.long).unsqueeze(0)

        # Only evaluate if there are ground truth targets in this window
        if len(meta['relative_target_positions']) > 0:
            results = localizer.localize_targets(
                window_tensor,
                meta['relative_target_positions'],
                meta['window_start']
            )

            results['participant_id'] = meta['participant_id']
            results['trial_id'] = meta['trial_id']
            results['ensemble_probability'] = meta['ensemble_probability']

            # Z-score detection for comparison
            z_score_anomalies = z_score_detection(window_df, feature_columns)
            z_score_target_matches = sum(
                1 for target in meta['relative_target_positions']
                if any(abs(target - anomaly) <= 5 for anomaly in z_score_anomalies)
            )

            z_score_precision = z_score_target_matches / len(z_score_anomalies) if z_score_anomalies else 0.0
            z_score_recall = z_score_target_matches / len(meta['relative_target_positions']) if len(
                meta['relative_target_positions']) > 0 else 0.0
            z_score_f1 = 2 * (z_score_precision * z_score_recall) / (z_score_precision + z_score_recall) if (
                                                                                                                        z_score_precision + z_score_recall) > 0 else 0.0

            results.update({
                'z_score_anomalies': z_score_anomalies,
                'z_score_precision': z_score_precision,
                'z_score_recall': z_score_recall,
                'z_score_f1': z_score_f1
            })

            total_z_score_precision += z_score_precision
            total_z_score_recall += z_score_recall
            total_z_score_f1 += z_score_f1
            total_z_score_windows += 1

            all_results.append(results)

    if len(all_results) == 0:
        print("No windows with ground truth targets found for evaluation!")
        return

    # Save and analyze results
    target_pick_method = 'threshold'
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(os.path.join(method_dir, f'{target_pick_method}_attention.csv'))

    print('############ percent similarity_score ############')
    print(all_results_df['similarity_score'].mean())

    metrics = calculate_metrics(all_results_df)
    for k, v in metrics.items():
        print(f'############ {k} ############')
        print(f'$$$$$$$$$$$$ {v} $$$$$$$$$$$$')

    if total_z_score_windows > 0:
        avg_z_score_precision = total_z_score_precision / total_z_score_windows
        avg_z_score_recall = total_z_score_recall / total_z_score_windows
        avg_z_score_f1 = total_z_score_f1 / total_z_score_windows

        print("\nAverage Z-Score Metrics:")
        print(f"Z-Score Precision: {avg_z_score_precision:.4f}")
        print(f"Z-Score Recall: {avg_z_score_recall:.4f}")
        print(f"Z-Score F1-Score: {avg_z_score_f1:.4f}")

    # Create visualizations
    sns.histplot(data=all_results_df, x='similarity_score', bins=10, kde=True)
    plt.xlabel('Percent Coincidences')
    plt.ylabel('Count')
    plt.title('Distribution of Percent Coincidences')
    plt.savefig(os.path.join(method_dir, f'similarity_score.png'))
    plt.close()

    windows_voting(all_results_df, method_dir)

    print(f"\nResults saved to: {method_dir}")


def windows_voting(all_results_df, method_dir,suffix = ''):
    """Same as original windows_voting function"""
    required_columns = ['participant_id', 'trial_id', 'abs_top_k_positions', 'abs_target_locations']
    for col in required_columns:
        if col not in all_results_df.columns:
            raise KeyError(f"Missing required column: {col}")

    def safe_literal_eval(val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                print(f"Warning: Skipping malformed entry: {val}")
                return []
        elif isinstance(val, list):
            return val
        else:
            print(f"Warning: Unexpected data type: {type(val)}")
            return []

    all_results_df['abs_top_k_positions'] = all_results_df['abs_top_k_positions'].apply(safe_literal_eval)
    all_results_df['abs_target_locations'] = all_results_df['abs_target_locations'].apply(safe_literal_eval)

    grouped_results = defaultdict(lambda: {'ground_truth': set(), 'predictions': defaultdict(int)})

    for _, row in all_results_df.iterrows():
        participant_id = row['participant_id']
        trial_id = row['trial_id']
        scan_id = (participant_id, trial_id)

        if isinstance(row['abs_target_locations'], list):
            grouped_results[scan_id]['ground_truth'].update(row['abs_target_locations'])

        if isinstance(row['abs_top_k_positions'], list):
            for position in row['abs_top_k_positions']:
                grouped_results[scan_id]['predictions'][position] += 1

    tolerance = 10
    agreement_thresholds = np.arange(0.1, 1.0, 0.1)
    threshold_results = []

    for agreement_threshold in agreement_thresholds:
        per_scan_results = []

        for scan_id, data in grouped_results.items():
            participant_id, trial_id = scan_id
            ground_truth = sorted(data['ground_truth'])

            total_windows = len(all_results_df[(all_results_df['participant_id'] == participant_id) &
                                               (all_results_df['trial_id'] == trial_id)])
            position_votes = data['predictions']
            position_agreement = {pos: count / total_windows for pos, count in position_votes.items()}

            predictions = sorted(
                [position for position, agreement in position_agreement.items() if agreement >= agreement_threshold]
            )

            matched_predictions = set()
            matched_ground_truth = set()

            for predicted in predictions:
                for target in ground_truth:
                    if target not in matched_ground_truth and abs(predicted - target) <= tolerance:
                        matched_predictions.add(predicted)
                        matched_ground_truth.add(target)
                        break

            true_positives = len(matched_predictions)
            predicted_positives = len(predictions)
            ground_truth_positives = len(ground_truth)

            precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
            recall = true_positives / ground_truth_positives if ground_truth_positives > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_scan_results.append({
                'participant_id': participant_id,
                'trial_id': trial_id,
                'ground_truth': ground_truth,
                'predictions': predictions,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'matched_predictions': list(matched_predictions),
                'matched_ground_truth': list(matched_ground_truth)
            })

        # Aggregate results for this threshold
        overall_true_positives = sum(len(set(row['matched_predictions'])) for row in per_scan_results)
        overall_predicted_positives = sum(len(set(row['predictions'])) for row in per_scan_results)
        overall_ground_truth_positives = sum(len(set(row['ground_truth'])) for row in per_scan_results)

        overall_precision = overall_true_positives / overall_predicted_positives if overall_predicted_positives > 0 else 0.0
        overall_recall = overall_true_positives / overall_ground_truth_positives if overall_ground_truth_positives > 0 else 0.0
        overall_f1 = (
            2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
            if (overall_precision + overall_recall) > 0 else 0.0
        )

        threshold_results.append({
            'agreement_threshold': agreement_threshold,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1
        })

    threshold_df = pd.DataFrame(threshold_results)

    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df['agreement_threshold'] * 100, threshold_df['precision'], marker='o', label="Precision")
    plt.plot(threshold_df['agreement_threshold'] * 100, threshold_df['recall'], marker='s', label="Recall")
    plt.plot(threshold_df['agreement_threshold'] * 100, threshold_df['f1_score'], marker='^', label="F1 Score")

    plt.title("Precision, Recall, and F1 Score over Voting Agreement Thresholds")
    plt.xlabel("Voting Agreement Threshold (%)")
    plt.ylabel("Metric Value")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(method_dir, f'voting_threshold_metrics_{suffix}.png')
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

    threshold_csv_path = os.path.join(method_dir, f'voting_threshold_metrics_{suffix}.csv')
    threshold_df.to_csv(threshold_csv_path, index=False)
    print(f"Threshold results saved to: {threshold_csv_path}")


if __name__ == "__main__":
    participant_id = 1
    target_fold = 0  # Which fold to use for evaluation
    seed = 0

    try:

        # MODIFIED: Read from CV results instead of original CSV
        cv_results_path =  Path(__file__).parent.parent/"outputs"/f"participant_{participant_id}"/"all_folds_results.csv"

        # Check if CV results file exists
        if not os.path.exists(cv_results_path):
            raise FileNotFoundError(f"CV results file not found: {cv_results_path}")

        print(f"Using CV results from: {cv_results_path}")

        # Run self-supervised transformer on ensemble positive predictions
        find_attention_from_cv_results(
            cv_results_path=cv_results_path,
            participant_id=participant_id,
            target_fold=target_fold,
            window_size=100,
            seed=seed
        )

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()