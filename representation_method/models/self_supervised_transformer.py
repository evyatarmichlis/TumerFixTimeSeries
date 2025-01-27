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
from scipy.signal import find_peaks

from representation_method.utils.data_loader import DataConfig, load_eye_tracking_data
from representation_method.utils.data_utils import create_dynamic_time_series, split_train_test_for_time_series, \
    create_dynamic_time_series_with_indices
from representation_method.utils.general_utils import seed_everything


TRAIN = False

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
        # config = BigBirdConfig.from_pretrained("google/bigbird-roberta-base")
        # config.hidden_size = d_model
        # config.num_attention_heads = n_heads
        # config.num_hidden_layers = n_layers
        # config.attention_type = "block_sparse"  # BigBird-specific
        # config.block_size = 64  # Example block size
        # config.num_random_blocks = 2  # Example random blocks
        #
        # self.transformer = BigBirdModel(config)
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
    """
    Custom collate function to handle variable-length target positions.
    Each element in batch is (tokens, label, sequence_id, target_positions)
    """
    tokens, labels = zip(*batch)
    tokens = torch.stack(tokens)
    labels = torch.stack(labels)
    return tokens, labels


def random_mask_mse_loss(reconstructed, original, mask_ratio=0.15, mask_strategy='random'):
    """
    Create a loss with random masking

    Args:
    - reconstructed: Reconstructed time series tensor
    - original: Original time series tensor
    - mask_ratio: Proportion of time steps to mask
    - mask_strategy: 'random', 'consecutive', or 'mixed'

    Returns:
    - Masked Mean Squared Error loss
    """
    batch_size, seq_len, n_features = original.shape
    mask = torch.zeros_like(original, dtype=torch.float32)

    for b in range(batch_size):
        if mask_strategy == 'random':
            # Completely random masking
            mask_indices = torch.rand(seq_len) < mask_ratio
            mask[b, mask_indices, :] = 1.0

        elif mask_strategy == 'consecutive':
            # Consecutive random masking
            num_masked_steps = int(seq_len * mask_ratio)
            start = random.randint(0, seq_len - num_masked_steps)
            mask[b, start:start + num_masked_steps, :] = 1.0

        elif mask_strategy == 'mixed':
            # Mix of random and consecutive masking
            random_mask_indices = torch.rand(seq_len) < (mask_ratio / 2)
            mask[b, random_mask_indices, :] = 1.0

            num_consecutive_steps = int(seq_len * (mask_ratio / 2))
            start = random.randint(0, seq_len - num_consecutive_steps)
            mask[b, start:start + num_consecutive_steps, :] = 1.0

    # Move mask to the same device as reconstructed
    mask = mask.to(reconstructed.device)

    # Compute masked MSE
    squared_error = (reconstructed - original) ** 2
    masked_squared_error = squared_error * mask

    # Compute loss
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
    criterion = nn.MSELoss()
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
            # loss = criterion(reconstructed, windows)
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

    def localize_targets(self, window_data: torch.Tensor, target_positions, window_start,supervised=False):
        """Return similarity score, actual values, and precision/recall metrics"""
        attentions = self.analyze_attention(window_data,supervised).squeeze()
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

        # top_k_indices = heapq.nlargest(len(target_positions), range(len(token_attentions)),
        #                                token_attentions.__getitem__)
        top_k_indices = dynamic_topk_by_threshold(token_attentions,1.7)
        results["top_k_positions"] = top_k_indices
        results["abs_top_k_positions"] = [idx + window_start for idx in top_k_indices]
        results["abs_target_locations"] = [pos + window_start for pos in target_positions]


        matched_targets = set()
        matched_predictions = set()
        tolerance = 10

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

        for target_pos in target_positions:
            min_distance = float('inf')
            for attention_argmax in top_k_indices:
                distance = abs(attention_argmax - target_pos)
                min_distance = min(min_distance, distance)

            similarity = 1 - (min_distance / seq_len)
            results["similarity_score"] += max(0, similarity)

        results["similarity_score"] /= len(target_positions) if len(target_positions) > 0 else 1

        # Add separate metrics to results
        results.update({
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "num_predictions": len(top_k_indices),
            "num_targets": len(target_positions)
        })

        return results


def calculate_confusion_metrics(df):
    """Calculate confusion matrix metrics from DataFrame containing TP, predictions and targets"""
    total_tp = df['true_positives'].sum()
    total_predictions = df['num_predictions'].sum()
    total_targets = df['num_targets'].sum()

    # Calculate other metrics
    fp = total_predictions - total_tp  # False Positives
    fn = total_targets - total_tp  # False Negatives

    # Create confusion matrix
    cm = np.array([
        [total_tp, fp],
        [fn, total_predictions - fp]
    ])

    # Calculate rates
    precision = total_tp / total_predictions if total_predictions > 0 else 0
    recall = total_tp / total_targets if total_targets > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


    return cm, {'precision': precision, 'recall': recall, 'f1': f1}


# Use with your DataFrame
def dynamic_topk_by_threshold(token_attentions: np.ndarray,
                              std_multiplier: float = 2.0) -> List[int]:
    """
    Return indices of all positions whose attention > mean + (std_multiplier * std).
    """
    mean_attn = np.mean(token_attentions)
    std_attn = np.std(token_attentions)
    threshold = mean_attn + std_multiplier * std_attn

    top_indices = np.where(token_attentions > threshold)[0].tolist()
    return top_indices


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
    """
    Detect anomalies based on Z-Score in a given window.

    Args:
    - window_df: DataFrame containing the features for the window.
    - feature_columns: List of feature names to consider for Z-Score detection.
    - threshold: Z-Score threshold for detecting anomalies.

    Returns:
    - List of indices in the window where anomalies are detected.
    """
    anomalies = []
    for feature in feature_columns:
        # Compute rolling mean and standard deviation
        mean = window_df[feature].mean()
        std = window_df[feature].std() + 1e-8  # Avoid division by zero
        z_scores = (window_df[feature] - mean) / std

        # Find indices exceeding the threshold
        anomaly_indices = window_df[z_scores.abs() > threshold].index.tolist()
        anomalies.extend(anomaly_indices)
    return sorted(set(anomalies))  # Remove duplicates


def find_attention(df, participant_id='1', filter_targets=True,tolerance = 5,  window_size = 100,top_k = 1):
    # Set parameters
    seed_everything(0)

    feature_columns = [
        'Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X',
        'CURRENT_FIX_IA_Y', 'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT',
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method_dir = os.path.join('results', f'self_supervised_transformer_participant{participant_id}_window_size_{window_size}')
    os.makedirs(method_dir, exist_ok=True)
    print(f"window_size is {window_size}")


    X_all, Y_all, window_metadata = create_dynamic_time_series_with_indices(
        df, feature_columns=feature_columns, window_size=window_size)
    feature_columns+= ['diff_pupil', 'diff_fix_duration']
    if filter_targets:
        target_mask = Y_all == 1
        X_all = X_all[target_mask]
        window_metadata = [meta for i, meta in enumerate(window_metadata) if target_mask[i]]
        Y_all = Y_all[target_mask]

    tokenizer = EyeTrackingTokenizer()
    X_all_flat = X_all.reshape(-1, X_all.shape[-1])
    X_all_df = pd.DataFrame(X_all_flat, columns=feature_columns)
    tokenizer.fit(X_all_df, feature_columns)
    all_dataset = create_dataset(X_all,Y_all,tokenizer,feature_columns)
    dataloader = DataLoader(all_dataset, batch_size=32, shuffle=True,collate_fn=custom_collate)

    model = IntegratedEyeTrackingTransformer(
        vocab_size=tokenizer.vocab_size,
        n_features=len(feature_columns),
        d_model=256,
        max_len=window_size,
        pretrained_model='xlnet-base-cased'
    )
    best_model_path =os.path.join(method_dir, "model.pth")

    if TRAIN:
        model = train_self_supervised(
            model=model,
            train_loader=dataloader,
            epochs=150,
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
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    localizer = TargetLocalizer(model, device)
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_windows = 0
    all_results = []
    total_z_score_precision = 0
    total_z_score_recall = 0
    total_z_score_f1 = 0
    total_z_score_windows = 0
    print("\nEvaluating target localization...")
    for i, (window, meta) in enumerate(zip(X_all, window_metadata)):
        window_2d = window.squeeze()
        window_df = pd.DataFrame(window_2d, columns=feature_columns)
        tokenized_window = tokenizer.tokenize(window_df, feature_columns)
        window_tensor = torch.tensor(tokenized_window, dtype=torch.long).unsqueeze(0)
        if len(meta['relative_target_positions'])>0:
            results = localizer.localize_targets(
                window_tensor,
                meta['relative_target_positions'],
                meta['window_start']
            )

            results['participant_id'] = meta['participant_id']
            results['trial_id'] = meta['trial_id']
            z_score_anomalies = z_score_detection(window_df, feature_columns)
            z_score_target_matches = sum(
                1 for target in meta['relative_target_positions']
                if any(abs(target - anomaly) <= 5 for anomaly in z_score_anomalies)  # Tolerance of 5 timesteps
            )

            z_score_precision = z_score_target_matches / len(z_score_anomalies) if z_score_anomalies else 0.0
            z_score_recall = z_score_target_matches / len(meta['relative_target_positions']) if len(
                meta['relative_target_positions']) > 0 else 0.0
            z_score_f1 = 2 * (z_score_precision * z_score_recall) / (z_score_precision + z_score_recall) if (z_score_precision + z_score_recall) > 0 else 0.0
            # Add Z-Score results to the results dictionary
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
            total_precision += results['precision']
            total_recall += results['recall']
            total_f1 += results['f1_score']
            total_windows += 1


    target_pick_method = 'threshold'
    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(os.path.join(method_dir, f'{target_pick_method}_attention.csv'))
    print('############ percent similarity_score ############')
    print(all_results_df['similarity_score'].mean())
    cm, metrics = calculate_confusion_metrics(all_results_df)
    print(cm)
    for k,v in metrics.items():
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
    else:
        print("\nNo windows with Z-Score results to calculate averages.")
    sns.histplot(data=all_results_df, x='similarity_score', bins=10, kde=True)
    plt.xlabel('Percent Coincidences')
    plt.ylabel('Count')
    plt.title('Distribution of Percent Coincidences')
    plt.savefig(os.path.join(method_dir, f'similarity_score.png'))
    plt.plot()



    # Debugging: Check for missing columns
    required_columns = ['participant_id', 'trial_id', 'abs_top_k_positions', 'abs_target_locations']
    for col in required_columns:
        if col not in all_results_df.columns:
            raise KeyError(f"Missing required column: {col}")

    # Step 1: Convert string representations of lists to actual lists
    def safe_literal_eval(val):
        if isinstance(val, str):  # Only parse strings
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                print(f"Warning: Skipping malformed entry: {val}")
                return []  # Return an empty list for malformed entries
        elif isinstance(val, list):  # If already a list, return as is
            return val
        else:
            print(f"Warning: Unexpected data type: {type(val)}")
            return []  # Return an empty list for unexpected data types

    all_results_df['abs_top_k_positions'] = all_results_df['abs_top_k_positions'].apply(safe_literal_eval)
    all_results_df['abs_target_locations'] = all_results_df['abs_target_locations'].apply(safe_literal_eval)

    # Step 2: Group by participant_id and trial_id
    grouped_results = defaultdict(lambda: {'ground_truth': set(), 'predictions': defaultdict(int)})

    for _, row in all_results_df.iterrows():
        participant_id = row['participant_id']
        trial_id = row['trial_id']
        scan_id = (participant_id, trial_id)

        # Aggregate ground truth
        if isinstance(row['abs_target_locations'], list):
            grouped_results[scan_id]['ground_truth'].update(row['abs_target_locations'])

        # Aggregate predictions
        if isinstance(row['abs_top_k_positions'], list):
            for position in row['abs_top_k_positions']:
                grouped_results[scan_id]['predictions'][position] += 1

    # Step 3: Majority voting and match predictions
    tolerance = 10  # Tolerance for matching positions


    agreement_thresholds = np.arange(0.1, 1.0, 0.1)  # Thresholds from 10% to 90%
    threshold_results = []

    for agreement_threshold in agreement_thresholds:
        per_scan_results = []

        for scan_id, data in grouped_results.items():
            participant_id, trial_id = scan_id
            ground_truth = sorted(data['ground_truth'])

            # Calculate window participation for each position
            total_windows = len(all_results_df[(all_results_df['participant_id'] == participant_id) &
                                               (all_results_df['trial_id'] == trial_id)])
            position_votes = data['predictions']  # Predictions and their counts from windows
            position_agreement = {pos: count / total_windows for pos, count in position_votes.items()}

            # Apply majority voting dynamically based on agreement
            predictions = sorted(
                [position for position, agreement in position_agreement.items() if agreement >= agreement_threshold]
            )

            # Match predictions to ground truth
            matched_predictions = set()
            matched_ground_truth = set()

            for predicted in predictions:
                for target in ground_truth:
                    if target not in matched_ground_truth and abs(predicted - target) <= tolerance:
                        matched_predictions.add(predicted)
                        matched_ground_truth.add(target)
                        break

            # Calculate metrics for this scan
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

    # Step 6: Convert results to a DataFrame
    threshold_df = pd.DataFrame(threshold_results)

    # Step 7: Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df['agreement_threshold'] * 100, threshold_df['precision'], marker='o', label="Precision")
    plt.plot(threshold_df['agreement_threshold'] * 100, threshold_df['recall'], marker='s', label="Recall")
    plt.plot(threshold_df['agreement_threshold'] * 100, threshold_df['f1_score'], marker='^', label="F1 Score")

    plt.title("Precision, Recall, and F1 Score over Voting Agreement Thresholds")
    plt.xlabel("Voting Agreement Threshold (%)")
    plt.ylabel("Metric Value")
    plt.ylim(0, 1.1)  # Ensure metrics range between 0 and 1
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(method_dir, 'voting_threshold_metrics.png')
    plt.savefig(plot_path)
    print(f"Plot saved to: {plot_path}")

    # Save threshold results to CSV
    threshold_csv_path = os.path.join(method_dir, 'voting_threshold_metrics.csv')
    threshold_df.to_csv(threshold_csv_path, index=False)
    print(f"Threshold results saved to: {threshold_csv_path}")




if __name__ == "__main__":
    config = DataConfig(
        data_path='data/Categorized_Fixation_Data_1_18.csv',
        approach_num=8,
        normalize=True,
        per_slice_target=True,
        participant_id=1
    )

    df = load_eye_tracking_data(
        data_path=config.data_path,
        approach_num=config.approach_num,
        participant_id=config.participant_id,
        data_format="legacy"
    )
    for window_size in [150]:
        find_attention(df, window_size=window_size)