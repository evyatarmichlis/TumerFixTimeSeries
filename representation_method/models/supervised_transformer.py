import heapq
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import matplotlib
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import KBinsDiscretizer
from transformers import XLNetModel, XLNetConfig
from torch.utils.data import Dataset
import seaborn as sns

import torch
import numpy as np
from torch.utils.data import DataLoader

from representation_method.models.self_supervised_transformer import TargetLocalizer, calculate_confusion_metrics

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
            return reconstructed, classification_logits,self.attention_threshold, attention_weights
        else:
            return reconstructed, classification_logits



def train_self_supervised(
        model: IntegratedEyeTrackingTransformer,
        train_loader: DataLoader,
        epochs: int = 150,
        learning_rate: float = 1e-4,
        device: str = 'cuda',
):
    """
    Train the model in a self-supervised manner using random mask MSE loss.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.006)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for tokens, _, _ in train_loader:  # Ignore labels
            tokens = tokens.to(device)

            # Forward pass
            reconstructed,_,_,_ = model(tokens, output_attentions=True)

            # Random mask MSE loss
            loss = random_mask_mse_loss(reconstructed, tokens)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Self-Supervised Loss: {avg_loss:.4f}")
    return model
def custom_collate(batch):
    tokens, reconstruction_labels, target_labels = zip(*batch)
    tokens = torch.stack(tokens)
    reconstruction_labels = torch.stack(reconstruction_labels)
    target_labels = torch.stack(target_labels)
    return tokens, reconstruction_labels, target_labels

def random_mask_mse_loss(reconstructed, original, mask_ratio=0.15, mask_strategy='random'):
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

    mask = mask.to(reconstructed.device)
    squared_error = (reconstructed - original) ** 2
    masked_squared_error = squared_error * mask

    loss = masked_squared_error.sum() / (mask.sum() + 1e-8)
    return loss


def target_focus_mse_loss(
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        target_locations: torch.Tensor,
        target_weight: float = 1.0
):
    """
    Efficiently compute MSE loss focused on target locations using vectorized operations.

    Args:
    - reconstructed: Predicted tensor (batch_size, seq_len, n_features)
    - original: Ground truth tensor (batch_size, seq_len, n_features)
    - target_locations: Tensor with target positions for each batch (batch_size, seq_len) - binary mask
    - target_weight: Weight applied to the target locations

    Returns:
    - loss: Computed loss value
    """
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
    """
    Compute supervised attention loss with a learned threshold for each head.

    Args:
    - attention_weights: Attention weights (batch_size, num_heads, seq_len, seq_len)
    - target_labels: Binary target labels (batch_size, seq_len)
    - seq_len: Length of the sequence.
    - num_heads: Number of attention heads.
    - device: Device (e.g., 'cuda').
    - threshold: Learnable threshold for attention.

    Returns:
    - loss: Averaged loss across all heads and all sequences.
    """
    bce_loss_fn = nn.BCEWithLogitsLoss()

    # No need to expand or squeeze target_labels here
    target_mask = target_labels.float().to(device)  # Shape: (batch_size, seq_len)

    total_loss = 0

    for head_idx in range(num_heads):
        # Extract attention for the current head: (batch_size, seq_len, seq_len)
        head_attention = attention_weights[:, head_idx, :, :]

        # Compute the average attention over rows (tokens attending to the diagonal)
        avg_attention = head_attention.mean(dim=1)  # Shape: (batch_size, seq_len)

        # Apply threshold to generate predictions
        pred_attention = torch.sigmoid(avg_attention - threshold).view(-1)  # Shape: (batch_size, seq_len)

        # Compute binary cross-entropy loss between predicted and true attention
        head_loss = bce_loss_fn(pred_attention, target_mask)
        total_loss += head_loss

    # Average loss across heads
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
        beta: float = 0.0,
        gamma: float = 0.0,
        patience: int = 10
):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.006)
    ce_loss_fn = nn.CrossEntropyLoss()

    best_loss = float('inf')
    patience_counter = 0  # Counter for epochs without improvement

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for tokens, _, target_labels in train_loader:
            tokens = tokens.to(device)
            target_labels = target_labels.to(device)

            reconstructed, classification_logits, threshold, attention_weights = model(tokens, output_attentions=True)

            recon_loss = target_focus_mse_loss(reconstructed, tokens, target_locations=target_labels)
            classification_logits = classification_logits.view(-1, 2)
            target_labels = target_labels.view(-1)

            class_loss = ce_loss_fn(classification_logits, target_labels)
            seq_len = tokens.shape[1]
            num_heads = attention_weights.shape[1]
            attention_loss = supervised_attention_loss(
                attention_weights, target_labels, seq_len, num_heads, device, threshold
            )

            # Total Loss
            loss = alpha * recon_loss + beta * class_loss + gamma * attention_loss

            optimizer.zero_grad()
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}')

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
                target_labels = target_labels.view(-1)

                class_loss = ce_loss_fn(classification_logits, target_labels)
                seq_len = tokens.shape[1]
                num_heads = attention_weights.shape[1]
                attention_loss = supervised_attention_loss(
                    attention_weights, target_labels, seq_len, num_heads, device, threshold
                )

                # Total Loss
                loss = alpha * recon_loss + beta * class_loss + gamma * attention_loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}')

        # Check for improvement
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0  # Reset patience counter
            print("Validation loss improved. Saving model...")
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience counter: {patience_counter}/{patience}")

            # Early stopping
            if patience_counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break


from sklearn.metrics import roc_curve

from sklearn.metrics import roc_curve, auc

def select_threshold_from_roc(model, val_loader, device):
    """
    Select the best threshold for attention weights based on ROC curve.
    """
    model.eval()
    all_labels = []
    all_attentions = []

    with torch.no_grad():
        for tokens, _, target_labels in val_loader:
            tokens = tokens.to(device)
            target_labels = target_labels.to(device)

            _, _, _, attention_weights = model(tokens, output_attentions=True)
            attention_weights = attention_weights.mean(dim=1)  # Average over heads
            attention_weights = attention_weights.cpu().numpy()  # Shape: (batch_size, seq_len, seq_len)

            # Collect diagonal attention (token attending to itself)
            diag_attention = np.diagonal(attention_weights, axis1=1, axis2=2)

            # Flatten labels and attention weights
            all_labels.extend(target_labels.cpu().numpy().flatten())
            all_attentions.extend(diag_attention.flatten())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_attentions = np.array(all_attentions)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(all_labels, all_attentions)
    roc_auc = auc(fpr, tpr)

    # Find the best threshold
    optimal_idx = np.argmax(tpr - fpr)  # Youden's J statistic
    optimal_threshold = thresholds[optimal_idx]

    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    return optimal_threshold

def evaluate_attention_based_model(model, test_loader, device, tolerance=10):
    model.eval()
    all_results = []

    with torch.no_grad():
        for tokens, _, target_labels in test_loader:
            tokens = tokens.to(device)
            target_labels = target_labels.to(device)

            reconstructed, classification_logits, attention_threshold, attention_weights = model(tokens, output_attentions=True)
            attention_weights = attention_weights.cpu().numpy()

            for batch_idx, attention in enumerate(attention_weights):
                pred_positions = np.where(attention > attention_threshold.detach().cpu().numpy())[1]
                true_positions = np.where(target_labels[batch_idx].cpu().numpy() == 1)[0]

                # Evaluate based on tolerance
                matched_positions = [
                    pred for pred in pred_positions if any(abs(pred - true) <= tolerance for true in true_positions)
                ]

                precision = len(matched_positions) / len(pred_positions) if len(pred_positions) > 0 else 0
                recall = len(matched_positions) / len(true_positions) if len(true_positions) > 0 else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                all_results.append({
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'true_positions': true_positions.tolist(),
                    'pred_positions': pred_positions.tolist(),
                })

    # Calculate average metrics
    avg_precision = np.mean([result['precision'] for result in all_results])
    avg_recall = np.mean([result['recall'] for result in all_results])
    avg_f1 = np.mean([result['f1_score'] for result in all_results])

    summary = {
        'average_precision': avg_precision,
        'average_recall': avg_recall,
        'average_f1_score': avg_f1,
        'detailed_results': all_results
    }

    return summary
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
def evaluate_with_roc_threshold(model, test_loader, device, roc_threshold, tolerance=10):
    """
    Evaluate the model on the test set using the ROC-selected threshold.

    Args:
    - model: The trained model.
    - test_loader: DataLoader for the test set.
    - device: Device (e.g., 'cuda').
    - roc_threshold: Threshold value selected using the ROC curve.
    - tolerance: Allowed range for matching true and predicted positions.

    Returns:
    - summary: A dictionary containing average precision, recall, and F1 score.
    """
    model.eval()
    all_results = []
    model = model.to(device)  # Move the entire model to the GPU
    with torch.no_grad():
        for tokens, _, target_labels in test_loader:
            tokens = tokens.to(device)
            target_labels = target_labels.to(device)

            # Forward pass to get attention weights
            _, _, _, attention_weights = model(tokens, output_attentions=True)
            attention_weights = attention_weights.mean(dim=1).cpu().numpy()  # Average over heads

            for batch_idx, attention in enumerate(attention_weights):
                # Use the ROC-selected threshold
                pred_positions = np.where(attention > roc_threshold)[0]
                true_positions = np.where(target_labels[batch_idx].cpu().numpy() == 1)[0]

                # Evaluate based on tolerance
                matched_positions = [
                    pred for pred in pred_positions if any(abs(pred - true) <= tolerance for true in true_positions)
                ]

                precision = len(matched_positions) / len(pred_positions) if len(pred_positions) > 0 else 0
                recall = len(matched_positions) / len(true_positions) if len(true_positions) > 0 else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                all_results.append({
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'true_positions': true_positions.tolist(),
                    'pred_positions': pred_positions.tolist(),
                })

    # Calculate average metrics
    avg_precision = np.mean([result['precision'] for result in all_results])
    avg_recall = np.mean([result['recall'] for result in all_results])
    avg_f1 = np.mean([result['f1_score'] for result in all_results])

    summary = {
        'average_precision': avg_precision,
        'average_recall': avg_recall,
        'average_f1_score': avg_f1,
        'detailed_results': all_results
    }

    # Print the results
    print(f"Test Set Evaluation with ROC Threshold: {roc_threshold:.4f}")
    print(f"Precision: {avg_precision:.4f}, Recall: {avg_recall:.4f}, F1 Score: {avg_f1:.4f}")

    return summary
def find_attention(df, participant_id='1', window_size=100, filter_targets=True):
    seed = 2024
    seed_everything(seed)

    train_df, test_df = split_train_test_for_time_series(df, test_size=0.2, random_state=seed)
    train_df, val_df = split_train_test_for_time_series(train_df, test_size=0.2, random_state=seed)

    feature_columns = [
        'Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X',
        'CURRENT_FIX_IA_Y', 'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT',
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method_dir = os.path.join('results', f'supervised_transformer_participant_{participant_id}_window_size_{window_size}')
    os.makedirs(method_dir, exist_ok=True)
    print(f"window_size is {window_size}")
    X_train, Y_train, train_metadata = create_dynamic_time_series_with_indices(train_df, feature_columns, window_size)
    X_val, Y_val, val_metadata = create_dynamic_time_series_with_indices(val_df, feature_columns, window_size)
    X_test, Y_test, test_metadata = create_dynamic_time_series_with_indices(test_df, feature_columns, window_size)
    X_all, Y_all, all_metadata = create_dynamic_time_series_with_indices(df, feature_columns, window_size)

    if filter_targets:
        target_mask = Y_train == 1
        X_train = X_train[target_mask]
        train_metadata = [meta for i, meta in enumerate(train_metadata) if target_mask[i]]
        Y_train = Y_train[target_mask]

        target_mask = Y_val == 1
        X_val = X_val[target_mask]
        val_metadata = [meta for i, meta in enumerate(val_metadata) if target_mask[i]]
        Y_val = Y_val[target_mask]

        target_mask = Y_test == 1
        X_test = X_test[target_mask]
        test_metadata = [meta for i, meta in enumerate(test_metadata) if target_mask[i]]
        Y_test = Y_test[target_mask]

        all_mask = Y_all == 1
        X_all = X_all[all_mask]


    feature_columns+= ['diff_pupil', 'diff_fix_duration']

    tokenizer = EyeTrackingTokenizer()
    X_flat = X_train.reshape(-1, X_train.shape[-1])
    X_df = pd.DataFrame(X_flat, columns=feature_columns)
    tokenizer.fit(X_df, feature_columns)

    def create_target_vector(window_size, relative_positions):
        target_vector = [0] * window_size
        for pos in relative_positions:
            if 0 <= pos < window_size:
                target_vector[pos] = 1
        return target_vector

    train_dataset = create_dataset(
        X_train,
        Y_train,
        [create_target_vector(window_size, meta['relative_target_positions']) for meta in train_metadata],
        tokenizer,
        feature_columns
    )
    val_dataset = create_dataset(
        X_val,
        Y_val,
        [create_target_vector(window_size, meta['relative_target_positions']) for meta in val_metadata],
        tokenizer,
        feature_columns
    )
    test_dataset = create_dataset(
        X_test,
        Y_test,
        [create_target_vector(window_size, meta['relative_target_positions']) for meta in test_metadata],
        tokenizer,
        feature_columns
    )
    all_dataset = create_dataset(
        X_all,
        np.zeros(X_all.shape[0]),
        np.zeros((X_all.shape[0], window_size)),
        tokenizer,
        feature_columns
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)
    all_dataloader = DataLoader(all_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)

    model = IntegratedEyeTrackingTransformer(
        vocab_size=tokenizer.vocab_size,
        n_features=len(feature_columns),
        d_model=256,
        max_len=window_size
    )
    best_model_path =os.path.join(method_dir, "model.pth")

    if TRAIN:
        model = train_self_supervised(
            model=model,
            train_loader=all_dataloader,
            epochs=200,
            learning_rate=1e-4,
            device='cuda'
        )
        train_multi_task(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=0,
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



    summary = evaluate_attention_based_model(model, test_loader, device)
    print(summary['average_f1_score'])
    print(summary['average_recall'])
    print(summary['average_precision'])

    # roc_threshold = select_threshold_from_roc(model, val_loader, device)
    localizer = TargetLocalizer(model, device)
    total_precision = 0
    total_recall = 0
    total_f1 = 0
    total_windows = 0
    all_results = []
    print("\nEvaluating target localization...")
    for i, (window, meta) in enumerate(zip(X_test, test_metadata)):
        window_2d = window.squeeze()
        window_df = pd.DataFrame(window_2d, columns=feature_columns)
        tokenized_window = tokenizer.tokenize(window_df, feature_columns)
        window_tensor = torch.tensor(tokenized_window, dtype=torch.long).unsqueeze(0)
        if len(meta['relative_target_positions']) > 0:
            results = localizer.localize_targets(
                window_tensor,
                meta['relative_target_positions'],
                meta['window_start'],
                supervised = True
            )
            results['participant_id'] = meta['participant_id']
            results['trial_id'] = meta['trial_id']
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
    for k, v in metrics.items():
        print(f'############ {k} ############')
        print(f'$$$$$$$$$$$$ {v} $$$$$$$$$$$$')

    sns.histplot(data=all_results_df, x='similarity_score', bins=10, kde=True)
    plt.xlabel('Percent Coincidences')
    plt.ylabel('Count')
    plt.title('Distribution of Percent Coincidences')
    plt.savefig(os.path.join(method_dir, f'similarity_score.png'))
    plt.plot()
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