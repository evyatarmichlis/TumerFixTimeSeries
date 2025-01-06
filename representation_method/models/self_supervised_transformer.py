import heapq
import os
import random
from pathlib import Path
from typing import List, Dict, Tuple

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

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from representation_method.utils.data_loader import DataConfig, load_eye_tracking_data
from representation_method.utils.data_utils import create_dynamic_time_series, split_train_test_for_time_series, \
    create_dynamic_time_series_with_indices
from representation_method.utils.general_utils import seed_everything


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
        learning_rate: float = 1e-4,
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
            windows = windows.float()  # Convert targets to torch.float32
            reconstructed = model(windows)
            # loss = criterion(reconstructed, windows)
            loss = random_mask_mse_loss(
                reconstructed,
                windows,
                mask_ratio=0.3,
                mask_strategy='random'
            )
            # Backward pass
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

    def analyze_attention(self, window_data: torch.Tensor) -> np.ndarray:
        """Analyze attention patterns to localize targets"""
        self.model.eval()
        with torch.no_grad():
            _, attention_weights = self.model(window_data.to(self.device), output_attentions=True)

        attention_scores = attention_weights[-1]
        return attention_scores.cpu().numpy()

    def localize_targets(self, window_data: torch.Tensor, target_positions,window_start):
        """Return both similarity score and actual values"""
        attentions = self.analyze_attention(window_data).squeeze()
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

        top_k_indices = heapq.nlargest(len(target_positions)*2, range(len(token_attentions)),
                                       token_attentions.__getitem__)
        results["top_k_positions"] = top_k_indices
        results["abs_top_k_positions"] = [top_k_indices[i] + window_start for i in range(len(top_k_indices))]
        results["abs_target_locations"] = [target_positions[i] + window_start for i in range(len(target_positions))]

        for target_pos in target_positions:
            min_distance = float('inf')
            for attention_argmax in top_k_indices:
                distance = abs(attention_argmax - target_pos)
                min_distance = min(min_distance, distance)

            similarity = 1 - (min_distance / seq_len)
            results["similarity_score"] += max(0, similarity)

        results["similarity_score"] /= len(target_positions)
        return results




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

def find_attention(df, participant_id='1', filter_targets=True,tolerance = 5,  window_size = 100,top_k = 1):
    # Set parameters
    seed_everything(0)

    feature_columns = [
        'Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X',
        'CURRENT_FIX_IA_Y', 'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT',
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    method_dir = os.path.join('results', f'self_supervised_transformer_participant{participant_id}_window_size_{window_size}_random_mask')
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

    model = train_self_supervised(
        model=model,
        train_loader=dataloader,
        epochs=1000,
        learning_rate=1e-4,
        device='cuda'
    )

    # Evaluation
    localizer = TargetLocalizer(model, device)

    all_results = []
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
            all_results.append(results)




    all_results_df = pd.DataFrame(all_results)
    all_results_df.to_csv(os.path.join(method_dir, 'attention.csv'))
    print('############ percent similarity_score ############')
    print(all_results_df['similarity_score'].mean())
    sns.histplot(data=all_results_df, x='similarity_score', bins=10, kde=True)
    plt.xlabel('Percent Coincidences')
    plt.ylabel('Count')
    plt.title('Distribution of Percent Coincidences')
    plt.savefig(os.path.join(method_dir, 'similarity_score.png'))




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
    for window_size in [100]:
        print(f"WINDOW_SIZE {window_size}")
        find_attention(df, window_size=window_size)