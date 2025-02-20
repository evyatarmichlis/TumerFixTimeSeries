import itertools
import os
from typing import List, Dict, Tuple, Optional

import matplotlib
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import KBinsDiscretizer
from transformers import XLNetModel, XLNetConfig
from torch.utils.data import Dataset

import torch
import numpy as np
from torch.utils.data import DataLoader

from representation_method.models.self_supervised_transformer import TargetLocalizer, calculate_metrics, \
    random_mask_mse_loss, windows_voting

matplotlib.use('Agg')


from representation_method.utils.data_loader import DataConfig, load_eye_tracking_data
from representation_method.utils.data_utils import create_dynamic_time_series, split_train_test_for_time_series, \
    create_dynamic_time_series_with_indices
from representation_method.utils.general_utils import seed_everything

TRAIN = True

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
        beta: float = 1.0,
        gamma: float = 1.0,
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
            return reconstructed, classification_logits, self.attention_threshold, attention_weights
        else:
            return reconstructed, classification_logits

# ============================================================================
# UPDATED SELF-SUPERVISED PRE-TRAINING FUNCTION
# This version now mirrors the self-supervised training procedure.
# ============================================================================
def train_self_supervised(
        model: IntegratedEyeTrackingTransformer,
        train_loader: DataLoader,
        epochs: int = 150,
        learning_rate: float = 1e-4,
        device: str = 'cuda',
        patience: int = 5
):
    """
    Pre-train the model in a self-supervised manner matching the self-supervised training pipeline.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.006)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for tokens, _, _ in train_loader:
            tokens = tokens.to(device).float()
            reconstructed, _, _, _ = model(tokens, output_attentions=True)
            loss = random_mask_mse_loss(
                reconstructed,
                tokens,
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

        print(f"Epoch {epoch + 1}/{epochs}, Self-Supervised Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    return model


def custom_collate(batch):
    tokens, reconstruction_labels, target_labels = zip(*batch)
    tokens = torch.stack(tokens)
    reconstruction_labels = torch.stack(reconstruction_labels)
    target_labels = torch.stack(target_labels)
    return tokens, reconstruction_labels, target_labels


def grid_search_loss_weights(
        base_model,  # a fresh instance of your IntegratedEyeTrackingTransformer
        pretrained_path,  # path to your self-supervised pre-trained checkpoint
        train_loader,
        val_loader,
        X_test,  # test windows (as produced by your create_dynamic_time_series_with_indices)
        test_metadata,
        tokenizer,
        feature_columns,
        device,
        grid_alpha=[0.1, 1.0, 10.0],
        grid_beta=[0.1, 1.0, 10.0],
        grid_gamma=[0.1, 1.0, 10.0],
        epochs=100,
        learning_rate=1e-5,
        patience=10
):
    best_f1 = -np.inf
    best_params = None
    all_results = []  # To store results for each combination

    # Loop over every combination in the grid
    for alpha, beta, gamma in itertools.product(grid_alpha, grid_beta, grid_gamma):
        print(f"\n--- Training with alpha={alpha}, beta={beta}, gamma={gamma} ---")
        # Create a fresh instance of the model.
        # (Assuming base_model is a callable that returns a new model instance.)
        model = base_model()


        pretrained_state = torch.load(pretrained_path)['model_state_dict']
        model_state = model.state_dict()
        filtered_state = {k: v for k, v in pretrained_state.items()
                          if k in model_state and model_state[k].shape == v.shape}
        model_state.update(filtered_state)
        model.load_state_dict(model_state)

        # Train the model using the current hyperparameter combination.
        train_multi_task(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            patience=patience
        )
        # Evaluate the fine-tuned model on the test set.
        test_f1 = evaluate_model(model, X_test, test_metadata, tokenizer, feature_columns, device)
        print(f"--> Average Test F1 score: {test_f1:.4f}")

        # Record the result.
        all_results.append({
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'test_f1': test_f1
        })

        if test_f1 > best_f1:
            best_f1 = test_f1
            best_params = (alpha, beta, gamma)

    print("\n--- Grid Search Complete ---")
    print(f"Best F1 Score: {best_f1:.4f} with alpha={best_params[0]}, beta={best_params[1]}, gamma={best_params[2]}")
    return all_results, best_params


def evaluate_model(model, X_test, test_metadata, tokenizer, feature_columns, device):
    localizer = TargetLocalizer(model, device)
    all_results = []
    for window, meta in zip(X_test, test_metadata):
        window_2d = window.squeeze()
        window_df = pd.DataFrame(window_2d, columns=feature_columns)
        tokenized_window = tokenizer.tokenize(window_df, feature_columns)
        window_tensor = torch.tensor(tokenized_window, dtype=torch.long).unsqueeze(0).to(device)
        if len(meta['relative_target_positions']) > 0:
            results = localizer.localize_targets(
                window_tensor,
                meta['relative_target_positions'],
                meta['window_start'],
                supervised=True
            )

            all_results.append(results)
    all_results_df = pd.DataFrame(all_results)
    metrics = calculate_metrics(all_results_df)
    return metrics["f1"]

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

def find_attention(df, participant_id='all', window_size=100, filter_targets=True):
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

    # If you are adding extra features ensure your raw arrays have them
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
    best_model_path = os.path.join(method_dir, "model.pth")
    best_model_path_unsupervised = os.path.join(method_dir, "model_supervised.pth")

    if TRAIN:
        # Pre-train the model using the updated self-supervised training function
        pretrained_state = torch.load(f'results/self_supervised_transformer_participant_{participant_id}_window_size_150/model.pth')['model_state_dict']
        def create_model_instance():
            return IntegratedEyeTrackingTransformer(
                vocab_size=tokenizer.vocab_size,
                n_features=len(feature_columns),
                d_model=256,
                max_len=window_size
            )

        # Path to your self-supervised pretrained checkpoint.
        pretrained_path = os.path.join('results', f'self_supervised_transformer_participant_{participant_id}_window_size_150', 'model.pth')
        #
        # # # Run grid search.
        grid_results, best_params = grid_search_loss_weights(
            base_model=create_model_instance,
            pretrained_path=pretrained_path,
            train_loader=train_loader,
            val_loader=val_loader,
            X_test=X_test,
            test_metadata=test_metadata,
            tokenizer=tokenizer,
            feature_columns=feature_columns,
            device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            grid_alpha=[0.1,1,5],
            grid_beta=[0.1,1,5],
            grid_gamma=[1,5, 10.0,15],
            epochs=100,
            learning_rate=1e-5,
            patience=10
        )

        # Optionally, save the grid search results to a CSV.
        grid_df = pd.DataFrame(grid_results)
        grid_df.to_csv(os.path.join(method_dir, "grid_search_results.csv"), index=False)

        supervised_state = model.state_dict()

        filtered_state = {k: v for k, v in pretrained_state.items() if
                          k in supervised_state and supervised_state[k].shape == v.shape}

        supervised_state.update(filtered_state)

        model.load_state_dict(supervised_state)
        train_multi_task(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=300,
            learning_rate=1e-5,
            device='cuda',
            alpha=0.1,
            beta=0.1,
            gamma=15.0,
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
    metrics = calculate_metrics(all_results_df)
    for k, v in metrics.items():
        print(f'############ {k} ############')
        print(f'$$$$$$$$$$$$ {v} $$$$$$$$$$$$')

    windows_voting(all_results_df, method_dir)

if __name__ == "__main__":

        participant_id = 37
        try:

            config = DataConfig(
                data_path='data/Categorized_Fixation_Data_1_18.csv',
                approach_num=8,
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
            find_attention(df, window_size=150, participant_id=str(participant_id))
        except Exception as e:
            print(e)

# ############ precision ############
# $$$$$$$$$$$$ 0.4720382634289919 $$$$$$$$$$$$
# ############ recall ############
# $$$$$$$$$$$$ 0.6752631578947368 $$$$$$$$$$$$
# ############ f1 ############
# $$$$$$$$$$$$ 0.5556517973148549 $$$$$$$$$$$$

# Best F1 Score: 0.6506 with alpha=1.0, beta=10.0, gamma=0.1

# all data - Evaluating target localization...
# ############ percent similarity_score ############
# 0.8090687757923062
# ############ precision ############
# $$$$$$$$$$$$ 0.3815837768397871 $$$$$$$$$$$$
# ############ recall ############
# $$$$$$$$$$$$ 0.1094500519812078 $$$$$$$$$$$$
# ############ f1 ############
# $$$$$$$$$$$$ 0.17010788975814287 $$$$$$$$$$$$