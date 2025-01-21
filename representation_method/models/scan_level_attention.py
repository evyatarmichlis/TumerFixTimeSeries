import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import KBinsDiscretizer
import pandas as pd
from typing import List, Dict, Tuple, Any
import seaborn as sns
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from representation_method.utils.data_loader import load_eye_tracking_data, DataConfig
from representation_method.utils.data_utils import split_train_test_for_time_series
import torch.nn.functional as F
import torch
import torch.nn as nn
from transformers import XLNetConfig, XLNetModel

from scipy.stats import pearsonr

class SupervisedAttentionLoss(nn.Module):
    def __init__(self, reconstruction_weight=1.0, attention_weight=0.5, contrastive_weight=0.3):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.attention_weight = attention_weight
        self.contrastive_weight = contrastive_weight
        self.reconstruction_criterion = nn.MSELoss()

    def attention_supervision_loss(self, attention_weights, window_labels, lengths):
        """Guide attention weights using window labels"""
        loss = 0
        batch_size = attention_weights[0].size(0)

        for i in range(batch_size):
            length = lengths[i]
            labels = window_labels[i][:length]

            # Average attention across layers and heads
            attn = torch.mean(torch.stack([layer[i, :, :length, :length]
                                           for layer in attention_weights]), dim=0)

            # Create target attention matrix based on labels
            target_attn = torch.zeros_like(attn)
            positive_indices = torch.where(labels == 1)[0]

            if len(positive_indices) > 0:
                # Encourage attention to positive windows
                target_attn[:, positive_indices] = 1.0
                target_attn = F.normalize(target_attn, dim=-1)

                # KL divergence between attention and target
                attn_log = torch.log(attn + 1e-10)
                loss += F.kl_div(attn_log, target_attn, reduction='batchmean')

        return loss / batch_size if batch_size > 0 else 0

    def contrastive_loss(self, features, window_labels, lengths, temperature=0.1):
        """Contrastive learning between positive and negative windows"""
        loss = 0
        batch_size = features.size(0)

        for i in range(batch_size):
            length = lengths[i]
            feat = features[i, :length]
            labels = window_labels[i][:length]

            positive_indices = torch.where(labels == 1)[0]
            negative_indices = torch.where(labels == 0)[0]

            if len(positive_indices) > 0 and len(negative_indices) > 0:
                # Normalize features
                feat = F.normalize(feat, dim=1)

                # Compute similarities
                similarity = torch.matmul(feat, feat.T) / temperature

                # Create positive and negative masks
                pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
                neg_mask = ~pos_mask

                # InfoNCE loss
                numerator = similarity[pos_mask].exp()
                denominator = similarity[neg_mask].exp().sum()

                loss += -torch.log(numerator / (numerator + denominator)).mean()

        return loss / batch_size if batch_size > 0 else 0

    def forward(self, reconstructed, inputs, attention_weights, features,
                window_labels, lengths):
        # Reconstruction loss
        recon_loss = self.reconstruction_criterion(reconstructed, inputs)

        # Attention supervision loss
        attn_loss = self.attention_supervision_loss(attention_weights, window_labels, lengths)

        # Contrastive loss
        contra_loss = self.contrastive_loss(features, window_labels, lengths)

        # Combine losses
        total_loss = (self.reconstruction_weight * recon_loss +
                      self.attention_weight * attn_loss +
                      self.contrastive_weight * contra_loss)

        return {
            'total': total_loss,
            'reconstruction': recon_loss.item(),
            'attention': float(attn_loss),
            'contrastive': float(contra_loss)
        }



class ScanLevelTransformer(nn.Module):
    """
    Hierarchical transformer model for scan-level self-supervised learning.
    Processes both window-level and scan-level patterns.
    """

    def __init__(
            self,
            vocab_size: int,
            n_features: int,
            d_model: int = 256,
            n_heads: int = 8,
            n_layers: int = 6,
            dropout: float = 0.1,
            max_windows: int = 100,
            window_size: int = 32,
            pretrained_model: str = "xlnet-base-cased"
    ):
        super().__init__()

        d_model = (d_model // n_features // n_heads) * n_features * n_heads
        self.d_model = d_model
        self.feature_dim = d_model // n_features

        # Window-level embeddings
        self.feature_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, self.feature_dim)
            for _ in range(n_features)
        ])

        # Position encodings for windows
        self.window_pos_embedding = nn.Parameter(torch.randn(1, max_windows, d_model))

        # Window-level transformer
        window_config = XLNetConfig.from_pretrained(pretrained_model)
        window_config.num_attention_heads = n_heads
        window_config.hidden_size = d_model
        window_config.num_hidden_layers = n_layers
        self.window_transformer = XLNetModel(window_config)

        # Scan-level transformer
        scan_config = XLNetConfig.from_pretrained(pretrained_model)
        scan_config.num_attention_heads = n_heads
        scan_config.hidden_size = d_model
        scan_config.num_hidden_layers = n_layers
        self.scan_transformer = XLNetModel(scan_config)

        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_features)
        )

    def embed_features(self, x):
        """Embed individual features and concatenate"""
        batch_size, n_windows, n_features = x.shape
        embeddings = []

        for i, embedding_layer in enumerate(self.feature_embeddings):
            feature_tokens = x[:, :, i].long()
            feature_embedding = embedding_layer(feature_tokens)
            embeddings.append(feature_embedding)

        return torch.cat(embeddings, dim=-1)

    def forward(self, x, window_lengths=None, output_attentions=False, output_hidden=False):
        """Forward pass with optional hidden features output"""
        batch_size, n_windows, n_features = x.shape

        attention_mask = torch.ones((batch_size, n_windows), device=x.device)
        if window_lengths is not None:
            for i, length in enumerate(window_lengths):
                attention_mask[i, :length] = 1

        x = self.embed_features(x)
        x = x + self.window_pos_embedding[:, :n_windows, :]

        window_outputs = self.window_transformer(
            inputs_embeds=x,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True
        )

        window_features = window_outputs.last_hidden_state
        scan_outputs = self.scan_transformer(
            inputs_embeds=window_features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            return_dict=True
        )

        scan_features = scan_outputs.last_hidden_state
        reconstructed = self.reconstruction_head(scan_features)

        if output_attentions and output_hidden:
            return reconstructed, window_outputs.attentions, scan_features
        elif output_attentions:
            return reconstructed, window_outputs.attentions
        elif output_hidden:
            return reconstructed, scan_features
        return reconstructed


class ScanLevelTrainer:
    def __init__(self, model, device, mask_probability=0.15):
        self.model = model.to(device)
        self.device = device
        self.mask_probability = mask_probability
        self.criterion = SupervisedAttentionLoss()

    def apply_masking(self, x):
        mask = torch.bernoulli(torch.full(x.shape, 1 - self.mask_probability)).to(self.device)
        return x * mask

    def train_epoch(self, train_loader, optimizer):
        self.model.train()
        total_losses = {'total': 0, 'reconstruction': 0, 'attention': 0, 'contrastive': 0}

        for batch in train_loader:
            tokens = batch['tokens'].to(self.device).float()
            window_labels = batch['window_labels'].to(self.device).float()
            lengths = batch['length']

            masked_tokens = self.apply_masking(tokens)

            # Get model outputs including attention and features
            reconstructed, attention_weights, features = self.model(
                masked_tokens, lengths, output_attentions=True, output_hidden=True)

            # Calculate losses
            losses = self.criterion(
                reconstructed, tokens,
                attention_weights, features,
                window_labels, lengths
            )

            # Backward pass
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            # Update running losses
            for k, v in losses.items():
                if isinstance(v, torch.Tensor):
                    total_losses[k] += v.item()
                else:
                    total_losses[k] += v

        return {k: v / len(train_loader) for k, v in total_losses.items()}

    def validate(self, val_loader):
        self.model.eval()
        total_losses = {'total': 0, 'reconstruction': 0, 'attention': 0, 'contrastive': 0}

        with torch.no_grad():
            for batch in val_loader:
                tokens = batch['tokens'].to(self.device)
                window_labels = batch['window_labels'].to(self.device)
                lengths = batch['length']

                reconstructed, attention_weights, features = self.model(
                    tokens, lengths, output_attentions=True, output_hidden=True)

                losses = self.criterion(
                    reconstructed, tokens,
                    attention_weights, features,
                    window_labels, lengths
                )

                for k, v in losses.items():
                    if isinstance(v, torch.Tensor):
                        total_losses[k] += v.item()
                    else:
                        total_losses[k] += v

        return {k: v / len(val_loader) for k, v in total_losses.items()}
def train_scan_transformer(
        model: ScanLevelTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str,
        epochs: int = 1000,
        learning_rate: float = 1e-4,
        mask_probability: float = 0.15,
        patience: int = 5
):
    """
    Train the scan-level transformer model

    Args:
        model: ScanLevelTransformer instance
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to train on
        epochs: Number of training epochs
        learning_rate: Learning rate
        mask_probability: Probability of masking input features
        patience: Early stopping patience
    """
    trainer = ScanLevelTrainer(
        model=model,
        device=device,
        mask_probability=mask_probability
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    patience_counter = 0

    print("Starting training...")
    try:
        for epoch in range(epochs):
            train_loss = trainer.train_epoch(train_loader, optimizer)


            val_loss = trainer.validate(val_loader)

            # Use the 'total' loss for the scheduler step
            scheduler.step(val_loss['total'])

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Train Loss: {train_loss['total']:.4f}")
            print(f"Val Loss: {val_loss['total']:.4f}")

            # Early stopping check
            if val_loss['total'] < best_val_loss:
                best_val_loss = val_loss['total']
                patience_counter = 0
                # Save best model here if needed
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered!")
                    break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    return model
class WindowFeatureExtractor:
    """Extract meaningful features from windows to capture key patterns"""

    def __init__(self):
        self.feature_names = []

    def extract_features(self, window: np.ndarray, feature_columns: List[str]) -> np.ndarray:
        """
        Extract meaningful features from a window

        Args:
            window: Array of shape [window_size, n_features]
            feature_columns: Names of original features
        """
        features = []
        self.feature_names = []

        # 1. Pupil Size Features
        pupil_idx = feature_columns.index('Pupil_Size')
        pupil_data = window[:, pupil_idx]

        # Basic statistics
        features.extend([
            np.mean(pupil_data),  # Average pupil size
            np.std(pupil_data),  # Variability
            np.max(pupil_data) - np.min(pupil_data),  # Range
        ])
        self.feature_names.extend(['pupil_mean', 'pupil_std', 'pupil_range'])

        # Rate of change
        pupil_diff = np.diff(pupil_data)
        features.extend([
            np.mean(np.abs(pupil_diff)),  # Average rate of change
            np.max(np.abs(pupil_diff)),  # Maximum rate of change
            len(pupil_diff[pupil_diff > np.std(pupil_diff)])  # Number of significant changes
        ])
        self.feature_names.extend(['pupil_change_rate', 'pupil_max_change', 'pupil_sig_changes'])

        # 2. Fixation Features
        fix_dur_idx = feature_columns.index('CURRENT_FIX_DURATION')
        fix_data = window[:, fix_dur_idx]

        # Duration patterns
        features.extend([
            np.mean(fix_data),  # Average fixation duration
            np.std(fix_data),  # Variability in fixation
            np.percentile(fix_data, 90),  # Long fixations
            len(fix_data[fix_data > np.mean(fix_data) + 2 * np.std(fix_data)])  # Unusually long fixations
        ])
        self.feature_names.extend(['fix_mean', 'fix_std', 'fix_90th', 'fix_outliers'])

        # 3. Gaze Pattern Features
        x_idx = feature_columns.index('CURRENT_FIX_IA_X')
        y_idx = feature_columns.index('CURRENT_FIX_IA_Y')
        x_data = window[:, x_idx]
        y_data = window[:, y_idx]

        # Spatial distribution
        features.extend([
            np.sqrt(np.var(x_data) + np.var(y_data)),  # Dispersion
            len(np.unique(np.column_stack((x_data, y_data)), axis=0)),  # Unique positions
            np.mean(np.sqrt(np.diff(x_data) ** 2 + np.diff(y_data) ** 2))  # Average movement
        ])
        self.feature_names.extend(['gaze_dispersion', 'unique_positions', 'avg_movement'])

        # 4. Pattern Complexity Features
        # Zero crossings in pupil size changes
        zero_crossings = np.where(np.diff(np.signbit(np.diff(pupil_data))))[0]
        features.append(len(zero_crossings))
        self.feature_names.append('pattern_complexity')

        # 5. Transition Features
        # Number of significant gaze transitions
        dx = np.diff(x_data)
        dy = np.diff(y_data)
        distances = np.sqrt(dx ** 2 + dy ** 2)
        significant_transitions = len(distances[distances > np.mean(distances) + np.std(distances)])
        features.append(significant_transitions)
        self.feature_names.append('significant_transitions')

        # 6. Temporal Features
        # Features about the distribution of events in time
        first_third = slice(0, len(window) // 3)
        last_third = slice(2 * len(window) // 3, None)

        features.extend([
            np.mean(pupil_data[first_third]) - np.mean(pupil_data[last_third]),  # Pupil size trend
            np.mean(np.abs(distances[first_third])) - np.mean(np.abs(distances[last_third]))  # Movement pattern change
        ])
        self.feature_names.extend(['pupil_trend', 'movement_trend'])

        return np.array(features)

    def get_feature_names(self) -> List[str]:
        """Get names of extracted features"""
        return self.feature_names


class WindowAggregator:
    """Process windows to extract meaningful aggregated features"""

    def __init__(self):
        self.feature_extractor = WindowFeatureExtractor()

    def process_windows(self, windows: np.ndarray, feature_columns: List[str]) -> np.ndarray:
        """
        Process multiple windows to extract features

        Args:
            windows: Array of shape [n_windows, window_size, n_features]
            feature_columns: Names of original features
        """
        processed_windows = []

        for window in windows:
            features = self.feature_extractor.extract_features(window, feature_columns)
            processed_windows.append(features)

        return np.array(processed_windows)

    def get_feature_info(self) -> Dict[str, Any]:
        """Get information about extracted features"""
        return {
            'feature_names': self.feature_extractor.get_feature_names(),
            'n_features': len(self.feature_extractor.get_feature_names())
        }


class DistributionTokenizer:
    """Tokenize windows based on feature distributions"""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.discretizers = {}
        self.feature_ranges = {}
        self.vocab_size = None

    def fit(self, features: np.ndarray, feature_names: List[str]):
        """
        Fit tokenizer based on feature distributions

        Args:
            features: Array of shape [n_windows, n_features]
            feature_names: Names of features
        """
        self.feature_columns = feature_names
        current_vocab_size = 0

        # For each feature
        for i, feature in enumerate(feature_names):
            # Get feature values
            feature_data = features[:, i].reshape(-1, 1)

            # Fit discretizer
            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode='ordinal',
                strategy='quantile'
            )
            discretizer.fit(feature_data)

            # Store range and discretizer
            self.feature_ranges[feature] = {
                'min': float(feature_data.min()),
                'max': float(feature_data.max()),
                'vocab_start': current_vocab_size
            }
            self.discretizers[feature] = discretizer
            current_vocab_size += self.n_bins

        self.vocab_size = current_vocab_size
        return self

    def tokenize(self, features: np.ndarray) -> np.ndarray:
        """
        Tokenize a single set of features

        Args:
            features: Array of shape [n_features]
        Returns:
            tokens: Array of shape [n_features]
        """
        tokens = np.zeros(len(self.feature_columns), dtype=np.int64)

        for i, feature in enumerate(self.feature_columns):
            feature_data = np.array([features[i]]).reshape(-1, 1)
            feature_tokens = self.discretizers[feature].transform(feature_data)
            tokens[i] = feature_tokens.flatten()[0] + self.feature_ranges[feature]['vocab_start']

        return tokens

    def get_feature_info(self) -> Dict:
        """Get information about feature tokenization"""
        return {
            'vocab_size': self.vocab_size,
            'n_features': len(self.feature_columns),
            'feature_ranges': self.feature_ranges,
            'n_bins': self.n_bins
        }

class WindowDataset(Dataset):
    """Dataset for windows with distribution-based tokens"""

    def __init__(self,
                 windows: np.ndarray,
                 labels: np.ndarray,
                 tokenizer: DistributionTokenizer):
        """
        Args:
            windows: Array of shape [n_windows, window_size, n_features]
            labels: Array of shape [n_windows]
            tokenizer: Fitted DistributionTokenizer
        """
        self.windows = windows
        self.labels = labels
        self.tokenizer = tokenizer

        # Pre-tokenize all windows
        self.tokenized_windows = []
        for window in windows:
            tokens = self.tokenizer.tokenize(window)
            self.tokenized_windows.append(tokens)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        tokens = torch.LongTensor(self.tokenized_windows[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return tokens, label


def extract_windows(df: pd.DataFrame,
                    feature_columns: List[str],
                    window_size: int,
                    stride: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract windows from DataFrame, focusing on windows with target=1

    Args:
        df: Input DataFrame
        feature_columns: Feature columns to use
        window_size: Size of windows to extract
        stride: Stride between windows (default: window_size//2)

    Returns:
        windows: Array of shape [n_windows, window_size, n_features]
        labels: Array of shape [n_windows]
    """
    if stride is None:
        stride = window_size // 2

    windows = []
    labels = []

    # Group by participant and trial
    for _, group in df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']):
        sequence = group[feature_columns].values
        targets = group['target'].values

        # Extract windows with stride
        for start in range(0, len(sequence) - window_size + 1, stride):
            window = sequence[start:start + window_size]
            window_targets = targets[start:start + window_size]

            # Label is 1 if any target=1 in window
            label = int(any(window_targets))

            windows.append(window)
            labels.append(label)

    return np.array(windows), np.array(labels)


def extract_windows_from_df(df: pd.DataFrame,
                            feature_columns: List[str],
                            window_size: int,
                            stride: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Extract windows from a DataFrame"""
    if stride is None:
        stride = window_size // 2

    windows = []
    labels = []

    # Group by participant and trial to maintain sequence integrity
    for _, group in df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']):
        sequence = group[feature_columns].values
        targets = group['target'].values

        # Skip if sequence is shorter than window size
        if len(sequence) < window_size:
            continue

        # Extract windows with stride
        for start in range(0, len(sequence) - window_size + 1, stride):
            window = sequence[start:start + window_size]
            window_targets = targets[start:start + window_size]

            # Label is 1 if any target=1 in window
            label = int(any(window_targets))

            windows.append(window)
            labels.append(label)

    # Convert to numpy arrays
    windows = np.array(windows)
    labels = np.array(labels)

    print(f"Extracted {len(windows)} windows from DataFrame")
    print(f"Window shape: {windows.shape}")
    print(f"Positive windows: {sum(labels)}")
    print(f"Negative windows: {len(labels) - sum(labels)}")

    return windows, labels


def extract_scan_windows(df: pd.DataFrame,
                         feature_columns: List[str],
                         window_size: int,
                         stride: int = None) -> Dict[Tuple[int, int], Dict]:
    """
    Extract windows while maintaining scan-level grouping

    Args:
        df: Input DataFrame
        feature_columns: Feature columns to use
        window_size: Size of each window
        stride: Stride between windows

    Returns:
        Dictionary mapping (participant_id, trial_id) to dict containing:
        - windows: Array of windows for this scan
        - labels: Array of labels for each window
        - scan_label: Overall label for the scan (1 if any target present)
    """
    if stride is None:
        stride = window_size // 2

    scan_data = {}

    # Group by participant and trial (each group is a "scan")
    for (participant_id, trial_id), group in df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']):
        sequence = group[feature_columns].values
        targets = group['target'].values
        scan_windows = []
        window_labels = []

        # Skip if sequence is too short
        if len(sequence) < window_size:
            continue

        # Extract windows for this scan
        for start in range(0, len(sequence) - window_size + 1, stride):
            window = sequence[start:start + window_size]
            window_targets = targets[start:start + window_size]

            # Label for this window
            window_label = int(any(window_targets))

            scan_windows.append(window)
            window_labels.append(window_label)

        # Convert to arrays
        scan_windows = np.array(scan_windows)
        window_labels = np.array(window_labels)

        # Store scan data
        scan_data[(participant_id, trial_id)] = {
            'windows': scan_windows,
            'window_labels': window_labels,
            'scan_label': int(any(targets)),  # Overall scan label
            'length': len(scan_windows)  # Number of windows in scan
        }

    return scan_data


class ScanDataset(Dataset):
    """Dataset that maintains scan-level structure"""

    def __init__(self,
                 scan_data: Dict[Tuple[int, int], Dict],
                 tokenizer: DistributionTokenizer,
                 aggregator: WindowAggregator,
                 feature_columns: List[str],
                 max_windows: int = None):
        """
        Args:
            scan_data: Dictionary mapping (participant_id, trial_id) to scan data
            tokenizer: Fitted tokenizer
            aggregator: Window feature aggregator
            feature_columns: Feature column names
            max_windows: Maximum number of windows per scan (for padding)
        """
        self.scan_data = scan_data
        self.tokenizer = tokenizer
        self.aggregator = aggregator
        self.feature_columns = feature_columns
        self.scan_ids = list(scan_data.keys())

        # Find max windows if not provided
        if max_windows is None:
            max_windows = max(data['length'] for data in scan_data.values())
        self.max_windows = max_windows

        # Process each scan
        self.processed_scans = {}
        for scan_id, data in scan_data.items():
            # Process windows
            processed_windows = self.aggregator.process_windows(
                data['windows'],
                feature_columns
            )

            # Tokenize processed windows
            tokenized_windows = []
            for window in processed_windows:
                tokens = self.tokenizer.tokenize(window)
                tokenized_windows.append(tokens)

            self.processed_scans[scan_id] = {
                'tokens': np.array(tokenized_windows),
                'window_labels': data['window_labels'],
                'scan_label': data['scan_label'],
                'length': data['length']
            }

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, idx):
        scan_id = self.scan_ids[idx]
        scan_data = self.processed_scans[scan_id]

        # Pad or truncate windows to max_windows
        n_windows = len(scan_data['tokens'])
        if n_windows > self.max_windows:
            tokens = scan_data['tokens'][:self.max_windows]
            window_labels = scan_data['window_labels'][:self.max_windows]
            length = self.max_windows
        else:
            # Pad with zeros
            tokens = np.pad(
                scan_data['tokens'],
                ((0, self.max_windows - n_windows), (0, 0)),
                mode='constant'
            )
            window_labels = np.pad(
                scan_data['window_labels'],
                (0, self.max_windows - n_windows),
                mode='constant'
            )
            length = n_windows

        return {
            'tokens': torch.LongTensor(tokens),
            'window_labels': torch.LongTensor(window_labels),
            'scan_label': torch.LongTensor([scan_data['scan_label']]),
            'length': length,
            'scan_id': scan_id
        }


if __name__ == "__main__":
    # Load data
    config = DataConfig(
        data_path='data/Categorized_Fixation_Data_1_18.csv',
        approach_num=8,
        normalize=True,
        per_slice_target=True,
        participant_id=None
    )

    df = load_eye_tracking_data(
        data_path=config.data_path,
        approach_num=config.approach_num,
        participant_id=config.participant_id,
        data_format="legacy"
    )

    # Define feature columns
    feature_columns = [
        'Pupil_Size',
        'CURRENT_FIX_DURATION',
        'CURRENT_FIX_IA_X',
        'CURRENT_FIX_IA_Y',
        'CURRENT_FIX_INDEX',
        'CURRENT_FIX_COMPONENT_COUNT'
    ]

    window_size = 100
    print(f"\nProcessing window size: {window_size}")
    output_dir = Path(f'results/scan_level_transformer_{window_size}')
    output_dir.mkdir(parents=True, exist_ok=True)

    # First split the data into train/val/test
    print("\nSplitting data into train/val/test sets...")
    train_df, test_df = split_train_test_for_time_series(df, test_size=0.2, random_state=0)
    train_df, val_df = split_train_test_for_time_series(train_df, test_size=0.2, random_state=0)

    # Extract scan-level windows
    print("\nExtracting scan windows...")
    train_scans = extract_scan_windows(train_df, feature_columns, window_size)
    val_scans = extract_scan_windows(val_df, feature_columns, window_size)
    test_scans = extract_scan_windows(test_df, feature_columns, window_size)

    # Initialize aggregator
    aggregator = WindowAggregator()

    # Process all training windows for tokenizer fitting
    all_processed_windows = []
    for scan_data in train_scans.values():
        processed_windows = aggregator.process_windows(scan_data['windows'], feature_columns)
        all_processed_windows.append(processed_windows)
    processed_sample = np.vstack(all_processed_windows)

    # Get feature information
    feature_info = aggregator.get_feature_info()
    print("\nExtracted Features:")
    for i, feature_name in enumerate(feature_info['feature_names']):
        print(f"{i + 1}. {feature_name}")

    # Initialize and fit tokenizer with dynamic bin sizing
    n_bins = min(20, len(processed_sample) // 10)  # Ensure enough samples per bin
    tokenizer = DistributionTokenizer(n_bins=n_bins)
    tokenizer.fit(processed_sample, feature_info['feature_names'])

    # Create datasets
    max_windows = max(
        max(data['length'] for data in train_scans.values()),
        max(data['length'] for data in val_scans.values()),
        max(data['length'] for data in test_scans.values())
    )
    print(f"\nMaximum windows per scan: {max_windows}")

    train_dataset = ScanDataset(train_scans, tokenizer, aggregator, feature_columns, max_windows)
    val_dataset = ScanDataset(val_scans, tokenizer, aggregator, feature_columns, max_windows)
    test_dataset = ScanDataset(test_scans, tokenizer, aggregator, feature_columns, max_windows)

    # Create dataloaders
    batch_size = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Get the number of features from the tokenizer
    n_features = len(tokenizer.feature_columns)

    model = ScanLevelTransformer(
        vocab_size=tokenizer.vocab_size,
        n_features=n_features,
        d_model=256,
        n_heads=8,
        n_layers=12,
        dropout=0.1,
        max_windows=max_windows,
        window_size=window_size,
        pretrained_model='xlnet-base-cased'
    ).to(device)

    # Training configuration
    train_config = {
        'epochs': 1000,
        'learning_rate': 1e-3,
        'mask_probability': 0.15,
        'patience': 5
    }


    print("\nStarting model training...")
    try:
        # Train model
        model = train_scan_transformer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            **train_config
        )

        # Save trained model
        torch.save(model.state_dict(), output_dir / 'final_model.pth')
        print("\nModel saved successfully!")

        # Evaluate attention patterns
        print("\nAnalyzing attention patterns...")
        model.eval()
        attention_results = []

        with torch.no_grad():
            for batch in test_loader:
                tokens = batch['tokens'].to(device)
                lengths = batch['length']
                scan_labels = batch['scan_label']
                window_labels = batch['window_labels']

                # Get reconstructions and attention weights
                reconstructed, window_attn, scan_attn = model(
                    tokens, lengths, output_attentions=True,output_hidden = True)

                # Store attention patterns for analysis
                for i in range(len(lengths)):
                    length = lengths[i]
                    scan_label = scan_labels[i].item()

                    # Average attention across heads and layers
                    avg_window_attn = torch.mean(torch.stack(window_attn)[:, i, :, :length, :length], dim=(0, 1))
                    avg_scan_attn = torch.mean(torch.stack(scan_attn)[:, i, :, :length, :length], dim=(0, 1))

                    attention_results.append({
                        'length': length,
                        'scan_label': scan_label,
                        'window_attention': avg_window_attn.cpu().numpy(),
                        'scan_attention': avg_scan_attn.cpu().numpy()
                    })

        # Visualize attention patterns

        print("\nAnalyzing attention patterns...")
        model.eval()
        attention_results = []

        # Find maximum sequence length
        max_len = max(batch['length'].max().item() for batch in test_loader)

        with torch.no_grad():
            for batch in test_loader:
                tokens = batch['tokens'].to(device)
                lengths = batch['length']
                scan_labels = batch['scan_label']
                window_labels = batch['window_labels']

                reconstructed, window_attn, scan_attn = model(tokens, lengths, output_attentions=True,output_hidden = True)

                for i in range(len(lengths)):
                    length = lengths[i]
                    scan_label = scan_labels[i].item()

                    # Pad attention matrices to max_len
                    avg_window_attn = torch.mean(torch.stack(window_attn)[:, i, :, :length, :length], dim=(0, 1))
                    avg_scan_attn = torch.mean(torch.stack(scan_attn)[:, i, :, :length, :length], dim=(0, 1))

                    padded_window_attn = torch.zeros((max_len, max_len), device=avg_window_attn.device)
                    padded_scan_attn = torch.zeros((max_len, max_len), device=avg_scan_attn.device)

                    padded_window_attn[:length, :length] = avg_window_attn
                    padded_scan_attn[:length, :length] = avg_scan_attn

                    attention_results.append({
                        'length': length,
                        'scan_label': scan_label,
                        'window_attention': padded_window_attn.cpu().numpy(),
                        'scan_attention': padded_scan_attn.cpu().numpy(),
                        'window_labels': F.pad(window_labels[i][:length], (0, max_len - length)).cpu().numpy()
                    })

        # Visualization code
        print("\nGenerating attention visualizations...")
        fig_dir = output_dir / 'attention_plots'
        fig_dir.mkdir(exist_ok=True)


        def plot_attention_comparison(attentions_pos, attentions_neg, title, filename):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            sns.heatmap(attentions_pos, ax=ax1, cmap='viridis')
            ax1.set_title('Positive Scans')

            sns.heatmap(attentions_neg, ax=ax2, cmap='viridis')
            ax2.set_title('Negative Scans')

            fig.suptitle(title)
            plt.tight_layout()
            plt.savefig(fig_dir / filename)
            plt.close()


        # Calculate padded average attention patterns
        pos_results = [r for r in attention_results if r['scan_label'] == 1]
        neg_results = [r for r in attention_results if r['scan_label'] == 0]

        if pos_results and neg_results:
            pos_window_attn = np.mean([r['window_attention'] for r in pos_results], axis=0)
            neg_window_attn = np.mean([r['window_attention'] for r in neg_results], axis=0)
            pos_scan_attn = np.mean([r['scan_attention'] for r in pos_results], axis=0)
            neg_scan_attn = np.mean([r['scan_attention'] for r in neg_results], axis=0)

            # Plot comparisons
            plot_attention_comparison(pos_window_attn, neg_window_attn,
                                      'Window-Level Attention Patterns',
                                      'window_attention_comparison.png')
            plot_attention_comparison(pos_scan_attn, neg_scan_attn,
                                      'Scan-Level Attention Patterns',
                                      'scan_attention_comparison.png')
            correlations = []
            for result in pos_results:
                length = result['length']
                window_attention_sum = np.sum(result['window_attention'][:length, :length], axis=1)
                window_labels = result['window_labels'][:length]

                # Compute correlation and check for NaN
                if len(window_attention_sum) > 1 and len(window_labels) > 1:  # Ensure enough data points
                    correlation, p_value = pearsonr(window_attention_sum, window_labels)
                    if not np.isnan(correlation):
                        correlations.append((correlation, p_value))

            # Average correlation analysis
            if correlations:
                avg_correlation = np.mean([c[0] for c in correlations])
                avg_p_value = np.mean([c[1] for c in correlations])
                print(f"Average correlation: {avg_correlation:.4f}")
                print(f"Average p-value: {avg_p_value:.4e}")

                with open(fig_dir / 'attention_correlation.txt', 'w') as f:
                    f.write(f"Attention-Label Correlation Analysis\n")
                    f.write(f"===================================\n\n")
                    f.write(f"Average correlation: {avg_correlation:.4f}\n")
                    f.write(f"Number of samples: {len(correlations)}\n")
                    f.write(f"Standard deviation: {np.std(correlations):.4f}\n")

        print("\nAttention analysis completed!")

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise

