import numpy as np
import pandas as pd
import torch
from typing import List, Dict, Tuple

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import KBinsDiscretizer
from torch_geometric.data import DataLoader


class EyeTrackingTokenizer:
    """
    Tokenizer for eye tracking data that converts continuous features into discrete tokens
    """

    def __init__(
            self,
            n_bins: int = 20,  # Number of bins for each feature
            strategy: str = 'quantile'  # 'uniform', 'quantile', or 'kmeans'
    ):
        self.n_bins = n_bins
        self.strategy = strategy
        self.discretizers = {}
        self.vocab_size = None
        self.feature_offsets = {}

    def fit(self, df: pd.DataFrame, feature_columns: List[str]):
        """Fit discretizers for each feature"""
        current_offset = 0

        for feature in feature_columns:
            # Create and fit discretizer for this feature
            discretizer = KBinsDiscretizer(
                n_bins=self.n_bins,
                encode='ordinal',
                strategy=self.strategy
            )

            # Reshape to 2D array as required by KBinsDiscretizer
            values = df[feature].values.reshape(-1, 1)
            discretizer.fit(values)

            # Store discretizer and offset for this feature
            self.discretizers[feature] = discretizer
            self.feature_offsets[feature] = current_offset

            # Update offset for next feature
            current_offset += self.n_bins

        # Total vocabulary size is n_bins * n_features
        self.vocab_size = current_offset

        return self

    def tokenize_feature(self, values: np.ndarray, feature: str) -> np.ndarray:
        """Tokenize a single feature"""
        # Get discretizer and offset for this feature
        discretizer = self.discretizers[feature]
        offset = self.feature_offsets[feature]

        # Discretize values and add offset
        tokens = discretizer.transform(values.reshape(-1, 1)).astype(int)
        return tokens + offset

    def tokenize(self, df: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
        """
        Tokenize all features in the dataframe
        Returns array of shape [n_samples, n_features] where each value is a token
        """
        tokens_list = []

        for feature in feature_columns:
            values = df[feature].values.reshape(-1, 1)
            feature_tokens = self.tokenize_feature(values, feature)
            tokens_list.append(feature_tokens)

        return np.column_stack(tokens_list)

    def get_feature_boundaries(self, feature: str) -> List[float]:
        """Get bin boundaries for a feature"""
        return self.discretizers[feature].bin_edges_[0].tolist()

    def decode_tokens(self, tokens: np.ndarray, feature_columns: List[str]) -> Dict[str, np.ndarray]:
        """Decode tokens back to approximate original values"""
        decoded = {}

        for i, feature in enumerate(feature_columns):
            feature_tokens = tokens[:, i]
            offset = self.feature_offsets[feature]

            # Remove offset to get original bin indices
            bin_indices = feature_tokens - offset

            # Get bin edges for this feature
            bin_edges = self.discretizers[feature].bin_edges_[0]

            # Use midpoint of bins as decoded values
            bin_midpoints = (bin_edges[1:] + bin_edges[:-1]) / 2
            decoded[feature] = bin_midpoints[bin_indices]

        return decoded


class EyeTrackingTokenDataset(torch.utils.data.Dataset):
    """Dataset for tokenized eye tracking data"""

    def __init__(
            self,
            tokens: np.ndarray,  # Shape: [n_samples, n_features]
            labels: np.ndarray,
            max_length: int = 512
    ):
        self.tokens = torch.LongTensor(tokens)
        self.labels = torch.LongTensor(labels)

        # Truncate if necessary
        if self.tokens.size(1) > max_length:
            self.tokens = self.tokens[:, :max_length]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.tokens[idx], self.labels[idx]


def prepare_tokenized_data(
        df: pd.DataFrame,
        feature_columns: List[str],
        n_bins: int = 20,
        strategy: str = 'quantile',
        test_size: float = 0.15,
        val_size: float = 0.15,
        batch_size: int = 32,
        max_length: int = 512,
        random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader, EyeTrackingTokenizer]:
    """
    Prepare tokenized data for transformer model
    """
    # Initialize and fit tokenizer
    tokenizer = EyeTrackingTokenizer(n_bins=n_bins, strategy=strategy)
    tokenizer.fit(df, feature_columns)

    # Group by trial
    trial_groups = []
    trial_labels = []

    for trial_id, group in df.groupby('TRIAL_INDEX'):
        # Tokenize features for this trial
        tokens = tokenizer.tokenize(group, feature_columns)
        label = (group['SCAN_TYPE'] != 'NORMAL').any().astype(int)

        trial_groups.append(tokens)
        trial_labels.append(label)

    # Convert to arrays
    trial_labels = np.array(trial_labels)

    # Create train/val/test splits
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    trial_indices = np.arange(len(trial_groups))
    train_val_idx, test_idx = next(splitter.split(trial_indices, trial_labels))

    # Further split train into train and validation
    val_size_adjusted = val_size / (1 - test_size)
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_seed)
    train_idx, val_idx = next(splitter_val.split(trial_indices[train_val_idx]))

    # Create datasets
    def create_dataset(indices):
        tokens = [trial_groups[i] for i in indices]
        labels = trial_labels[indices]
        return EyeTrackingTokenDataset(
            np.vstack(tokens),
            labels,
            max_length=max_length
        )

    train_dataset = create_dataset(train_idx)
    val_dataset = create_dataset(val_idx)
    test_dataset = create_dataset(test_idx)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader, tokenizer