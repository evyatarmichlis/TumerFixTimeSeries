import gc
from typing import Optional, Tuple, Union, Dict, List, Any
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging

from tqdm import tqdm

from transformer_method.data_utils.patch_handler import  PatchCache


class TimeSeriesDataset(Dataset):
    def __init__(
            self,
            data: Union[np.ndarray, pd.DataFrame],
            targets: Optional[Union[np.ndarray, pd.Series]] = None,
            trial_ids: Optional[Union[np.ndarray, pd.Series]] = None,
            patch_len: int = 128,
            stride: int = 2,
            seq_len: int = 1028,
            seq_stride: Optional[int] = None,
            scaler: Optional[StandardScaler] = None,
            feature_columns: Optional[List[str]] = None,
            scale_method: str = 'standard',
            cache_dir: str = 'cached_patches',
            split: str = 'train',
            verbose: bool = False,
            logger: Optional[logging.Logger] = None,
    ):
        # Initialize basic attributes
        self.patch_cache = PatchCache(cache_dir)
        self.patch_len = patch_len
        self.stride = stride
        self.seq_len = seq_len
        self.seq_stride = seq_stride or seq_len // 2
        self.split = split
        self.logger = logger or self._setup_logging()
        self.scaler = scaler

        if self.patch_cache.cache_exists(seq_len,patch_len, stride, split):
            self.sequences, self.sequence_targets = \
                self.patch_cache.load_patches(seq_len,patch_len, stride, split)
            if verbose:
                print(f"Loaded cached sequences for {split} split")
            return

        # Prepare data
        self.data = self._prepare_data(data, feature_columns)
        self._targets = self._prepare_targets(targets)
        self.trial_ids = self._prepare_trial_ids(trial_ids)

        # Scale features if needed
        if scale_method is not None:
            self.data = self._scale_features(scale_method)

        all_sequences = []
        all_sequence_targets = []

        # Process each trial
        unique_trials = np.unique(self.trial_ids)
        for trial in tqdm(unique_trials, desc="Processing trials"):
            trial_mask = self.trial_ids == trial
            trial_data = self.data[trial_mask]
            trial_targets = self._targets[trial_mask]

            # Create sequences with stride
            for seq_start in range(0, len(trial_data) - seq_len + 1, self.seq_stride):
                seq_end = seq_start + seq_len
                sequence_data = trial_data[seq_start:seq_end]
                sequence_targets = trial_targets[seq_start:seq_end]

                # Get sequence-level target (1 if any target in sequence is 1)
                sequence_target = 1 if np.any(sequence_targets == 1) else 0

                # Create patches for this sequence
                sequence_patches = []
                num_patches = (seq_len - patch_len) // stride + 1

                for i in range(num_patches):
                    start_idx = i * stride
                    end_idx = start_idx + patch_len
                    patch = sequence_data[start_idx:end_idx]
                    sequence_patches.append(patch)

                # Stack patches for this sequence
                sequence_patches = np.stack(sequence_patches)  # [num_patches, patch_len, features]
                all_sequences.append(sequence_patches)
                all_sequence_targets.append(sequence_target)

            if verbose and len(trial_data) > seq_len:
                num_sequences = (len(trial_data) - seq_len) // self.seq_stride + 1
                self.logger.info(f"Trial {trial}: Created {num_sequences} sequences from {len(trial_data)} samples")

            gc.collect()

        # Convert to arrays
        self.sequences = np.stack(all_sequences)  # [num_sequences, num_patches, patch_len, features]
        self.sequence_targets = np.array(all_sequence_targets)  # [num_sequences]

        if verbose:
            self.logger.info(f"\nDataset Summary:")
            self.logger.info(f"Total sequences: {len(self.sequences)}")
            self.logger.info(f"Patches per sequence: {num_patches}")
            self.logger.info(f"Sequence shape: {self.sequences[0].shape}")
            self.logger.info(f"Positive sequences: {np.sum(self.sequence_targets == 1)}")
            self.logger.info(f"Negative sequences: {np.sum(self.sequence_targets == 0)}")
            self.logger.info(f"Memory usage: {self.sequences.nbytes / 1e9:.2f} GB")

        # Save to cache
        self.patch_cache.save_patches(self.sequences, self.sequence_targets,self.seq_len, patch_len, stride, split)

    def __len__(self) -> int:
        """Return number of sequences"""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sequence and its target"""
        sequence = self.sequences[idx]  # [num_patches, patch_len, features]

        if not isinstance(sequence, torch.Tensor):
            sequence = torch.FloatTensor(sequence)

        target = torch.tensor(self.sequence_targets[idx], dtype=torch.long)
        return sequence, target


    def _scale_features(self, scale_method: str) -> np.ndarray:
        """Scale features using specified method"""
        if self.scaler is None:
            if scale_method == 'standard':
                self.scaler = StandardScaler()
                return self.scaler.fit_transform(self.data)
            else:
                raise ValueError(f"Unsupported scaling method: {scale_method}")
        return self.scaler.transform(self.data)

    def _prepare_data(self, data: Union[np.ndarray, pd.DataFrame],
                      feature_columns: Optional[List[str]]) -> np.ndarray:
        """Prepare input data"""
        if isinstance(data, pd.DataFrame):
            if feature_columns is not None:
                data = data[feature_columns].values
            else:
                data = data.values
        return data.astype(np.float32)

    def _prepare_targets(self, targets: Optional[Union[np.ndarray, pd.Series]]) -> Optional[np.ndarray]:
        """Prepare target values"""
        if targets is None:
            return None
        if isinstance(targets, pd.Series):
            targets = targets.values
        return targets.astype(np.int64)

    def _prepare_trial_ids(self, trial_ids: Optional[Union[np.ndarray, pd.Series]]) -> np.ndarray:
        """Prepare trial IDs"""
        if trial_ids is None:
            return np.zeros(len(self.data))
        if isinstance(trial_ids, pd.Series):
            trial_ids = trial_ids.values
        return trial_ids

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)
        return logger

