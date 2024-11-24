from typing import Optional, Tuple, Union, Dict, List, Any
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import logging
from pathlib import Path


class TimeSeriesDataset(Dataset):
    """
    Dataset for time series data with support for patch creation and trial-based processing.

    Features:
    - Handles continuous time series data
    - Creates patches within trial boundaries
    - Supports various feature scaling options
    - Handles missing data
    - Provides dataset statistics
    """

    def __init__(
            self,
            data: Union[np.ndarray, pd.DataFrame],
            targets: Optional[Union[np.ndarray, pd.Series]] = None,
            trial_ids: Optional[Union[np.ndarray, pd.Series]] = None,
            patch_len: int = 32,
            stride: int = 8,
            scaler: Optional[StandardScaler] = None,
            feature_columns: Optional[List[str]] = None,
            scale_method: str = 'standard',
            min_samples: int = 1,
            verbose: bool = False
    ):
        """
        Initialize TimeSeriesDataset.

        Args:
            data: Input time series data
            targets: Target values (optional)
            trial_ids: Trial identifiers (optional)
            patch_len: Length of each patch
            stride: Stride between patches
            scaler: Pre-fitted scaler (optional)
            feature_columns: List of feature column names
            scale_method: Scaling method ('standard', 'minmax', 'robust', None)
            min_samples: Minimum number of samples in a trial
            verbose: Whether to print dataset information
        """
        # 1. First, set basic parameters
        self.patch_len = patch_len
        self.stride = stride
        self.min_samples = min_samples

        # 2. Setup logging
        self.logger = self._setup_logging()

        # 3. Initialize patch-related attributes
        self.patches: Optional[torch.Tensor] = None
        self.patch_targets: Optional[torch.Tensor] = None
        self.patch_trials: Optional[np.ndarray] = None
        self.patch_indices: List[Tuple[int, int]] = []

        # 4. Prepare input data, targets, and trial IDs
        self.data = self._prepare_data(data, feature_columns)
        self._targets = self._prepare_targets(targets)
        self.trial_ids = self._prepare_trial_ids(trial_ids)

        # 5. Scale features if needed
        self.scaler = scaler
        if scale_method is not None:
            self.data = self._scale_features(scale_method)

        # 6. Create patches from prepared data
        self._create_patches()

        # 7. Log dataset information if verbose
        if verbose:
            self._log_dataset_info()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)
        return logger

    def _prepare_data(
            self,
            data: Union[np.ndarray, pd.DataFrame],
            feature_columns: Optional[List[str]]
    ) -> np.ndarray:
        """Prepare input data"""
        if isinstance(data, pd.DataFrame):
            if feature_columns is not None:
                data = data[feature_columns].values
            else:
                data = data.values
        return data.astype(np.float32)

    def _prepare_targets(
            self,
            curr_target: Optional[Union[np.ndarray, pd.Series]]
    ) -> Optional[np.ndarray]:
        """Prepare target values"""
        if curr_target is None:
            return None
        if isinstance(curr_target, pd.Series):
            curr_target = curr_target.values
        return curr_target.astype(np.int64)

    def _prepare_trial_ids(
            self,
            trial_ids: Optional[Union[np.ndarray, pd.Series]]
    ) -> Optional[np.ndarray]:
        """Prepare trial IDs"""
        if trial_ids is None:
            return np.zeros(len(self.data))
        if isinstance(trial_ids, pd.Series):
            trial_ids = trial_ids.values
        return trial_ids

    def _scale_features(self, scale_method: str) -> np.ndarray:
        """Scale features using specified method"""
        if self.scaler is None:
            if scale_method == 'standard':
                self.scaler = StandardScaler()
                return self.scaler.fit_transform(self.data)
            else:
                raise ValueError(f"Unsupported scaling method: {scale_method}")
        return self.scaler.transform(self.data)



    def _log_dataset_info(self):
        """Log dataset information"""
        info = self.get_dataset_info()

        self.logger.info("\nDataset Information:")
        self.logger.info(f"Number of samples: {info['n_samples']}")
        self.logger.info(f"Number of features: {info['n_features']}")
        self.logger.info(f"Number of trials: {info['n_trials']}")
        self.logger.info(f"Number of patches: {info['n_patches']}")

        if self.targets is not None:
            self.logger.info("\nTarget Distribution:")
            for label, count in info['target_distribution'].items():
                self.logger.info(f"Class {label}: {count} ({count / info['n_patches'] * 100:.1f}%)")

    def get_dataset_info(self) -> Dict:
        """Get dataset information"""
        info = {
            'n_samples': len(self.data),
            'n_features': self.data.shape[1],
            'n_trials': len(np.unique(self.trial_ids)),
            'n_patches': len(self.patches),
            'patch_len': self.patch_len,
            'stride': self.stride,
        }

        if self.targets is not None:
            info['target_distribution'] = {
                int(label): int((self.patch_targets == label).sum())
                for label in torch.unique(self.patch_targets)
            }

        return info

    def _create_patches(self) -> None:
        """Create patches from time series data"""
        patches_list: List[np.ndarray] = []
        targets_list: List[int] = []
        trials_list: List[Any] = []

        unique_trials = np.unique(self.trial_ids)
        for trial in unique_trials:
            trial_mask = self.trial_ids == trial
            trial_data = self.data[trial_mask]

            if len(trial_data) < self.patch_len:
                continue

            for i in range(0, len(trial_data) - self.patch_len + 1, self.stride):
                patch = trial_data[i:i + self.patch_len]
                patches_list.append(patch)
                self.patch_indices.append((i, i + self.patch_len))
                trials_list.append(trial)

                if self._targets is not None:
                    trial_targets = self._targets[trial_mask]
                    patch_targets = trial_targets[i:i + self.patch_len]
                    target = np.max(patch_targets)
                    targets_list.append(target)

        # Convert to tensors with explicit types
        self.patches = torch.FloatTensor(np.stack(patches_list))
        if targets_list:
            self.patch_targets = torch.LongTensor(targets_list)
        self.patch_trials = np.array(trials_list)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if self.patches is None:
            raise ValueError("No patches available")

        # Get and reshape patch
        curr_patch = self.patches[idx].clone().float()  # Ensure we return a copy
        curr_patch = curr_patch.transpose(0, 1)  # Reshape to [n_features, patch_len]

        if self.patch_targets is not None:
            patch_target = self.patch_targets[idx].clone().long()  # Ensure we return a copy
            return curr_patch, patch_target
        return curr_patch

    @property
    def targets(self) -> Optional[torch.Tensor]:
        """Get all targets in the dataset"""
        return self.patch_targets.clone() if self.patch_targets is not None else None

    @property
    def all_data(self) -> Optional[torch.Tensor]:
        """Get all patches in the dataset"""
        return self.patches.clone() if self.patches is not None else None

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for the dataset"""
        if self.patch_targets is None:
            raise ValueError("Dataset has no targets")

        # Convert to numpy, calculate weights, and return as tensor
        targets_np = self.patch_targets.cpu().numpy()
        class_counts = np.bincount(targets_np)
        total = len(targets_np)
        weights = total / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)

    def __len__(self) -> int:
        return len(self.patches) if self.patches is not None else 0

    def to(self, device: torch.device) -> 'TimeSeriesDataset':
        """Move dataset tensors to specified device"""
        if self.patches is not None:
            self.patches = self.patches.to(device)
        if self.patch_targets is not None:
            self.patch_targets = self.patch_targets.to(device)
        return self

    def get_patch_info(self) -> Dict[str, Any]:
        """Get detailed information about patches"""
        if self.patches is None:
            return {'error': 'No patches available'}

        info = {
            'n_patches': len(self.patches),
            'n_features': self.patches.size(-1),
            'patch_len': self.patch_len,
            'stride': self.stride,
            'n_trials': len(np.unique(self.patch_trials))
        }

        if self.patch_targets is not None:
            target_counts = torch.bincount(self.patch_targets)
            info['target_distribution'] = {
                int(i): int(count.item())
                for i, count in enumerate(target_counts)
            }

        return info


# Example usage
# if __name__ == "__main__":
#     # Create sample data
#     np.random.seed(42)
#     n_samples = 1000
#     n_features = 6
#
#     data = np.random.randn(n_samples, n_features)
#     targets = np.random.randint(0, 2, size=n_samples)
#     trial_ids = np.repeat(range(5), n_samples // 5)
#
#     # Create dataset
#     dataset = TimeSeriesDataset(
#         data=data,
#         targets=targets,
#         trial_ids=trial_ids,
#         patch_len=32,
#         stride=8,
#         verbose=True
#     )
#
#     # Get a sample
#     patch, target = dataset[0]
#     print(f"\nSample patch shape: {patch.shape}")
#     print(f"Sample target: {target}")
#
#     # Get dataset info
#     info = dataset.get_dataset_info()
#     print("\nDataset Info:", info)