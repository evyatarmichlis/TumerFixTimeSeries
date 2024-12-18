import numpy as np
import os
import gc
from pathlib import Path
import hashlib



class PatchCache:
    """Handles caching and loading of pre-computed sequences with patches"""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_patches(self, sequences, sequence_targets,seq_len, patch_len, stride, split):
        """
        Save sequences and their targets to cache

        Args:
            sequences: Array of shape [num_sequences, num_patches, patch_len, features]
            sequence_targets: Array of shape [num_sequences]
            patch_len: Length of each patch
            stride: Stride between patches
            split: Data split identifier ('train', 'val', 'test')
        """
        cache_path = self.get_cache_path(seq_len,patch_len, stride, split)
        np.savez(
            cache_path,
            sequences=sequences,
            targets=sequence_targets
        )

    def load_patches(self, seq_len,patch_len, stride, split):
        """
        Load sequences and targets from cache

        Returns:
            sequences: Array of shape [num_sequences, num_patches, patch_len, features]
            sequence_targets: Array of shape [num_sequences]
        """
        cache_path = self.get_cache_path(seq_len,patch_len, stride, split)
        data = np.load(cache_path)
        return data['sequences'], data['targets']

    def get_cache_path(self,seq_len, patch_len: int, stride: int, split: str) -> Path:
        """Generate cache file path"""
        return self.cache_dir / f"sequences_{seq_len}_{patch_len}_{stride}_{split}.npz"

    def cache_exists(self,seq_len, patch_len: int, stride: int, split: str) -> bool:
        """Check if cache file exists"""
        return self.get_cache_path(seq_len,patch_len, stride, split).exists()


def create_sequence_patches(data: np.ndarray, targets: np.ndarray,
                            seq_len: int, patch_len: int, stride: int) -> tuple:
    """
    Create patches for a sequence with proper dimensionality.

    Args:
        data: Input data of shape [seq_len, features]
        targets: Target values of shape [seq_len]
        seq_len: Length of full sequence
        patch_len: Length of each patch
        stride: Stride between patches

    Returns:
        patches: Array of shape [num_patches, patch_len, features]
        sequence_target: Single target value for the sequence
    """
    # Calculate sequence-level target
    sequence_target = 1 if np.any(targets == 1) else 0

    # Calculate number of patches
    num_patches = (seq_len - patch_len) // stride + 1
    features = data.shape[1]

    # Initialize array for patches
    sequence_patches = np.zeros((num_patches, patch_len, features))

    # Create patches
    for i in range(num_patches):
        start_idx = i * stride
        end_idx = start_idx + patch_len
        sequence_patches[i] = data[start_idx:end_idx]

    return sequence_patches, sequence_target