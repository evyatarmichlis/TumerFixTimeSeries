from torch.utils.data import Dataset, DataLoader
import torch
from typing import List, Optional
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from representation_method.utils.data_loader import load_eye_tracking_data, DataConfig


class EyeTrackingPatchDataset(Dataset):
    """Dataset for eye tracking data using PatchTST-style patching"""

    def __init__(
            self,
            features: np.ndarray,  # [n_trials, max_len, n_features]
            labels: np.ndarray,  # [n_trials]
            trial_ids: np.ndarray,  # [n_trials]
            patch_len: int = 16,  # Length of each patch
            stride: int = 8,  # Stride between patches
            padding_mask: Optional[np.ndarray] = None
    ):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.trial_ids = trial_ids
        self.patch_len = patch_len
        self.stride = stride
        self.padding_mask = torch.BoolTensor(padding_mask) if padding_mask is not None else None

        # Create patches for each trial
        self.patches = []
        self.patch_labels = []
        self.patch_trial_ids = []

        for i in range(len(features)):
            # Get valid length (non-padded)
            valid_len = features[i].shape[0]
            if self.padding_mask is not None:
                valid_len = self.padding_mask[i].sum()

            # Create patches for this trial
            trial_patches = []
            for start in range(0, valid_len - patch_len + 1, stride):
                end = start + patch_len
                patch = self.features[i, start:end]
                trial_patches.append(patch)

            if trial_patches:  # If we have any patches
                self.patches.append(torch.stack(trial_patches))
                self.patch_labels.extend([self.labels[i]] * len(trial_patches))
                self.patch_trial_ids.extend([self.trial_ids[i]] * len(trial_patches))

        self.patches = torch.cat(self.patches, dim=0)
        self.patch_labels = torch.tensor(self.patch_labels)
        self.patch_trial_ids = np.array(self.patch_trial_ids)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx], self.patch_labels[idx]


def prepare_patchtst_data(
        df,
        feature_columns: List[str],
        patch_len: int = 16,
        stride: int = 8,
        test_size: float = 0.15,
        val_size: float = 0.15,
        batch_size: int = 32,
        random_seed: int = 42
):
    """
    Prepare data for PatchTST model with trial-based splitting
    """

    # Group by trial and prepare data
    grouped_data = []
    grouped_labels = []
    trial_ids = []

    for (trial_id, group) in df.groupby('TRIAL_INDEX'):
        # Label based on SCAN_TYPE
        label = (group['SCAN_TYPE'] != 'NORMAL').any().astype(int)
        trial_features = group[feature_columns].values

        grouped_data.append(trial_features)
        grouped_labels.append(label)
        trial_ids.append(trial_id)

    # Convert to arrays
    labels = np.array(grouped_labels)
    trial_ids = np.array(trial_ids)

    # Pad sequences
    max_len = max(len(trial) for trial in grouped_data)
    padded_features = []
    padding_masks = []

    for trial in grouped_data:
        pad_len = max_len - len(trial)
        if pad_len > 0:
            padded_trial = np.pad(trial, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
            mask = np.ones(max_len, dtype=bool)
            mask[len(trial):] = False
        else:
            padded_trial = trial
            mask = np.ones(max_len, dtype=bool)
        padded_features.append(padded_trial)
        padding_masks.append(mask)

    features = np.array(padded_features)
    padding_masks = np.array(padding_masks)

    # Use GroupShuffleSplit for trial-based splitting
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_val_idx, test_idx = next(splitter.split(features, labels, groups=trial_ids))

    # Further split train into train and validation
    val_size_adjusted = val_size / (1 - test_size)  # Adjust val_size for remaining data
    splitter_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adjusted, random_state=random_seed)
    train_idx, val_idx = next(splitter_val.split(
        features[train_val_idx],
        labels[train_val_idx],
        groups=trial_ids[train_val_idx]
    ))

    # Convert relative indices to absolute indices
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    # Print split statistics
    print(f"\nData split statistics:")
    print(f"Total trials: {len(trial_ids)}")
    print(f"Training trials: {len(train_idx)} ({len(train_idx) / len(trial_ids) * 100:.1f}%)")
    print(f"Validation trials: {len(val_idx)} ({len(val_idx) / len(trial_ids) * 100:.1f}%)")
    print(f"Test trials: {len(test_idx)} ({len(test_idx) / len(trial_ids) * 100:.1f}%)")

    # Print class distribution for each split
    for name, indices in [('Train', train_idx), ('Val', val_idx), ('Test', test_idx)]:
        split_labels = labels[indices]
        normal = (split_labels == 0).sum()
        abnormal = (split_labels == 1).sum()
        print(f"\n{name} split:")
        print(f"Normal scans: {normal} ({normal / len(split_labels) * 100:.1f}%)")
        print(f"Abnormal scans: {abnormal} ({abnormal / len(split_labels) * 100:.1f}%)")

    # Create datasets
    train_dataset = EyeTrackingPatchDataset(
        features[train_idx],
        labels[train_idx],
        trial_ids[train_idx],
        patch_len=patch_len,
        stride=stride,
        padding_mask=padding_masks[train_idx]
    )

    val_dataset = EyeTrackingPatchDataset(
        features[val_idx],
        labels[val_idx],
        trial_ids[val_idx],
        patch_len=patch_len,
        stride=stride,
        padding_mask=padding_masks[val_idx]
    )

    test_dataset = EyeTrackingPatchDataset(
        features[test_idx],
        labels[test_idx],
        trial_ids[test_idx],
        patch_len=patch_len,
        stride=stride,
        padding_mask=padding_masks[test_idx]
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


feature_columns = [
    'Pupil_Size',
    'CURRENT_FIX_DURATION',
    'CURRENT_FIX_IA_X',
    'CURRENT_FIX_IA_Y',
    'CURRENT_FIX_INDEX',
    'CURRENT_FIX_COMPONENT_COUNT'
]

config = DataConfig(
    data_path='data/Categorized_Fixation_Data_1_18.csv',
    approach_num=6,
    normalize=True,
    per_slice_target=True,
    participant_id=1
)


df = load_eye_tracking_data(data_path=config.data_path,
                            approach_num=config.approach_num,
                            participant_id=config.participant_id,
                            data_format="legacy")

train_loader, val_loader, test_loader = prepare_patchtst_data(
    df,
    feature_columns,
    patch_len=16,
    stride=8,
    test_size=0.15,
    val_size=0.15
)