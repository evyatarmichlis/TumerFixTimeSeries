"""
Utilities for data processing and window creation.
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X', 'CURRENT_FIX_IA_Y',
                   'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT']
def resample_func(time_series_df, interval, feature_columns=None):
    if feature_columns is None:
        feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X', 'CURRENT_FIX_IA_Y',
                           'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT']
    resampled_data = []

    for (session, trial), group in time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX','CURRENT_FIX_INDEX']):
        group = group.set_index('Cumulative_Time')

        # Resample features to a fixed interval (e.g., 1 second)
        resampled_features = group[feature_columns].resample(interval).mean()

        # Resample the target column using the max function
        resampled_target = group['target'].resample(interval).max()

        resampled_group = resampled_features.merge(resampled_target, on='Cumulative_Time', how='left')

        resampled_group.ffill(inplace=True)
        resampled_group.bfill(inplace=True)

        # Add 'RECORDING_SESSION_LABEL' and 'TRIAL_INDEX' back to the resampled group
        resampled_group['RECORDING_SESSION_LABEL'] = session
        resampled_group['TRIAL_INDEX'] = trial

        # Append the resampled group to the list
        resampled_data.append(resampled_group)

    # Concatenate all resampled groups into a single DataFrame
    time_series_df = pd.concat(resampled_data).reset_index()

    return time_series_df

def create_windows(grouped, window_size, feature_columns=None):
    """
    Create windows from grouped time series data.

    Args:
        grouped: Grouped DataFrame
        window_size: Size of the sliding window
        feature_columns: List of feature column names

    Returns:
        Tuple of (samples, labels, weights)
    """
    if feature_columns is None:
        feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X', 'CURRENT_FIX_IA_Y',
                           'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT']
    max_possible_time_length = 0

    # First pass to find maximum time length
    for _, group in grouped:
        group = group.sort_values(by='Cumulative_Time')
        for start in range(0, len(group) - window_size + 1):
            window = group.iloc[start:start + window_size]
            time_length = window['Cumulative_Time'].max() - window['Cumulative_Time'].min()
            time_length_seconds = time_length.total_seconds()
            if time_length_seconds > max_possible_time_length:
                max_possible_time_length = time_length_seconds

    samples = []
    labels = []
    weights = []

    # Second pass to create windows
    for _, group in grouped:
        group = group.sort_values(by='Cumulative_Time')
        for start in range(0, len(group) - window_size + 1):
            window = group.iloc[start:start + window_size]
            features = window[feature_columns].values
            samples.append(features)

            label = window['target'].max()
            labels.append(label)

            time_length_seconds = (
                    window['Cumulative_Time'].max().total_seconds() -
                    window['Cumulative_Time'].min().total_seconds()
            )
            normalized_weight = time_length_seconds / max_possible_time_length
            weights.append(normalized_weight)

    return (np.array(samples), np.array(labels), np.array(weights))


def create_time_series(time_series_df, participant_id, interval='30ms', window_size=5,
                       resample=False, feature_columns=None, load_existing=False):
    """
    Create time series windows from DataFrame with option to load from cache.

    Args:
        time_series_df: Input DataFrame
        participant_id: Participant ID for file naming
        interval: Resampling interval
        window_size: Size of sliding window
        resample: Whether to resample the data
        feature_columns: List of feature column names
        load_existing: Whether to load existing processed data if available

    Returns:
        Tuple of (samples, labels, weights)
    """
    # Define file paths
    save_dir = f'data/legacy/participant{participant_id}'
    samples_path = os.path.join(save_dir, f'samples_w{window_size}.npy')
    labels_path = os.path.join(save_dir, f'labels_w{window_size}.npy')
    weights_path = os.path.join(save_dir, f'weights_w{window_size}.npy')

    # Try loading existing files if requested
    if load_existing:
        if all(os.path.exists(f) for f in [samples_path, labels_path, weights_path]):
            print(f"Loading existing processed data for participant {participant_id}")
            samples = np.load(samples_path)
            labels = np.load(labels_path)
            weights = np.load(weights_path)
            return samples, labels, weights

    if feature_columns is None:
        feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X',
                           'CURRENT_FIX_IA_Y', 'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT']

    time_series_df = time_series_df.copy()

    # Calculate cumulative time
    time_series_df['Cumulative_Time_Update'] = (
            time_series_df['CURRENT_FIX_COMPONENT_INDEX'] == 1).astype(int)
    time_series_df['Cumulative_Time'] = 0.0

    # Calculate cumulative times for each group
    for _, group in time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']):
        cumulative_time = 0
        cumulative_times = []

        for _, row in group.iterrows():
            if row['Cumulative_Time_Update'] == 1:
                cumulative_time += row['CURRENT_FIX_DURATION']
            cumulative_times.append(cumulative_time)

        time_series_df.loc[group.index, 'Cumulative_Time'] = cumulative_times

    # Add unique index and adjust cumulative time
    time_series_df['Unique_Index'] = time_series_df.groupby(
        ['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).cumcount()
    time_series_df['Cumulative_Time'] += time_series_df['Unique_Index'] * 1e-4

    # Convert to timedelta
    time_series_df['Cumulative_Time'] = pd.to_timedelta(
        time_series_df['Cumulative_Time'], unit='s')

    if resample:
        time_series_df = resample_func(time_series_df, interval)

    grouped = time_series_df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'])
    samples, labels, weights = create_windows(grouped, window_size, feature_columns)

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save processed data
    np.save(samples_path, samples)
    np.save(labels_path, labels)
    np.save(weights_path, weights)
    print(f"Saved processed data for participant {participant_id}")

    return samples, labels, weights

def split_train_test_for_time_series(df, input_columns= None, target_column='target',
                                     split_columns=None,
                                     split_type='random', test_size=0.2, random_state=0):
    """
    Split time series data into train and test sets using either random or temporal splitting.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing features and target
    input_columns : list
        List of column names to use as input features
    target_column : str
        Name of the target column
    split_columns : list
        Columns to use for grouping (e.g., ['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'])
    split_type : str
        'random': Use GroupShuffleSplit to randomly split while preserving groups
        'temporal': Split based on trial order (first N% train, last N% test)
    test_size : float
        Proportion of data for testing (0.0 to 1.0)
    random_state : int
        Random seed for reproducibility

    Returns:
    --------
    train_df, test_df : pd.DataFrame
        Split DataFrames containing both features and target
    """
    # Create group labels
    if split_columns is None:
        split_columns = ['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']

    if input_columns is None:
        input_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X', 'CURRENT_FIX_IA_Y',
                   'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT']

    df['group'] = df[split_columns].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1
    )

    if split_type == 'temporal':
        # Get unique groups in order
        unique_groups = df['group'].unique()
        n_test_groups = int(len(unique_groups) * test_size)

        # Split groups into train/test
        train_groups = unique_groups[:-n_test_groups]
        test_groups = unique_groups[-n_test_groups:]

        # Create masks for train/test
        train_mask = df['group'].isin(train_groups)
        test_mask = df['group'].isin(test_groups)

        # Split the data
        train_df = df[train_mask]
        test_df = df[test_mask]

    else:  # random split
        # Use GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(gss.split(
            X=df[input_columns],
            y=df[target_column],
            groups=df['group']
        ))

        # Split the data
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

    return train_df, test_df


class DataSplitter:
    def __init__(self, window_data, labels, meta_data, random_state=42):
        self.window_data = window_data
        self.labels = labels
        self.meta_data = meta_data
        self.random_state = random_state
        np.random.seed(random_state)

    def print_split_stats(self, name, labels):
        total = len(labels)
        ones = np.sum(labels)
        zeros = total - ones
        print(f"\n{name} Statistics:")
        print(f"Total samples: {total}")
        print(f"Class 0 (no target): {zeros} ({zeros / total * 100:.2f}%)")
        print(f"Class 1 (target): {ones} ({ones / total * 100:.2f}%)")

    def split_by_trials(self, train_size=0.7, val_size=0.15):
        """Split data by allocating whole trials to train/val/test sets."""
        unique_trials = list(set(info['trial'] for info in self.meta_data.values()))
        np.random.shuffle(unique_trials)

        n_trials = len(unique_trials)
        train_idx = int(n_trials * train_size)
        val_idx = int(n_trials * (train_size + val_size))

        train_trials = set(unique_trials[:train_idx])
        val_trials = set(unique_trials[train_idx:val_idx])
        test_trials = set(unique_trials[val_idx:])

        splits = self._split_data_by_trial_sets(train_trials, val_trials, test_trials)

        print(f"\nSplit by Trials (random_state={self.random_state}):")
        self.print_split_stats("Training", splits[0][1])
        self.print_split_stats("Validation", splits[1][1])
        self.print_split_stats("Testing", splits[2][1])

        return splits

    def split_within_trials(self, train_size=0.7, val_size=0.15):
        """Split each trial's timeline into train/val/test sets."""
        trial_groups = {}

        for window_idx, info in self.meta_data.items():
            trial_id = (info['participant'], info['trial'])
            if trial_id not in trial_groups:
                trial_groups[trial_id] = []
            trial_groups[trial_id].append(window_idx)

        train_indices = []
        val_indices = []
        test_indices = []

        for indices in trial_groups.values():
            indices.sort(key=lambda x: self.meta_data[x]['start_time'])
            n_windows = len(indices)

            train_idx = int(n_windows * train_size)
            val_idx = int(n_windows * (train_size + val_size))

            train_indices.extend(indices[:train_idx])
            val_indices.extend(indices[train_idx:val_idx])
            test_indices.extend(indices[val_idx:])

        splits = self._get_split_data(train_indices, val_indices, test_indices)

        print(f"\nSplit Within Trials (random_state={self.random_state}):")
        self.print_split_stats("Training", splits[0][1])
        self.print_split_stats("Validation", splits[1][1])
        self.print_split_stats("Testing", splits[2][1])

        return splits

    def _split_data_by_trial_sets(self, train_trials, val_trials, test_trials):
        train_indices = []
        val_indices = []
        test_indices = []

        for window_idx, info in self.meta_data.items():
            trial = info['trial']
            if trial in train_trials:
                train_indices.append(window_idx)
            elif trial in val_trials:
                val_indices.append(window_idx)
            else:
                test_indices.append(window_idx)

        return self._get_split_data(train_indices, val_indices, test_indices)

    def _get_split_data(self, train_indices, val_indices, test_indices):
        return (
            (self.window_data[train_indices], self.labels[train_indices]),
            (self.window_data[val_indices], self.labels[val_indices]),
            (self.window_data[test_indices], self.labels[test_indices])
        )

