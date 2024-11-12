"""
Utilities for data processing and window creation.
"""

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


def create_time_series(time_series_df, interval='30ms', window_size=5, resample=False, feature_columns=None):
    """
    Create time series windows from DataFrame.

    Args:
        time_series_df: Input DataFrame
        interval: Resampling interval
        window_size: Size of sliding window
        resample: Whether to resample the data
        feature_columns: List of feature column names

    Returns:
        Tuple of (samples, labels, weights)
    """
    if feature_columns is None:
        feature_columns = ['Pupil_Size', 'CURRENT_FIX_DURATION', 'CURRENT_FIX_IA_X', 'CURRENT_FIX_IA_Y',
                           'CURRENT_FIX_INDEX', 'CURRENT_FIX_COMPONENT_COUNT']
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
    return create_windows(grouped, window_size, feature_columns)


def split_train_test_for_time_series(time_series_df, input_data_points, test_size=0.2, random_state=0,
                                     split_columns=None):
    if split_columns is None:
        split_columns = ['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']
    df = time_series_df
    df['group'] = df[split_columns].apply(
        lambda row: '_'.join(row.values.astype(str)), axis=1
    )
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    split_indices = list(gss.split(X=df[input_data_points], y=df['target'], groups=df['group']))[0]

    train_index, test_index = split_indices

    x_train = df.iloc[train_index].drop(columns='target', errors='ignore')
    x_test = df.iloc[test_index].drop(columns='target', errors='ignore')

    y_train = df['target'].iloc[train_index]
    y_test = df['target'].iloc[test_index]

    train_df = pd.concat([x_train, y_train], axis=1)
    test_df = pd.concat([x_test, y_test], axis=1)

    return train_df, test_df
