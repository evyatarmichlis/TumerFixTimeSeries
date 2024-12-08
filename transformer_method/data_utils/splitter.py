from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass
import json

from keras.src.utils.config import Config

from transformer_method.configs.config import SplitConfig


class TrialSplitter:
    """
    Handles splitting of time series data while maintaining temporal order within trials.

    Features:
    - Splits data by time within each trial
    - Maintains temporal ordering
    - Supports saving/loading split indices
    - Provides split statistics
    """

    def __init__(self, config:SplitConfig):
        """
        Initialize the splitter with configuration.

        Args:
            config: Either a SplitConfig object or a dictionary with configuration parameters
        """
        self.config = config
        self.split_indices = {}
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging for the splitter"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
            self.logger.setLevel(logging.INFO)

    def split_by_time(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame by time while maintaining temporal order within each trial.

        Args:
            df: Input DataFrame containing trial and time information

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

        trials = df[self.config.trial_column].unique()
        train_dfs, val_dfs, test_dfs = [], [], []
        split_info = {'trials': {}}

        for trial in trials:
            trial_df = df[df[self.config.trial_column] == trial].sort_values(self.config.time_column)
            n_samples = len(trial_df)

            # Calculate split indices
            train_idx = int(n_samples * self.config.train_ratio)
            val_idx = train_idx + int(n_samples * self.config.val_ratio)

            # Store split indices
            self.split_indices[trial] = {
                'train': (0, train_idx),
                'val': (train_idx, val_idx),
                'test': (val_idx, n_samples)
            }

            # Split the data
            train_dfs.append(trial_df.iloc[:train_idx])
            val_dfs.append(trial_df.iloc[train_idx:val_idx])
            test_dfs.append(trial_df.iloc[val_idx:])

            # Collect split information
            split_info['trials'][trial] = {
                'total_samples': n_samples,
                'train_samples': train_idx,
                'val_samples': val_idx - train_idx,
                'test_samples': n_samples - val_idx,
                'time_range': {
                    'start': trial_df[self.config.time_column].iloc[0],
                    'end': trial_df[self.config.time_column].iloc[-1]
                }
            }

        # Combine splits
        train_df = pd.concat(train_dfs, ignore_index=True)
        val_df = pd.concat(val_dfs, ignore_index=True)
        test_df = pd.concat(test_dfs, ignore_index=True)
        train_df.drop(columns=self.config.time_column, inplace=True)
        val_df.drop(columns=self.config.time_column, inplace=True)
        test_df.drop(columns=self.config.time_column, inplace=True)

        # Log split statistics
        self._log_split_statistics(split_info)

        return train_df, val_df, test_df

    def split_by_trials(self, df: pd.DataFrame,seed = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame by trials maintaining temporal order with 70/15/15 split.
        """
        self.logger.info("\nSplitting data by trials...")
        np.random.seed(seed)
        # Get unique trials in temporal order
        trials = df[['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']].drop_duplicates()

        # Calculate split indices
        n_trials = len(trials)
        train_idx = int(0.7 * n_trials)
        val_idx = int(0.85 * n_trials)

        # Split trials maintaining order
        train_trials = trials.iloc[:train_idx]
        val_trials = trials.iloc[train_idx:val_idx]
        test_trials = trials.iloc[val_idx:]

        # Create splits maintaining temporal order
        train_mask = df.set_index(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).index.isin(
            train_trials.set_index(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).index
        )
        val_mask = df.set_index(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).index.isin(
            val_trials.set_index(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).index
        )
        test_mask = df.set_index(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).index.isin(
            test_trials.set_index(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']).index
        )

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

        # Log split information
        self.logger.info("\nSplit Statistics:")
        self.logger.info(f"Total trials: {n_trials}")
        self.logger.info(f"Train trials: {len(train_trials)} ({len(train_trials) / n_trials:.1%})")
        self.logger.info(f"Val trials: {len(val_trials)} ({len(val_trials) / n_trials:.1%})")
        self.logger.info(f"Test trials: {len(test_trials)} ({len(test_trials) / n_trials:.1%})")

        for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
            total = len(split_df)
            class_1_count = (split_df['target'] == 1).sum()
            self.logger.info(f"\n{name}:")
            self.logger.info(f"Total samples: {total}")
            self.logger.info(f"Class 1 samples: {class_1_count} ({class_1_count / total:.1%})")

        return train_df, val_df, test_df

    def save_split_indices(self, output_path: Union[str, Path]):
        """
        Save split indices to file for reproducibility.

        Args:
            output_path: Path to save the split indices
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(self.split_indices, f, indent=4)

        self.logger.info(f"Split indices saved to {output_path}")

    def load_split_indices(self, input_path: Union[str, Path]):
        """
        Load previously saved split indices.

        Args:
            input_path: Path to the split indices file
        """
        with open(input_path, 'r') as f:
            self.split_indices = json.load(f)

        self.logger.info(f"Split indices loaded from {input_path}")

    def _log_split_statistics(self, split_info: dict):
        """Log statistics about the split"""
        total_samples = sum(trial['total_samples'] for trial in split_info['trials'].values())
        total_train = sum(trial['train_samples'] for trial in split_info['trials'].values())
        total_val = sum(trial['val_samples'] for trial in split_info['trials'].values())
        total_test = sum(trial['test_samples'] for trial in split_info['trials'].values())

        self.logger.info("\nData Split Statistics:")
        self.logger.info(f"Total Trials: {len(split_info['trials'])}")
        self.logger.info(f"Total Samples: {total_samples}")
        self.logger.info(f"Train Samples: {total_train} ({total_train / total_samples:.1%})")
        self.logger.info(f"Val Samples: {total_val} ({total_val / total_samples:.1%})")
        self.logger.info(f"Test Samples: {total_test} ({total_test / total_samples:.1%})")





def create_temporal_splits(
        df: pd.DataFrame,
        config:Config = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to create temporal splits with default configuration.

    Args:
        df: Input DataFrame
        config: Optional configuration (will use defaults if not provided)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if config is None:
        config = SplitConfig()
    elif isinstance(config, dict):
        config = SplitConfig(**config)

    splitter = TrialSplitter(config)
    return splitter.split_by_time(df)


