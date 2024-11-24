from typing import Tuple, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from dataclasses import dataclass
import json

from keras.src.utils.config import Config


@dataclass
class SplitConfig:
    """Configuration for data splitting"""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    time_column: str = 'SAMPLE_START_TIME'
    trial_column: str = 'RECORDING_SESSION_LABEL'
    random_seed: Optional[int] = None

    def __post_init__(self):
        assert np.isclose(self.train_ratio + self.val_ratio + self.test_ratio, 1.0), \
            "Split ratios must sum to 1"


class TrialSplitter:
    """
    Handles splitting of time series data while maintaining temporal order within trials.

    Features:
    - Splits data by time within each trial
    - Maintains temporal ordering
    - Supports saving/loading split indices
    - Provides split statistics
    """

    def __init__(self, config:Config):
        """
        Initialize the splitter with configuration.

        Args:
            config: Either a SplitConfig object or a dictionary with configuration parameters
        """
        self.config = config if isinstance(config, SplitConfig) else SplitConfig(**config)
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

        # Log split statistics
        self._log_split_statistics(split_info)

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

        # Per-trial statistics
        trial_stats = pd.DataFrame.from_dict(
            {trial: {
                'samples': info['total_samples'],
                'train_ratio': info['train_samples'] / info['total_samples'],
                'val_ratio': info['val_samples'] / info['total_samples'],
                'test_ratio': info['test_samples'] / info['total_samples']
            }
                for trial, info in split_info['trials'].items()},
            orient='index'
        )

        self.logger.info("\nPer-Trial Statistics:")
        self.logger.info(f"\nSample Distribution:\n{trial_stats.describe()}")

    def get_split_summary(self) -> dict:
        """
        Get a summary of the current split configuration and statistics.

        Returns:
            Dictionary containing split configuration and statistics
        """
        if not self.split_indices:
            return {"status": "No splits performed yet"}

        summary = {
            "config": vars(self.config),
            "splits": {
                "n_trials": len(self.split_indices),
                "trials": {}
            }
        }

        for trial, indices in self.split_indices.items():
            summary["splits"]["trials"][trial] = {
                "train_samples": indices["train"][1] - indices["train"][0],
                "val_samples": indices["val"][1] - indices["val"][0],
                "test_samples": indices["test"][1] - indices["test"][0]
            }

        return summary


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


