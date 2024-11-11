import os
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from data_process import IdentSubRec


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_path: str
    approach_num: int
    participant_id :int
    test_data_path: Optional[str] = None
    augment: bool = False
    join_train_test: bool = False
    normalize: bool = True
    remove_surrounding_hits: int = 0
    update_surrounding_hits: int = 0
    feature_columns: List[str] = None

    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                'Pupil_Size',
                'CURRENT_FIX_DURATION',
                'CURRENT_FIX_IA_X',
                'CURRENT_FIX_IA_Y',
                'CURRENT_FIX_INDEX',
                'CURRENT_FIX_COMPONENT_COUNT'
            ]


class DataLoader:
    """Handles data loading and preprocessing for the eye tracking dataset."""

    def __init__(self, config: DataConfig):
        """
        Initialize DataLoader with configuration.

        Args:
            config: DataConfig object containing data loading parameters
        """
        self.config = config
        self._validate_config()
        self.df = None

    def _validate_config(self):
        """Validate the configuration parameters."""
        if not os.path.exists(self.config.data_path):
            raise FileNotFoundError(f"Data file not found: {self.config.data_path}")

        if self.config.test_data_path and not os.path.exists(self.config.test_data_path):
            raise FileNotFoundError(f"Test data file not found: {self.config.test_data_path}")

    def load_data(self) -> pd.DataFrame:
        """
        Load and preprocess the data using IdentSubRec.

        Returns:
            pd.DataFrame: Processed dataframe
        """
        print("Loading data...")
        init_kwargs = {
            'data_file_path': self.config.data_path,
            'test_data_file_path': self.config.test_data_path,
            'augment': self.config.augment,
            'join_train_and_test_data': self.config.join_train_test,
            'normalize': self.config.normalize,
            'remove_surrounding_to_hits': self.config.remove_surrounding_hits,
            'update_surrounding_to_hits': self.config.update_surrounding_hits,
            'approach_num': self.config.approach_num,
        }

        try:
            ident_sub_rec = IdentSubRec(**init_kwargs)
            self.df = ident_sub_rec.df
            print("Data loaded successfully.")
            self._print_data_info()
            return self.df
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def _print_data_info(self):
        """Print information about the loaded data."""
        if self.df is not None:
            print("\nDataset Information:")
            print(f"Total samples: {len(self.df)}")
            print("\nClass distribution:")
            print(self.df['target'].value_counts())
            print("\nFeature columns:")
            print(self.config.feature_columns)
            print("\nMissing values:")
            print(self.df[self.config.feature_columns].isnull().sum())

    def get_participant_data(self, participant_id: int) -> pd.DataFrame:
        """
        Get data for a specific participant.

        Args:
            participant_id: The participant ID to filter by

        Returns:
            pd.DataFrame: Filtered dataframe for the specified participant
        """
        if self.df is None:
            self.load_data()

        participant_df = self.df[self.df['RECORDING_SESSION_LABEL'] == participant_id].copy()
        print(f"\nParticipant {participant_id} data:")
        print(f"Number of samples: {len(participant_df)}")
        print("Class distribution:")
        print(participant_df['target'].value_counts())

        return participant_df

    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate basic statistics for each feature.

        Returns:
            Dict containing statistics for each feature
        """
        if self.df is None:
            self.load_data()

        stats = {}
        for feature in self.config.feature_columns:
            stats[feature] = {
                'mean': self.df[feature].mean(),
                'std': self.df[feature].std(),
                'min': self.df[feature].min(),
                'max': self.df[feature].max()
            }
        return stats

    def validate_data_quality(self) -> bool:
        """
        Perform basic data quality checks.

        Returns:
            bool: True if all checks pass
        """
        if self.df is None:
            self.load_data()

        checks = {
            'missing_values': self.df[self.config.feature_columns].isnull().sum().sum() == 0,
            'target_present': 'target' in self.df.columns,
            'participant_ids': 'RECORDING_SESSION_LABEL' in self.df.columns,
            'feature_columns': all(col in self.df.columns for col in self.config.feature_columns)
        }

        for check_name, passed in checks.items():
            if not passed:
                print(f"Data quality check failed: {check_name}")
                return False

        print("All data quality checks passed.")
        return True


def load_eye_tracking_data(
        data_path: str,
        approach_num: int,
        participant_id: Optional[int] = None,
        **kwargs
) -> pd.DataFrame:
    """
    Convenience function to load eye tracking data.

    Args:
        data_path: Path to the data file
        approach_num: Approach number for processing
        participant_id: Optional participant ID to filter by
        **kwargs: Additional arguments for DataConfig

    Returns:
        pd.DataFrame: Processed dataframe
    """
    config = DataConfig(
        data_path=data_path,
        approach_num=approach_num,
        participant_id = participant_id,
        **kwargs
    )

    loader = DataLoader(config)
    if participant_id is not None:
        return loader.get_participant_data(participant_id)
    return loader.load_data()


# Example usage
if __name__ == "__main__":
    # Example configuration
    config = DataConfig(
        data_path='data/Categorized_Fixation_Data_1_18.csv',
        approach_num=6,
        normalize=True
    )

    loader = DataLoader(config)
    df = loader.load_data()

    participant_df = loader.get_participant_data(1)

    stats = loader.get_feature_statistics()

    is_valid = loader.validate_data_quality()
