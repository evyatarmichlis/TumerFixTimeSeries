import os
import pickle

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
from abc import ABC, abstractmethod
from data_process import IdentSubRec
import gc

from representation_method.utils.data_utils import DataSplitter


@dataclass
class DataConfig:
    """Configuration for data loading."""
    data_path: str
    approach_num: int
    participant_id: Optional[Union[int, List[int]]] = None
    test_data_path: Optional[str] = None
    augment: bool = False
    join_train_test: bool = False
    normalize: bool = True
    per_slice_target: bool = False
    window_size: int = 32
    stride: int = 8
    invalid_value: int = -1

    def __post_init__(self):
        # Updated feature columns with new normalized features
        self.feature_columns = [
            'Pupil_Size',
            'CURRENT_FIX_INDEX',
            'relative_x',
            'relative_y',
            'gaze_velocity'
        ]


class BaseDataLoader(ABC):
    """Abstract base class for data loaders."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.logger = self._setup_logging()
        self._validate_config()
        self.df = None

    def _setup_logging(self):
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


    def _validate_config(self):
        """Validate the configuration parameters."""
        try:
            if not os.path.isabs(self.config.data_path):
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                abs_path = os.path.join(project_root, self.config.data_path)
            else:
                abs_path = self.config.data_path
            self.config.data_path = abs_path
            self.logger.info(f"Using data path: {self.config.data_path}")
            if not os.path.exists(self.config.data_path):
                self.logger.warning(f"Path validation failed for: {self.config.data_path}")
                self.logger.warning("Directory contents:")
                try:
                    parent_dir = os.path.dirname(self.config.data_path)
                    for file in os.listdir(parent_dir):
                        self.logger.info(f"  - {file}")
                except Exception as e:
                    self.logger.warning(f"Could not list directory contents: {str(e)}")
        except Exception as e:
            self.logger.error(f"Path validation error: {str(e)}")

    def _print_data_info(self):
        """Print information about the loaded data."""
        if self.df is not None:
            # Basic dataset info
            self.logger.info("\nDataset Information:")
            self.logger.info(f"Total samples: {len(self.df)}")

            # Detailed target distribution
            target_counts = self.df['target'].value_counts()
            target_ratios = self.df['target'].value_counts(normalize=True)

            self.logger.info("\nTarget Distribution:")
            self.logger.info(f"Number of positive samples (1s): {target_counts.get(1, 0)}")
            self.logger.info(f"Number of negative samples (0s): {target_counts.get(0, 0)}")
            self.logger.info(f"Positive ratio: {target_ratios.get(1, 0):.4f}")
            self.logger.info(f"Negative ratio: {target_ratios.get(0, 0):.4f}")
            self.logger.info(f"Positive to Negative ratio: 1:{(target_counts.get(0, 0) / target_counts.get(1, 1)):.2f}")

            # Feature info
            self.logger.info("\nFeature columns:")
            self.logger.info(self.config.feature_columns)

            # Missing values
            self.logger.info("\nMissing values:")
            self.logger.info(self.df[self.config.feature_columns].isnull().sum())

            # Per participant statistics if available
            if 'RECORDING_SESSION_LABEL' in self.df.columns:
                self.logger.info("\nPer Participant Statistics:")
                for participant in self.df['RECORDING_SESSION_LABEL'].unique():
                    participant_data = self.df[self.df['RECORDING_SESSION_LABEL'] == participant]
                    participant_targets = participant_data['target'].value_counts()
                    participant_ratios = participant_data['target'].value_counts(normalize=True)

                    self.logger.info(f"\nParticipant {participant}:")
                    self.logger.info(f"Total samples: {len(participant_data)}")
                    self.logger.info(f"Positive samples: {participant_targets.get(1, 0)} "
                                     f"({participant_ratios.get(1, 0):.4f})")
                    self.logger.info(f"Negative samples: {participant_targets.get(0, 0)} "
                                     f"({participant_ratios.get(0, 0):.4f})")
    @abstractmethod
    def load_data(self) -> pd.DataFrame:
        pass



    @abstractmethod
    def get_participant_data(self, participant_id: int) -> pd.DataFrame:
        pass


    def get_feature_statistics(self) -> Dict[str, Dict[str, float]]:
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
                self.logger.warning(f"Data quality check failed: {check_name}")
                return False

        self.logger.info("All data quality checks passed.")
        return True

    def create_rolling_windows(self, df_new):
        pass

    def cleanup_temp_files(self, param):
        pass


class TimeSeriesDataLoader(BaseDataLoader):
    @staticmethod
    def letter_to_num(letter: str) -> int:
        return ord(letter.upper()) - ord('A') + 1
    def _normalize_pupil_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize pupil size to 0-1 range"""
        # Ensure we're not dividing by zero
        df['Pupil_Size'] = df['Pupil_Size'].apply(lambda x: float(x))
        min_pupil = df['Pupil_Size'].min()
        max_pupil = df['Pupil_Size'].max()

        if max_pupil > min_pupil:
            df['Pupil_Size'] = (df['Pupil_Size'] - min_pupil) / (max_pupil - min_pupil)
        else:
            self.logger.warning("Could not normalize Pupil_Size: min equals max")
        df['Pupil_Size'] = df['Pupil_Size'].clip(1e-6, 1 - 1e-6)
        df['Pupil_Size'] = df['Pupil_Size'].astype(np.float32)
        return df

    def _add_gaze_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add relative gaze coordinates and velocity features by recording session and trial"""
        grouping_cols = ['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']

        # Calculate relative X and Y positions
        df['relative_x'] = df.groupby(grouping_cols)['GAZE_IA_X'].transform(lambda x: x - x.mean())
        df['relative_y'] = df.groupby(grouping_cols)['GAZE_IA_Y'].transform(lambda x: x - x.mean())

        # Calculate gaze velocity
        df['gaze_velocity'] = df.groupby(grouping_cols).apply(
            lambda group: np.sqrt(
                group['GAZE_IA_X'].diff() ** 2 +
                group['GAZE_IA_Y'].diff() ** 2
            )
        ).reset_index(level=[0, 1], drop=True)

        # Fill NaN values in velocity
        df['gaze_velocity'] = df['gaze_velocity'].fillna(0)

        # Convert to float32 for memory efficiency
        df['relative_x'] = df['relative_x'].astype(np.float32)
        df['relative_y'] = df['relative_y'].astype(np.float32)
        df['gaze_velocity'] = df['gaze_velocity'].astype(np.float32)

        return df
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types and preprocessing for memory efficiency."""
        self.logger.info("Starting data preprocessing...")
        df = df.copy()

        # Initial cleanup
        df = df.replace(['.', np.nan], self.config.invalid_value)

        # Handle per-slice targeting
        if self.config.per_slice_target:
            if 'AILMENT_NUMBER' in df.columns:
                df['has_ailment'] = df.groupby('CURRENT_IMAGE')['AILMENT_NUMBER'].transform(
                    lambda x: (x != self.config.invalid_value).any()
                )
                df['target'] = df['has_ailment'].astype(np.int8)
                df = df.drop('has_ailment', axis=1)
        else:
            df = df.rename(columns={'Hit': 'target'})

        # Convert GAZE_IA to coordinates if needed
        if 'GAZE_IA' in df.columns:
            df['GAZE_IA_X'] = df['GAZE_IA'].apply(
                lambda x: ord(str(x)[0].upper()) - ord('A') if str(x)[0].isalpha()
                else self.config.invalid_value
            ).astype(np.float32)

            df['GAZE_IA_Y'] = df['GAZE_IA'].apply(
                lambda x: int(str(x)[1:]) if str(x)[1:].isdigit()
                else self.config.invalid_value
            ).astype(np.float32)

            df = df.drop('GAZE_IA', axis=1)

        # Normalize pupil size
        df = self._normalize_pupil_size(df)

        # Add normalized gaze features
        df = self._add_gaze_features(df)

        # Convert numeric columns
        numeric_conversions = {
            'RECORDING_SESSION_LABEL': np.int8,
            'TRIAL_INDEX': np.int8,
            'CURRENT_FIX_INDEX': np.float32,
            'target': np.int8
        }

        for col, dtype in numeric_conversions.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        # Drop unnecessary columns
        columns_to_drop = [
            'AILMENT_NUMBER', 'TARGET_ZONE', 'TARGET_XY', 'GAZE_XY',
            'CURRENT_IMAGE', 'IN_SACCADE', 'Hit', 'GAZE_IA_X', 'GAZE_IA_Y'
        ]
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if existing_columns_to_drop:
            df = df.drop(existing_columns_to_drop, axis=1)

        # Keep only required feature columns plus metadata columns
        required_columns = (
                self.config.feature_columns +
                ['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'target', 'SAMPLE_START_TIME']
        )
        df = df[required_columns]

        self.logger.info(f"Final columns: {df.columns.tolist()}")
        return df

    def create_rolling_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
        window_dir = Path(self.config.data_path).parent / 'window_data'
        window_dir.mkdir(exist_ok=True)
        exclude_columns = ['RECORDING_SESSION_LABEL', 'TRIAL_INDEX', 'target']
        features = [col for col in df.columns if col not in exclude_columns]
        global_trial_map = {}
        window_offset = 0
        chunk_size = 1000

        for participant_id in tqdm(df['RECORDING_SESSION_LABEL'].unique()):
            participant_dir = window_dir / f'participant_{participant_id}'
            participant_dir.mkdir(exist_ok=True)

            participant_data = df[df['RECORDING_SESSION_LABEL'] == participant_id].copy()
            trials = list(participant_data.groupby('TRIAL_INDEX'))
            del participant_data

            # Process trials in chunks
            for chunk_start in range(0, len(trials), chunk_size):
                chunk_trials = trials[chunk_start:chunk_start + chunk_size]

                for trial_idx, trial_data in chunk_trials:
                    windows, labels, trial_map = self._process_trial(
                        trial_data, features, window_offset, participant_id, trial_idx
                    )

                    if len(windows) > 0:
                        np.save(f"{participant_dir}/trial_{trial_idx}_windows.npy", windows)
                        np.save(f"{participant_dir}/trial_{trial_idx}_labels.npy", labels)
                        global_trial_map.update(trial_map)
                        window_offset += len(windows)

                    del windows, labels, trial_data
                gc.collect()

            del trials
            gc.collect()

        with open(f"{window_dir}/global_trial_map.pkl", 'wb') as f:
            pickle.dump(global_trial_map, f)

        return self._load_windows(window_dir)


    def _process_trial(self, trial_data: pd.DataFrame, features: List[str],
                       window_offset: int, participant_id: int, trial_idx: int) :
        windows, labels, trial_map = [], [], {}
        current_offset = 0

        trial_data = trial_data.sort_values('SAMPLE_START_TIME')
        for i in range(0, len(trial_data) - self.config.window_size + 1, self.config.stride):
            window = trial_data.iloc[i:i + self.config.window_size]
            window_data = np.round(window[features].values.astype(np.float32), 4)
            has_target = bool(window['target'].any())

            windows.append(window_data)
            labels.append(int(has_target))

            trial_map[window_offset + current_offset] = {
                'participant': participant_id,
                'trial': trial_idx,
                'start_time': int(window['SAMPLE_START_TIME'].iloc[0]),
                'end_time': int(window['SAMPLE_START_TIME'].iloc[-1]),
                'has_target': has_target
            }
            current_offset += 1

        return (np.array(windows, dtype=np.float32) if windows else np.array([]),
                np.array(labels, dtype=np.int8) if labels else np.array([]),
                trial_map)

    def _load_windows(self, window_dir: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load saved windows in chunks to avoid memory issues."""
        all_samples, all_labels = [], []

        for participant_dir in sorted(window_dir.glob('participant_*')):
            for trial_windows in sorted(participant_dir.glob('trial_*_windows.npy')):
                trial_labels = trial_windows.parent / f"trial_{trial_windows.stem.split('_')[1]}_labels.npy"

                samples = np.load(trial_windows)
                labels = np.load(trial_labels)

                all_samples.append(samples)
                all_labels.append(labels)

        with open(window_dir / 'global_trial_map.pkl', 'rb') as f:
            global_trial_map = pickle.load(f)

        return np.concatenate(all_samples), np.concatenate(all_labels), global_trial_map

    def load_data(self, data_type: str = 'raw') -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray, Dict]]:
        """Load data either as raw DataFrame or processed windows."""
        if data_type == 'raw':
            base_path = Path(self.config.data_path)
            if isinstance(self.config.participant_id, (int, str)):
                file_path = base_path / f"{self.config.participant_id}_Formatted_Sample.csv"
                if not file_path.exists():
                    raise FileNotFoundError(f"No data file found for participant {self.config.participant_id}")
                df = pd.read_csv(file_path)
            else:
                # Load all available participants
                files = list(base_path.glob("*_Formatted_Sample.csv"))
                if not files:
                    raise ValueError(f"No formatted sample files found in {base_path}")
                df = pd.concat([pd.read_csv(f) for f in tqdm(files, desc='Loading data')], ignore_index=True)

            self.df = self._preprocess_data(df)
            return self.df

        elif data_type == 'windowed':
            window_dir = Path(self.config.data_path).parent / 'window_data'
            return self._load_windows(window_dir)

        raise ValueError("data_type must be either 'raw' or 'windowed'")

    def get_participant_data(self, participant_id: int) -> pd.DataFrame:
        """Load and preprocess data for a specific participant."""
        file_path = Path(self.config.data_path) / f"{participant_id}_Formatted_Sample.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"No data file found for participant {participant_id}")

        df = pd.read_csv(file_path)
        return self._preprocess_data(df)
class LegacyDataLoader(BaseDataLoader):
    """Handles loading and preprocessing of legacy eye tracking data format."""

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess legacy format data."""
        try:
            self.df = IdentSubRec.get_df_for_training(data_file_path = self.config.data_path,approach_num=self.config.approach_num)
            #
            # self.df['Pupil_Size_Diff_1'] = self.df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'])[
            #     'Pupil_Size'].diff()
            # self.df['Pupil_Size_Diff_2'] = self.df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'])[
            #     'Pupil_Size'].diff(2)
            # self.df['CURRENT_FIX_DURATION_1'] = self.df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'])[
            #     'CURRENT_FIX_DURATION'].diff()
            # self.df['CURRENT_FIX_DURATION_2'] = self.df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX'])[
            #     'CURRENT_FIX_DURATION'].diff(2)
            # self.df['Pupil_Size_Diff_1'] = self.df['Pupil_Size_Diff_1'].fillna(0)
            # self.df['Pupil_Size_Diff_2'] = self.df['Pupil_Size_Diff_2'].fillna(0)
            # self.df['CURRENT_FIX_DURATION_1'] = self.df['CURRENT_FIX_DURATION_1'].fillna(0)
            # self.df['CURRENT_FIX_DURATION_2'] = self.df['CURRENT_FIX_DURATION_2'].fillna(0)
            # Add your legacy preprocessing steps here
            self.df = self._normalize_pupil_size(self.df)
            self._print_data_info()
            return self.df


        except Exception as e:
            self.logger.error(f"Error loading legacy data: {str(e)}")
            raise

    def get_participant_data(self, participant_id: int) -> pd.DataFrame:
        """Get data for a specific participant from legacy format."""
        if self.df is None:
            self.load_data()

        participant_df = self.df[self.df['RECORDING_SESSION_LABEL'] == participant_id].copy()

        self.logger.info(f"\nParticipant {participant_id} data:")
        self.logger.info(f"Number of samples: {len(participant_df)}")
        self.logger.info("Class distribution:")
        self.logger.info(participant_df['target'].value_counts())

        return participant_df


    def _print_data_info(self):
            """Print information about the loaded data."""
            if self.df is not None:
                # Basic dataset info
                self.logger.info("\nDataset Information:")
                self.logger.info(f"Total samples: {len(self.df)}")

                # Detailed target distribution
                target_counts = self.df['target'].value_counts()
                target_ratios = self.df['target'].value_counts(normalize=True)

                self.logger.info("\nTarget Distribution:")
                self.logger.info(f"Number of positive samples (1s): {target_counts.get(1, 0)}")
                self.logger.info(f"Number of negative samples (0s): {target_counts.get(0, 0)}")
                self.logger.info(f"Positive ratio: {target_ratios.get(1, 0):.4f}")
                self.logger.info(f"Negative ratio: {target_ratios.get(0, 0):.4f}")
                self.logger.info(f"Positive to Negative ratio: 1:{(target_counts.get(0, 0) / target_counts.get(1, 1)):.2f}")

                # Feature info
                self.logger.info("\nFeature columns:")
                self.logger.info(self.config.feature_columns)

    def _normalize_pupil_size(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize pupil size to 0-1 range"""
        # Ensure we're not dividing by zero
        df['Pupil_Size'] = df['Pupil_Size'].apply(lambda x: float(x))
        min_pupil = df['Pupil_Size'].min()
        max_pupil = df['Pupil_Size'].max()

        if max_pupil > min_pupil:
            df['Pupil_Size'] = (df['Pupil_Size'] - min_pupil) / (max_pupil - min_pupil)
        else:
            self.logger.warning("Could not normalize Pupil_Size: min equals max")
        df['Pupil_Size'] = df['Pupil_Size'].clip(1e-6, 1 - 1e-6)
        df['Pupil_Size'] = df['Pupil_Size'].astype(np.float32)
        return df

def create_data_loader(data_format: str, config: DataConfig) -> BaseDataLoader:
    """Factory function to create appropriate data loader based on format."""
    # Convert relative paths to absolute using project root
    if not os.path.isabs(config.data_path):
        project_root = find_project_root()
        config.data_path = os.path.join(project_root, config.data_path)

    if data_format.lower() == 'time_series':
        return TimeSeriesDataLoader(config)
    elif data_format.lower() == 'legacy':
        return LegacyDataLoader(config)
    else:
        raise ValueError(f"Unknown data format: {data_format}. Use 'time_series' or 'legacy'.")

def load_eye_tracking_data(
        data_path: str,
        approach_num: int,
        data_format: str = 'time_series',
        participant_id: Optional[int] = None,
        **kwargs
) -> pd.DataFrame:
    """Convenience function to load eye tracking data."""
    config = DataConfig(
        data_path=data_path,
        approach_num=approach_num,
        participant_id=participant_id,
        **kwargs
    )

    loader = create_data_loader(data_format, config)
    if participant_id is not None:
        return loader.get_participant_data(participant_id)
    return loader.load_data()



def find_project_root():
    """Find the project root directory (TumerFixTimeSeries)."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        if os.path.basename(current_dir) == 'TumerFixTimeSeries':
            return current_dir
        parent = os.path.dirname(current_dir)
        if parent == current_dir:  # Reached root directory
            raise ValueError("Could not find project root directory (TumerFixTimeSeries)")
        current_dir = parent





# Example usage
if __name__ == "__main__":
    # Example for new time series data
    config_new = DataConfig(
        data_path='data/Formatted_Samples_ML',
        approach_num=6,
        normalize=True,
        per_slice_target=True,
        participant_id=1,
        window_size= 1000,
        stride=5
    )

    # Using the new time series format
    loader_new = create_data_loader('time_series', config_new)
    df_new = loader_new.load_data(data_type = 'raw')
    # window_df,labels,meta_data =loader_new.create_rolling_windows(df_new)
    # Add cleanup
    window_dir = Path(loader_new.config.data_path).parent / 'window_data'

    # splitter = DataSplitter(window_df, labels, meta_data)
    #
    # # Split by trials
    # (train_data_trial, train_labels_trial), (val_data_trial, val_labels_trial), (test_data_trial, test_labels_trial) = \
    #     splitter.split_by_trials()
    #
    # # Split within trials timeline
    # (train_data_time, train_labels_time), (val_data_time, val_labels_time), (test_data_time, test_labels_time) = \
    #     splitter.split_within_trials()