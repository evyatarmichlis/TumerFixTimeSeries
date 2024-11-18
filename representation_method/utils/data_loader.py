import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
import logging
from tqdm import tqdm
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from functools import partial

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
    remove_surrounding_hits: int = 0
    update_surrounding_hits: int = 0
    per_slice_target: bool = False
    remove_saccade: bool = True
    window_size: int = 100  # New parameter for rolling window size
    stride: int = 1  # New parameter for window stride
    feature_columns: List[str] = None
    invalid_value: int = -1

    def __post_init__(self):

        if self.feature_columns is None and not  self.remove_saccade:
            self.feature_columns = [
                'Pupil_Size',
                'GAZE_IA_X',
                'GAZE_IA_Y',
                'CURRENT_FIX_INDEX',
                'SAMPLE_START_TIME',
                'IN_BLINK',
                'IN_SACCADE'
            ]
        else:
            self.feature_columns = [
                'Pupil_Size',
                'GAZE_IA_X',
                'GAZE_IA_Y',
                'CURRENT_FIX_INDEX',
                'SAMPLE_START_TIME',
                'IN_BLINK',
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


class TimeSeriesDataLoader(BaseDataLoader):
    """Handles loading and preprocessing of time series formatted eye tracking data."""

    @staticmethod
    def letter_to_num(letter: str) -> int:
        """Convert a letter to its corresponding number (A=1, B=2, etc.)."""
        return ord(letter.upper()) - ord('A') + 1

    def load_data(self, data_type: str = 'raw') -> Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Load data either as raw DataFrame or as processed windows.

        Parameters:
        -----------
        data_type : str
            'raw': Load raw CSV data
            'windowed': Load preprocessed windowed data

        Returns:
        --------
        Union[pd.DataFrame, Tuple[np.ndarray, np.ndarray, Dict]]:
            For raw: Returns DataFrame
            For windowed: Returns (samples, labels, trial_maps)
        """
        if data_type == 'raw':
            return self._load_raw_data()
        elif data_type == 'windowed':
            return self._load_windowed_data()
        else:
            raise ValueError("data_type must be either 'raw' or 'windowed'")

    def _load_raw_data(self) -> pd.DataFrame:
        """Load and preprocess raw time series data."""
        base_path = Path(self.config.data_path)

        if isinstance(self.config.participant_id, (int, str)):
            # Single participant
            file_path = base_path / f"{self.config.participant_id}_Formatted_Sample.csv"
            if not file_path.exists():
                raise FileNotFoundError(f"No data file found for participant {self.config.participant_id}")

            self.logger.info(f"Loading data for participant {self.config.participant_id}")
            df = pd.read_csv(file_path)
            self.df = self._preprocess_data(df)

        elif isinstance(self.config.participant_id, (list, tuple)):
            # Multiple specific participants
            self.logger.info(f"Loading data for participants {self.config.participant_id}")
            dfs = []
            for participant_id in tqdm(self.config.participant_id, desc='Loading participant data'):
                file_path = base_path / f"{participant_id}_Formatted_Sample.csv"
                if not file_path.exists():
                    raise FileNotFoundError(f"No data file found for participant {participant_id}")
                df = pd.read_csv(file_path)
                dfs.append(df)

            self.df = pd.concat(dfs, ignore_index=True)
            self.df = self._preprocess_data(self.df)

        else:
            # Load all available participants
            files = list(base_path.glob("*_Formatted_Sample.csv"))
            if not files:
                raise ValueError(f"No formatted sample files found in {base_path}")

            self.logger.info("Loading data for all participants")
            dfs = []
            for file in tqdm(files, desc='Loading participant data'):
                df = pd.read_csv(file)
                dfs.append(df)

            self.df = pd.concat(dfs, ignore_index=True)
            self.df = self._preprocess_data(self.df)

        self._print_data_info()
        return self.df

    def _load_windowed_data(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load all participants' windowed data and combine."""
        if not isinstance(self.config.participant_id, (list, tuple)):
            self.config.participant_id = [self.config.participant_id]

        all_samples = []
        all_labels = []
        all_trial_maps = {}
        window_idx_offset = 0

        for participant_id in self.config.participant_id:
            samples, labels, trial_map = self.load_participant_windows(
                self.config.data_path,
                participant_id,
                self.config.window_size,
                self.config.stride
            )

            # Adjust window indices in trial map
            adjusted_trial_map = {
                (k + window_idx_offset): v
                for k, v in trial_map.items()
            }

            all_samples.append(samples)
            all_labels.append(labels)
            all_trial_maps.update(adjusted_trial_map)

            window_idx_offset += len(samples)

        return (np.concatenate(all_samples),
                np.concatenate(all_labels),
                all_trial_maps)

    def get_participant_data(self, participant_id: int) -> pd.DataFrame:
        """Load and preprocess data for a specific participant."""
        file_path = Path(self.config.data_path) / f"{participant_id}_Formatted_Sample.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"No data file found for participant {participant_id}")

        df = pd.read_csv(file_path)
        self.df = self._preprocess_data(df)

        self.logger.info(f"\nParticipant {participant_id} data:")
        self.logger.info(f"Number of samples: {len(self.df)}")
        self.logger.info("Class distribution:")
        self.logger.info(self.df['target'].value_counts())

        return self.df

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the time series formatted data."""
        # Clean and handle missing values
        df = df.replace(['.', np.nan], self.config.invalid_value)

        # Drop unnecessary columns
        columns_to_drop = ['TARGET_XY', 'GAZE_XY', 'TARGET_ZONE']
        if self.config.remove_saccade:
            columns_to_drop.extend(['IN_SACCADE'])

        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

        # Convert GAZE_IA to numeric coordinates
        invalid_ia = '?-1'
        df['GAZE_IA'] = df['GAZE_IA'].replace(self.config.invalid_value, invalid_ia)
        df['GAZE_IA_X'] = df['GAZE_IA'].apply(
            lambda x: self.letter_to_num(str(x)[0]) if str(x)[0].isalpha() else self.config.invalid_value)
        df['GAZE_IA_Y'] = df['GAZE_IA'].apply(
            lambda x: int(str(x)[1:]) if str(x)[1:].isdigit() else self.config.invalid_value)
        df = df.drop('GAZE_IA', axis=1)

        # Convert boolean columns
        bool_columns = ['IN_BLINK', 'CURRENT_FIX_INDEX']
        if not self.config.remove_saccade:
            bool_columns.append('IN_SACCADE')
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype(float).astype(bool)

        # Convert numeric columns
        df['RECORDING_SESSION_LABEL'] = df['RECORDING_SESSION_LABEL'].astype(np.int16)
        df['TRIAL_INDEX'] = df['TRIAL_INDEX'].astype(np.int16)
        df['SAMPLE_INDEX'] = df['SAMPLE_INDEX'].astype(np.int32)
        df['SAMPLE_START_TIME'] = df['SAMPLE_START_TIME'].astype(np.int32)
        df['Pupil_Size'] = df['Pupil_Size'].astype(float).astype(np.int16)

        # Handle targets
        df = df.rename(columns={'Hit': 'target'})
        df['target'] = df['target'].astype(int)

        if self.config.per_slice_target:
            # Create slice-based targets using AILMENT_NUMBER
            # Mark a slice as target if it has any non-NaN AILMENT_NUMBER
            has_ailment = df.groupby(['RECORDING_SESSION_LABEL', 'TRIAL_INDEX',
                                      'CURRENT_IMAGE'])['AILMENT_NUMBER'].transform(
                lambda x: (x != self.config.invalid_value).any()
            )
            df['target'] = has_ailment.astype(int)

        # Remove AILMENT_NUMBER and CURRENT_IMAGE after using them for target calculation
        if 'AILMENT_NUMBER' in df.columns:
            df = df.drop('AILMENT_NUMBER', axis=1)
        if 'CURRENT_IMAGE' in df.columns:
            df = df.drop('CURRENT_IMAGE', axis=1)

        # Clean up coordinates outside grid
        df = df[
            (df['GAZE_IA_X'] >= 0) &
            (df['GAZE_IA_X'] <= 12) &
            (df['GAZE_IA_Y'] >= 0) &
            (df['GAZE_IA_Y'] <= 12)
            ].reset_index(drop=True)

        return df

    class TimeSeriesDataLoader(BaseDataLoader):
        def _process_trial_parallel(self,
                                    trial_data: Tuple[int, pd.DataFrame],
                                    participant: int,
                                    window_dir: Path,
                                    features: List[str],
                                    window_size: int,
                                    stride: int) -> Tuple[int, Dict]:
            """
            Process and save a single trial's windows immediately.

            Parameters:
            -----------
            trial_data : Tuple[int, pd.DataFrame]
                Tuple of (trial_number, trial_dataframe)
            participant : int
                Participant ID
            window_dir : Path
                Directory to save window data
            features : List[str]
                List of feature columns
            window_size : int
                Size of each window
            stride : int
                Stride between windows

            Returns:
            --------
            Tuple[int, Dict]
                Number of windows created and trial mapping
            """
            trial, group = trial_data
            group = group.sort_values('SAMPLE_START_TIME')

            windows_list = []
            labels_list = []
            trial_map = {}
            window_count = 0

            for i in range(0, len(group) - window_size + 1, stride):
                window = group.iloc[i:i + window_size]
                window_data = window[features].values
                windows_list.append(window_data)
                labels_list.append(int(window['target'].any()))

                # Create metadata for this window
                trial_map[window_count] = {
                    'participant': participant,
                    'trial': trial,
                    'window_number': window_count,
                    'start_time': window['SAMPLE_START_TIME'].iloc[0],
                    'end_time': window['SAMPLE_START_TIME'].iloc[-1],
                    'start_idx': i,
                    'end_idx': i + window_size,
                    'has_target': bool(window['target'].any())
                }
                window_count += 1

                # Save windows in batches to manage memory
                if len(windows_list) >= 1000:  # Adjust batch size as needed
                    self._save_window_batch(windows_list, labels_list, participant, trial,
                                            window_count - len(windows_list), window_dir)
                    windows_list = []
                    labels_list = []

            # Save any remaining windows
            if windows_list:
                self._save_window_batch(windows_list, labels_list, participant, trial,
                                        window_count - len(windows_list), window_dir)

            # Save trial mapping
            trial_map_file = window_dir / f'participant_{participant}_trial_{trial}_map.npy'
            np.save(str(trial_map_file), trial_map)

            return window_count, trial_map

        def _save_window_batch(self, windows: List[np.ndarray], labels: List[int],
                               participant: int, trial: int, start_idx: int, window_dir: Path):
            """Save a batch of windows to disk."""
            if not windows:
                return

            samples = np.array(windows)
            labels_arr = np.array(labels)

            # Save with batch index
            np.save(str(window_dir / f'participant_{participant}_trial_{trial}_start_{start_idx}_samples.npy'),
                    samples)
            np.save(str(window_dir / f'participant_{participant}_trial_{trial}_start_{start_idx}_labels.npy'),
                    labels_arr)

        def create_rolling_windows(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict]:
            """Create rolling windows using parallel processing, saving each trial separately."""
            self.logger.info(f"Creating rolling windows with size {self.config.window_size} "
                             f"and stride {self.config.stride}")

            window_dir = Path(self.config.data_path).parent / 'window_data'
            window_dir.mkdir(exist_ok=True)

            features = [col for col in df.columns if col not in ['RECORDING_SESSION_LABEL',
                                                                 'TRIAL_INDEX',
                                                                 'target']]

            global_trial_map = {}
            total_windows = 0

            participants = df['RECORDING_SESSION_LABEL'].unique()

            for participant in tqdm(participants, desc="Processing participants"):
                participant_data = df[df['RECORDING_SESSION_LABEL'] == participant]
                trials = list(participant_data.groupby('TRIAL_INDEX'))

                # Process trials in parallel
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for trial_data in trials:
                        future = executor.submit(
                            self._process_trial_parallel,
                            trial_data=trial_data,
                            participant=participant,
                            window_dir=window_dir,
                            features=features,
                            window_size=self.config.window_size,
                            stride=self.config.stride
                        )
                        futures.append(future)

                    # Process results with progress bar
                    for future in tqdm(futures, desc=f"Processing trials for participant {participant}", leave=False):
                        n_windows, trial_map = future.result()
                        # Adjust window indices for global mapping
                        adjusted_map = {
                            (k + total_windows): v
                            for k, v in trial_map.items()
                        }
                        global_trial_map.update(adjusted_map)
                        total_windows += n_windows

            # Save feature list
            feature_file = window_dir / f'features_w{self.config.window_size}.txt'
            with open(str(feature_file), 'w') as f:
                f.write('\n'.join(features))

            # Save global trial mapping
            np.save(
                str(window_dir / f'all_participants_w{self.config.window_size}_s{self.config.stride}_trial_map.npy'),
                global_trial_map)

            self.logger.info(f"\nTotal windows created: {total_windows}")
            self.logger.info(f"Data saved in: {window_dir}")

            # Load and concatenate all files for return value
            return self._concatenate_all_files(window_dir, total_windows, global_trial_map)

        def _concatenate_all_files(self, window_dir: Path, total_windows: int,
                                   global_trial_map: Dict) -> Tuple[np.ndarray, np.ndarray, Dict]:
            """Concatenate all saved window files into final arrays."""
            self.logger.info("Concatenating all window files...")

            all_samples = []
            all_labels = []

            # Get all sample files sorted by participant and trial
            sample_files = sorted(window_dir.glob('*_samples.npy'))
            label_files = sorted(window_dir.glob('*_labels.npy'))

            for sample_file, label_file in tqdm(zip(sample_files, label_files),
                                                desc="Concatenating files",
                                                total=len(sample_files)):
                samples = np.load(str(sample_file))
                labels = np.load(str(label_file))
                all_samples.append(samples)
                all_labels.append(labels)

            return (np.concatenate(all_samples),
                    np.concatenate(all_labels),
                    global_trial_map)

    def _log_dataset_info(self, samples, labels, features):
        """Log information about the dataset."""
        self.logger.info(f"\nOverall dataset shape:")
        self.logger.info(f"Samples shape: {samples.shape}")
        self.logger.info(f"Labels shape: {labels.shape}")
        self.logger.info(f"\nFeatures used: {features}")
        self.logger.info(f"\nClass distribution:")
        unique_labels, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique_labels, counts):
            self.logger.info(f"Class {label}: {count} samples ({count / len(labels):.4f})")

    @staticmethod
    def load_participant_windows(data_path: str, participant_id: int, window_size: int, stride: int):
        """Load pre-saved windowed data for a specific participant, including trial information."""
        window_dir = Path(data_path).parent / 'window_data'

        samples_file = window_dir / f'participant_{participant_id}_w{window_size}_s{stride}_samples.npy'
        labels_file = window_dir / f'participant_{participant_id}_w{window_size}_s{stride}_labels.npy'
        trial_map_file = window_dir / f'participant_{participant_id}_w{window_size}_s{stride}_trial_map.npy'

        if not all(f.exists() for f in [samples_file, labels_file, trial_map_file]):
            raise FileNotFoundError(f"Windowed data not found for participant {participant_id}")

        samples = np.load(str(samples_file))
        labels = np.load(str(labels_file))
        trial_map = np.load(str(trial_map_file), allow_pickle=True).item()

        return samples, labels, trial_map



class LegacyDataLoader(BaseDataLoader):
    """Handles loading and preprocessing of legacy eye tracking data format."""

    def load_data(self) -> pd.DataFrame:
        """Load and preprocess legacy format data."""
        try:
            self.df = pd.read_csv(self.config.data_path)
            # Add your legacy preprocessing steps here
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
        window_size= 500
    )

    # Using the new time series format
    loader_new = create_data_loader('time_series', config_new)
    df_new = loader_new.load_data()
    window_df =loader_new.create_rolling_windows(df_new[:1000])
    print(window_df)