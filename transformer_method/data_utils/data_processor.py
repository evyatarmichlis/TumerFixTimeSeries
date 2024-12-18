import pandas as pd
import numpy as np
from typing import Optional
import logging

from transformer_method.configs.config import DataConfig


class DataProcessor:
    """
    Data processor for eye tracking data.
    Handles preprocessing, type optimization, and feature engineering.

    The processor uses configuration from DataConfig to determine:
    - Invalid value handling
    - Target processing
    - Coordinate processing
    - Column type conversions
    - Column filtering
    """

    def __init__(self, config: DataConfig):
        """
        Initialize the DataProcessor with configuration.

        Args:
            config: DataConfig object containing processing parameters
        """
        self.config = config
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            logger.addHandler(logging.StreamHandler())
            logger.setLevel(logging.INFO)
        return logger

    @staticmethod
    def letter_to_num(letter: str) -> int:
        """Convert letter to number (A->0, B->1, etc.)"""
        return ord(letter.upper()) - ord('A') if letter.isalpha() else -1

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize data types and preprocessing for memory efficiency.

        Args:
            df: Input DataFrame

        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Starting data preprocessing...")
        self.logger.info(f"Initial columns: {df.columns.tolist()}")
        self.logger.info(f"Initial memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")

        # Make a copy to avoid modifying the original
        df = df.copy()

        # Initial cleanup
        df = df.replace(['.', np.nan], self.config['invalid_value'])

        # Process in specific order
        df = self._process_targets(df)

        # Process GAZE_IA coordinates
        df = self._process_gaze_coordinates(df)

        # Add relative coordinates and velocity per trial
        df = self._add_gaze_features(df)

        # Normalize pupil size
        df = self._normalize_pupil_size(df)

        # Convert numeric columns
        df = self._convert_numeric_columns(df)

        # Convert boolean columns
        df = self._convert_boolean_columns(df)

        # Drop unnecessary columns
        df = self._drop_columns(df)

        # Filter valid coordinates
        df = self._filter_valid_coordinates(df)

        self.logger.info(f"Final columns: {df.columns.tolist()}")
        self.logger.info(f"Processed data shape: {df.shape}")

        return df

    def _process_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process target variables based on configuration"""
        if self.config['per_slice_target']:
            if 'AILMENT_NUMBER' in df.columns:
                df['has_ailment'] = df.groupby('CURRENT_IMAGE')['AILMENT_NUMBER'].transform(
                    lambda x: (x != self.config['invalid_value']).any()
                )
                df['target'] = df['has_ailment'].astype(np.int8)
                df = df.drop('has_ailment', axis=1)
        else:
            for source, target in self.config['target_mapping'].items():
                if source in df.columns:
                    df = df.rename(columns={source: target})

        return df

    def _process_gaze_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process GAZE_IA coordinates"""
        if 'GAZE_IA' in df.columns:
            invalid_ia = '?-1'
            df['GAZE_IA'] = df['GAZE_IA'].replace(self.config['invalid_value'], invalid_ia)

            # Extract X coordinate (letter)
            df['GAZE_IA_X'] = df['GAZE_IA'].apply(
                lambda x: self.letter_to_num(str(x)[0]) if str(x)[0].isalpha()
                else self.config['invalid_value']
            ).astype(np.int8)

            # Extract Y coordinate (number)
            df['GAZE_IA_Y'] = df['GAZE_IA'].apply(
                lambda x: int(str(x)[1:]) if str(x)[1:].isdigit()
                else self.config['invalid_value']
            ).astype(np.int8)

            df = df.drop('GAZE_IA', axis=1)

        return df


    def _convert_boolean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert boolean columns to int8 based on configuration"""
        for col in self.config['bool_columns']:
            if col in df.columns:
                try:
                    df[col] = df[col].astype(np.int8)
                except Exception as e:
                    self.logger.warning(f"Failed to convert {col} to boolean: {str(e)}")
        return df

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unnecessary columns specified in configuration"""
        existing_columns = [col for col in self.config['columns_to_drop'] if col in df.columns]
        if existing_columns:
            self.logger.info(f"Dropping columns: {existing_columns}")
            df = df.drop(existing_columns, axis=1)
        return df

    def _filter_valid_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter rows with valid coordinates based on configuration range"""
        min_coord, max_coord = self.config['coordinate_range']
        valid_coords = (
                (df['GAZE_IA_X'] >= min_coord) &
                (df['GAZE_IA_X'] <= max_coord) &
                (df['GAZE_IA_Y'] >= min_coord) &
                (df['GAZE_IA_Y'] <= max_coord)
        )

        initial_rows = len(df)
        df = df[valid_coords].reset_index(drop=True)
        filtered_rows = initial_rows - len(df)

        self.logger.info(f"Filtered {filtered_rows} rows with invalid coordinates")
        return df

    def get_feature_info(self, df: pd.DataFrame) -> dict:
        """Get information about features"""
        info = {
            'n_features': len(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024 ** 2,  # MB
            'feature_columns': [col for col in self.config['feature_columns'] if col in df.columns],
            'invalid_values': {
                col: (df[col] == self.config['invalid_value']).sum()
                for col in df.columns
            }
        }
        return info

    def _add_gaze_features(self, df: pd.DataFrame) -> pd.DataFrame: #just for approach 6
        """Add relative gaze coordinates and velocity features by recording session and trial"""
        # Calculate relative X and Y positions using both grouping levels
        grouping_cols = ['RECORDING_SESSION_LABEL', 'TRIAL_INDEX']

        # Relative positions
        df['relative_x'] = df.groupby(grouping_cols)['GAZE_IA_X'].transform(lambda x: x - x.mean())
        df['relative_y'] = df.groupby(grouping_cols)['GAZE_IA_Y'].transform(lambda x: x - x.mean())

        # Calculate gaze velocity within each trial
        df['gaze_velocity'] = df.groupby(grouping_cols).apply(
            lambda group: np.sqrt(
                group['GAZE_IA_X'].diff() ** 2 +
                group['GAZE_IA_Y'].diff() ** 2
            )
        ).reset_index(level=[0, 1], drop=True)

        # Fill NaN values in velocity (first row of each trial)
        df['gaze_velocity'] = df['gaze_velocity'].fillna(0)

        # Convert to float32 for memory efficiency
        df['relative_x'] = df['relative_x'].astype(np.float32)
        df['relative_y'] = df['relative_y'].astype(np.float32)
        df['gaze_velocity'] = df['gaze_velocity'].astype(np.float32)

        return df

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

    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric columns to appropriate dtypes"""
        numeric_conversions = {
            'RECORDING_SESSION_LABEL': np.int8,
            'TRIAL_INDEX': np.int8,
            'Pupil_Size': np.float32,
            'GAZE_IA_X': np.float32,
            'GAZE_IA_Y': np.float32,
            'relative_x': np.float32,
            'relative_y': np.float32,
            'gaze_velocity': np.float32,
            'target': np.int8,
            'CURRENT_FIX_INDEX': np.int32
        }

        for col, dtype in numeric_conversions.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        return df