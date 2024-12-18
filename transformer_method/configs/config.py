from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from pathlib import Path

import numpy as np
import yaml

@dataclass
class SplitConfig:
    """Configuration for data splitting"""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    time_column: str = 'SAMPLE_START_TIME'
    trial_column: str = 'TRIAL_INDEX'
    random_seed: Optional[int] = None

    def __post_init__(self):
        assert np.isclose(self.train_ratio + self.val_ratio + self.test_ratio, 1.0), \
            "Split ratios must sum to 1"

@dataclass
class DataConfig:
    """Configuration for data processing and features"""
    # Data loading
    data_path: str = 'data/Formatted_Samples_ML'
    file_pattern: str = '*_Formatted_Sample.csv'
    participant_ids: Optional[List[str]] = None
    # Data processing
    invalid_value: Union[str, float] = -1
    per_slice_target: bool = False
    coordinate_range: tuple = (0, 12)

    # Feature configuration
    feature_columns: List[str] = field(default_factory=lambda: [
        'Pupil_Size',
        'CURRENT_FIX_INDEX',
        'relative_x',
        'relative_y',
        'gaze_velocity'
    ])

    # Column configurations
    numeric_conversions: Dict[str, type] = field(default_factory=lambda: {
        'RECORDING_SESSION_LABEL': np.int8,
        'TRIAL_INDEX': np.int8,
        'SAMPLE_START_TIME': np.int32,
        'SAMPLE_INDEX': np.int32,
        'Pupil_Size': np.float32,
        'target': np.int8
    })

    bool_columns: List[str] = field(default_factory=lambda: [
        'IN_BLINK',
        'IN_SACCADE',
        'CURRENT_FIX_INDEX'
    ])

    columns_to_drop: List[str] = field(default_factory=lambda: [
        'AILMENT_NUMBER',
        'TARGET_ZONE',
        'TARGET_XY',
        'GAZE_XY',
        'CURRENT_IMAGE',
        'IN_SACCADE',
        'Hit'
    ])

    # Target configuration
    target_column: str = 'target'
    target_mapping: Dict[str, int] = field(default_factory=lambda: {
        'Hit': 'target'
    })


@dataclass
class Config:
    # Data parameters
    data: DataConfig = field(default_factory=DataConfig)
    splitter:SplitConfig = field(default_factory=SplitConfig)
    cache_dir = 'cache_data'
    patch_len: int = 32
    stride: int = 8
    seq_len: int = 128
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Model parameters
    n_layers: int = 3
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 256
    norm_type: str = 'BatchNorm'
    activation: str = 'gelu'
    attn_dropout: float = 0.1
    dropout: float = 0.1
    pos_encoding: str = 'learnable'

    # Training parameters
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    epochs: int = 50
    grad_clip: float = 1.0
    seed: int = 42
    head_dropout: float = 0.1
    num_patch: Optional[int] = None
    # Feature columns
    feature_columns: list = None
    patience: int = 10
    model_save_dir: Optional[str] = None
    output_dir: str = 'outputs'
    log_dir: Optional[str] = None
    experiment_name: Optional[str] = None


    def __post_init__(self):
        """Post initialization processing"""
        # Set default feature columns if none provided
        if self.feature_columns is None:
            self.feature_columns = [
                'Pupil_Size',
                 'CURRENT_FIX_INDEX',
                'relative_x',
                 'relative_y',
                 'gaze_velocity'
            ]

        # Calculate num_patch if not provided
        if self.num_patch is None:
            self.num_patch = (self.seq_len - self.patch_len) // self.stride + 1
        if self.model_save_dir is None:
            self.model_save_dir = str(Path(self.output_dir) / 'models')
        if self.log_dir is None:
            self.log_dir = str(Path(self.output_dir) / 'logs')
        # Validate configurations
        self._validate_config()

    def _validate_config(self):
        """Validate configuration values"""
        assert self.patch_len > 0, "patch_len must be positive"
        assert self.stride > 0, "stride must be positive"
        assert self.seq_len >= self.patch_len, "seq_len must be >= patch_len"
        assert 0 <= self.dropout <= 1, "dropout must be between 0 and 1"
        assert 0 <= self.head_dropout <= 1, "head_dropout must be between 0 and 1"
        assert 0 <= self.attn_dropout <= 1, "attn_dropout must be between 0 and 1"
        assert len(self.feature_columns) > 0, "must have at least one feature column"

    def get_output_dirs(self) -> dict:
        """Get all output directories"""
        return {
            'base': self.output_dir,
            'models': self.model_save_dir,
            'logs': self.log_dir
        }

    def create_output_dirs(self):
        """Create all necessary output directories"""
        for dir_path in self.get_output_dirs().values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def save(self, path: Union[str, Path]):
        """Save configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dictionary
        config_dict = {
            'data': vars(self.data),
            **{k: v for k, v in vars(self).items() if k != 'data'}
        }

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'Config':
        """Load configuration from YAML file"""
        with open(path) as f:
            config_dict = yaml.safe_load(f)

        # Extract data config
        data_config = config_dict.pop('data', {})
        data_config = DataConfig(**data_config)

        # Create main config
        return cls(data=data_config, **config_dict)