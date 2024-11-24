from dataclasses import dataclass


@dataclass
class Config:
    # Data parameters
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

    # Feature columns
    feature_columns: list = None

    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                'Pupil_Size',
                'GAZE_IA_X',
                'GAZE_IA_Y',
                'CURRENT_FIX_INDEX',
                'SAMPLE_START_TIME',
                'IN_BLINK',
            ]