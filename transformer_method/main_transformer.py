import argparse
from typing import Union, Tuple, cast, Optional, List, Dict
from datetime import datetime

import yaml
import logging
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from datetime import datetime

from matplotlib import pyplot as plt
from scipy.stats import stats
from sympy.codegen.ast import break_
from tensorflow.python.data import Dataset
from torch.utils.data import DataLoader

# Local imports
from transformer_method.configs.config import Config
from models.classifier import PatchTSTClassifier, TripletPatchTSTClassifier
from data_utils.splitter import TrialSplitter
from data_utils.datasets import TimeSeriesDataset
from transformer_method.data_utils.data_processor import DataProcessor
from transformer_method.models.llm_transformer import LLMPatchtTransformer
from utils.training import setup_logging, seed_everything
import torch.multiprocessing as mp
import torch


class Args:
    """Simple class to mimic parsed arguments"""
    def __init__(self):
        self.data_path = 'data/Formatted_Samples_ML'
        self.config = 'configs/default.yaml'
        self.participant_ids = ['1']
        self.output_dir = 'outputs'
        self.experiment_name = None
        self.seed = 42
        self.pretrained_path = 'PatchTST/PatchTST_self_supervised/saved_models/etth1/masked_patchtst/based_model/patchtst_pretrained_cw512_patch12_stride12_epochs-pretrain100_mask0.4_model1.pth'


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)




def setup_experiment_dir(config: Config) -> Path:
    """
    Setup experiment directory with timestamp.

    Args:
        config: Configuration object

    Returns:
        Path: Path to experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    if config.experiment_name:
        exp_name = f"{config.experiment_name}_{timestamp}"
    else:
        exp_name = timestamp

    output_dir = Path(config.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Also create subdirectories
    (output_dir / 'models').mkdir(exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)

    return output_dir


def load_participant_data(
        participant_id: str,
        data_path: str,
        config: Config,
        logger: logging.Logger
) -> Optional[pd.DataFrame]:
    """
    Load data for a specific participant.

    Args:
        participant_id: ID of the participant
        data_path: Base path to data directory
        config: Configuration object
        logger: Logger instance

    Returns:
        DataFrame with participant's data or None if file not found
    """
    try:
        # Construct file path for participant
        base_path = Path(__file__).parent.parent / data_path
        filename = f"{participant_id}_Formatted_Sample.csv"
        file_path = base_path / filename
        if not file_path.exists():
            logger.warning(f"Data file not found for participant {participant_id}: {file_path}")
            return None

        df = pd.read_csv(file_path)
        logger.info(f"Loaded data for participant {participant_id}, shape: {df.shape}")

        return df

    except Exception as e:
        logger.error(f"Error loading data for participant {participant_id}: {str(e)}")
        return None


def oversample_fixed_sequences(df, seq_len, logger):
    """
    Oversample sequences containing minority class while maintaining temporal order
    and patch positions.
    """
    n_sequences = len(df) // seq_len
    df = df.iloc[:n_sequences * seq_len]

    sequences = [df.iloc[i:i + seq_len] for i in range(0, len(df), seq_len)]

    # Identify sequences containing minority class (1)
    minority_sequences = []
    majority_sequences = []

    for seq in sequences:
        if 1 in seq['target'].values:
            minority_sequences.append(seq)
        else:
            majority_sequences.append(seq)

    logger.info(f"Found {len(minority_sequences)} minority sequences and {len(majority_sequences)} majority sequences")

    target_ratio = 0.4
    n_copies = int((target_ratio * len(sequences) - len(minority_sequences)) / len(minority_sequences))

    logger.info(f"Making {n_copies} copies of each minority sequence")

    new_sequences = sequences.copy()
    minority_copies = minority_sequences * n_copies

    for seq in minority_copies:
        insert_pos = np.random.randint(0, len(new_sequences) - 1)
        new_sequences.insert(insert_pos, seq)

    return pd.concat(new_sequences, ignore_index=True)


def analyze_class_1_intervals(df: pd.DataFrame, logger: logging.Logger):
    """
    Analyze intervals between class 1 events in the data.
    """
    logger.info("\nAnalyzing intervals between class 1 events...")

    # Find indices where target = 1
    class_1_indices = df.index[df['target'] == 1].tolist()

    if not class_1_indices:
        logger.info("No class 1 events found in the data")
        return

    # Calculate intervals
    intervals = []
    for i in range(1, len(class_1_indices)):
        interval = class_1_indices[i] - class_1_indices[i - 1]
        intervals.append(interval)

    if not intervals:
        logger.info("Not enough class 1 events to calculate intervals")
        return

    # Calculate statistics
    stats = {
        'min_interval': np.min(intervals),
        'max_interval': np.max(intervals),
        'mean_interval': np.mean(intervals),
        'median_interval': np.median(intervals),
        'std_interval': np.std(intervals),
        'quartiles': np.percentile(intervals, [25, 50, 75])
    }

    # Log results
    logger.info(f"Class 1 event statistics:")
    logger.info(f"Total class 1 events: {len(class_1_indices)}")
    logger.info(f"Minimum interval: {stats['min_interval']}")
    logger.info(f"Maximum interval: {stats['max_interval']}")
    logger.info(f"Mean interval: {stats['mean_interval']:.2f}")
    logger.info(f"Median interval: {stats['median_interval']}")
    logger.info(f"Standard deviation: {stats['std_interval']:.2f}")
    logger.info(f"25th percentile: {stats['quartiles'][0]}")
    logger.info(f"75th percentile: {stats['quartiles'][2]}")

    # Optional: Create histogram of intervals
    plt.figure(figsize=(10, 6))
    plt.hist(intervals, bins=50)
    plt.title('Distribution of Intervals Between Class 1 Events')
    plt.xlabel('Interval Length')
    plt.ylabel('Frequency')
    plt.show()

    return stats, intervals


def compare_feature_distributions(train_df, val_df, logger):
    import  seaborn as sns
    """
    Compare feature distributions between train and validation sets,
    particularly for class 1 examples.
    """
    logger.info("\nAnalyzing feature distributions between train and validation sets")

    # Get numerical features
    features = [col for col in train_df.columns
                if col not in ['target', 'cluster_start', 'cluster_id']
                and np.issubdtype(train_df[col].dtype, np.number)]

    # Setup plotting
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(15, 5 * n_rows))

    for idx, feature in enumerate(features, 1):
        # Get class 1 data for both sets
        train_values = train_df[train_df['target'] == 1][feature]
        val_values = val_df[val_df['target'] == 1][feature]

        # Calculate statistics
        train_stats = {
            'mean': train_values.mean(),
            'median': train_values.median(),
            'std': train_values.std(),
            'q1': train_values.quantile(0.25),
            'q3': train_values.quantile(0.75),
            'min': train_values.min(),
            'max': train_values.max()
        }

        val_stats = {
            'mean': val_values.mean(),
            'median': val_values.median(),
            'std': val_values.std(),
            'q1': val_values.quantile(0.25),
            'q3': val_values.quantile(0.75),
            'min': val_values.min(),
            'max': val_values.max()
        }

        # Log statistics
        logger.info(f"\nFeature: {feature}")
        logger.info(f"{'Statistic':<10} {'Train':<15} {'Val':<15} {'Diff %':<15}")
        logger.info("-" * 55)
        for stat in ['mean', 'median', 'std', 'q1', 'q3', 'min', 'max']:
            train_val = train_stats[stat]
            val_val = val_stats[stat]
            diff_pct = abs((train_val - val_val) / train_val * 100) if train_val != 0 else 0
            logger.info(f"{stat:<10} {train_val:15.4f} {val_val:15.4f} {diff_pct:15.2f}")

        # Create subplot
        plt.subplot(n_rows, n_cols, idx)

        # Plot distributions
        # KDE plot
        sns.kdeplot(data=train_values, label='Train', alpha=0.6)
        sns.kdeplot(data=val_values, label='Val', alpha=0.6)

        # Box plot on top
        plt.boxplot([train_values, val_values],
                    positions=[0.2, 0.4],
                    widths=0.1,
                    patch_artist=True,
                    labels=['Train', 'Val'])

        plt.title(f'{feature} Distribution (Class 1)')
        plt.legend()

        # Perform Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(train_values, val_values)
        logger.info(f"KS test - statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")

        # Calculate distribution overlap
        hist_train, bins = np.histogram(train_values, bins=50, density=True)
        hist_val, _ = np.histogram(val_values, bins=bins, density=True)
        overlap = np.minimum(hist_train, hist_val).sum() * (bins[1] - bins[0])
        logger.info(f"Distribution overlap: {overlap:.4f}")

        # Check for significant value ranges unique to either set
        train_range = (train_values.min(), train_values.max())
        val_range = (val_values.min(), val_values.max())
        if train_range[0] < val_range[0] or train_range[1] > val_range[1]:
            logger.warning(f"Warning: Training set contains values outside validation range")
            logger.warning(f"Train range: {train_range}")
            logger.warning(f"Val range: {val_range}")

    plt.tight_layout()
    plt.show()

    # Additional analysis: Feature correlations
    logger.info("\nFeature Correlations Analysis:")
    train_corr = train_df[train_df['target'] == 1][features].corr()
    val_corr = val_df[val_df['target'] == 1][features].corr()

    # Plot correlation differences
    plt.figure(figsize=(10, 8))
    corr_diff = train_corr - val_corr
    sns.heatmap(corr_diff, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Differences (Train - Val) for Class 1')
    plt.show()

    return train_stats, val_stats

def load_data(
        data_path: str,
        config: Config,
        logger: logging.Logger,
        participant_ids: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load, process, and split data for specified participants.

    Args:
        data_path: Path to the data directory
        config: Configuration object containing DataConfig and split parameters
        logger: Logger instance for tracking progress
        participant_ids: Optional list of specific participant IDs to load

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    try:
        # Use participant IDs from config if not explicitly provided
        if participant_ids is None:
            participant_ids = config.data.participant_ids

        # If still None, try to find all participant files
        if participant_ids is None:
            data_dir = Path(data_path)
            pattern = config.data.file_pattern.replace('{}', '*')
            files = list(data_dir.glob(pattern))
            participant_ids = [f.stem.split('_')[0] for f in files]
            logger.info(f"Found {len(participant_ids)} participant files")

        # Load data for each participant
        all_dfs = []
        for participant_id in participant_ids:
            df = load_participant_data(participant_id, data_path, config, logger)
            if df is not None:
                all_dfs.append(df)

        if not all_dfs:
            raise ValueError("No data was loaded for any participant")

        # Combine all participant data
        combined_df = pd.concat(all_dfs, ignore_index=True)
        logger.info(f"\nCombined data shape: {combined_df.shape}")
        logger.info(f"Participants included: {', '.join(str(combined_df['RECORDING_SESSION_LABEL'].unique()))}")

        processor = DataProcessor(config=config.data)
        processed_df = processor.preprocess_data(combined_df)
        logger.info(f"Processed data shape: {processed_df.shape}")
        # analyze_class_1_intervals(processed_df, logger)
        # Get and log feature information
        feature_info = processor.get_feature_info(processed_df)
        logger.info("\nFeature Information:")
        logger.info(f"Number of features: {feature_info['n_features']}")
        logger.info(f"Memory usage: {feature_info['memory_usage']:.2f} MB")

        # Create splitter and split data
        splitter = TrialSplitter(config=config.splitter)
        train_df, val_df, test_df = splitter.split_by_trials(processed_df)
        # Log split sizes
        logger.info("\nSplit sizes:")
        logger.info(f"Train: {len(train_df)} ({len(train_df) / len(processed_df):.1%})")
        logger.info(f"Val: {len(val_df)} ({len(val_df) / len(processed_df):.1%})")
        logger.info(f"Test: {len(test_df)} ({len(test_df) / len(processed_df):.1%})")

        logger.info("Original train class distribution:")
        logger.info(train_df['target'].value_counts())

        # # Oversample training sequences
        # train_df = oversample_fixed_sequences(
        #     df=train_df,
        #     seq_len=config.seq_len,
        #     logger=logger
        # )

        logger.info("After oversampling train class distribution:")
        logger.info(train_df['target'].value_counts())

        return train_df, val_df, test_df

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def collate_fn(batch):
    """Custom collate function to ensure correct shapes"""
    # Separate inputs and targets
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]

    # Stack correctly
    inputs = torch.stack(inputs)  # [batch_size, features, patch_len]
    targets = torch.stack(targets)  # [batch_size]


    return inputs, targets
def create_dataloaders(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: Config,
        logger: logging.Logger
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoader objects with caching"""
    logger.info("\nCreating/loading datasets and dataloaders...")
    # Create datasets with caching
    train_dataset = TimeSeriesDataset(
        data=train_df[config.feature_columns].values,
        targets=train_df['target'].values,
        trial_ids=train_df['TRIAL_INDEX'].values,
        patch_len=config.patch_len,
        seq_len=config.seq_len,
        stride=config.stride,
        cache_dir=config.cache_dir,
        split='train',
        verbose=True,
        logger = logger,
        seq_stride=64
    )

    val_dataset = TimeSeriesDataset(
        data=val_df[config.feature_columns].values,
        targets=val_df['target'].values,
        trial_ids=val_df['TRIAL_INDEX'].values,
        patch_len=config.patch_len,
        seq_len=config.seq_len,
        stride=config.stride,
        scaler=train_dataset.scaler,
        cache_dir=config.cache_dir,
        split='val',
        verbose=True,
        logger=logger,
        seq_stride = 64

    )

    test_dataset = TimeSeriesDataset(
        data=test_df[config.feature_columns].values,
        targets=test_df['target'].values,
        trial_ids=test_df['TRIAL_INDEX'].values,
        patch_len=config.patch_len,
        seq_len=config.seq_len,
        stride=config.stride,
        scaler=train_dataset.scaler,
        cache_dir=config.cache_dir,
        split='test',
        verbose=True,
        logger=logger,
        seq_stride = 64

    )
    train_targets = train_dataset.sequence_targets
    class_counts = np.bincount(train_targets)
    total_samples = len(train_targets)
    class_weights = total_samples / (len(class_counts) * class_counts)
    sample_weights = torch.FloatTensor([class_weights[t] for t in train_targets])
    # Create weighted sampler for training
    train_sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_dataset_info(self) -> Dict:
    """Get dataset information"""
    info = {
        'n_samples': len(self.data),
        'n_features': self.data.shape[1],
        'n_trials': len(np.unique(self.trial_ids)),
        'n_patches': len(self.valid_indices),  # Changed from self.patches
        'patch_len': self.patch_len,
        'stride': self.stride,
    }

    if self._targets is not None:
        # Calculate target distribution on-the-fly
        all_targets = [self.__getitem__(i)[1].item() for i in range(len(self))]
        unique_targets = torch.unique(torch.tensor(all_targets))
        target_counts = torch.bincount(torch.tensor(all_targets))
        info['target_distribution'] = {
            int(label): int(count.item())
            for label, count in zip(unique_targets, target_counts)
        }

    return info


def calculate_class_weights(dataset: TimeSeriesDataset, device: torch.device,logger:logging.Logger) -> torch.Tensor:
    """
    Calculate class weights for imbalanced data using raw targets.

    Args:
        dataset: TimeSeriesDataset instance
        device: torch device

    Returns:
        torch.Tensor: Class weights tensor
    """

    # Use the raw targets (before patch creation)
    targets = dataset.sequence_targets  # This is the original numpy array of targets
    class_counts = np.bincount(targets)
    total = len(targets)
    weights = total / (len(class_counts) * class_counts)

    logger.info(f"Target distribution in original data:")
    for class_idx, count in enumerate(class_counts):
        logger.info(f"Class {class_idx}: {count} samples ({count / total:.2%})")

    return torch.FloatTensor(weights)


def create_model(config, device='cpu',model_type = "patch"):
    """Create model on CPU first, then transfer to GPU in chunks"""
    # Reduce model size based on available GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory
        if gpu_mem < 8e9:  # Less than 8GB
            config.d_model = 64
            config.d_ff = 128
            config.n_heads = 4

    if model_type == "llm":
        model = LLMPatchtTransformer(
            c_in=len(config.feature_columns),
            patch_len=config.patch_len,
            stride=config.stride,
            num_patch=config.num_patch,
            n_classes=2,
            pos_encoding='learned'  # or 'sinusoidal' or 'relative'
        )
    else: #model name == patch
        model = PatchTSTClassifier(
            c_in=len(config.feature_columns),
            patch_len=config.patch_len,
            stride=config.stride,
            num_patch=config.num_patch,
            n_classes=2,  # Binary classification
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            head_dropout=config.head_dropout,
            attn_dropout=config.attn_dropout,
            norm=config.norm_type,
            act=config.activation,
            pe='zeros',  # Changed from config.pos_encoding to 'zeros'
            learn_pe=True,  # This will make the positional encoding learnable
            verbose=True
        )

    # Transfer to GPU in chunks if needed
    if device != 'cpu':
        model.backbone = model.backbone.to(device)
        torch.cuda.empty_cache()
        model.head = model.head.to(device)
        torch.cuda.empty_cache()

    return model
def main():
    torch.cuda.empty_cache()

    args = Args()
    config = load_config(args.config)
    output_dir = setup_experiment_dir(config=config)
    logger = setup_logging(output_dir / 'train.log')
    logger.info("Starting experiment...")
    output_dir = setup_experiment_dir(config)
    seed_everything(args.seed)
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Experiment directory created at: {output_dir}")
    config.create_output_dirs()
    setup_logging(output_dir / 'logs' / 'train.log')
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(vars(config), f, default_flow_style=False)
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
    logger.info(f"Using device: {device}")

    try:
        train_df, val_df, test_df = load_data(args.data_path, config, logger,args.participant_ids)

        train_loader, val_loader, test_loader = create_dataloaders(
            train_df, val_df, test_df, config, logger
        )
        train_dataset = cast(TimeSeriesDataset, train_loader.dataset)
        class_weights = calculate_class_weights(train_dataset, device,logger)

        logger.info(f"\nClass weights: {class_weights}")
        device = torch.device('cuda' if torch.cuda.is_available()  else 'cpu')
        model = create_model(config, device="cuda",model_type="patch")
        # Check memory usage
        print(torch.cuda.memory_summary(device="cuda"))
        if args.pretrained_path:
            pre_train_path = Path(__file__).parent.parent/args.pretrained_path
            logger.info(f"\nLoading pretrained weights from {pre_train_path}")

            try:
                pre_train_path = Path(__file__).parent.parent / args.pretrained_path

                pretrained_state = torch.load(pre_train_path, map_location=device)
                if 'state_dict' in pretrained_state:
                    pretrained_state = pretrained_state['state_dict']

                model.load_pretrained(pretrained_state, logger)
                logger.info("Pretrained weights loaded successfully")
            except Exception as e:
                logger.warning(f"Error loading pretrained weights: {str(e)}")
                logger.warning("Training will continue with random initialization")
        # Train model
        history = model.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            epochs=config.epochs,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            patience=config.patience,
            output_dir=output_dir,
            device=device,
            class_weights=class_weights
        )

        logger.info("\nTraining completed successfully!")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

    finally:
        # Save final outputs
        logger.info(f"\nExperiment outputs saved to {output_dir}")


if __name__ == "__main__":
    # mp.set_start_method('spawn', force=True)
    main()