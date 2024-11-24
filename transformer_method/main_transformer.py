import argparse
from typing import Union, Tuple

import yaml
import logging
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from datetime import datetime

from tensorflow.python.data import Dataset
from torch.utils.data import DataLoader

# Local imports
from transformer_method.configs.config import Config
from models.classifier import PatchTSTClassifier
from data_utils.splitter import TrialSplitter
from data_utils.datasets import TimeSeriesDataset
from utils.training import setup_logging, seed_everything


def parse_args():
    parser = argparse.ArgumentParser(description='Train PatchTST Classifier for Eye Tracking Data')

    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')

    # Training arguments
    parser.add_argument('--pretrained_path', type=str, default=None,
                        help='Path to pretrained weights')
    parser.add_argument('--output_dir', type=str, default='outputs',
                        help='Output directory')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for logging')

    # Runtime arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Disable CUDA')

    return parser.parse_args()


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file"""
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    return Config(**config_dict)


def setup_experiment_dir(base_dir: str, experiment_name: str = None) -> Path:
    """Create experiment directory with timestamp"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if experiment_name:
        exp_name = f"{experiment_name}_{timestamp}"
    else:
        exp_name = timestamp

    output_dir = Path(base_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def load_data(data_path: Path, config: Config, logger: logging.Logger):
    """Load and preprocess data"""
    logger.info(f"\nLoading data from {data_path}")

    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data shape: {df.shape}")

        # Split data
        splitter = TrialSplitter(config= config)


        train_df, val_df, test_df = splitter.split_by_time(df)
        logger.info(f"\nSplit sizes:")
        logger.info(f"Train: {len(train_df)}")
        logger.info(f"Val: {len(val_df)}")
        logger.info(f"Test: {len(test_df)}")

        return train_df, val_df, test_df

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def create_dataloaders(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        config: Config,
        logger: logging.Logger
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoader objects ensuring TimeSeriesDataset type"""
    logger.info("\nCreating datasets and dataloaders...")

    # Create datasets
    train_dataset = TimeSeriesDataset(
        data=train_df[config.feature_columns].values,
        targets=train_df['target'].values,
        trial_ids=train_df['RECORDING_SESSION_LABEL'].values,
        patch_len=config.patch_len,
        stride=config.stride,
        verbose=True
    )

    val_dataset = TimeSeriesDataset(
        data=val_df[config.feature_columns].values,
        targets=val_df['target'].values,
        trial_ids=val_df['RECORDING_SESSION_LABEL'].values,
        patch_len=config.patch_len,
        stride=config.stride,
        scaler=train_dataset.scaler  # Use training set scaler
    )

    test_dataset = TimeSeriesDataset(
        data=test_df[config.feature_columns].values,
        targets=test_df['target'].values,
        trial_ids=test_df['RECORDING_SESSION_LABEL'].values,
        patch_len=config.patch_len,
        stride=config.stride,
        scaler=train_dataset.scaler  # Use training set scaler
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
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


def calculate_class_weights(dataset: Union[Dataset, TimeSeriesDataset], device: torch.device) -> torch.Tensor:
    """Calculate class weights for imbalanced data"""
    targets = dataset.patch_targets.numpy()
    class_counts = np.bincount(targets)
    total = len(targets)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights).to(device)


def main():
    # Parse arguments
    args = parse_args()

    # Create experiment directory
    output_dir = setup_experiment_dir(args.output_dir, args.experiment_name)

    # Setup logging
    logger = setup_logging(output_dir / 'train.log')
    logger.info("Starting experiment...")

    # Set random seed
    seed_everything(args.seed)
    logger.info(f"Random seed: {args.seed}")

    config = load_config(args.config)

    # Save configuration
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(vars(config), f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logger.info(f"Using device: {device}")

    try:
        # Load data
        train_df, val_df, test_df = load_data(args.data_path, config, logger)

        # Create dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            train_df, val_df, test_df, config, logger
        )

        # Calculate class weights
        class_weights = calculate_class_weights(train_loader.dataset, device)
        logger.info(f"\nClass weights: {class_weights}")

        # Create model
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
            verbose=True
        )

        # Load pretrained weights if specified
        if args.pretrained_path:
            logger.info(f"\nLoading pretrained weights from {args.pretrained_path}")
            model.load_pretrained(torch.load(args.pretrained_path))

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
    main()