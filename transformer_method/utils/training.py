import logging
import random
import numpy as np
import torch
from pathlib import Path
import json


def setup_logging(log_file):
    """
    Setup logging configuration and return logger instance.

    Args:
        log_file: Path to log file

    Returns:
        logging.Logger: Configured logger instance
    """
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('experiment_logger')
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def seed_everything(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def save_checkpoint(state, is_best, output_dir):
    """Save training checkpoint"""
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / 'checkpoint.pth'
    torch.save(state, checkpoint_path)
    if is_best:
        best_path = output_dir / 'best_model.pth'
        torch.save(state, best_path)

def load_checkpoint(path, model, optimizer=None):
    """Load checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0)

def save_training_history(history, output_dir):
    """Save training history to JSON"""
    output_dir = Path(output_dir)
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=4)