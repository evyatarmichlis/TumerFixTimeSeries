"""
General utility functions.
"""

import random
import numpy as np
import torch


def seed_everything(seed):
    """
    Set seeds for reproducibility.

    Args:
        seed: Seed value for random number generators
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)