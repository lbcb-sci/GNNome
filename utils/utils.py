import os
import pickle
import random

import torch
import numpy as np
import dgl


def set_seed(seed=42):
    """Set random seed to enable reproducibility.
    
    Parameters
    ----------
    seed : int, optional
        A number used to set the random seed

    Returns
    -------
    None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)


def timedelta_to_str(delta):
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}h {minutes}m {seconds}s'
