# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by elnmax, 2026

__all__ = [
    'setup_random_seed',
    'setup_environment',
    'seed_worker',
    'setup_deterministic_training',
]

import os
import random
from typing import Callable, Tuple

import torch
import numpy as np


def setup_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Initialize NumPy and random seeds inside a DataLoader worker."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup_deterministic_training(
    seed_model: int,
    seed_data: int,
) -> Tuple[torch.Generator, Callable[[int], None]]:
    """
    Configure deterministic training.

    https://docs.pytorch.org/docs/stable/notes/randomness.html

    Parameters
    ----------
    seed_model : int
        Seed for the model and global RNGs.
    seed_data : int
        Seed for the DataLoader generator.

    Returns
    -------
    tuple[torch.Generator, Callable[[int], None]]
        DataLoader generator and worker_init_fn.
    """
    os.environ['PYTHONHASHSEED'] = str(seed_model)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')

    random.seed(seed_model)
    np.random.seed(seed_model)
    torch.manual_seed(seed_model)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_model)
        torch.cuda.manual_seed_all(seed_model)

    generator = torch.Generator()
    generator.manual_seed(seed_data)

    return generator, seed_worker


def setup_environment(cfg):
    # Do not overwrite CUDA_VISIBLE_DEVICES when a subprocess already set the GPU
    # (e.g. test_net_optim.py workers).
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg.SEED is not None:
        setup_random_seed(cfg.SEED)
    return device
