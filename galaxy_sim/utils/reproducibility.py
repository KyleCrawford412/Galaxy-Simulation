"""Reproducibility utilities for deterministic simulations."""

import random
import numpy as np
from typing import Optional, Dict
from galaxy_sim.backends.base import Backend


def set_all_seeds(seed: int, backend: Optional[Backend] = None):
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
        backend: Optional backend to set seed for
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if backend is not None:
        backend.set_seed(seed)
    
    # Set seeds for optional backends if available
    try:
        import jax
        jax.random.PRNGKey(seed)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_seed_info(seed: Optional[int] = None) -> Dict[str, any]:
    """Get information about current seed state.
    
    Args:
        seed: Optional seed to include in info
        
    Returns:
        Dictionary with seed information
    """
    info = {}
    
    if seed is not None:
        info['seed'] = seed
    
    info['numpy_state'] = np.random.get_state()[1][0] if hasattr(np.random.get_state(), '__getitem__') else None
    info['python_random_state'] = random.getstate()[1][0] if hasattr(random.getstate(), '__getitem__') else None
    
    return info
