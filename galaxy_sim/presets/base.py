"""Base class for preset scenarios."""

from abc import ABC, abstractmethod
from typing import Tuple
from galaxy_sim.backends.base import Backend


class Preset(ABC):
    """Abstract base class for preset scenarios."""
    
    def __init__(self, backend: Backend, n_particles: int = 1000, seed: int = None):
        """Initialize preset.
        
        Args:
            backend: Compute backend
            n_particles: Number of particles
            seed: Random seed for reproducibility
        """
        self.backend = backend
        self.n_particles = n_particles
        self.seed = seed
        
        if seed is not None:
            backend.set_seed(seed)
    
    @abstractmethod
    def generate(self) -> Tuple:
        """Generate initial conditions.
        
        Returns:
            Tuple of (positions, velocities, masses)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this preset."""
        pass
