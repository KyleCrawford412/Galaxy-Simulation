"""Abstract base class for numerical integrators."""

from abc import ABC, abstractmethod
from typing import Tuple


class Integrator(ABC):
    """Abstract interface for numerical integrators."""
    
    @abstractmethod
    def step(self, positions, velocities, masses, forces, dt: float, backend) -> Tuple:
        """Perform one integration step.
        
        Args:
            positions: Current positions array
            velocities: Current velocities array
            masses: Masses array
            forces: Current forces (tuple of components)
            dt: Time step
            backend: Compute backend
            
        Returns:
            Tuple of (new_positions, new_velocities)
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this integrator."""
        pass
    
    @property
    @abstractmethod
    def order(self) -> int:
        """Return the order of accuracy (e.g., 1 for Euler, 2 for Verlet, 4 for RK4)."""
        pass
