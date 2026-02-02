"""Base renderer interface."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np


class Renderer(ABC):
    """Abstract base class for renderers."""
    
    @abstractmethod
    def render(self, positions: np.ndarray, velocities: Optional[np.ndarray] = None, masses: Optional[np.ndarray] = None):
        """Render current frame.
        
        Args:
            positions: Particle positions (n, 2) or (n, 3)
            velocities: Optional velocities for coloring
            masses: Optional masses for sizing
        """
        pass
    
    @abstractmethod
    def capture_frame(self) -> np.ndarray:
        """Capture current frame as image array.
        
        Returns:
            Image array (H, W, 3) uint8
        """
        pass
    
    @abstractmethod
    def clear(self):
        """Clear the renderer."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the renderer."""
        pass
