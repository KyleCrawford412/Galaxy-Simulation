"""Render manager for switching between 2D and 3D modes."""

from typing import Optional, Literal
import numpy as np
from galaxy_sim.render.base import Renderer
from galaxy_sim.render.renderer_2d import Renderer2D
from galaxy_sim.render.renderer_3d import Renderer3D


class RenderManager:
    """Manages rendering and mode switching."""
    
    def __init__(self, mode: Literal["2d", "3d"] = "2d", **renderer_kwargs):
        """Initialize render manager.
        
        Args:
            mode: Rendering mode ('2d' or '3d')
            **renderer_kwargs: Additional arguments for renderer
        """
        self.mode = mode
        self.renderer: Optional[Renderer] = None
        self.renderer_kwargs = renderer_kwargs
        self._create_renderer()
    
    def _create_renderer(self):
        """Create appropriate renderer based on mode."""
        if self.renderer is not None:
            self.renderer.close()
        
        if self.mode == "2d":
            self.renderer = Renderer2D(**self.renderer_kwargs)
        elif self.mode == "3d":
            self.renderer = Renderer3D(**self.renderer_kwargs)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def set_mode(self, mode: Literal["2d", "3d"]):
        """Switch rendering mode.
        
        Args:
            mode: New mode ('2d' or '3d')
        """
        if mode != self.mode:
            self.mode = mode
            self._create_renderer()
    
    def render(self, positions: np.ndarray, velocities: Optional[np.ndarray] = None, masses: Optional[np.ndarray] = None):
        """Render current frame."""
        if self.renderer is None:
            raise RuntimeError("Renderer not initialized")
        self.renderer.render(positions, velocities, masses)
    
    def capture_frame(self) -> np.ndarray:
        """Capture current frame."""
        if self.renderer is None:
            raise RuntimeError("Renderer not initialized")
        return self.renderer.capture_frame()
    
    def close(self):
        """Close renderer."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
