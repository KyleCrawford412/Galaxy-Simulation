"""GIF export functionality."""

import numpy as np
from typing import List, Optional


class GIFExporter:
    """Export simulation frames to animated GIF."""
    
    def __init__(self, output_path: str, fps: int = 10, duration: Optional[float] = None):
        """Initialize GIF exporter.
        
        Args:
            output_path: Output file path (.gif)
            fps: Frames per second (used if duration is None)
            duration: Frame duration in seconds (overrides fps)
        """
        self.output_path = output_path
        self.fps = fps
        self.duration = duration if duration is not None else (1.0 / fps)
        self.frames: List[np.ndarray] = []
    
    def add_frame(self, frame: np.ndarray):
        """Add a frame to the export queue.
        
        Args:
            frame: Image array (H, W, 3) uint8
        """
        if frame.dtype != np.uint8:
            # Normalize to 0-255
            frame = (frame * 255).astype(np.uint8)
        self.frames.append(frame.copy())
    
    def export(self):
        """Export all frames to GIF file."""
        if not self.frames:
            raise ValueError("No frames to export")
        
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "GIF export requires imageio. Install with: pip install imageio"
            )
        
        # Convert frames to list of arrays
        frames_list = [frame for frame in self.frames]
        
        imageio.mimsave(
            self.output_path,
            frames_list,
            duration=self.duration,
            loop=0  # Infinite loop
        )
