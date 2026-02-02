"""Video export functionality."""

import numpy as np
from typing import List, Optional
import os


class VideoExporter:
    """Export simulation frames to MP4 video."""
    
    def __init__(self, output_path: str, fps: int = 30, codec: str = 'mp4v'):
        """Initialize video exporter.
        
        Args:
            output_path: Output file path (.mp4)
            fps: Frames per second
            codec: Video codec
        """
        self.output_path = output_path
        self.fps = fps
        self.codec = codec
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
        """Export all frames to video file."""
        if not self.frames:
            raise ValueError("No frames to export")
        
        try:
            import cv2
            use_cv2 = True
        except ImportError:
            use_cv2 = False
        
        if use_cv2:
            self._export_cv2()
        else:
            self._export_imageio()
    
    def _export_cv2(self):
        """Export using OpenCV."""
        import cv2
        
        height, width = self.frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
        
        for frame in self.frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        
        out.release()
    
    def _export_imageio(self):
        """Export using imageio (fallback)."""
        try:
            import imageio
        except ImportError:
            raise ImportError(
                "Video export requires either opencv-python or imageio. "
                "Install with: pip install opencv-python or pip install imageio[ffmpeg]"
            )
        
        # Convert frames to list of arrays
        frames_list = [frame for frame in self.frames]
        
        imageio.mimsave(
            self.output_path,
            frames_list,
            fps=self.fps,
            codec='libx264',
            quality=8
        )
