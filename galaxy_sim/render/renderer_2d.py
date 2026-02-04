"""2D renderer using matplotlib."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from typing import Optional, Tuple
import time
from galaxy_sim.render.base import Renderer


class Renderer2D(Renderer):
    """2D real-time renderer using matplotlib."""
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 10),
        dpi: int = 100,
        show_trails: bool = False,
        trail_length: int = 100,
        color_by_velocity: bool = True,
        size_by_mass: bool = True
    ):
        """Initialize 2D renderer.
        
        Args:
            figsize: Figure size (width, height)
            dpi: Dots per inch
            show_trails: Whether to show particle trails
            trail_length: Number of previous positions to keep
            color_by_velocity: Color particles by velocity magnitude
            size_by_mass: Size particles by mass
        """
        self.figsize = figsize
        self.dpi = dpi
        self.show_trails = show_trails
        self.trail_length = trail_length
        self.color_by_velocity = color_by_velocity
        self.size_by_mass = size_by_mass
        
        self.fig: Optional[Figure] = None
        self.ax = None
        self.scatter = None
        self.trails = []  # List of trail arrays
        self.initialized = False
        
        # Frame rate limiting
        self.target_fps = 30.0
        self.frame_time = 1.0 / self.target_fps
        self.last_render_time = 0.0
    
    def _initialize(self, positions: np.ndarray):
        """Initialize plot if not already done."""
        if not self.initialized:
            self.fig, self.ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            self.ax.set_aspect('equal')
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_title('Galaxy Simulation (2D)')
            self.ax.grid(True, alpha=0.3)
            
            # Extract 2D positions (use first 2 dimensions)
            if positions.shape[1] == 3:
                pos_2d = positions[:, :2]
            else:
                pos_2d = positions
            
            # Set initial axis limits
            margin = 0.1
            x_range = pos_2d[:, 0].max() - pos_2d[:, 0].min()
            y_range = pos_2d[:, 1].max() - pos_2d[:, 1].min()
            x_center = (pos_2d[:, 0].max() + pos_2d[:, 0].min()) / 2
            y_center = (pos_2d[:, 1].max() + pos_2d[:, 1].min()) / 2
            
            max_range = max(x_range, y_range) * (1 + margin)
            self.ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
            self.ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
            
            # Show the window (non-blocking)
            plt.show(block=False)
            plt.pause(0.1)  # Give it time to appear
            
            self.initialized = True
    
    def _is_figure_open(self) -> bool:
        """Check if the figure window is still open."""
        if self.fig is None:
            return False
        try:
            # Check if figure number still exists in matplotlib's figure manager
            if not plt.fignum_exists(self.fig.number):
                # Reset initialized flag if figure was closed
                self.initialized = False
                self.fig = None
                self.ax = None
                return False
            return True
        except (AttributeError, ValueError, RuntimeError):
            # If any error occurs, assume figure is closed
            self.initialized = False
            self.fig = None
            self.ax = None
            return False
    
    def render(self, positions: np.ndarray, velocities: Optional[np.ndarray] = None, masses: Optional[np.ndarray] = None):
        """Render current frame."""
        # Check if figure was closed - if so, stop rendering
        if self.initialized and not self._is_figure_open():
            return
        
        # Frame rate limiting: skip frame if rendering too fast
        current_time = time.time()
        if self.initialized and (current_time - self.last_render_time) < self.frame_time:
            return  # Skip frame to maintain target FPS
        
        self.last_render_time = current_time
        self._initialize(positions)
        
        # Extract 2D positions
        if positions.shape[1] == 3:
            pos_2d = positions[:, :2]
        else:
            pos_2d = positions
        
        # Prepare colors
        if self.color_by_velocity and velocities is not None:
            if velocities.shape[1] == 3:
                vel_mag = np.linalg.norm(velocities, axis=1)
            else:
                vel_mag = np.linalg.norm(velocities, axis=1)
            colors = vel_mag
            colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-10)
            cmap = plt.cm.viridis
        else:
            colors = 'blue'
            cmap = None
        
        # Prepare sizes - must match number of particles
        n_particles = pos_2d.shape[0]
        if self.size_by_mass and masses is not None:
            masses_np = np.asarray(masses).flatten()
            if len(masses_np) == n_particles and len(masses_np) > 0 and masses_np.max() > 0:
                sizes = (10 + 50 * (masses_np / masses_np.max())).astype(float)
            else:
                sizes = 5.0  # Scalar for all particles
        else:
            sizes = 5.0  # Scalar for all particles
        
        # Update trails
        if self.show_trails:
            self.trails.append(pos_2d.copy())
            if len(self.trails) > self.trail_length:
                self.trails.pop(0)
        
        # Fast path: update only scatter offsets (no full redraw) when already initialized
        if self.initialized and self.scatter is not None and not self.show_trails:
            self.scatter.set_offsets(pos_2d)
            if cmap is not None and hasattr(colors, '__len__'):
                rgba = cmap(np.asarray(colors).flatten())
                self.scatter.set_facecolors(rgba)
            if hasattr(sizes, '__len__'):
                self.scatter.set_sizes(sizes)
            margin = 0.15
            x_min, x_max = pos_2d[:, 0].min(), pos_2d[:, 0].max()
            y_min, y_max = pos_2d[:, 1].min(), pos_2d[:, 1].max()
            x_range, y_range = x_max - x_min, y_max - y_min
            if x_range > 1e-6 and y_range > 1e-6:
                max_range = max(x_range, y_range, 10.0) * (1 + margin)
                x_center = (x_max + x_min) / 2
                y_center = (y_max + y_min) / 2
                self.ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
                self.ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
            return
        
        # Full redraw (first frame or when trails enabled)
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Galaxy Simulation (2D)')
        self.ax.grid(True, alpha=0.3)
        
        if self.show_trails and len(self.trails) > 1:
            for trail in self.trails[:-1]:
                self.ax.plot(trail[:, 0], trail[:, 1], 'k-', alpha=0.1, linewidth=0.5)
        
        if cmap is not None:
            self.scatter = self.ax.scatter(
                pos_2d[:, 0], pos_2d[:, 1],
                c=colors, s=sizes, cmap=cmap,
                alpha=0.6, edgecolors='black', linewidths=0.5
            )
        else:
            self.scatter = self.ax.scatter(
                pos_2d[:, 0], pos_2d[:, 1],
                c=colors, s=sizes,
                alpha=0.6, edgecolors='black', linewidths=0.5
            )
        
        margin = 0.15
        x_min, x_max = pos_2d[:, 0].min(), pos_2d[:, 0].max()
        y_min, y_max = pos_2d[:, 1].min(), pos_2d[:, 1].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_center = (x_max + x_min) / 2
        y_center = (y_max + y_min) / 2
        if x_range > 1e-6 and y_range > 1e-6:
            max_range = max(x_range, y_range, 10.0) * (1 + margin)
            self.ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
            self.ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
        
        plt.draw()
        plt.pause(0.001)
    
    def capture_frame(self) -> np.ndarray:
        """Capture current frame as image array."""
        if self.fig is None:
            raise RuntimeError("Renderer not initialized. Call render() first.")
        
        # Convert figure to image array
        self.fig.canvas.draw()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return buf
    
    def clear(self):
        """Clear the renderer."""
        if self.ax is not None:
            self.ax.clear()
    
    def close(self):
        """Close the renderer."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.initialized = False
