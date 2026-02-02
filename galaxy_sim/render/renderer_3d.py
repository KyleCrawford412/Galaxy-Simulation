"""3D renderer using matplotlib."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from typing import Optional, Tuple
from galaxy_sim.render.base import Renderer


class Renderer3D(Renderer):
    """3D real-time renderer using matplotlib 3D."""
    
    def __init__(
        self,
        figsize: Tuple[int, int] = (10, 10),
        dpi: int = 100,
        show_trails: bool = False,
        trail_length: int = 50,
        color_by_velocity: bool = True,
        size_by_mass: bool = True,
        elevation: float = 20.0,
        azimuth: float = 45.0
    ):
        """Initialize 3D renderer.
        
        Args:
            figsize: Figure size
            dpi: Dots per inch
            show_trails: Whether to show particle trails
            trail_length: Number of previous positions to keep
            color_by_velocity: Color particles by velocity magnitude
            size_by_mass: Size particles by mass
            elevation: Camera elevation angle
            azimuth: Camera azimuth angle
        """
        self.figsize = figsize
        self.dpi = dpi
        self.show_trails = show_trails
        self.trail_length = trail_length
        self.color_by_velocity = color_by_velocity
        self.size_by_mass = size_by_mass
        self.elevation = elevation
        self.azimuth = azimuth
        
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes3D] = None
        self.scatter = None
        self.trails = []
        self.initialized = False
    
    def _initialize(self, positions: np.ndarray):
        """Initialize plot if not already done."""
        if not self.initialized:
            self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('Galaxy Simulation (3D)')
            
            # Set initial limits
            margin = 0.1
            x_range = positions[:, 0].max() - positions[:, 0].min()
            y_range = positions[:, 1].max() - positions[:, 1].min()
            z_range = positions[:, 2].max() - positions[:, 2].min()
            
            x_center = (positions[:, 0].max() + positions[:, 0].min()) / 2
            y_center = (positions[:, 1].max() + positions[:, 1].min()) / 2
            z_center = (positions[:, 2].max() + positions[:, 2].min()) / 2
            
            max_range = max(x_range, y_range, z_range) * (1 + margin)
            self.ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
            self.ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
            self.ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
            
            self.ax.view_init(elev=self.elevation, azim=self.azimuth)
            
            # Show the window (non-blocking)
            plt.show(block=False)
            plt.pause(0.1)  # Give it time to appear
            
            self.initialized = True
    
    def render(self, positions: np.ndarray, velocities: Optional[np.ndarray] = None, masses: Optional[np.ndarray] = None):
        """Render current frame."""
        self._initialize(positions)
        
        # Prepare colors
        if self.color_by_velocity and velocities is not None:
            vel_mag = np.linalg.norm(velocities, axis=1)
            colors = vel_mag
            colors = (colors - colors.min()) / (colors.max() - colors.min() + 1e-10)
            cmap = plt.cm.viridis
        else:
            colors = 'blue'
            cmap = None
        
        # Prepare sizes - must match number of particles
        n_particles = positions.shape[0]
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
            self.trails.append(positions.copy())
            if len(self.trails) > self.trail_length:
                self.trails.pop(0)
        
        # Clear and redraw
        self.ax.clear()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Galaxy Simulation (3D)')
        
        # Draw trails
        if self.show_trails and len(self.trails) > 1:
            for trail in self.trails[:-1]:
                self.ax.plot(trail[:, 0], trail[:, 1], trail[:, 2], 'k-', alpha=0.1, linewidth=0.5)
        
        # Draw particles
        if cmap is not None:
            self.scatter = self.ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=colors, s=sizes, cmap=cmap,
                alpha=0.6, edgecolors='black', linewidths=0.5
            )
        else:
            self.scatter = self.ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=colors, s=sizes,
                alpha=0.6, edgecolors='black', linewidths=0.5
            )
        
        # Auto-adjust limits
        margin = 0.1
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        z_range = positions[:, 2].max() - positions[:, 2].min()
        
        x_center = (positions[:, 0].max() + positions[:, 0].min()) / 2
        y_center = (positions[:, 1].max() + positions[:, 1].min()) / 2
        z_center = (positions[:, 2].max() + positions[:, 2].min()) / 2
        
        max_range = max(x_range, y_range, z_range) * (1 + margin)
        if max_range > 0:
            self.ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
            self.ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
            self.ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
        
        self.ax.view_init(elev=self.elevation, azim=self.azimuth)
        
        plt.draw()
        plt.pause(0.001)
    
    def capture_frame(self) -> np.ndarray:
        """Capture current frame as image array."""
        if self.fig is None:
            raise RuntimeError("Renderer not initialized. Call render() first.")
        
        self.fig.canvas.draw()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        return buf
    
    def set_view(self, elevation: float, azimuth: float):
        """Set camera view angles.
        
        Args:
            elevation: Elevation angle
            azimuth: Azimuth angle
        """
        self.elevation = elevation
        self.azimuth = azimuth
        if self.ax is not None:
            self.ax.view_init(elev=elevation, azim=azimuth)
    
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
