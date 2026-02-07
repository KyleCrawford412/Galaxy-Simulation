"""3D renderer using matplotlib."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.figure import Figure
from typing import Optional, Tuple
import time
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
        azimuth: float = 45.0,
        space_theme: bool = True,
        trails: bool = True,
        density: bool = True,
        density_res: int = 256,
        density_blur_sigma: float = 1.2,
        density_alpha: float = 0.25,
        starfield: bool = True,
        starfield_count: int = 5000,
        starfield_layers: int = 3,
        color_mode: str = "component",
        camera_follow_com: bool = True,
        auto_zoom: bool = True,
        render_every_k_steps: int = 10,
        fps_overlay: bool = True
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
        self.initialized = False
        self.figsize = figsize
        self.dpi = dpi
        self.show_trails = show_trails or trails
        self.trail_length = trail_length
        self.color_by_velocity = color_by_velocity
        self.size_by_mass = size_by_mass
        self.elevation = elevation
        self.azimuth = azimuth
        self.space_theme = space_theme
        self.density = density
        self.density_res = density_res
        self.density_blur_sigma = density_blur_sigma
        self.density_alpha = density_alpha
        self.starfield = starfield
        self.starfield_count = starfield_count
        self.starfield_layers = starfield_layers
        self.color_mode = color_mode
        self.camera_follow_com = camera_follow_com
        self.auto_zoom = auto_zoom
        self.render_every_k_steps = max(1, render_every_k_steps)
        self.fps_overlay = fps_overlay
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps = 0.0
        self._com_smooth = None
        self._zoom_smooth = None
        self._starfield = None
        self.fig: Optional[Figure] = None
        self.ax: Optional[Axes3D] = None
        self.scatter = None
        self.trails = []
        self.trail_scatter = None
        self.density_im = None
        self.haze_ax = None
        self.fps_text = None

    def _blur2d(self, data: np.ndarray, sigma: float) -> np.ndarray:
        """Simple Gaussian blur using separable kernel."""
        if sigma <= 0:
            return data
        radius = int(max(1, sigma * 2))
        x = np.arange(-radius, radius + 1)
        kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        data = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=0, arr=data)
        data = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode='same'), axis=1, arr=data)
        return data
    
    def _initialize(self, positions: np.ndarray):
        """Initialize plot if not already done."""
        if not self.initialized:
            self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
            self.ax = self.fig.add_subplot(111, projection='3d')
            if self.space_theme:
                self.fig.patch.set_facecolor('black')
                self.ax.set_facecolor('black')
                self.ax.set_axis_off()
                if self.density and self.haze_ax is None:
                    # 2D overlay axis for haze (avoids 3D zdir issues)
                    self.haze_ax = self.fig.add_axes(self.ax.get_position(), frameon=False)
                    self.haze_ax.set_axis_off()
            else:
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
            if self.starfield:
                self._init_starfield()
            
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
        self._frame_count += 1
        if self._frame_count % self.render_every_k_steps != 0:
            return
        
        self._initialize(positions)
        
        # Prepare colors
        colors = 'white'
        cmap = None
        if self.color_mode == "speed" and velocities is not None:
            vel_mag = np.linalg.norm(velocities, axis=1)
            colors = (vel_mag - vel_mag.min()) / (vel_mag.max() - vel_mag.min() + 1e-10)
            cmap = plt.cm.plasma
        elif self.color_mode == "radius":
            radii = np.linalg.norm(positions[:, :2], axis=1)
            colors = (radii - radii.min()) / (radii.max() - radii.min() + 1e-10)
            cmap = plt.cm.magma
        elif self.color_mode == "component":
            # Color by type if mass is provided (simple proxy)
            if masses is not None:
                colors = (masses - np.min(masses)) / (np.max(masses) - np.min(masses) + 1e-10)
                cmap = plt.cm.plasma
        elif self.color_mode == "bound" and velocities is not None:
            # Approximate bound by speed vs median (proxy)
            vel_mag = np.linalg.norm(velocities, axis=1)
            colors = (vel_mag < np.median(vel_mag)).astype(float)
            cmap = plt.cm.coolwarm
        elif self.color_by_velocity and velocities is not None:
            vel_mag = np.linalg.norm(velocities, axis=1)
            colors = (vel_mag - vel_mag.min()) / (vel_mag.max() - vel_mag.min() + 1e-10)
            cmap = plt.cm.viridis
        
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
        if self.space_theme:
            self.fig.patch.set_facecolor('black')
            self.ax.set_facecolor('black')
            self.ax.set_axis_off()
        else:
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title('Galaxy Simulation (3D)')
        if self.starfield:
            self._draw_starfield()
        
        # Draw trails
        if self.show_trails and len(self.trails) > 1:
            trail_pts = np.concatenate(self.trails[:-1], axis=0)
            self.ax.scatter(trail_pts[:, 0], trail_pts[:, 1], trail_pts[:, 2],
                            s=1.0, c='white', alpha=0.05)
        
        # Draw particles (two-pass glow)
        self.ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c=colors if cmap is not None else '#fff2b0',
            s=np.asarray(sizes) * 4.0,
            cmap=cmap,
            alpha=0.08,
            edgecolors='none'
        )
        # Draw particles
        if cmap is not None:
            self.scatter = self.ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c=colors, s=sizes, cmap=cmap,
                alpha=0.8, edgecolors='none'
            )
        else:
            self.scatter = self.ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                c='#ffdca0', s=sizes,
                alpha=0.8, edgecolors='none'
            )
        
        # Auto-adjust limits / camera follow COM
        margin = 0.1
        x_range = positions[:, 0].max() - positions[:, 0].min()
        y_range = positions[:, 1].max() - positions[:, 1].min()
        z_range = positions[:, 2].max() - positions[:, 2].min()
        
        x_center = (positions[:, 0].max() + positions[:, 0].min()) / 2
        y_center = (positions[:, 1].max() + positions[:, 1].min()) / 2
        z_center = (positions[:, 2].max() + positions[:, 2].min()) / 2
        
        max_range = max(x_range, y_range, z_range) * (1 + margin)
        if self.camera_follow_com:
            com = np.mean(positions, axis=0)
            if self._com_smooth is None:
                self._com_smooth = com
            self._com_smooth = 0.9 * self._com_smooth + 0.1 * com
            x_center, y_center, z_center = self._com_smooth
        if self.auto_zoom:
            radii = np.linalg.norm(positions - np.array([x_center, y_center, z_center]), axis=1)
            zoom = np.percentile(radii, 90) * 2.0
            if self._zoom_smooth is None:
                self._zoom_smooth = zoom
            self._zoom_smooth = 0.9 * self._zoom_smooth + 0.1 * zoom
            max_range = max(max_range, self._zoom_smooth)
        if max_range > 0:
            self.ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
            self.ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
            self.ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
        
        # Density haze (projected)
        if self.density and self.haze_ax is not None:
            x = positions[:, 0]
            y = positions[:, 1]
            H, xedges, yedges = np.histogram2d(x, y, bins=self.density_res)
            H = np.log1p(H)
            H = self._blur2d(H, self.density_blur_sigma)
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            if self.density_im is None:
                self.density_im = self.haze_ax.imshow(
                    H.T, extent=extent, origin='lower', cmap='magma', alpha=self.density_alpha
                )
            else:
                self.density_im.set_data(H.T)
                self.density_im.set_extent(extent)
        
        self.ax.view_init(elev=self.elevation, azim=self.azimuth)
        if self.fps_overlay:
            now = time.time()
            dt = now - self._last_fps_time
            if dt > 0:
                self._fps = 1.0 / dt
            self._last_fps_time = now
            self.ax.text2D(0.02, 0.95, f"{self._fps:.1f} FPS",
                           transform=self.ax.transAxes, color='white')
        
        plt.draw()
        plt.pause(0.001)

    def _init_starfield(self):
        """Initialize static starfield."""
        rng = np.random.default_rng(42)
        stars = []
        for layer in range(self.starfield_layers):
            count = int(self.starfield_count / self.starfield_layers)
            coords = rng.uniform(-1, 1, size=(count, 3)) * (6 + layer * 4)
            size = rng.uniform(1, 3, size=count) * (1 + layer * 0.5)
            alpha = 0.15 + 0.15 * layer
            colors = rng.choice(['#fff2b0', '#ffd1a3', '#ffffff'], size=count)
            stars.append((coords, size, alpha, colors))
        self._starfield = stars

    def _draw_starfield(self):
        if self._starfield is None:
            return
        for coords, size, alpha, colors in self._starfield:
            self.ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                            s=size, c=colors, alpha=alpha, depthshade=False)
    
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
