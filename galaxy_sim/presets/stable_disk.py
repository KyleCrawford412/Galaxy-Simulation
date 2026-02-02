"""Stable rotating disk preset with central mass."""

import numpy as np
from typing import Tuple
from galaxy_sim.backends.base import Backend
from galaxy_sim.presets.base import Preset


class StableDisk(Preset):
    """Stable rotating disk with exponential profile and central mass.
    
    Uses normalized units: G=1, M_center=1000, star mass=1, r in [1, 50]
    """
    
    def __init__(
        self,
        backend: Backend,
        n_particles: int = 1000,
        seed: int = None,
        M_center: float = 1000.0,
        star_mass: float = 1.0,
        r_min: float = 1.0,
        r_max: float = 50.0,
        disk_scale_radius: float = 10.0,
        velocity_noise: float = 0.05,
        epsilon: float = 0.1
    ):
        """Initialize stable disk preset.
        
        Args:
            backend: Compute backend
            n_particles: Number of star particles
            seed: Random seed
            M_center: Central mass (default: 1000)
            star_mass: Mass of each star (default: 1.0)
            r_min: Minimum radius (default: 1.0)
            r_max: Maximum radius (default: 50.0)
            disk_scale_radius: Exponential disk scale radius (default: 10.0)
            velocity_noise: Fractional noise in velocity (default: 0.05 = 5%)
            epsilon: Softening parameter for velocity calculation (default: 0.1)
        """
        super().__init__(backend, n_particles, seed)
        self.M_center = M_center
        self.star_mass = star_mass
        self.r_min = r_min
        self.r_max = r_max
        self.disk_scale_radius = disk_scale_radius
        self.velocity_noise = velocity_noise
        self.epsilon = epsilon
    
    @property
    def name(self) -> str:
        return "stable_disk"
    
    def generate(self) -> Tuple:
        """Generate stable rotating disk with central mass.
        
        Returns:
            Tuple of (positions, velocities, masses)
        """
        n = self.n_particles
        rng = np.random.default_rng(self.seed)
        
        # Sample positions from exponential disk profile
        # Use inverse transform sampling for exponential distribution
        u = rng.uniform(0, 1, n)
        # CDF: F(r) = 1 - exp(-(r - r_min) / scale_radius) for r in [r_min, r_max]
        # Inverse: r = r_min - scale_radius * ln(1 - u * (1 - exp(-(r_max - r_min) / scale_radius)))
        max_u = 1 - np.exp(-(self.r_max - self.r_min) / self.disk_scale_radius)
        u_scaled = u * max_u
        radii = self.r_min - self.disk_scale_radius * np.log(1 - u_scaled)
        radii = np.clip(radii, self.r_min, self.r_max)
        
        # Uniform angles
        angles = rng.uniform(0, 2 * np.pi, n)
        
        # Convert to Cartesian (2D disk)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        z = np.zeros(n)  # Flat disk
        
        positions = np.column_stack([x, y, z])
        
        # Calculate tangential velocities for circular orbits
        # v = sqrt(G * M_center / (r + eps))
        # Direction: perpendicular to radius, (-y, x) normalized
        G = 1.0  # Normalized units
        v_mag = np.sqrt(G * self.M_center / (radii + self.epsilon))
        
        # Add velocity noise (5% random variation)
        v_noise = rng.normal(1.0, self.velocity_noise, n)
        v_mag = v_mag * v_noise
        
        # Tangential velocity: perpendicular to radius vector
        # For position (x, y), tangent is (-y, x) / r
        vx = -v_mag * np.sin(angles)  # -y/r * v_mag
        vy = v_mag * np.cos(angles)    # x/r * v_mag
        vz = np.zeros(n)
        
        velocities = np.column_stack([vx, vy, vz])
        
        # Masses: all stars have same mass
        masses = np.full(n, self.star_mass)
        
        # Add central mass particle at origin
        center_pos = np.array([[0.0, 0.0, 0.0]])
        center_vel = np.array([[0.0, 0.0, 0.0]])
        center_mass = np.array([self.M_center])
        
        # Combine
        positions = np.vstack([center_pos, positions])
        velocities = np.vstack([center_vel, velocities])
        masses = np.concatenate([center_mass, masses])
        
        # Convert to backend arrays
        positions = self.backend.array(positions)
        velocities = self.backend.array(velocities)
        masses = self.backend.array(masses)
        
        return positions, velocities, masses
