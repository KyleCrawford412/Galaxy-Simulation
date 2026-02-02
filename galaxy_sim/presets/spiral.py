"""Spiral galaxy preset."""

import numpy as np
from typing import Tuple
from galaxy_sim.backends.base import Backend
from galaxy_sim.presets.base import Preset


class SpiralGalaxy(Preset):
    """Spiral galaxy with exponential disk and central bulge."""
    
    def __init__(
        self,
        backend: Backend,
        n_particles: int = 10000,
        seed: int = None,
        disk_radius: float = 10.0,
        bulge_radius: float = 2.0,
        disk_mass: float = 1.0,
        bulge_mass: float = 0.3,
        spiral_arms: int = 2
    ):
        """Initialize spiral galaxy preset.
        
        Args:
            backend: Compute backend
            n_particles: Number of particles
            seed: Random seed
            disk_radius: Outer radius of disk
            bulge_radius: Radius of central bulge
            disk_mass: Total mass of disk
            bulge_mass: Total mass of bulge
            spiral_arms: Number of spiral arms
        """
        super().__init__(backend, n_particles, seed)
        self.disk_radius = disk_radius
        self.bulge_radius = bulge_radius
        self.disk_mass = disk_mass
        self.bulge_mass = bulge_mass
        self.spiral_arms = spiral_arms
    
    @property
    def name(self) -> str:
        return "spiral"
    
    def generate(self) -> Tuple:
        """Generate spiral galaxy initial conditions."""
        n = self.n_particles
        n_bulge = int(n * 0.2)  # 20% in bulge
        n_disk = n - n_bulge
        
        positions = []
        velocities = []
        masses = []
        
        # Generate bulge (spherical distribution)
        for _ in range(n_bulge):
            # Spherical coordinates
            r = self.backend.random_uniform((1,), 0, self.bulge_radius, self.seed)[0]
            theta = self.backend.random_uniform((1,), 0, 2 * np.pi, self.seed)[0]
            phi = np.arccos(2 * self.backend.random_uniform((1,), 0, 1, self.seed)[0] - 1)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi) * 0.1  # Flatten slightly
            
            # Circular velocity (simplified)
            v_mag = np.sqrt(self.bulge_mass / max(r, 0.1))
            vx = -v_mag * np.sin(theta)
            vy = v_mag * np.cos(theta)
            vz = 0.0
            
            positions.append([x, y, z])
            velocities.append([vx, vy, vz])
            masses.append(self.bulge_mass / n_bulge)
        
        # Generate disk (exponential disk with spiral arms)
        for i in range(n_disk):
            # Exponential radial distribution
            r = -self.disk_radius * np.log(1 - self.backend.random_uniform((1,), 0, 1, self.seed)[0])
            r = min(r, self.disk_radius)
            
            # Spiral arm modulation
            theta_base = self.backend.random_uniform((1,), 0, 2 * np.pi, self.seed)[0]
            arm_phase = self.spiral_arms * np.log(r / 0.5 + 1)
            theta = theta_base + arm_phase
            
            # Add some random scatter
            theta += self.backend.random_normal((1,), 0, 0.2, self.seed)[0]
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = self.backend.random_normal((1,), 0, 0.1, self.seed)[0]  # Thin disk
            
            # Circular velocity (Keplerian approximation)
            total_mass = self.disk_mass + self.bulge_mass
            v_mag = np.sqrt(total_mass / max(r, 0.1))
            vx = -v_mag * np.sin(theta)
            vy = v_mag * np.cos(theta)
            vz = 0.0
            
            positions.append([x, y, z])
            velocities.append([vx, vy, vz])
            masses.append(self.disk_mass / n_disk)
        
        # Convert to backend arrays
        positions = self.backend.array(positions)
        velocities = self.backend.array(velocities)
        masses = self.backend.array(masses)
        
        return positions, velocities, masses
