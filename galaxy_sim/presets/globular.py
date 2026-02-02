"""Globular cluster preset."""

import numpy as np
from typing import Tuple
from galaxy_sim.backends.base import Backend
from galaxy_sim.presets.base import Preset


class GlobularCluster(Preset):
    """Globular cluster - dense spherical distribution."""
    
    def __init__(
        self,
        backend: Backend,
        n_particles: int = 5000,
        seed: int = None,
        radius: float = 5.0,
        total_mass: float = 1.0
    ):
        """Initialize globular cluster preset.
        
        Args:
            backend: Compute backend
            n_particles: Number of particles
            seed: Random seed
            radius: Cluster radius
            total_mass: Total mass
        """
        super().__init__(backend, n_particles, seed)
        self.radius = radius
        self.total_mass = total_mass
    
    @property
    def name(self) -> str:
        return "globular"
    
    def generate(self) -> Tuple:
        """Generate globular cluster initial conditions."""
        n = self.n_particles
        
        positions = []
        velocities = []
        masses = []
        
        for _ in range(n):
            # Spherical distribution (uniform in volume)
            r = self.radius * (self.backend.random_uniform((1,), 0, 1, self.seed)[0] ** (1/3))
            theta = self.backend.random_uniform((1,), 0, 2 * np.pi, self.seed)[0]
            phi = np.arccos(2 * self.backend.random_uniform((1,), 0, 1, self.seed)[0] - 1)
            
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            # Isotropic velocity distribution (virialized)
            # Velocity magnitude scales with radius
            v_mag = np.sqrt(self.total_mass * r / (self.radius ** 2))
            v_theta = self.backend.random_uniform((1,), 0, 2 * np.pi, self.seed)[0]
            v_phi = np.arccos(2 * self.backend.random_uniform((1,), 0, 1, self.seed)[0] - 1)
            
            vx = v_mag * np.sin(v_phi) * np.cos(v_theta)
            vy = v_mag * np.sin(v_phi) * np.sin(v_theta)
            vz = v_mag * np.cos(v_phi)
            
            positions.append([x, y, z])
            velocities.append([vx, vy, vz])
            masses.append(self.total_mass / n)
        
        positions = self.backend.array(positions)
        velocities = self.backend.array(velocities)
        masses = self.backend.array(masses)
        
        return positions, velocities, masses
