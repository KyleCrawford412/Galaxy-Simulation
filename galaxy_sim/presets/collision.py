"""Galaxy collision preset."""

import numpy as np
from typing import Tuple
from galaxy_sim.backends.base import Backend
from galaxy_sim.presets.base import Preset


class CollisionScenario(Preset):
    """Two galaxies colliding."""
    
    def __init__(
        self,
        backend: Backend,
        n_particles: int = 10000,
        seed: int = None,
        galaxy_separation: float = 20.0,
        relative_velocity: float = 0.5,
        galaxy_radius: float = 5.0
    ):
        """Initialize collision scenario.
        
        Args:
            backend: Compute backend
            n_particles: Total number of particles (split between galaxies)
            seed: Random seed
            galaxy_separation: Initial separation between galaxy centers
            relative_velocity: Relative velocity magnitude
            galaxy_radius: Radius of each galaxy
        """
        super().__init__(backend, n_particles, seed)
        self.galaxy_separation = galaxy_separation
        self.relative_velocity = relative_velocity
        self.galaxy_radius = galaxy_radius
    
    @property
    def name(self) -> str:
        return "collision"
    
    def generate(self) -> Tuple:
        """Generate collision scenario initial conditions."""
        n = self.n_particles
        n_per_galaxy = n // 2
        
        positions = []
        velocities = []
        masses = []
        
        # Galaxy 1 (left, moving right)
        center1 = np.array([-self.galaxy_separation / 2, 0, 0])
        # Generate all random numbers at once for better distribution
        rng = np.random.default_rng(self.seed) if self.seed is not None else np.random.default_rng()
        r_vals1 = rng.uniform(0, 1, n_per_galaxy) ** (1/3)
        theta_vals1 = rng.uniform(0, 2 * np.pi, n_per_galaxy)
        phi_vals1 = np.arccos(2 * rng.uniform(0, 1, n_per_galaxy) - 1)
        
        for idx in range(n_per_galaxy):
            # Spherical distribution
            r = self.galaxy_radius * r_vals1[idx]
            theta = theta_vals1[idx]
            phi = phi_vals1[idx]
            
            x = center1[0] + r * np.sin(phi) * np.cos(theta)
            y = center1[1] + r * np.sin(phi) * np.sin(theta)
            z = center1[2] + r * np.cos(phi) * 0.2
            
            # Circular motion + bulk velocity
            v_mag = np.sqrt(1.0 / max(r, 0.1))
            vx = self.relative_velocity / 2 - v_mag * np.sin(theta)
            vy = v_mag * np.cos(theta)
            vz = 0.0
            
            positions.append([x, y, z])
            velocities.append([vx, vy, vz])
            masses.append(1.0 / n_per_galaxy)
        
        # Galaxy 2 (right, moving left)
        center2 = np.array([self.galaxy_separation / 2, 0, 0])
        n_galaxy2 = n - n_per_galaxy
        r_vals2 = rng.uniform(0, 1, n_galaxy2) ** (1/3)
        theta_vals2 = rng.uniform(0, 2 * np.pi, n_galaxy2)
        phi_vals2 = np.arccos(2 * rng.uniform(0, 1, n_galaxy2) - 1)
        
        for idx in range(n_galaxy2):
            r = self.galaxy_radius * r_vals2[idx]
            theta = theta_vals2[idx]
            phi = phi_vals2[idx]
            
            x = center2[0] + r * np.sin(phi) * np.cos(theta)
            y = center2[1] + r * np.sin(phi) * np.sin(theta)
            z = center2[2] + r * np.cos(phi) * 0.2
            
            v_mag = np.sqrt(1.0 / max(r, 0.1))
            vx = -self.relative_velocity / 2 - v_mag * np.sin(theta)
            vy = v_mag * np.cos(theta)
            vz = 0.0
            
            positions.append([x, y, z])
            velocities.append([vx, vy, vz])
            masses.append(1.0 / (n - n_per_galaxy))
        
        positions = self.backend.array(positions)
        velocities = self.backend.array(velocities)
        masses = self.backend.array(masses)
        
        return positions, velocities, masses
