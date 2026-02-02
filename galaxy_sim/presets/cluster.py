"""Galaxy cluster preset."""

import numpy as np
from typing import Tuple
from galaxy_sim.backends.base import Backend
from galaxy_sim.presets.base import Preset


class GalaxyCluster(Preset):
    """Galaxy cluster - multiple galaxies in a cluster."""
    
    def __init__(
        self,
        backend: Backend,
        n_particles: int = 20000,
        seed: int = None,
        n_galaxies: int = 5,
        cluster_radius: float = 30.0,
        galaxy_radius: float = 3.0
    ):
        """Initialize galaxy cluster preset.
        
        Args:
            backend: Compute backend
            n_particles: Total number of particles
            seed: Random seed
            n_galaxies: Number of galaxies in cluster
            cluster_radius: Radius of cluster
            galaxy_radius: Radius of each galaxy
        """
        super().__init__(backend, n_particles, seed)
        self.n_galaxies = n_galaxies
        self.cluster_radius = cluster_radius
        self.galaxy_radius = galaxy_radius
    
    @property
    def name(self) -> str:
        return "cluster"
    
    def generate(self) -> Tuple:
        """Generate galaxy cluster initial conditions."""
        n = self.n_particles
        n_per_galaxy = n // self.n_galaxies
        
        positions = []
        velocities = []
        masses = []
        
        # Generate galaxy centers
        galaxy_centers = []
        for _ in range(self.n_galaxies):
            r = self.cluster_radius * (self.backend.random_uniform((1,), 0, 1, self.seed)[0] ** (1/3))
            theta = self.backend.random_uniform((1,), 0, 2 * np.pi, self.seed)[0]
            phi = np.arccos(2 * self.backend.random_uniform((1,), 0, 1, self.seed)[0] - 1)
            
            center = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi) * 0.3  # Flatten cluster
            ])
            galaxy_centers.append(center)
        
        # Generate particles in each galaxy
        for gal_idx, center in enumerate(galaxy_centers):
            n_this_galaxy = n_per_galaxy if gal_idx < self.n_galaxies - 1 else n - gal_idx * n_per_galaxy
            
            for _ in range(n_this_galaxy):
                # Spherical distribution around galaxy center
                r = self.galaxy_radius * (self.backend.random_uniform((1,), 0, 1, self.seed)[0] ** (1/3))
                theta = self.backend.random_uniform((1,), 0, 2 * np.pi, self.seed)[0]
                phi = np.arccos(2 * self.backend.random_uniform((1,), 0, 1, self.seed)[0] - 1)
                
                x = center[0] + r * np.sin(phi) * np.cos(theta)
                y = center[1] + r * np.sin(phi) * np.sin(theta)
                z = center[2] + r * np.cos(phi) * 0.2
                
                # Circular motion around galaxy center + cluster motion
                v_mag = np.sqrt(1.0 / max(r, 0.1))
                vx = -v_mag * np.sin(theta)
                vy = v_mag * np.cos(theta)
                vz = 0.0
                
                positions.append([x, y, z])
                velocities.append([vx, vy, vz])
                masses.append(1.0 / n_this_galaxy)
        
        positions = self.backend.array(positions)
        velocities = self.backend.array(velocities)
        masses = self.backend.array(masses)
        
        return positions, velocities, masses
