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
        
        # Use numpy RNG for proper random number generation
        rng = np.random.default_rng(self.seed)
        
        positions = []
        velocities = []
        masses = []
        
        # Generate galaxy centers
        galaxy_centers = []
        center_radii, center_theta, center_phi = generate_spherical_particles(
            self.n_galaxies, self.cluster_radius, rng
        )
        
        for i in range(self.n_galaxies):
            r = center_radii[i]
            th = center_theta[i]
            ph = center_phi[i]
            
            center = np.array([
                r * np.sin(ph) * np.cos(th),
                r * np.sin(ph) * np.sin(th),
                r * np.cos(ph) * 0.3  # Flatten cluster
            ])
            galaxy_centers.append(center)
        
        # Generate particles in each galaxy
        for gal_idx, center in enumerate(galaxy_centers):
            n_this_galaxy = n_per_galaxy if gal_idx < self.n_galaxies - 1 else n - gal_idx * n_per_galaxy
            
            # Generate particles around this galaxy center
            gal_radii, gal_theta, gal_phi = generate_spherical_particles(
                n_this_galaxy, self.galaxy_radius, rng
            )
            
            for i in range(n_this_galaxy):
                r = gal_radii[i]
                th = gal_theta[i]
                ph = gal_phi[i]
                
                x = center[0] + r * np.sin(ph) * np.cos(th)
                y = center[1] + r * np.sin(ph) * np.sin(th)
                z = center[2] + r * np.cos(ph) * 0.2
                
                # Circular motion around galaxy center
                v_mag = np.sqrt(1.0 / max(r, 0.1))
                vx = -v_mag * np.sin(th)
                vy = v_mag * np.cos(th)
                vz = 0.0
                
                positions.append([x, y, z])
                velocities.append([vx, vy, vz])
                masses.append(1.0 / n_this_galaxy)
        
        positions = self.backend.array(positions)
        velocities = self.backend.array(velocities)
        masses = self.backend.array(masses)
        
        return positions, velocities, masses
