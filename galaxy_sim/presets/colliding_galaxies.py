"""Colliding galaxies preset using multi-component galaxies."""

import numpy as np
from typing import Tuple
from galaxy_sim.backends.base import Backend
from galaxy_sim.presets.multi_component import MultiComponentGalaxy


class CollidingGalaxies(MultiComponentGalaxy):
    """Two multi-component galaxies colliding."""
    
    def __init__(
        self,
        backend: Backend,
        n_particles: int = 2000,
        seed: int = None,
        # Galaxy 1 parameters
        galaxy1_offset: Tuple[float, float, float] = (-20.0, 0.0, 0.0),
        galaxy1_velocity: Tuple[float, float, float] = (0.5, 0.0, 0.0),
        # Galaxy 2 parameters
        galaxy2_offset: Tuple[float, float, float] = (20.0, 0.0, 0.0),
        galaxy2_velocity: Tuple[float, float, float] = (-0.5, 0.0, 0.0),
        # Component ratios (per galaxy)
        disk_fraction: float = 0.7,
        bulge_fraction: float = 0.2,
        halo_fraction: float = 0.1,
        **kwargs
    ):
        # Each galaxy gets half the particles
        n_per_galaxy = n_particles // 2
        
        # Store collision parameters
        self.galaxy1_offset = np.array(galaxy1_offset)
        self.galaxy1_velocity = np.array(galaxy1_velocity)
        self.galaxy2_offset = np.array(galaxy2_offset)
        self.galaxy2_velocity = np.array(galaxy2_velocity)
        
        # Initialize base class with per-galaxy particle counts
        kwargs.setdefault('n_disk', int(n_per_galaxy * disk_fraction))
        kwargs.setdefault('n_bulge', int(n_per_galaxy * bulge_fraction))
        kwargs.setdefault('n_halo', int(n_per_galaxy * halo_fraction))
        kwargs.setdefault('central_mass', 0.0)  # No central mass for collision
        
        super().__init__(backend, n_per_galaxy, seed, **kwargs)
        
        # Update total particle count
        self.n_particles = n_per_galaxy * 2
    
    @property
    def name(self) -> str:
        return "colliding_galaxies"
    
    def generate(self) -> Tuple:
        """Generate two colliding galaxies."""
        # Generate first galaxy
        pos1, vel1, mass1 = super().generate()
        
        # Generate second galaxy with different seed
        seed2 = self.seed + 1000 if self.seed is not None else None
        rng = np.random.default_rng(seed2)
        
        # Temporarily change seed to generate different galaxy
        old_seed = self.seed
        self.seed = seed2
        pos2, vel2, mass2 = super().generate()
        self.seed = old_seed
        
        # Convert to numpy for manipulation
        pos1_np = np.asarray(self.backend.to_numpy(pos1))
        vel1_np = np.asarray(self.backend.to_numpy(vel1))
        mass1_np = np.asarray(self.backend.to_numpy(mass1))
        pos2_np = np.asarray(self.backend.to_numpy(pos2))
        vel2_np = np.asarray(self.backend.to_numpy(vel2))
        mass2_np = np.asarray(self.backend.to_numpy(mass2))
        
        # Offset positions and add bulk velocities
        pos1_np += self.galaxy1_offset
        pos2_np += self.galaxy2_offset
        vel1_np += self.galaxy1_velocity
        vel2_np += self.galaxy2_velocity
        
        # Combine
        positions = np.vstack([pos1_np, pos2_np])
        velocities = np.vstack([vel1_np, vel2_np])
        masses = np.concatenate([mass1_np, mass2_np])
        
        return self.backend.array(positions), self.backend.array(velocities), self.backend.array(masses)
