"""Spiral galaxy preset."""

import numpy as np
from typing import Tuple
from galaxy_sim.backends.base import Backend
from galaxy_sim.presets.base import Preset
from galaxy_sim.presets.utils import (
    generate_exponential_disk_particles,
    generate_spherical_particles,
    generate_core_particles,
    enclosed_mass_exponential,
    enclosed_mass_hernquist,
    circular_velocity
)


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
        spiral_arms: int = 2,
        core_mass: float = None,
        core_radius: float = None,
        n_core_particles: int = 8
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
            core_mass: Mass of central core (default: 0.25 * (disk_mass + bulge_mass))
            core_radius: Radius of core region (default: 0.1 * bulge_radius)
            n_core_particles: Number of core particles
        """
        super().__init__(backend, n_particles, seed)
        self.disk_radius = disk_radius
        self.bulge_radius = bulge_radius
        self.disk_mass = disk_mass
        self.bulge_mass = bulge_mass
        self.spiral_arms = spiral_arms
        total_mass = disk_mass + bulge_mass
        self.core_mass = core_mass if core_mass is not None else 0.25 * total_mass
        self.core_radius = core_radius if core_radius is not None else 0.1 * bulge_radius
        self.n_core_particles = n_core_particles
    
    @property
    def name(self) -> str:
        return "spiral"
    
    def generate(self) -> Tuple:
        """Generate spiral galaxy initial conditions with proper rotation."""
        n = self.n_particles
        n_bulge = int(n * 0.2)  # 20% in bulge
        n_disk = n - n_bulge
        
        # Use numpy RNG for proper random number generation
        rng = np.random.default_rng(self.seed)
        
        # Disk scale radius (typically ~0.2 * disk_radius for realistic galaxies)
        disk_scale_radius = self.disk_radius * 0.2
        
        positions = []
        velocities = []
        masses = []
        
        # Generate core at center
        center = np.array([0.0, 0.0, 0.0])
        core_pos, core_vel, core_mass = generate_core_particles(
            self.n_core_particles,
            self.core_radius,
            self.core_mass,
            center,
            rng
        )
        positions.extend(core_pos)
        velocities.extend(core_vel)
        masses.extend(core_mass)
        
        # Generate bulge (Hernquist profile, spherical)
        if n_bulge > 0:
            bulge_radii, bulge_theta, bulge_phi = generate_spherical_particles(
                n_bulge, self.bulge_radius, rng
            )
            
            for i in range(n_bulge):
                r = bulge_radii[i]
                theta = bulge_theta[i]
                phi = bulge_phi[i]
                
                # Convert to Cartesian
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi) * 0.1  # Flatten slightly
                
                # Calculate circular velocity from enclosed mass (including core and disk)
                M_enc_bulge = enclosed_mass_hernquist(np.array([r]), self.bulge_radius * 0.3, self.bulge_mass)[0]
                M_enc_disk = enclosed_mass_exponential(np.array([r]), disk_scale_radius, self.disk_mass)[0]
                M_enc_total = M_enc_bulge + M_enc_disk + self.core_mass
                v_mag = circular_velocity(np.array([r]), np.array([M_enc_total]))[0]
                
                # Tangential velocity in xy plane
                vx = -v_mag * np.sin(theta)
                vy = v_mag * np.cos(theta)
                vz = 0.0
                
                positions.append([x, y, z])
                velocities.append([vx, vy, vz])
                masses.append(self.bulge_mass / n_bulge)
        
        # Generate disk (exponential disk with spiral arms)
        if n_disk > 0:
            disk_radii, disk_angles = generate_exponential_disk_particles(
                n_disk, disk_scale_radius, self.disk_radius, self.disk_mass, rng
            )
            
            # Add spiral arm modulation
            for i in range(n_disk):
                r = disk_radii[i]
                theta_base = disk_angles[i]
                
                # Spiral arm phase
                arm_phase = self.spiral_arms * np.log(r / (disk_scale_radius * 0.5) + 1)
                theta = theta_base + arm_phase
                
                # Add random scatter for spiral arm width
                theta += rng.normal(0, 0.15)
                
                # Convert to Cartesian
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                z = rng.normal(0, 0.1)  # Thin disk
                
                # Calculate circular velocity from total enclosed mass (including core)
                M_enc_disk = enclosed_mass_exponential(np.array([r]), disk_scale_radius, self.disk_mass)[0]
                M_enc_bulge = enclosed_mass_hernquist(np.array([r]), self.bulge_radius * 0.3, self.bulge_mass)[0]
                M_enc_total = M_enc_disk + M_enc_bulge + self.core_mass
                
                v_mag = circular_velocity(np.array([r]), np.array([M_enc_total]))[0]
                
                # Tangential velocity (perpendicular to radius)
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
