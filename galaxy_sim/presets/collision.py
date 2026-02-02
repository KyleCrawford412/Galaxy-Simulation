"""Galaxy collision preset."""

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


class CollisionScenario(Preset):
    """Two galaxies colliding."""
    
    def __init__(
        self,
        backend: Backend,
        n_particles: int = 10000,
        seed: int = None,
        galaxy_separation: float = 20.0,
        relative_velocity: float = 0.5,
        galaxy_radius: float = 5.0,
        core_mass: float = None,
        core_radius: float = None,
        n_core_particles: int = 8
    ):
        """Initialize collision scenario.
        
        Args:
            backend: Compute backend
            n_particles: Total number of particles (split between galaxies)
            seed: Random seed
            galaxy_separation: Initial separation between galaxy centers
            relative_velocity: Relative velocity magnitude
            galaxy_radius: Radius of each galaxy
            core_mass: Mass of central core (default: 0.25 * galaxy_mass)
            core_radius: Radius of core region (default: 0.1 * galaxy_radius)
            n_core_particles: Number of core particles per galaxy
        """
        super().__init__(backend, n_particles, seed)
        self.galaxy_separation = galaxy_separation
        self.relative_velocity = relative_velocity
        self.galaxy_radius = galaxy_radius
        self.core_mass = core_mass if core_mass is not None else 0.5  # 50% of galaxy mass for better binding
        self.core_radius = core_radius if core_radius is not None else 0.1 * galaxy_radius
        self.n_core_particles = n_core_particles
    
    @property
    def name(self) -> str:
        return "collision"
    
    def generate(self) -> Tuple:
        """Generate collision scenario with two proper rotating galaxies."""
        n = self.n_particles
        n_per_galaxy = n // 2
        
        # Use numpy RNG for proper random number generation
        rng = np.random.default_rng(self.seed)
        
        # Galaxy parameters
        disk_scale_radius = self.galaxy_radius * 0.3
        bulge_radius = self.galaxy_radius * 0.4
        galaxy_mass = 1.0
        disk_mass = galaxy_mass * 0.7
        bulge_mass = galaxy_mass * 0.3
        
        positions = []
        velocities = []
        masses = []
        
        # Galaxy 1 (left, moving right)
        center1 = np.array([-self.galaxy_separation / 2, 0, 0])
        n_bulge1 = int(n_per_galaxy * 0.2)
        n_disk1 = n_per_galaxy - n_bulge1
        
        # Core of galaxy 1
        core_pos1, core_vel1, core_mass1 = generate_core_particles(
            self.n_core_particles,
            self.core_radius,
            self.core_mass,
            center1,
            rng
        )
        # Add core velocity offset for galaxy motion
        core_vel1[:, 0] += self.relative_velocity / 2
        positions.extend(core_pos1)
        velocities.extend(core_vel1)
        masses.extend(core_mass1)
        
        # Bulge of galaxy 1
        if n_bulge1 > 0:
            bulge_radii, bulge_theta, bulge_phi = generate_spherical_particles(
                n_bulge1, bulge_radius, rng
            )
            for i in range(n_bulge1):
                r = bulge_radii[i]
                theta = bulge_theta[i]
                phi = bulge_phi[i]
                
                x = center1[0] + r * np.sin(phi) * np.cos(theta)
                y = center1[1] + r * np.sin(phi) * np.sin(theta)
                z = center1[2] + r * np.cos(phi) * 0.1
                
                # For bulge particles, use total galaxy mass for circular velocity
                total_galaxy_mass = disk_mass + bulge_mass + self.core_mass
                v_mag = np.sqrt(total_galaxy_mass / max(r, 0.1))
                
                vx = self.relative_velocity / 2 - v_mag * np.sin(theta)
                vy = v_mag * np.cos(theta)
                vz = 0.0
                
                positions.append([x, y, z])
                velocities.append([vx, vy, vz])
                masses.append(bulge_mass / n_bulge1)
        
        # Disk of galaxy 1
        if n_disk1 > 0:
            disk_radii, disk_angles = generate_exponential_disk_particles(
                n_disk1, disk_scale_radius, self.galaxy_radius, disk_mass, rng
            )
            for i in range(n_disk1):
                r = disk_radii[i]
                theta = disk_angles[i]
                
                x = center1[0] + r * np.cos(theta)
                y = center1[1] + r * np.sin(theta)
                z = center1[2] + rng.normal(0, 0.1)
                
                # For stable circular orbits, use total galaxy mass
                # The core provides the primary binding, and distributed mass adds to it
                total_galaxy_mass = disk_mass + bulge_mass + self.core_mass
                # Use total mass for circular velocity (core + all distributed mass)
                v_mag = circular_velocity(np.array([r]), np.array([total_galaxy_mass]))[0]
                
                vx = self.relative_velocity / 2 - v_mag * np.sin(theta)
                vy = v_mag * np.cos(theta)
                vz = 0.0
                
                positions.append([x, y, z])
                velocities.append([vx, vy, vz])
                masses.append(disk_mass / n_disk1)
        
        # Galaxy 2 (right, moving left)
        center2 = np.array([self.galaxy_separation / 2, 0, 0])
        n_galaxy2 = n - n_per_galaxy
        n_bulge2 = int(n_galaxy2 * 0.2)
        n_disk2 = n_galaxy2 - n_bulge2
        
        # Core of galaxy 2
        core_pos2, core_vel2, core_mass2 = generate_core_particles(
            self.n_core_particles,
            self.core_radius,
            self.core_mass,
            center2,
            rng
        )
        # Add core velocity offset for galaxy motion
        core_vel2[:, 0] += -self.relative_velocity / 2
        positions.extend(core_pos2)
        velocities.extend(core_vel2)
        masses.extend(core_mass2)
        
        # Bulge of galaxy 2
        if n_bulge2 > 0:
            bulge_radii, bulge_theta, bulge_phi = generate_spherical_particles(
                n_bulge2, bulge_radius, rng
            )
            for i in range(n_bulge2):
                r = bulge_radii[i]
                theta = bulge_theta[i]
                phi = bulge_phi[i]
                
                x = center2[0] + r * np.sin(phi) * np.cos(theta)
                y = center2[1] + r * np.sin(phi) * np.sin(theta)
                z = center2[2] + r * np.cos(phi) * 0.1
                
                # For bulge particles, use total galaxy mass for circular velocity
                total_galaxy_mass = disk_mass + bulge_mass + self.core_mass
                v_mag = np.sqrt(total_galaxy_mass / max(r, 0.1))
                
                vx = -self.relative_velocity / 2 - v_mag * np.sin(theta)
                vy = v_mag * np.cos(theta)
                vz = 0.0
                
                positions.append([x, y, z])
                velocities.append([vx, vy, vz])
                masses.append(bulge_mass / n_bulge2)
        
        # Disk of galaxy 2
        if n_disk2 > 0:
            disk_radii, disk_angles = generate_exponential_disk_particles(
                n_disk2, disk_scale_radius, self.galaxy_radius, disk_mass, rng
            )
            for i in range(n_disk2):
                r = disk_radii[i]
                theta = disk_angles[i]
                
                x = center2[0] + r * np.cos(theta)
                y = center2[1] + r * np.sin(theta)
                z = center2[2] + rng.normal(0, 0.1)
                
                # For stable circular orbits, use total galaxy mass
                # The core provides the primary binding, and distributed mass adds to it
                total_galaxy_mass = disk_mass + bulge_mass + self.core_mass
                # Use total mass for circular velocity (core + all distributed mass)
                v_mag = circular_velocity(np.array([r]), np.array([total_galaxy_mass]))[0]
                
                vx = -self.relative_velocity / 2 - v_mag * np.sin(theta)
                vy = v_mag * np.cos(theta)
                vz = 0.0
                
                positions.append([x, y, z])
                velocities.append([vx, vy, vz])
                masses.append(disk_mass / n_disk2)
        
        positions = self.backend.array(positions)
        velocities = self.backend.array(velocities)
        masses = self.backend.array(masses)
        
        return positions, velocities, masses
