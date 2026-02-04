"""Galaxy collision preset."""

import numpy as np
from typing import Tuple
from galaxy_sim.backends.base import Backend
from galaxy_sim.presets.base import Preset
from galaxy_sim.presets.utils import (
    generate_exponential_disk_particles,
    generate_spherical_particles,
    generate_core_particles,
    enclosed_mass_hernquist,
    circular_velocity,
    circular_velocity_from_acceleration,
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
        
        # Bulge of galaxy 1 (build first so disk v_circ uses actual bulge field)
        bulge1_pos_list = []
        bulge1_vel_list = []
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
                bulge1_pos_list.append([x, y, z])
                M_enc_bulge = enclosed_mass_hernquist(np.array([r]), bulge_radius * 0.3, bulge_mass)[0]
                M_enc_total = M_enc_bulge + self.core_mass
                v_mag = circular_velocity(np.array([r]), np.array([M_enc_total]))[0]
                vx = self.relative_velocity / 2 - v_mag * np.sin(theta)
                vy = v_mag * np.cos(theta)
                vz = 0.0
                bulge1_vel_list.append([vx, vy, vz])
                positions.append([x, y, z])
                velocities.append([vx, vy, vz])
                masses.append(bulge_mass / n_bulge1)
        bulge1_pos = np.array(bulge1_pos_list) - center1 if bulge1_pos_list else np.empty((0, 3))
        bulge1_mass = np.full(len(bulge1_pos_list), bulge_mass / n_bulge1) if bulge1_pos_list else np.empty(0)
        
        # Disk of galaxy 1: v_circ from a_central + a_bulge_field (no M_enc_disk)
        if n_disk1 > 0:
            disk_radii, disk_angles = generate_exponential_disk_particles(
                n_disk1, disk_scale_radius, self.galaxy_radius, disk_mass, rng
            )
            x_d1 = center1[0] + disk_radii * np.cos(disk_angles)
            y_d1 = center1[1] + disk_radii * np.sin(disk_angles)
            z_d1 = center1[2] + rng.normal(0, 0.1, n_disk1)
            disk1_pos_abs = np.column_stack([x_d1, y_d1, z_d1])
            disk1_pos_rel = disk1_pos_abs - center1
            v_circ = circular_velocity_from_acceleration(
                disk1_pos_rel,
                central_mass=self.core_mass,
                halo_potential=None,
                backend=self.backend,
                source_positions=bulge1_pos,
                source_masses=bulge1_mass,
                G=1.0,
                eps=1e-6,
                use_analytic_disk=False,
            )
            vx_d1 = self.relative_velocity / 2 - v_circ * np.sin(disk_angles)
            vy_d1 = v_circ * np.cos(disk_angles)
            vz_d1 = np.zeros(n_disk1)
            for i in range(n_disk1):
                positions.append([x_d1[i], y_d1[i], z_d1[i]])
                velocities.append([vx_d1[i], vy_d1[i], vz_d1[i]])
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
        
        # Bulge of galaxy 2 (build first so disk v_circ uses actual bulge field)
        bulge2_pos_list = []
        bulge2_vel_list = []
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
                bulge2_pos_list.append([x, y, z])
                M_enc_bulge = enclosed_mass_hernquist(np.array([r]), bulge_radius * 0.3, bulge_mass)[0]
                M_enc_total = M_enc_bulge + self.core_mass
                v_mag = circular_velocity(np.array([r]), np.array([M_enc_total]))[0]
                vx = -self.relative_velocity / 2 - v_mag * np.sin(theta)
                vy = v_mag * np.cos(theta)
                vz = 0.0
                bulge2_vel_list.append([vx, vy, vz])
                positions.append([x, y, z])
                velocities.append([vx, vy, vz])
                masses.append(bulge_mass / n_bulge2)
        bulge2_pos = np.array(bulge2_pos_list) - center2 if bulge2_pos_list else np.empty((0, 3))
        bulge2_mass = np.full(len(bulge2_pos_list), bulge_mass / n_bulge2) if bulge2_pos_list else np.empty(0)
        
        # Disk of galaxy 2: v_circ from a_central + a_bulge_field (no M_enc_disk)
        if n_disk2 > 0:
            disk_radii, disk_angles = generate_exponential_disk_particles(
                n_disk2, disk_scale_radius, self.galaxy_radius, disk_mass, rng
            )
            x_d2 = center2[0] + disk_radii * np.cos(disk_angles)
            y_d2 = center2[1] + disk_radii * np.sin(disk_angles)
            z_d2 = center2[2] + rng.normal(0, 0.1, n_disk2)
            disk2_pos_rel = np.column_stack([x_d2 - center2[0], y_d2 - center2[1], z_d2 - center2[2]])
            v_circ = circular_velocity_from_acceleration(
                disk2_pos_rel,
                central_mass=self.core_mass,
                halo_potential=None,
                backend=self.backend,
                source_positions=bulge2_pos,
                source_masses=bulge2_mass,
                G=1.0,
                eps=1e-6,
                use_analytic_disk=False,
            )
            vx_d2 = -self.relative_velocity / 2 - v_circ * np.sin(disk_angles)
            vy_d2 = v_circ * np.cos(disk_angles)
            vz_d2 = np.zeros(n_disk2)
            for i in range(n_disk2):
                positions.append([x_d2[i], y_d2[i], z_d2[i]])
                velocities.append([vx_d2[i], vy_d2[i], vz_d2[i]])
                masses.append(disk_mass / n_disk2)
        
        positions = self.backend.array(positions)
        velocities = self.backend.array(velocities)
        masses = self.backend.array(masses)
        
        return positions, velocities, masses
