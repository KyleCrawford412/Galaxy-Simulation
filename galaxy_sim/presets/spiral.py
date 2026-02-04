"""Spiral galaxy preset."""

import numpy as np
from typing import Tuple
from galaxy_sim.backends.base import Backend
from galaxy_sim.presets.base import Preset
from galaxy_sim.presets.utils import (
    generate_exponential_disk_particles,
    generate_spherical_particles,
    generate_core_particles,
    enclosed_mass_hernquist,
    enclosed_mass_exponential,
    circular_velocity,
    circular_velocity_from_acceleration,
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
        n_core_particles: int = 1,
        disk_sigma_r: float = 0.08,
        disk_sigma_t: float = 0.05,
        bulge_velocity_dispersion: float = 0.6,
        use_analytic_disk: bool = True,
        use_analytic_bulge: bool = False
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
            n_core_particles: Number of core particles (default 1 for stable axisymmetric center; 8+ can cause scattering)
            disk_sigma_r: Radial velocity dispersion as fraction of v_circ (default 0.08, keeps disk stable)
            disk_sigma_t: Tangential velocity dispersion as fraction of v_circ (default 0.05)
            bulge_velocity_dispersion: Bulge random velocity scale as fraction of local v_circ (default 0.6)
        """
        super().__init__(backend, n_particles, seed)
        self.disk_radius = disk_radius
        self.disk_scale_radius = disk_radius * 0.2
        self.bulge_radius = bulge_radius
        self.disk_mass = disk_mass
        self.bulge_mass = bulge_mass
        self.spiral_arms = spiral_arms
        total_mass = disk_mass + bulge_mass
        self.core_mass = core_mass if core_mass is not None else 0.25 * total_mass
        self.core_radius = core_radius if core_radius is not None else 0.1 * bulge_radius
        # Enforce n_core_particles=1 for stability (multiple particles cause lumpy potential and scattering)
        if n_core_particles > 1:
            import warnings
            warnings.warn(
                f"n_core_particles={n_core_particles} causes instability (lumpy potential, particle scattering). "
                f"Using n_core_particles=1 for stable axisymmetric center.",
                UserWarning
            )
        self.n_core_particles = 1  # Always use 1 for stable axisymmetric center
        self.disk_sigma_r = disk_sigma_r
        self.disk_sigma_t = disk_sigma_t
        self.bulge_velocity_dispersion = bulge_velocity_dispersion
        self.use_analytic_disk = use_analytic_disk
        self.use_analytic_bulge = use_analytic_bulge
        self.min_radius_non_core = None
        self.eps_estimate = None
        self.eps_central_est = None
    
    @property
    def name(self) -> str:
        return "spiral"
    
    def generate(self) -> Tuple:
        """Generate spiral galaxy initial conditions with proper rotation."""
        n = self.n_particles
        n_bulge = int(n * 0.2)  # 20% in bulge
        if self.use_analytic_bulge:
            n_bulge = 0
        n_disk = n - n_bulge
        
        # Use numpy RNG for proper random number generation
        rng = np.random.default_rng(self.seed)
        
        # Disk scale radius (typically ~0.2 * disk_radius for realistic galaxies)
        disk_scale_radius = self.disk_radius * 0.2
        
        positions = []
        velocities = []
        masses = []
        particle_types_list = []
        
        # Central mass: single particle at origin for stable axisymmetric potential (no lumpy scattering)
        if self.n_core_particles == 1:
            positions.append([0.0, 0.0, 0.0])
            velocities.append([0.0, 0.0, 0.0])
            masses.append(self.core_mass)
            particle_types_list.append('core')
        else:
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
            particle_types_list.extend(['core'] * self.n_core_particles)
        
        # Minimum radius for non-core particles: ensure safe separation from core
        # Use larger value to account for epsilon (~0.05-0.1) and prevent collapse
        min_radius_non_core = max(0.3, 2.0 * self.core_radius)
        # Pre-estimate epsilon from spacing to enforce r_min >= 3 * eps (stability rule)
        if n_disk > 0:
            avg_spacing = self.disk_radius / max(np.sqrt(n_disk), 1.0)
            eps_pre = max(0.7 * avg_spacing, 1e-3)
            min_radius_non_core = max(min_radius_non_core, 3.0 * eps_pre)
            self.eps_central_est = eps_pre
        min_radius_width = max(0.1 * min_radius_non_core, 1e-3)
        self.min_radius_non_core = min_radius_non_core
        
        # Build bulge first so disk v_circ can use actual bulge field (no M_enc heuristic)
        bulge_pos_list = []
        bulge_vel_list = []
        bulge_mass_list = []
        if n_bulge > 0:
            bulge_radii, bulge_theta, bulge_phi = generate_spherical_particles(
                n_bulge,
                self.bulge_radius,
                rng,
                min_radius=min_radius_non_core,
                min_radius_width=min_radius_width,
            )
            for i in range(n_bulge):
                r = bulge_radii[i]
                theta = bulge_theta[i]
                phi = bulge_phi[i]
                x = r * np.sin(phi) * np.cos(theta)
                y = r * np.sin(phi) * np.sin(theta)
                z = r * np.cos(phi) * 0.1
                bulge_pos_list.append([x, y, z])
                # Bulge velocity from a_central + a_bulge_analytic (no M_enc_disk)
                M_enc_bulge = enclosed_mass_hernquist(np.array([r]), self.bulge_radius * 0.3, self.bulge_mass)[0]
                M_enc_total = M_enc_bulge + self.core_mass
                v_circ = circular_velocity(np.array([r]), np.array([M_enc_total]))[0]
                v_mag = v_circ * self.bulge_velocity_dispersion * (0.5 + np.abs(rng.standard_normal()))
                v_mag = max(v_mag, v_circ * 0.3)
                vx = -v_mag * np.sin(theta)
                vy = v_mag * np.cos(theta)
                vz = rng.normal(0, v_mag * 0.2)
                bulge_vel_list.append([vx, vy, vz])
                bulge_mass_list.append(self.bulge_mass / n_bulge)
            positions.extend(bulge_pos_list)
            velocities.extend(bulge_vel_list)
            masses.extend(bulge_mass_list)
            particle_types_list.extend(['bulge'] * n_bulge)
        bulge_pos = np.array(bulge_pos_list) if bulge_pos_list else np.empty((0, 3))
        bulge_mass = np.array(bulge_mass_list) if bulge_mass_list else np.empty(0)
        
        # Disk: v_circ from same acceleration field used in simulation (a_central + a_bulge_field; no M_enc_disk)
        if n_disk > 0:
            disk_radii, disk_angles = generate_exponential_disk_particles(
                n_disk,
                disk_scale_radius,
                self.disk_radius,
                self.disk_mass,
                rng,
                min_radius=min_radius_non_core,
                min_radius_width=min_radius_width,
            )
            theta_arr = disk_angles + self.spiral_arms * np.log(disk_radii / (disk_scale_radius * 0.5) + 1) + rng.normal(0, 0.15, n_disk)
            x_disk = disk_radii * np.cos(theta_arr)
            y_disk = disk_radii * np.sin(theta_arr)
            z_disk = rng.normal(0, 0.1, n_disk)
            disk_positions = np.column_stack([x_disk, y_disk, z_disk])
            
            # Estimate epsilon (same logic as NBodySystem._calculate_adaptive_epsilon)
            # Compute median nearest neighbor distance to match simulation epsilon
            all_positions_temp = np.array(positions + [[x, y, z] for x, y, z in zip(x_disk, y_disk, z_disk)])
            try:
                r_diff = all_positions_temp[np.newaxis, :, :] - all_positions_temp[:, np.newaxis, :]
                distances_sq = np.sum(r_diff ** 2, axis=2)
                np.fill_diagonal(distances_sq, np.inf)
                nearest_neighbor_distances_sq = np.min(distances_sq, axis=1)
                nearest_neighbor_distances = np.sqrt(nearest_neighbor_distances_sq)
                median_nn_distance = np.median(nearest_neighbor_distances)
                # Epsilon floor: use 10th percentile to protect inner particles (matches nbody.py)
                percentile_10_nn = np.percentile(nearest_neighbor_distances, 10)
                epsilon_floor = 0.5 * percentile_10_nn
            except Exception:
                center = np.mean(all_positions_temp, axis=0)
                distances_from_center = np.linalg.norm(all_positions_temp - center, axis=1)
                char_size = np.max(distances_from_center) if len(distances_from_center) > 0 else 1.0
                avg_spacing = char_size / (len(all_positions_temp) ** (1/3))
                median_nn_distance = avg_spacing
                epsilon_floor = 0.5 * avg_spacing
            
            eta = 0.7  # Default eta from NBodySystem
            eps_estimate = eta * median_nn_distance
            # Use max of: floor (for inner particles), median-based (for outer), and default
            eps_estimate = max(epsilon_floor, eps_estimate, 1e-3)
            
            self.eps_estimate = eps_estimate
            self.eps_central_est = self.eps_central_est or eps_estimate

            v_circ = circular_velocity_from_acceleration(
                disk_positions,
                central_mass=self.core_mass,
                halo_potential=getattr(self, 'halo_potential', None),
                backend=self.backend,
                source_positions=bulge_pos,
                source_masses=bulge_mass,
                G=1.0,
                eps=eps_estimate,  # Use estimated epsilon matching simulation
                use_analytic_disk=self.use_analytic_disk,
                disk_scale_radius=disk_scale_radius,
                disk_mass=self.disk_mass,
                use_analytic_bulge=self.use_analytic_bulge,
                bulge_mass=self.bulge_mass,
                bulge_scale_radius=self.bulge_radius * 0.3,
            )
            sigma_r = self.disk_sigma_r * v_circ
            v_mag = v_circ * (1.0 + rng.normal(0, self.disk_sigma_t, n_disk))
            v_rad = rng.normal(0, sigma_r, n_disk)
            # Cap tangential speed by escape speed for inner radii
            if self.eps_central_est is not None:
                r_safe = np.maximum(disk_radii, 1e-6)
                phi_central = -1.0 * self.core_mass / np.sqrt(r_safe ** 2 + self.eps_central_est ** 2)
                phi_bulge = np.zeros_like(r_safe)
                if self.use_analytic_bulge:
                    phi_bulge = -1.0 * self.bulge_mass / (r_safe + self.bulge_radius * 0.3)
                phi_disk = np.zeros_like(r_safe)
                if self.use_analytic_disk:
                    M_enc_disk = enclosed_mass_exponential(r_safe, disk_scale_radius, self.disk_mass)
                    phi_disk = -1.0 * M_enc_disk / r_safe
                phi_total = phi_central + phi_bulge + phi_disk
                v_escape = np.sqrt(2.0 * np.abs(phi_total))
                v_mag = np.minimum(v_mag, 0.9 * v_escape)
            cos_t, sin_t = np.cos(theta_arr), np.sin(theta_arr)
            vx = -v_mag * sin_t + v_rad * cos_t
            vy = v_mag * cos_t + v_rad * sin_t
            vz = rng.normal(0, 0.02 * v_circ, n_disk)
            for i in range(n_disk):
                positions.append([x_disk[i], y_disk[i], z_disk[i]])
                velocities.append([vx[i], vy[i], vz[i]])
                masses.append(self.disk_mass / n_disk)
            particle_types_list.extend(['disk'] * n_disk)
        
        self.particle_types = np.array(particle_types_list, dtype=object)
        positions = self.backend.array(positions)
        velocities = self.backend.array(velocities)
        masses = self.backend.array(masses)
        return positions, velocities, masses
