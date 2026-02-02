"""Multi-component galaxy presets with disk, bulge, and halo."""

import numpy as np
from typing import Tuple, Optional
from galaxy_sim.backends.base import Backend
from galaxy_sim.presets.base import Preset


class MultiComponentGalaxy(Preset):
    """Galaxy with disk, bulge, and optional halo components.
    
    Disk: Thin, rotating, exponential radial distribution
    Bulge: Compact, high velocity dispersion, low rotation
    Halo: Large radius, random velocities (or analytic potential)
    """
    
    def __init__(
        self,
        backend: Backend,
        n_particles: int = 1000,
        seed: int = None,
        # Disk parameters
        n_disk: int = None,  # Number of disk particles (default: 70% of total)
        disk_scale_radius: float = 10.0,  # Exponential disk scale radius
        disk_mass: float = 1000.0,  # Total disk mass
        disk_velocity_factor: float = 1.0,  # Multiplier for circular velocity
        disk_thickness: float = 0.5,  # Vertical scale height (z-direction)
        # Bulge parameters
        n_bulge: int = None,  # Number of bulge particles (default: 20% of total)
        bulge_scale_radius: float = 2.0,  # Bulge scale radius (Plummer or Gaussian)
        bulge_mass: float = 500.0,  # Total bulge mass
        bulge_velocity_dispersion: float = 0.5,  # Velocity dispersion (random velocities)
        # Halo parameters
        n_halo: int = None,  # Number of halo particles (default: 10% of total, or 0 if using analytic)
        halo_scale_radius: float = 20.0,  # Halo scale radius
        halo_mass: float = 2000.0,  # Total halo mass
        halo_velocity_dispersion: float = 0.3,  # Halo velocity dispersion
        use_analytic_halo: bool = False,  # Use analytic halo potential instead of particles
        halo_v0: float = 2.0,  # Analytic halo circular velocity
        halo_rc: float = 5.0,  # Analytic halo core radius
        # Central mass
        central_mass: float = 100.0,  # Central black hole / massive core
        # Mass variation
        vary_masses: bool = False,  # If True, use lognormal mass distribution
        mass_lognormal_sigma: float = 0.2,  # Standard deviation for lognormal mass distribution
        # Halo potential (for velocity calculation)
        halo_potential = None,  # HaloPotential instance for computing circular velocities
        # Other
        G: float = 1.0,  # Gravitational constant
        epsilon: float = 0.1,  # Softening parameter
        velocity_noise: float = 0.05,  # Random noise in velocities (5%)
    ):
        super().__init__(backend, n_particles, seed)
        
        # Set default particle counts if not specified
        if n_disk is None:
            n_disk = int(n_particles * 0.75)  # More disk particles
        if n_bulge is None:
            n_bulge = int(n_particles * 0.1)  # Fewer bulge particles (was 0.2)
        if n_halo is None:
            n_halo = int(n_particles * 0.1) if not use_analytic_halo else 0
        
        # Ensure we have at least one particle per component
        n_disk = max(1, n_disk)
        n_bulge = max(1, n_bulge)
        n_halo = max(0, n_halo)
        
        # If no central mass specified, use a moderate one to provide clean center
        # But not too large to avoid over-binding
        if central_mass == 0.0:
            central_mass = max(disk_mass * 0.15, 100.0)  # 15% of disk mass or minimum 100
        self.central_mass = central_mass
        
        # Adjust total to match actual counts
        # Always add central mass particle
        self.n_particles = n_disk + n_bulge + n_halo + 1
        
        # Store parameters
        self.n_disk = n_disk
        self.n_bulge = n_bulge
        self.n_halo = n_halo
        self.disk_scale_radius = disk_scale_radius
        self.disk_mass = disk_mass
        self.disk_velocity_factor = disk_velocity_factor
        self.disk_thickness = disk_thickness
        self.bulge_scale_radius = bulge_scale_radius
        self.bulge_mass = bulge_mass
        self.bulge_velocity_dispersion = bulge_velocity_dispersion
        self.halo_scale_radius = halo_scale_radius
        self.halo_mass = halo_mass
        self.halo_velocity_dispersion = halo_velocity_dispersion
        self.use_analytic_halo = use_analytic_halo
        self.halo_v0 = halo_v0
        self.halo_rc = halo_rc
        self.vary_masses = vary_masses
        self.mass_lognormal_sigma = mass_lognormal_sigma
        self.halo_potential = halo_potential
        self.G = G
        self.epsilon = epsilon
        self.velocity_noise = velocity_noise
    
    @property
    def name(self) -> str:
        return "multi_component"
    
    def _generate_disk_particles(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate disk particles with exponential radial distribution.
        
        Disk radius: r = -R_d * log(1-u) (exponential disk)
        Angle: uniform in [0, 2π]
        """
        # Exponential disk: r = -R_d * log(1-u) where u ~ Uniform(0,1)
        u = rng.uniform(0, 1, self.n_disk)
        r = -self.disk_scale_radius * np.log(1 - u)
        
        # Uniform angles
        theta = rng.uniform(0, 2 * np.pi, self.n_disk)
        
        # Convert to Cartesian (thin disk with some vertical thickness)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = rng.normal(0, self.disk_thickness, self.n_disk)
        
        positions = np.column_stack([x, y, z])
        
        # Tangential velocities for rotation
        # Compute circular velocity from actual potential (halo + central mass + disk)
        # v_circ = sqrt(r * |a_r|) where a_r is radial acceleration
        r_min_orbital = max(self.disk_scale_radius * 0.3, 2.0)  # Minimum orbital radius
        r_safe = np.maximum(r, r_min_orbital)
        
        # Vectorized computation of radial acceleration
        # Acceleration from central mass: a = -GM/r² (radial, inward)
        a_central = self.G * self.central_mass / (r_safe ** 2)  # Positive magnitude (inward direction)
        
        # Acceleration from disk (approximate enclosed mass)
        M_enc_disk = self.disk_mass * (1 - np.exp(-r_safe / self.disk_scale_radius))
        a_disk = self.G * M_enc_disk / (r_safe ** 2)  # Positive magnitude
        
        # Acceleration from halo (if present) - compute for all radii at once
        a_halo = np.zeros_like(r_safe)
        if self.halo_potential is not None and self.halo_potential.enabled:
            # Create test positions along x-axis for each radius
            test_positions = np.column_stack([r_safe, np.zeros(len(r_safe)), np.zeros(len(r_safe))])
            test_positions_backend = self.backend.array(test_positions)
            a_halo_vec = self.halo_potential.compute_acceleration(test_positions_backend, self.backend)
            if a_halo_vec is not None:
                a_halo_np = np.asarray(self.backend.to_numpy(a_halo_vec))
                # Radial component: dot product with unit radial vector [1, 0, 0]
                # Since positions are along x-axis, acceleration x-component is radial
                a_halo = np.abs(a_halo_np[:, 0])  # Magnitude of radial acceleration (inward)
        
        # Total radial acceleration magnitude (all components point inward)
        a_r_total = a_central + a_disk + a_halo
        
        # Circular velocity: v_circ = sqrt(r * |a_r|)
        v_circ = np.sqrt(r_safe * a_r_total) * self.disk_velocity_factor
        
        # For very small r, use a larger safety radius
        v_circ = np.where(r < r_min_orbital,
                         np.sqrt(r_min_orbital * (self.G * (self.central_mass + self.disk_mass * 0.1) / (r_min_orbital ** 2))) * self.disk_velocity_factor,
                         v_circ)
        
        # Add noise
        v_circ *= (1 + rng.normal(0, self.velocity_noise, self.n_disk))
        
        # Tangential direction: perpendicular to radius
        vx = -v_circ * np.sin(theta)  # -v * sin(theta) = -v * y/r
        vy = v_circ * np.cos(theta)   # v * cos(theta) = v * x/r
        vz = rng.normal(0, 0.1 * self.disk_thickness, self.n_disk)  # Small vertical motion
        
        velocities = np.column_stack([vx, vy, vz])
        
        # Masses: uniform by default, or lognormal if vary_masses is True
        if self.vary_masses:
            # Lognormal distribution: log(m) ~ N(0, sigma²), mean = exp(sigma²/2)
            # To get mean = 1.0, we need to adjust
            log_masses = rng.normal(0, self.mass_lognormal_sigma, self.n_disk)
            masses_raw = np.exp(log_masses)
            # Normalize to get correct total mass
            masses = masses_raw * (self.disk_mass / np.sum(masses_raw))
        else:
            # Uniform masses
            mass_per_particle = self.disk_mass / self.n_disk
            masses = np.full(self.n_disk, mass_per_particle)
        
        return positions, velocities, masses
    
    def _generate_bulge_particles(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate bulge particles with compact distribution (Plummer sphere).
        
        Bulge: small radii, high velocity dispersion, low rotation
        Made less dense by increasing minimum radius and scale radius
        """
        # Plummer sphere distribution: r follows (1 + r²/a²)^(-5/2)
        # Sample using inverse transform: r = a * sqrt(u^(-2/3) - 1) where u ~ Uniform(0,1)
        # Use larger scale radius and minimum radius to avoid dense blob
        effective_scale = self.bulge_scale_radius * 2.0  # Make bulge even more extended
        u = rng.uniform(0, 1, self.n_bulge)
        r = effective_scale * np.sqrt(np.power(u, -2/3) - 1)
        # Add larger minimum radius to keep bulge particles away from center
        # This prevents the "mushed" center effect
        r_min = max(self.bulge_scale_radius * 1.0, 3.0)  # Minimum distance from center (increased)
        r = np.maximum(r, r_min)
        r = np.clip(r, r_min, 5 * effective_scale)  # Cap at 5 scale radii
        
        # Uniform on sphere
        cos_theta = rng.uniform(-1, 1, self.n_bulge)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        phi = rng.uniform(0, 2 * np.pi, self.n_bulge)
        
        x = r * sin_theta * np.cos(phi)
        y = r * sin_theta * np.sin(phi)
        z = r * cos_theta
        
        positions = np.column_stack([x, y, z])
        
        # Random velocities (high dispersion, low rotation)
        # Compute velocities from actual potential to achieve better virial balance
        r_safe = np.maximum(r, 3.0)  # Minimum radius for velocity calculation
        
        # Vectorized computation of circular velocity from actual potential
        # Acceleration from central mass
        a_central = self.G * self.central_mass / (r_safe ** 2)
        
        # Acceleration from bulge (approximate enclosed mass for Plummer)
        # For Plummer: M_enc ≈ M * r²/(r+a)²
        M_enc_bulge = self.bulge_mass * (r_safe ** 2) / ((r_safe + self.bulge_scale_radius) ** 2)
        a_bulge = self.G * M_enc_bulge / (r_safe ** 2)
        
        # Acceleration from halo (if present) - compute for all radii at once
        a_halo = np.zeros_like(r_safe)
        if self.halo_potential is not None and self.halo_potential.enabled:
            test_positions = np.column_stack([r_safe, np.zeros(len(r_safe)), np.zeros(len(r_safe))])
            test_positions_backend = self.backend.array(test_positions)
            a_halo_vec = self.halo_potential.compute_acceleration(test_positions_backend, self.backend)
            if a_halo_vec is not None:
                a_halo_np = np.asarray(self.backend.to_numpy(a_halo_vec))
                a_halo = np.abs(a_halo_np[:, 0])  # Radial component
        
        # Total radial acceleration
        a_r_total = a_central + a_bulge + a_halo
        
        # Circular velocity: v_circ = sqrt(r * a_r)
        v_circ = np.sqrt(r_safe * a_r_total)
        
        # For virialized spherical system, velocity dispersion σ ≈ v_circ
        # Use dispersion parameter as a multiplier: v ≈ σ_b * v_circ
        # But ensure minimum is at least 0.7 * v_circ for stability
        v_disp_base = v_circ * self.bulge_velocity_dispersion
        # Add random variation around this base
        v_disp_factor = rng.normal(1.0, 0.3, self.n_bulge)  # 30% variation
        v_mag = np.maximum(v_disp_base * np.abs(v_disp_factor), v_circ * 0.7)  # At least 70% of circular velocity
        
        # Random direction (mostly isotropic, some rotation)
        v_cos_theta = rng.uniform(-1, 1, self.n_bulge)
        v_sin_theta = np.sqrt(1 - v_cos_theta ** 2)
        v_phi = rng.uniform(0, 2 * np.pi, self.n_bulge)
        
        # Add some tangential component (10% rotation)
        rotation_factor = 0.1
        vx = v_mag * (v_sin_theta * np.cos(v_phi) + rotation_factor * (-np.sin(phi)))
        vy = v_mag * (v_sin_theta * np.sin(v_phi) + rotation_factor * np.cos(phi))
        vz = v_mag * v_cos_theta
        
        velocities = np.column_stack([vx, vy, vz])
        
        # Masses: uniform by default, or lognormal if vary_masses is True
        if self.vary_masses:
            # Lognormal distribution
            log_masses = rng.normal(0, self.mass_lognormal_sigma, self.n_bulge)
            masses_raw = np.exp(log_masses)
            # Normalize to get correct total mass
            masses = masses_raw * (self.bulge_mass / np.sum(masses_raw))
        else:
            # Uniform masses
            mass_per_particle = self.bulge_mass / self.n_bulge
            masses = np.full(self.n_bulge, mass_per_particle)
        
        return positions, velocities, masses
    
    def _generate_halo_particles(self, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate halo particles with large radius and random velocities."""
        if self.n_halo == 0:
            return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)
        
        # Halo: large radii, extended distribution
        # Use NFW-like or isothermal-like distribution
        # Simple: uniform in log space from r_min to r_max
        r_min = 5.0
        r_max = self.halo_scale_radius * 3
        log_r = rng.uniform(np.log(r_min), np.log(r_max), self.n_halo)
        r = np.exp(log_r)
        
        # Uniform on sphere
        cos_theta = rng.uniform(-1, 1, self.n_halo)
        sin_theta = np.sqrt(1 - cos_theta ** 2)
        phi = rng.uniform(0, 2 * np.pi, self.n_halo)
        
        x = r * sin_theta * np.cos(phi)
        y = r * sin_theta * np.sin(phi)
        z = r * cos_theta
        
        positions = np.column_stack([x, y, z])
        
        # Random velocities (low dispersion, mostly random)
        v_mag = rng.normal(0, self.halo_velocity_dispersion, self.n_halo)
        v_mag = np.abs(v_mag)
        
        # Random direction (isotropic)
        v_cos_theta = rng.uniform(-1, 1, self.n_halo)
        v_sin_theta = np.sqrt(1 - v_cos_theta ** 2)
        v_phi = rng.uniform(0, 2 * np.pi, self.n_halo)
        
        vx = v_mag * v_sin_theta * np.cos(v_phi)
        vy = v_mag * v_sin_theta * np.sin(v_phi)
        vz = v_mag * v_cos_theta
        
        velocities = np.column_stack([vx, vy, vz])
        
        # Masses: uniform by default, or lognormal if vary_masses is True
        if self.vary_masses:
            # Lognormal distribution
            log_masses = rng.normal(0, self.mass_lognormal_sigma, self.n_halo)
            masses_raw = np.exp(log_masses)
            # Normalize to get correct total mass
            masses = masses_raw * (self.halo_mass / np.sum(masses_raw))
        else:
            # Uniform masses
            mass_per_particle = self.halo_mass / self.n_halo
            masses = np.full(self.n_halo, mass_per_particle)
        
        return positions, velocities, masses
    
    def generate(self) -> Tuple:
        """Generate multi-component galaxy.
        
        Returns:
            Tuple of (positions, velocities, masses, particle_types)
            where particle_types is an array of 'disk', 'bulge', 'halo', 'core' for each particle
        """
        rng = np.random.default_rng(self.seed)
        
        all_positions = []
        all_velocities = []
        all_masses = []
        particle_types = []  # Track particle types for virialization
        
        # Central mass particle (always add for clean center)
        # This provides a point mass that particles orbit around
        all_positions.append([0.0, 0.0, 0.0])
        all_velocities.append([0.0, 0.0, 0.0])
        all_masses.append(self.central_mass)
        particle_types.append('core')
        
        # Generate components
        disk_pos, disk_vel, disk_mass = self._generate_disk_particles(rng)
        bulge_pos, bulge_vel, bulge_mass = self._generate_bulge_particles(rng)
        halo_pos, halo_vel, halo_mass = self._generate_halo_particles(rng)
        
        # Combine all components with particle type tracking
        all_positions_list = []
        all_velocities_list = []
        all_masses_list = []
        
        # Add central mass (already added above)
        all_positions_list.append(np.array([[0.0, 0.0, 0.0]]))
        all_velocities_list.append(np.array([[0.0, 0.0, 0.0]]))
        all_masses_list.append(np.array([self.central_mass]))
        
        # Add disk particles
        if len(disk_pos) > 0:
            all_positions_list.append(disk_pos)
            all_velocities_list.append(disk_vel)
            all_masses_list.append(disk_mass)
            particle_types.extend(['disk'] * len(disk_mass))
        
        # Add bulge particles
        if len(bulge_pos) > 0:
            all_positions_list.append(bulge_pos)
            all_velocities_list.append(bulge_vel)
            all_masses_list.append(bulge_mass)
            particle_types.extend(['bulge'] * len(bulge_mass))
        
        # Add halo particles (if not using analytic halo)
        if not self.use_analytic_halo and len(halo_pos) > 0:
            all_positions_list.append(halo_pos)
            all_velocities_list.append(halo_vel)
            all_masses_list.append(halo_mass)
            particle_types.extend(['halo'] * len(halo_mass))
        
        # Stack all particles
        positions = np.vstack(all_positions_list)
        velocities = np.vstack(all_velocities_list)
        masses = np.concatenate(all_masses_list)
        particle_types_array = np.array(particle_types, dtype=object)
        
        # Store particle types as attribute for virialization
        self.particle_types = particle_types_array
        
        return self.backend.array(positions), self.backend.array(velocities), self.backend.array(masses)


class SpiralDiskGalaxy(MultiComponentGalaxy):
    """Spiral galaxy with prominent disk, small bulge, no halo particles."""
    
    def __init__(
        self,
        backend: Backend,
        n_particles: int = 1000,
        seed: int = None,
        **kwargs
    ):
        # Default: 85% disk, 10% bulge, no halo particles
        # Central mass provides clean center
        kwargs.setdefault('n_disk', int(n_particles * 0.85))
        kwargs.setdefault('n_bulge', int(n_particles * 0.1))  # Reduced from 0.15
        kwargs.setdefault('n_halo', 0)
        kwargs.setdefault('use_analytic_halo', True)
        kwargs.setdefault('disk_scale_radius', 15.0)
        kwargs.setdefault('bulge_scale_radius', 4.0)  # Increased further to make less dense
        kwargs.setdefault('central_mass', 150.0)  # Moderate central mass (reduced from 200)
        kwargs.setdefault('bulge_velocity_dispersion', 0.8)  # Higher dispersion for stability
        super().__init__(backend, n_particles, seed, **kwargs)
    
    @property
    def name(self) -> str:
        return "spiral_disk"


class DiskPlusBulgeGalaxy(MultiComponentGalaxy):
    """Galaxy with disk and bulge, no halo.
    
    Realistic galaxy structure with:
    - Disk: exponential radial distribution r = -R_d * log(1-u)
    - Bulge: compact Plummer distribution
    - Configurable parameters: N_disk, N_bulge, R_d, R_b, σ_b
    """
    
    def __init__(
        self,
        backend: Backend,
        n_particles: int = 1000,
        seed: int = None,
        # Explicit parameter names for clarity
        N_disk: int = None,
        N_bulge: int = None,
        R_d: float = None,  # Disk scale radius
        R_b: float = None,  # Bulge scale radius
        sigma_b: float = None,  # Bulge velocity dispersion
        **kwargs
    ):
        # Map explicit parameters to internal names
        if N_disk is not None:
            kwargs['n_disk'] = N_disk
        if N_bulge is not None:
            kwargs['n_bulge'] = N_bulge
        if R_d is not None:
            kwargs['disk_scale_radius'] = R_d
        if R_b is not None:
            kwargs['bulge_scale_radius'] = R_b
        if sigma_b is not None:
            kwargs['bulge_velocity_dispersion'] = sigma_b
        
        # Default: 75% disk, 15% bulge, no halo (reduced bulge)
        kwargs.setdefault('n_disk', int(n_particles * 0.75))
        kwargs.setdefault('n_bulge', int(n_particles * 0.15))  # Reduced from 0.3
        kwargs.setdefault('n_halo', 0)
        kwargs.setdefault('use_analytic_halo', False)
        kwargs.setdefault('disk_scale_radius', 15.0)  # R_d default
        kwargs.setdefault('bulge_scale_radius', 4.0)  # R_b default
        kwargs.setdefault('central_mass', 150.0)  # Moderate central mass
        kwargs.setdefault('bulge_velocity_dispersion', 0.8)  # σ_b default
        kwargs.setdefault('vary_masses', False)  # Uniform masses by default
        kwargs.setdefault('mass_lognormal_sigma', 0.2)  # Lognormal sigma if vary_masses=True
        super().__init__(backend, n_particles, seed, **kwargs)
    
    @property
    def name(self) -> str:
        return "disk_plus_bulge"
