"""Utility functions for galaxy generation."""

import numpy as np
from typing import Tuple, Optional


def exponential_disk_profile(r: np.ndarray, scale_radius: float, total_mass: float) -> np.ndarray:
    """Generate exponential disk surface density profile.
    
    Σ(r) = Σ₀ * exp(-r / R_d)
    where Σ₀ is chosen so that total mass is correct.
    
    Args:
        r: Radial distances
        scale_radius: Disk scale radius R_d
        total_mass: Total disk mass
        
    Returns:
        Surface density at each radius
    """
    # Normalization constant: M = 2π * Σ₀ * R_d²
    sigma_0 = total_mass / (2 * np.pi * scale_radius ** 2)
    return sigma_0 * np.exp(-r / scale_radius)


def enclosed_mass_exponential(r: np.ndarray, scale_radius: float, total_mass: float) -> np.ndarray:
    """Calculate enclosed mass for exponential disk.
    
    M_enc(r) = M * (1 - (1 + r/R_d) * exp(-r/R_d))
    
    Args:
        r: Radial distances
        scale_radius: Disk scale radius R_d
        total_mass: Total disk mass
        
    Returns:
        Enclosed mass at each radius
    """
    x = r / scale_radius
    return total_mass * (1 - (1 + x) * np.exp(-x))


def circular_velocity(r: np.ndarray, enclosed_mass: np.ndarray, G: float = 1.0) -> np.ndarray:
    """Calculate circular velocity from enclosed mass.
    
    v_circ(r) = sqrt(G * M_enc(r) / r)
    
    Args:
        r: Radial distances
        enclosed_mass: Enclosed mass at each radius
        G: Gravitational constant
        
    Returns:
        Circular velocity at each radius
    """
    # Avoid division by zero
    r_safe = np.maximum(r, 1e-6)
    return np.sqrt(G * enclosed_mass / r_safe)


def acceleration_from_particles(
    target_positions: np.ndarray,
    source_positions: np.ndarray,
    source_masses: np.ndarray,
    G: float = 1.0,
    eps: float = 1e-6
) -> np.ndarray:
    """Compute gravitational acceleration at target positions from source particles.
    
    Uses Plummer softening: a_i = sum_j G * m_j * (r_j - r_i) / (|r_ij|^2 + eps^2)^(3/2).
    Matches the force law used in the simulation.
    
    Args:
        target_positions: (n, dim) positions where acceleration is evaluated
        source_positions: (m, dim) positions of source particles
        source_masses: (m,) masses of source particles
        G: Gravitational constant
        eps: Softening length (same as simulation eps0)
        
    Returns:
        (n, dim) acceleration vectors at each target (inward = negative direction from target to source)
    """
    n_target = target_positions.shape[0]
    n_source = source_positions.shape[0]
    dim = target_positions.shape[1]
    if n_source == 0:
        return np.zeros((n_target, dim))
    # r_diff[i,j] = source_j - target_i  (direction from target i to source j)
    # Shape (n_target, n_source, dim)
    r_diff = source_positions[np.newaxis, :, :] - target_positions[:, np.newaxis, :]
    r_sq = np.sum(r_diff ** 2, axis=2)  # (n_target, n_source)
    r_soft_cubed = (r_sq + eps ** 2) ** 1.5
    # a_i = sum_j G * m_j * (r_j - r_i) / r_soft_cubed_ij
    # (n_target, n_source, 1) * (n_target, n_source, dim) -> sum over axis=1 -> (n_target, dim)
    acc = np.sum(G * source_masses[np.newaxis, :, np.newaxis] * r_diff / r_soft_cubed[:, :, np.newaxis], axis=1)
    return acc


def radial_acceleration_magnitude(positions: np.ndarray, acceleration_vectors: np.ndarray) -> np.ndarray:
    """Radial (inward) component magnitude of acceleration at each position.
    
    a_r = - (a_vec · r_hat) with r_hat = position/|position| outward; returns positive = inward.
    
    Args:
        positions: (n, dim) positions
        acceleration_vectors: (n, dim) acceleration at each position
        
    Returns:
        (n,) radial acceleration magnitude (inward positive)
    """
    r = np.linalg.norm(positions, axis=1)
    r_safe = np.maximum(r, 1e-10)
    r_hat = positions / r_safe[:, np.newaxis]  # outward
    # Inward component: - (a · r_hat); we want magnitude (positive)
    a_radial = -np.sum(acceleration_vectors * r_hat, axis=1)
    return np.maximum(a_radial, 0.0)  # clamp to non-negative


def circular_velocity_from_acceleration(
    positions: np.ndarray,
    central_mass: float,
    halo_potential: Optional[object],
    backend: Optional[object],
    source_positions: np.ndarray,
    source_masses: np.ndarray,
    G: float = 1.0,
    eps: float = 1e-6,
    use_analytic_disk: bool = False,
    disk_scale_radius: float = None,
    disk_mass: float = None,
    use_analytic_bulge: bool = False,
    bulge_mass: float = None,
    bulge_scale_radius: float = None
) -> np.ndarray:
    """Compute v_circ at positions from the same acceleration field used in the simulation.
    
    a_total = a_central + a_halo + a_bulge_field (+ optional analytic disk).
    No M_enc_disk heuristic unless use_analytic_disk is True (analytic disk in dynamics).
    
    Args:
        positions: (n, dim) positions (e.g. disk particle positions)
        central_mass: Point mass at origin (core)
        halo_potential: HaloPotential instance or None
        backend: Backend for halo_potential.compute_acceleration
        source_positions: (m, dim) e.g. bulge particle positions
        source_masses: (m,) e.g. bulge masses
        G: Gravitational constant
        eps: Softening (match simulation eps0)
        use_analytic_disk: If True, add analytic disk radial acceleration
        disk_scale_radius: For analytic disk (required if use_analytic_disk)
        disk_mass: For analytic disk (required if use_analytic_disk)
        
    Returns:
        (n,) circular velocity at each position: v_circ = sqrt(r * a_r_total)
    """
    r = np.linalg.norm(positions, axis=1)
    r_safe = np.maximum(r, 1e-6)
    
    # a_central with Plummer softening (matches simulation force law: F = G*m1*m2*r / (r^2 + eps^2)^1.5)
    # So a = F/m = G*M*r / (r^2 + eps^2)^1.5 (inward positive)
    r_sq = r_safe ** 2
    r_soft_cubed = (r_sq + eps ** 2) ** 1.5
    a_central = G * central_mass * r_safe / r_soft_cubed
    
    # a_halo from HaloPotential (radial magnitude)
    a_halo = np.zeros_like(r_safe)
    if halo_potential is not None and getattr(halo_potential, 'enabled', False) and backend is not None:
        acc_halo = halo_potential.compute_acceleration(backend.array(positions), backend)
        if acc_halo is not None:
            acc_halo_np = np.asarray(backend.to_numpy(acc_halo))
            a_halo = radial_acceleration_magnitude(positions, acc_halo_np)
    
    # a_bulge_field = acceleration from source particles (e.g. bulge)
    a_sources = np.zeros_like(r_safe)
    if source_positions is not None and source_masses is not None and len(source_positions) > 0:
        acc_sources = acceleration_from_particles(positions, source_positions, source_masses, G, eps)
        a_sources = radial_acceleration_magnitude(positions, acc_sources)
    # Optional analytic bulge (Hernquist-like; a = G*M/(r+a)^2)
    a_bulge_analytic = np.zeros_like(r_safe)
    if use_analytic_bulge and bulge_mass is not None and bulge_scale_radius is not None:
        a_bulge_analytic = G * bulge_mass / ((r_safe + bulge_scale_radius) ** 2)
    
    # Optional analytic disk (only if used in dynamics)
    a_disk = np.zeros_like(r_safe)
    if use_analytic_disk and disk_scale_radius is not None and disk_mass is not None:
        M_enc_disk = enclosed_mass_exponential(r_safe, disk_scale_radius, disk_mass)
        a_disk = G * M_enc_disk / (r_safe ** 2)
    
    a_r_total = a_central + a_halo + a_sources + a_bulge_analytic + a_disk
    v_circ = np.sqrt(r_safe * a_r_total)
    return v_circ


def hernquist_bulge_profile(r: np.ndarray, scale_radius: float, total_mass: float) -> np.ndarray:
    """Generate Hernquist profile for bulge (spherical).
    
    ρ(r) = (M * a) / (2π * r * (r + a)³)
    
    Args:
        r: Radial distances
        scale_radius: Scale radius 'a'
        total_mass: Total bulge mass
        
    Returns:
        Density at each radius
    """
    r_safe = np.maximum(r, 1e-6)
    return (total_mass * scale_radius) / (2 * np.pi * r_safe * (r_safe + scale_radius) ** 3)


def enclosed_mass_hernquist(r: np.ndarray, scale_radius: float, total_mass: float) -> np.ndarray:
    """Calculate enclosed mass for Hernquist profile.
    
    M_enc(r) = M * r² / (r + a)²
    
    Args:
        r: Radial distances
        scale_radius: Scale radius 'a'
        total_mass: Total bulge mass
        
    Returns:
        Enclosed mass at each radius
    """
    r_safe = np.maximum(r, 1e-6)
    return total_mass * (r_safe ** 2) / ((r_safe + scale_radius) ** 2)


def generate_exponential_disk_particles(
    n_particles: int,
    scale_radius: float,
    disk_radius: float,
    total_mass: float,
    rng: np.random.Generator,
    min_radius: float = 0.0,
    min_radius_width: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate particle positions for exponential disk.
    
    Uses inverse transform sampling to generate particles
    with proper exponential distribution.
    
    Args:
        n_particles: Number of particles
        scale_radius: Disk scale radius
        disk_radius: Outer cutoff radius
        total_mass: Total disk mass
        rng: Random number generator
        min_radius: Minimum radius (prevents particles at center, default 0.0)
        min_radius_width: Width of soft transition above min_radius (default: 0.1 * min_radius)
        
    Returns:
        Tuple of (radii, angles) arrays
    """
    # Generate uniform random numbers
    u = rng.uniform(0, 1, n_particles)
    
    # Inverse CDF for exponential disk: r = -R_d * ln(1 - u * (1 - exp(-R_max/R_d)))
    # Simplified: r = -R_d * ln(1 - u) with cutoff
    max_u = 1 - np.exp(-disk_radius / scale_radius)
    u_scaled = u * max_u
    radii = -scale_radius * np.log(1 - u_scaled)
    if min_radius > 0.0:
        # Rejection resample to preserve original PDF without pile-up at min_radius
        mask = radii < min_radius
        max_iter = 10
        iter_count = 0
        while np.any(mask) and iter_count < max_iter:
            u_new = rng.uniform(0, 1, np.sum(mask))
            u_scaled_new = u_new * max_u
            radii[mask] = -scale_radius * np.log(1 - u_scaled_new)
            mask = radii < min_radius
            iter_count += 1
        if np.any(mask):
            width = min_radius_width if min_radius_width is not None else 0.1 * min_radius
            width = max(width, 1e-6)
            u_inner = rng.uniform(0, 1, np.sum(mask))
            radii[mask] = min_radius + u_inner * width
    radii = np.clip(radii, 0.0, disk_radius)
    
    # Uniform angles
    angles = rng.uniform(0, 2 * np.pi, n_particles)
    
    return radii, angles


def generate_spherical_particles(
    n_particles: int,
    radius: float,
    rng: np.random.Generator,
    min_radius: float = 0.0,
    min_radius_width: float = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate particle positions for spherical distribution.
    
    Args:
        n_particles: Number of particles
        radius: Outer radius
        rng: Random number generator
        min_radius: Minimum radius (prevents particles at center, default 0.0)
        min_radius_width: Width of soft transition above min_radius (default: 0.1 * min_radius)
        
    Returns:
        Tuple of (radii, theta, phi) arrays
    """
    # Uniform in volume: r ~ U(0,1)^(1/3) * R
    u = rng.uniform(0, 1, n_particles)
    radii = radius * (u ** (1/3))
    if min_radius > 0.0:
        mask = radii < min_radius
        max_iter = 10
        iter_count = 0
        while np.any(mask) and iter_count < max_iter:
            u_new = rng.uniform(0, 1, np.sum(mask))
            radii[mask] = radius * (u_new ** (1/3))
            mask = radii < min_radius
            iter_count += 1
        if np.any(mask):
            width = min_radius_width if min_radius_width is not None else 0.1 * min_radius
            width = max(width, 1e-6)
            u_inner = rng.uniform(0, 1, np.sum(mask))
            radii[mask] = min_radius + u_inner * width
    
    # Uniform on sphere
    theta = rng.uniform(0, 2 * np.pi, n_particles)
    u_phi = rng.uniform(0, 1, n_particles)
    phi = np.arccos(2 * u_phi - 1)
    
    return radii, theta, phi


def generate_core_particles(
    n_core_particles: int,
    core_radius: float,
    core_mass: float,
    center: np.ndarray,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate central core particles for galaxy.
    
    Creates multiple high-mass particles in a small spherical region
    at the galaxy center. The core provides gravitational binding
    to keep the galaxy stable.
    
    Args:
        n_core_particles: Number of core particles
        core_radius: Radius of core region (typically very small)
        core_mass: Total mass of core
        center: Center position of galaxy (x, y, z) or (x, y)
        rng: Random number generator
        
    Returns:
        Tuple of (positions, velocities, masses) arrays
        - positions: (n_core_particles, dim) array
        - velocities: (n_core_particles, dim) array (near zero)
        - masses: (n_core_particles,) array (each particle gets core_mass / n_core_particles)
    """
    dim = len(center)
    
    # Generate particles in small sphere around center
    # Use uniform distribution in volume for dense core
    u = rng.uniform(0, 1, n_core_particles)
    radii = core_radius * (u ** (1/3))
    
    # Uniform angles on sphere
    theta = rng.uniform(0, 2 * np.pi, n_core_particles)
    u_phi = rng.uniform(0, 1, n_core_particles)
    phi = np.arccos(2 * u_phi - 1)
    
    # Convert to Cartesian coordinates
    if dim == 3:
        x = center[0] + radii * np.sin(phi) * np.cos(theta)
        y = center[1] + radii * np.sin(phi) * np.sin(theta)
        z = center[2] + radii * np.cos(phi)
        positions = np.column_stack([x, y, z])
    else:
        # 2D: use theta for angle, ignore phi
        x = center[0] + radii * np.cos(theta)
        y = center[1] + radii * np.sin(theta)
        positions = np.column_stack([x, y])
    
    # Core particles have near-zero velocities (dense, slow-moving)
    # Add small random velocities to prevent exact overlap issues
    velocity_scale = 0.01 * core_radius  # Very small velocities
    if dim == 3:
        v_theta = rng.uniform(0, 2 * np.pi, n_core_particles)
        v_phi = np.arccos(2 * rng.uniform(0, 1, n_core_particles) - 1)
        v_mag = rng.uniform(0, velocity_scale, n_core_particles)
        vx = v_mag * np.sin(v_phi) * np.cos(v_theta)
        vy = v_mag * np.sin(v_phi) * np.sin(v_theta)
        vz = v_mag * np.cos(v_phi)
        velocities = np.column_stack([vx, vy, vz])
    else:
        v_theta = rng.uniform(0, 2 * np.pi, n_core_particles)
        v_mag = rng.uniform(0, velocity_scale, n_core_particles)
        vx = v_mag * np.cos(v_theta)
        vy = v_mag * np.sin(v_theta)
        velocities = np.column_stack([vx, vy])
    
    # Each core particle gets equal share of core mass
    masses = np.full(n_core_particles, core_mass / n_core_particles)
    
    return positions, velocities, masses
