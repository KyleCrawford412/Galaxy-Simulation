"""Utility functions for galaxy generation."""

import numpy as np
from typing import Tuple


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
    rng: np.random.Generator
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
    radii = np.clip(radii, 0, disk_radius)
    
    # Uniform angles
    angles = rng.uniform(0, 2 * np.pi, n_particles)
    
    return radii, angles


def generate_spherical_particles(
    n_particles: int,
    radius: float,
    rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate particle positions for spherical distribution.
    
    Args:
        n_particles: Number of particles
        radius: Outer radius
        rng: Random number generator
        
    Returns:
        Tuple of (radii, theta, phi) arrays
    """
    # Uniform in volume: r ~ U(0,1)^(1/3) * R
    u = rng.uniform(0, 1, n_particles)
    radii = radius * (u ** (1/3))
    
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
