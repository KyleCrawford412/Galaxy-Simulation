"""Component-wise virialization for disk+bulge galaxies."""

import numpy as np
from typing import Tuple, Optional
from galaxy_sim.backends.base import Backend
from galaxy_sim.physics.diagnostics import Diagnostics


def virialize_component_wise(
    positions,
    velocities,
    masses,
    backend: Backend,
    diagnostics: Diagnostics,
    target_Q: float = 1.0,
    particle_types: Optional[np.ndarray] = None,
    v_tan_scale_factor: float = 1.05,  # Small factor to increase tangential velocity
    sigma_r_scale: float = 0.1  # Small radial dispersion to add
) -> Tuple:
    """Virialize system with component-wise scaling for disk particles.
    
    For disk particles:
    - Decompose velocity into radial and tangential components
    - Scale tangential component by s_t (anchored to v_circ)
    - Scale radial component by s_r (add dispersion)
    - Default: increase v_tan first, then add small sigma_r
    
    For bulge particles:
    - Uniform isotropic scaling
    
    Args:
        positions: Particle positions (n, dim)
        velocities: Particle velocities (n, dim)
        masses: Particle masses (n,)
        backend: Compute backend
        diagnostics: Diagnostics instance for Q calculation
        target_Q: Target virial ratio
        particle_types: Array of 'disk', 'bulge', 'halo', 'core' for each particle (n,)
                       If None, all particles are treated as bulge (uniform scaling)
        v_tan_scale_factor: Factor to scale tangential velocity (default: 1.05)
        sigma_r_scale: Scale for radial dispersion (default: 0.1)
    
    Returns:
        Tuple of (new_velocities, Q_final, scale_info)
    """
    positions_np = np.asarray(backend.to_numpy(positions))
    velocities_np = np.asarray(backend.to_numpy(velocities))
    masses_np = np.asarray(backend.to_numpy(masses)).flatten()
    
    n = len(masses_np)
    dim = positions_np.shape[1]
    
    # Compute initial Q
    Q_initial = diagnostics.compute_virial_ratio(positions, velocities, masses)
    
    if Q_initial <= 0 or np.isinf(Q_initial):
        return velocities, Q_initial, {"error": "Invalid initial Q"}
    
    # If no particle types provided, use uniform scaling
    if particle_types is None:
        # Uniform scaling for all particles
        scale_factor = np.sqrt(target_Q / Q_initial)
        new_velocities_np = velocities_np * scale_factor
        new_velocities = backend.array(new_velocities_np)
        Q_final = diagnostics.compute_virial_ratio(positions, new_velocities, masses)
        return new_velocities, Q_final, {"scale": scale_factor, "method": "uniform"}
    
    # Separate disk and bulge particles
    is_disk = (particle_types == 'disk')
    is_bulge = (particle_types == 'bulge')
    is_other = ~(is_disk | is_bulge)  # halo, core, etc.
    
    new_velocities_np = velocities_np.copy()
    
    # For disk particles: component-wise scaling
    if np.any(is_disk):
        disk_positions = positions_np[is_disk]
        disk_velocities = velocities_np[is_disk]
        
        # Decompose velocities into radial and tangential components
        # Radial: v_r = (v · r_hat) * r_hat
        # Tangential: v_t = v - v_r
        
        # Compute radial unit vectors
        r_mag = np.linalg.norm(disk_positions, axis=1, keepdims=True)
        r_safe = np.maximum(r_mag, 1e-6)  # Avoid division by zero
        r_hat = disk_positions / r_safe  # (n_disk, dim)
        
        # Radial velocity component: v_r = (v · r_hat) * r_hat
        v_radial_mag = np.sum(disk_velocities * r_hat, axis=1, keepdims=True)  # (n_disk, 1)
        v_radial = v_radial_mag * r_hat  # (n_disk, dim)
        
        # Tangential velocity component: v_t = v - v_r
        v_tangential = disk_velocities - v_radial  # (n_disk, dim)
        v_tan_mag = np.linalg.norm(v_tangential, axis=1, keepdims=True)  # (n_disk, 1)
        
        # Scale tangential velocity (anchor to v_circ, scale by small factor)
        v_tan_new = v_tangential * v_tan_scale_factor
        
        # Scale radial velocity (add dispersion)
        # If radial velocity is small, add some dispersion
        v_rad_mag = np.abs(v_radial_mag).flatten()
        v_rad_scale = np.ones_like(v_rad_mag)
        
        # For particles with small radial velocity, add dispersion
        small_rad = v_rad_mag < (v_tan_mag.flatten() * 0.1)  # Less than 10% of tangential
        if np.any(small_rad):
            # Add small random radial component
            rng = np.random.default_rng(42)  # Fixed seed for reproducibility
            n_small = np.sum(small_rad)
            random_rad = rng.normal(0, sigma_r_scale, n_small)
            v_rad_scale[small_rad] = 1.0 + np.abs(random_rad) / (v_tan_mag.flatten()[small_rad] + 1e-6)
        
        # Scale radial component
        v_radial_new = v_radial * v_rad_scale[:, np.newaxis]
        
        # Combine: v_new = v_tan_new + v_radial_new
        new_velocities_np[is_disk] = v_tan_new + v_radial_new
    
    # For bulge particles: uniform isotropic scaling
    # Iteratively adjust all velocities to hit target Q precisely
    max_iter = 20
    for iter in range(max_iter):
        test_velocities_backend = backend.array(new_velocities_np)
        Q_test = diagnostics.compute_virial_ratio(positions, test_velocities_backend, masses)
        
        if abs(Q_test - target_Q) < 0.001:
            break
        
        # Adjust all velocities proportionally to hit target Q
        scale_adjust = np.sqrt(target_Q / Q_test)
        
        # Apply adjustment: disk tangential already scaled, so adjust everything proportionally
        # But for disk, we want to preserve the tangential/radial ratio
        if np.any(is_disk):
            # Re-extract disk velocities and scale both components
            disk_velocities = new_velocities_np[is_disk]
            disk_positions = positions_np[is_disk]
            
            # Re-decompose
            r_mag = np.linalg.norm(disk_positions, axis=1, keepdims=True)
            r_safe = np.maximum(r_mag, 1e-6)
            r_hat = disk_positions / r_safe
            
            v_radial_mag = np.sum(disk_velocities * r_hat, axis=1, keepdims=True)
            v_radial = v_radial_mag * r_hat
            v_tangential = disk_velocities - v_radial
            
            # Scale both components by adjustment factor
            v_tan_new = v_tangential * scale_adjust
            v_radial_new = v_radial * scale_adjust
            new_velocities_np[is_disk] = v_tan_new + v_radial_new
        else:
            # For non-disk particles, uniform scaling
            if np.any(is_bulge):
                new_velocities_np[is_bulge] *= scale_adjust
            if np.any(is_other):
                new_velocities_np[is_other] *= scale_adjust
    
    # For other particles (halo, core): uniform scaling
    if np.any(is_other):
        other_velocities = velocities_np[is_other]
        scale_factor = np.sqrt(target_Q / Q_initial)
        new_velocities_np[is_other] = other_velocities * scale_factor
    
    # Convert back to backend array
    new_velocities = backend.array(new_velocities_np)
    
    # Compute final Q
    Q_final = diagnostics.compute_virial_ratio(positions, new_velocities, masses)
    
    scale_info = {
        "Q_initial": Q_initial,
        "Q_final": Q_final,
        "v_tan_scale": v_tan_scale_factor,
        "method": "component_wise"
    }
    
    return new_velocities, Q_final, scale_info
