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
    f_rot: float = 1.1,  # Rotation factor for disk: v_tan = f_rot * v_circ
    sigma_r_fraction: float = 0.05,  # Radial dispersion as fraction of v_circ
    sigma_t_fraction: float = 0.03,  # Tangential dispersion as fraction of v_circ
    halo_potential = None,  # Halo potential for computing v_circ
    G: float = 1.0,  # Gravitational constant
    central_mass: float = 100.0,  # Central mass for computing v_circ
    disk_mass: float = 1000.0,  # Disk mass for computing v_circ
    disk_scale_radius: float = 10.0  # Disk scale radius for computing v_circ
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
    
    # For disk particles: recompute velocities from v_circ (do NOT globally rescale)
    if np.any(is_disk):
        disk_positions = positions_np[is_disk]
        n_disk = np.sum(is_disk)
        rng = np.random.default_rng(42)  # Fixed seed for reproducibility
        
        # Compute radial distances and angles
        r_mag = np.linalg.norm(disk_positions, axis=1)  # (n_disk,)
        r_safe = np.maximum(r_mag, 1e-6)
        r_hat = disk_positions / r_safe[:, np.newaxis]  # (n_disk, dim)
        
        # Compute angles for tangential direction
        # For 2D/3D: theta = atan2(y, x), tangent = (-sin(theta), cos(theta))
        if disk_positions.shape[1] == 3:
            theta = np.arctan2(disk_positions[:, 1], disk_positions[:, 0])
        else:
            theta = np.arctan2(disk_positions[:, 1], disk_positions[:, 0])
        
        # Compute circular velocity from potential
        # v_circ = sqrt(r * |a_r|) where a_r is radial acceleration
        r_min_orbital = max(disk_scale_radius * 0.3, 2.0)
        r_safe_orbital = np.maximum(r_safe, r_min_orbital)
        
        # Acceleration from central mass
        a_central = G * central_mass / (r_safe_orbital ** 2)
        
        # Acceleration from disk (approximate enclosed mass)
        M_enc_disk = disk_mass * (1 - np.exp(-r_safe_orbital / disk_scale_radius))
        a_disk = G * M_enc_disk / (r_safe_orbital ** 2)
        
        # Acceleration from halo (if present)
        a_halo = np.zeros_like(r_safe_orbital)
        if halo_potential is not None and halo_potential.enabled:
            test_positions = np.column_stack([r_safe_orbital, np.zeros(n_disk), np.zeros(n_disk)])
            test_positions_backend = backend.array(test_positions)
            a_halo_vec = halo_potential.compute_acceleration(test_positions_backend, backend)
            if a_halo_vec is not None:
                a_halo_np = np.asarray(backend.to_numpy(a_halo_vec))
                a_halo = np.abs(a_halo_np[:, 0])
        
        # Total radial acceleration
        a_r_total = a_central + a_disk + a_halo
        
        # Circular velocity
        v_circ = np.sqrt(r_safe_orbital * a_r_total)
        
        # Set tangential velocity: v_tan = f_rot * v_circ
        v_tan_mag = f_rot * v_circ
        
        # Add tangential dispersion: v_tan *= 1 + Normal(0, sigma_t)
        sigma_t = sigma_t_fraction * v_circ
        v_tan_factor = 1.0 + rng.normal(0, sigma_t)
        v_tan_mag = v_tan_mag * v_tan_factor
        
        # Tangential direction: perpendicular to radius
        # For 2D/3D: tangent = (-sin(theta), cos(theta), 0)
        vx_tan = -v_tan_mag * np.sin(theta)
        vy_tan = v_tan_mag * np.cos(theta)
        if disk_positions.shape[1] == 3:
            vz_tan = np.zeros(n_disk)
        else:
            vz_tan = np.zeros(n_disk)
        
        # Add radial velocity dispersion: v_rad = Normal(0, sigma_r) in radial direction
        sigma_r = sigma_r_fraction * v_circ
        v_rad_mag = rng.normal(0, sigma_r, n_disk)
        v_rad_x = v_rad_mag * r_hat[:, 0]
        v_rad_y = v_rad_mag * r_hat[:, 1]
        if disk_positions.shape[1] == 3:
            v_rad_z = v_rad_mag * r_hat[:, 2]
        else:
            v_rad_z = np.zeros(n_disk)
        
        # Combine tangential and radial components
        vx = vx_tan + v_rad_x
        vy = vy_tan + v_rad_y
        vz = vz_tan + v_rad_z
        
        # Stack velocities
        if disk_positions.shape[1] == 3:
            new_velocities_np[is_disk] = np.column_stack([vx, vy, vz])
        else:
            new_velocities_np[is_disk] = np.column_stack([vx, vy])
    
    # For bulge particles: uniform isotropic scaling
    # Iteratively adjust velocities to hit target Q precisely
    # Adjust f_rot for disk particles and scale factor for bulge
    max_iter = 20
    f_rot_current = f_rot
    bulge_scale = 1.0
    r_min_orbital = max(disk_scale_radius * 0.3, 2.0) if np.any(is_disk) else 2.0
    
    for iter in range(max_iter):
        # Recompute disk velocities with current f_rot
        if np.any(is_disk):
            disk_positions = positions_np[is_disk]
            n_disk = np.sum(is_disk)
            
            r_mag = np.linalg.norm(disk_positions, axis=1)
            r_safe = np.maximum(r_mag, 1e-6)
            r_hat = disk_positions / r_safe[:, np.newaxis]
            
            if disk_positions.shape[1] == 3:
                theta = np.arctan2(disk_positions[:, 1], disk_positions[:, 0])
            else:
                theta = np.arctan2(disk_positions[:, 1], disk_positions[:, 0])
            
            r_safe_orbital = np.maximum(r_safe, r_min_orbital)
            a_central = G * central_mass / (r_safe_orbital ** 2)
            M_enc_disk = disk_mass * (1 - np.exp(-r_safe_orbital / disk_scale_radius))
            a_disk = G * M_enc_disk / (r_safe_orbital ** 2)
            a_halo = np.zeros_like(r_safe_orbital)
            if halo_potential is not None and halo_potential.enabled:
                test_positions = np.column_stack([r_safe_orbital, np.zeros(n_disk), np.zeros(n_disk)])
                test_positions_backend = backend.array(test_positions)
                a_halo_vec = halo_potential.compute_acceleration(test_positions_backend, backend)
                if a_halo_vec is not None:
                    a_halo_np = np.asarray(backend.to_numpy(a_halo_vec))
                    a_halo = np.abs(a_halo_np[:, 0])
            a_r_total = a_central + a_disk + a_halo
            v_circ = np.sqrt(r_safe_orbital * a_r_total)
            v_tan_mag = f_rot_current * v_circ
            sigma_t = sigma_t_fraction * v_circ
            v_tan_factor = 1.0 + rng.normal(0, sigma_t)
            v_tan_mag = v_tan_mag * v_tan_factor
            vx_tan = -v_tan_mag * np.sin(theta)
            vy_tan = v_tan_mag * np.cos(theta)
            sigma_r = sigma_r_fraction * v_circ
            v_rad_mag = rng.normal(0, sigma_r, n_disk)
            v_rad_x = v_rad_mag * r_hat[:, 0]
            v_rad_y = v_rad_mag * r_hat[:, 1]
            if disk_positions.shape[1] == 3:
                vz_tan = np.zeros(n_disk)
                v_rad_z = v_rad_mag * r_hat[:, 2]
                new_velocities_np[is_disk] = np.column_stack([vx_tan + v_rad_x, vy_tan + v_rad_y, vz_tan + v_rad_z])
            else:
                new_velocities_np[is_disk] = np.column_stack([vx_tan + v_rad_x, vy_tan + v_rad_y])
        
        # Scale bulge velocities
        if np.any(is_bulge):
            bulge_velocities = velocities_np[is_bulge]
            new_velocities_np[is_bulge] = bulge_velocities * bulge_scale
        
        # Test Q
        test_velocities_backend = backend.array(new_velocities_np)
        Q_test = diagnostics.compute_virial_ratio(positions, test_velocities_backend, masses)
        
        if abs(Q_test - target_Q) < 0.001:
            break
        
        # Adjust f_rot and bulge_scale to hit target Q
        scale_adjust = np.sqrt(target_Q / Q_test)
        f_rot_current *= scale_adjust
        bulge_scale *= scale_adjust
    
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
        "f_rot": f_rot_current if np.any(is_disk) else f_rot,
        "sigma_r_fraction": sigma_r_fraction,
        "sigma_t_fraction": sigma_t_fraction,
        "method": "component_wise"
    }
    
    return new_velocities, Q_final, scale_info
