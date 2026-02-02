"""Verlet/Leapfrog integrator (better energy conservation, O(h²) accuracy)."""

from typing import Tuple
import numpy as np
from galaxy_sim.backends.base import Backend
from galaxy_sim.physics.integrators.base import Integrator


class VerletIntegrator(Integrator):
    """Velocity Verlet integrator - second-order, symplectic.
    
    Canonical Velocity Verlet algorithm:
    1. x_new = x + v*dt + 0.5*a_old*dt^2
    2. (recompute forces to get a_new)
    3. v_new = v + 0.5*(a_old + a_new)*dt
    
    This is implemented as:
    - step(): computes x_new and v_half = v + 0.5*a_old*dt
    - complete_step(): computes v_new = v_half + 0.5*a_new*dt
    
    Better energy conservation than Euler. Good default choice.
    """
    
    @property
    def name(self) -> str:
        return "verlet"
    
    @property
    def order(self) -> int:
        return 2
    
    def step(self, positions, velocities, masses, forces, dt: float, backend: Backend) -> Tuple:
        """Velocity Verlet step (first half).
        
        Computes:
        - x_new = x + v*dt + 0.5*a_old*dt^2
        - v_half = v + 0.5*a_old*dt
        
        Args:
            positions: Current positions
            velocities: Current velocities
            masses: Particle masses
            forces: Tuple of force components (fx, fy, [fz]) at current positions
            dt: Time step
            backend: Compute backend
            
        Returns:
            Tuple of (new_positions, v_half) where v_half is intermediate velocity
        """
        # Unpack forces
        if len(forces) == 3:
            fx, fy, fz = forces
            dim = 3
        else:
            fx, fy = forces
            fz = None
            dim = 2
        
        # Compute accelerations: a = F / m
        # Use backend operations only (no NumPy conversions)
        # Forces are 1D (n,), masses are 1D (n,), divide element-wise
        masses_1d = backend.array(masses)
        if len(masses_1d.shape) > 1:
            masses_1d = masses_1d.flatten()
        
        # Divide forces by masses to get accelerations
        # Forces fx, fy, fz are 1D (n,), masses is 1D (n,)
        # Element-wise division gives 1D (n,)
        ax = backend.divide(fx, masses_1d)
        ay = backend.divide(fy, masses_1d)
        
        if dim == 3:
            az = backend.divide(fz, masses_1d)
            # Stack accelerations into (n, 3) array
            # Use numpy conversion temporarily for stacking, then convert back
            # This is a limitation of the backend API - we need column_stack
            ax_np = backend.to_numpy(ax).flatten()
            ay_np = backend.to_numpy(ay).flatten()
            az_np = backend.to_numpy(az).flatten()
            acc_np = np.column_stack([ax_np, ay_np, az_np])
            accelerations = backend.array(acc_np)
        else:
            # Stack accelerations into (n, 2) array
            ax_np = backend.to_numpy(ax).flatten()
            ay_np = backend.to_numpy(ay).flatten()
            acc_np = np.column_stack([ax_np, ay_np])
            accelerations = backend.array(acc_np)
        
        # Velocity Verlet: x_new = x + v*dt + 0.5*a_old*dt^2
        dt_sq = dt * dt
        new_positions = backend.add(
            positions,
            backend.add(
                backend.multiply(velocities, dt),
                backend.multiply(accelerations, 0.5 * dt_sq)
            )
        )
        
        # v_half = v + 0.5*a_old*dt (intermediate velocity for complete_step)
        v_half = backend.add(velocities, backend.multiply(accelerations, 0.5 * dt))
        
        return new_positions, v_half
    
    def complete_step(
        self,
        v_half,
        masses,
        forces_old,
        forces_new,
        dt: float,
        backend: Backend
    ):
        """Complete Velocity Verlet step (second half).
        
        Computes: v_new = v_half + 0.5*a_new*dt
        where v_half = v_old + 0.5*a_old*dt (from step())
        
        This gives: v_new = v_old + 0.5*(a_old + a_new)*dt
        
        Args:
            v_half: Intermediate velocity from step() (v_old + 0.5*a_old*dt)
            masses: Particle masses
            forces_old: Forces at old positions (unused, kept for API compatibility)
            forces_new: Forces at new positions (tuple of components)
            dt: Time step
            backend: Compute backend
            
        Returns:
            Final velocities v_new
        """
        # Unpack new forces
        if len(forces_new) == 3:
            fx_new, fy_new, fz_new = forces_new
            dim = 3
        else:
            fx_new, fy_new = forces_new
            fz_new = None
            dim = 2
        
        # Compute new accelerations: a_new = F_new / m
        # Use backend operations only (no NumPy conversions)
        masses_1d = backend.array(masses)
        if len(masses_1d.shape) > 1:
            masses_1d = masses_1d.flatten()
        
        # Divide forces by masses (both 1D)
        ax_new = backend.divide(fx_new, masses_1d)
        ay_new = backend.divide(fy_new, masses_1d)
        
        if dim == 3:
            az_new = backend.divide(fz_new, masses_1d)
            # Stack accelerations into (n, 3) array
            ax_np = backend.to_numpy(ax_new).flatten()
            ay_np = backend.to_numpy(ay_new).flatten()
            az_np = backend.to_numpy(az_new).flatten()
            a_new_np = np.column_stack([ax_np, ay_np, az_np])
            a_new = backend.array(a_new_np)
        else:
            # Stack accelerations into (n, 2) array
            ax_np = backend.to_numpy(ax_new).flatten()
            ay_np = backend.to_numpy(ay_new).flatten()
            a_new_np = np.column_stack([ax_np, ay_np])
            a_new = backend.array(a_new_np)
        
        # Complete Velocity Verlet: v_new = v_half + 0.5*a_new*dt
        # This gives: v_new = v_old + 0.5*(a_old + a_new)*dt
        v_new = backend.add(v_half, backend.multiply(a_new, 0.5 * dt))
        
        return v_new
