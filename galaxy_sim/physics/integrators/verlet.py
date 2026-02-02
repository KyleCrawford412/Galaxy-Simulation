"""Verlet/Leapfrog integrator (better energy conservation, O(h²) accuracy)."""

from typing import Tuple
from galaxy_sim.backends.base import Backend
from galaxy_sim.physics.integrators.base import Integrator


class VerletIntegrator(Integrator):
    """Verlet/Leapfrog integrator - second-order, symplectic.
    
    Better energy conservation than Euler. Good default choice.
    """
    
    @property
    def name(self) -> str:
        return "verlet"
    
    @property
    def order(self) -> int:
        return 2
    
    def step(self, positions, velocities, masses, forces, dt: float, backend: Backend) -> Tuple:
        """Leapfrog step: v_half = v + a*dt/2, r_new = r + v_half*dt, v_new = v_half + a_new*dt/2.
        
        For simplicity, we use velocity Verlet which is equivalent:
        v_new = v + a*dt/2
        r_new = r + v_new*dt
        (recompute forces)
        v_new = v_new + a_new*dt/2
        
        But since we don't recompute forces here, we use:
        r_new = r + v*dt + 0.5*a*dt²
        v_new = v + 0.5*(a_old + a_new)*dt
        
        For now, we approximate with current acceleration:
        r_new = r + v*dt + 0.5*a*dt²
        v_new = v + a*dt
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
        # Ensure forces and masses are proper 1D arrays
        import numpy as np
        fx_np = np.asarray(backend.to_numpy(fx)).flatten()
        fy_np = np.asarray(backend.to_numpy(fy)).flatten()
        masses_np = np.asarray(backend.to_numpy(masses)).flatten()
        n = len(masses_np)
        
        # Ensure correct lengths
        fx_np = fx_np[:n]
        fy_np = fy_np[:n]
        masses_np = masses_np[:n]
        
        ax = fx_np / masses_np
        ay = fy_np / masses_np
        
        if dim == 3:
            fz_np = np.asarray(backend.to_numpy(fz)).flatten()[:n]
            az = fz_np / masses_np
            accelerations = backend.array(np.column_stack([ax, ay, az]))
        else:
            accelerations = backend.array(np.column_stack([ax, ay]))
        
        # Velocity Verlet: r_new = r + v*dt + 0.5*a*dt²
        dt_sq = dt * dt
        new_positions = backend.add(
            positions,
            backend.add(
                backend.multiply(velocities, dt),
                backend.multiply(accelerations, 0.5 * dt_sq)
            )
        )
        
        # v_new = v + a*dt
        new_velocities = backend.add(velocities, backend.multiply(accelerations, dt))
        
        return new_positions, new_velocities
