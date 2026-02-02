"""Euler method integrator (baseline, O(h) accuracy)."""

from typing import Tuple
from galaxy_sim.backends.base import Backend
from galaxy_sim.physics.integrators.base import Integrator


class EulerIntegrator(Integrator):
    """Euler method - simple first-order integrator.
    
    Fast but less accurate. Good for baseline comparisons.
    """
    
    @property
    def name(self) -> str:
        return "euler"
    
    @property
    def order(self) -> int:
        return 1
    
    def step(self, positions, velocities, masses, forces, dt: float, backend: Backend) -> Tuple:
        """Euler step: v_new = v + a*dt, r_new = r + v*dt.
        
        Args:
            positions: Current positions
            velocities: Current velocities
            masses: Particle masses
            forces: Tuple of force components (fx, fy, [fz])
            dt: Time step
            backend: Compute backend
            
        Returns:
            Tuple of (new_positions, new_velocities)
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
        masses_expanded = backend.array(masses)[:, None]
        ax = backend.divide(fx, masses_expanded)
        ay = backend.divide(fy, masses_expanded)
        
        if dim == 3:
            az = backend.divide(fz, masses_expanded)
            accelerations = backend.array([ax, ay, az]).T
        else:
            accelerations = backend.array([ax, ay]).T
        
        # Update velocities: v_new = v + a*dt
        new_velocities = backend.add(velocities, backend.multiply(accelerations, dt))
        
        # Update positions: r_new = r + v*dt
        new_positions = backend.add(positions, backend.multiply(velocities, dt))
        
        return new_positions, new_velocities
