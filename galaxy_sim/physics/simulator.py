"""Main simulator controller."""

from typing import Optional, Callable
from galaxy_sim.backends.base import Backend
from galaxy_sim.physics.nbody import NBodySystem
from galaxy_sim.physics.integrators.base import Integrator
from galaxy_sim.physics.integrators.euler import EulerIntegrator
from galaxy_sim.physics.integrators.verlet import VerletIntegrator
from galaxy_sim.physics.integrators.rk4 import RK4Integrator


class Simulator:
    """Main simulation controller.
    
    Orchestrates physics, integrators, and timestep management.
    """
    
    def __init__(
        self,
        backend: Backend,
        integrator: Optional[Integrator] = None,
        dt: float = 0.01
    ):
        """Initialize simulator.
        
        Args:
            backend: Compute backend
            integrator: Integrator to use (default: Verlet)
            dt: Time step
        """
        self.backend = backend
        self.integrator = integrator or VerletIntegrator()
        self.dt = dt
        
        self.system = NBodySystem(backend)
        self.time = 0.0
        self.paused = False
        self.step_count = 0
        
        # Callbacks
        self.on_step_callback: Optional[Callable] = None
        self.on_energy_callback: Optional[Callable] = None
    
    def initialize(self, positions, velocities, masses):
        """Initialize particle system.
        
        Args:
            positions: Initial positions
            velocities: Initial velocities
            masses: Particle masses
        """
        self.system.initialize(positions, velocities, masses)
        self.time = 0.0
        self.step_count = 0
    
    def step(self):
        """Perform one simulation step."""
        if self.paused:
            return
        
        # Compute forces at current positions
        forces_old = self.system.compute_forces()
        
        # Velocity Verlet step (first half): computes x_new and v_half
        new_positions, v_half = self.integrator.step(
            self.system.positions,
            self.system.velocities,
            self.system.masses,
            forces_old,
            self.dt,
            self.backend
        )
        
        # For Velocity Verlet, we need forces at new positions
        # Temporarily update positions to compute new forces
        old_positions = self.system.positions
        self.system.positions = new_positions
        forces_new = self.system.compute_forces()
        
        # Complete velocity update with new forces (for proper Velocity Verlet)
        if hasattr(self.integrator, 'complete_step'):
            # Pass v_half (not v_old) to complete_step
            new_velocities = self.integrator.complete_step(
                v_half,  # Use v_half from step(), not self.system.velocities
                self.system.masses,
                forces_old,
                forces_new,
                self.dt,
                self.backend
            )
        else:
            # Fallback: use v_half as new velocities (not correct, but better than nothing)
            new_velocities = v_half
        
        # Restore positions (will be set below)
        self.system.positions = old_positions
        
        # Update state
        self.system.positions = new_positions
        self.system.velocities = new_velocities
        self.time += self.dt
        self.step_count += 1
        
        # Callbacks
        if self.on_step_callback:
            self.on_step_callback(self)
        
        if self.on_energy_callback:
            energy = self.system.compute_total_energy()
            self.on_energy_callback(self, energy)
    
    def run(self, n_steps: int):
        """Run simulation for specified number of steps.
        
        Args:
            n_steps: Number of steps to run
        """
        for _ in range(n_steps):
            self.step()
    
    def pause(self):
        """Pause simulation."""
        self.paused = True
    
    def resume(self):
        """Resume simulation."""
        self.paused = False
    
    def set_timestep(self, dt: float):
        """Set time step.
        
        Args:
            dt: New time step
        """
        self.dt = dt
    
    def set_integrator(self, integrator: Integrator):
        """Set integrator.
        
        Args:
            integrator: New integrator
        """
        self.integrator = integrator
    
    def get_state(self):
        """Get current simulation state.
        
        Returns:
            Tuple of (positions, velocities, masses, time, step_count)
        """
        pos, vel, mass = self.system.get_state()
        return pos, vel, mass, self.time, self.step_count
    
    def get_energy(self):
        """Get current total energy.
        
        Returns:
            Total energy (kinetic + potential)
        """
        return self.system.compute_total_energy()
    
    def get_kinetic_energy(self):
        """Get current kinetic energy."""
        return self.system.compute_kinetic_energy()
    
    def get_potential_energy(self):
        """Get current potential energy."""
        return self.system.compute_potential_energy()
