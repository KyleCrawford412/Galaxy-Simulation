"""Main simulator controller."""

from typing import Optional, Callable
import numpy as np
from galaxy_sim.backends.base import Backend
from galaxy_sim.physics.nbody import NBodySystem
from galaxy_sim.physics.integrators.base import Integrator
from galaxy_sim.physics.integrators.euler import EulerIntegrator
from galaxy_sim.physics.integrators.verlet import VerletIntegrator
from galaxy_sim.physics.integrators.rk4 import RK4Integrator
from galaxy_sim.physics.halo_potential import HaloPotential


class Simulator:
    """Main simulation controller.
    
    Orchestrates physics, integrators, and timestep management.
    """
    
    def __init__(
        self,
        backend: Backend,
        integrator: Optional[Integrator] = None,
        dt: float = 0.01,
        halo_potential: Optional[HaloPotential] = None
    ):
        """Initialize simulator.
        
        Args:
            backend: Compute backend
            integrator: Integrator to use (default: Verlet)
            dt: Time step
            halo_potential: Optional analytic halo potential
        """
        self.backend = backend
        self.integrator = integrator or VerletIntegrator()
        self.dt = dt
        
        self.system = NBodySystem(backend, halo_potential=halo_potential)
        self.time = 0.0
        self.paused = False
        self.step_count = 0
        
        # Callbacks
        self.on_step_callback: Optional[Callable] = None
        self.on_energy_callback: Optional[Callable] = None
    
    def initialize(self, positions, velocities, masses, virialize: bool = False, target_Q: float = 1.0):
        """Initialize particle system.
        
        Args:
            positions: Initial positions
            velocities: Initial velocities
            masses: Particle masses
            virialize: If True, rescale velocities to achieve target virial ratio
            target_Q: Target virial ratio (default: 1.0 for equilibrium)
        """
        self.system.initialize(positions, velocities, masses)
        
        # Use consistent diagnostics for virial ratio
        from galaxy_sim.physics.diagnostics import Diagnostics
        
        diagnostics = Diagnostics(
            self.backend,
            G=self.system.G,
            epsilon=self.system.epsilon,
            halo_potential=self.system.halo_potential
        )
        
        # Compute initial virial ratio using consistent diagnostics
        Q_initial = diagnostics.compute_virial_ratio(
            self.system.positions,
            self.system.velocities,
            self.system.masses
        )
        
        if virialize:
            # Use component-wise virialization if preset provides particle types
            from galaxy_sim.physics.virialization import virialize_component_wise
            
            # Check if preset has particle types (for disk+bulge galaxies)
            particle_types = None
            if hasattr(self, 'preset') and hasattr(self.preset, 'particle_types'):
                particle_types = self.preset.particle_types
            
            if particle_types is not None and len(particle_types) == len(masses):
                # Component-wise virialization for disk+bulge
                new_velocities, Q_final, scale_info = virialize_component_wise(
                    self.system.positions,
                    self.system.velocities,
                    self.system.masses,
                    self.backend,
                    diagnostics,
                    target_Q=target_Q,
                    particle_types=particle_types
                )
                self.system.velocities = new_velocities
                print(f"Virialization (component-wise): Q_initial = {Q_initial:.4f}, Q_target = {target_Q:.4f}, Q_final = {Q_final:.4f}")
                if 'v_tan_scale' in scale_info:
                    print(f"  Disk: v_tan scaled by {scale_info['v_tan_scale']:.4f}")
            else:
                # Uniform scaling for other presets
                if Q_initial > 0 and not np.isinf(Q_initial):
                    scale_factor = np.sqrt(target_Q / Q_initial)
                    velocities = self.backend.multiply(velocities, scale_factor)
                    self.system.velocities = velocities
                    Q_final = diagnostics.compute_virial_ratio(
                        self.system.positions,
                        self.system.velocities,
                        self.system.masses
                    )
                    print(f"Virialization (uniform): Q_initial = {Q_initial:.4f}, Q_target = {target_Q:.4f}, Q_final = {Q_final:.4f}, scale = {scale_factor:.4f}")
                else:
                    print(f"Warning: Cannot virialize, Q_initial = {Q_initial:.4f}")
        else:
            print(f"Initial virial ratio: Q = {Q_initial:.4f} (target: 1.0 for equilibrium)")
        
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
