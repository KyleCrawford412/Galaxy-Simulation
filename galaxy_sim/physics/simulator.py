"""Main simulator controller."""

from typing import Optional, Callable
import time
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
        halo_potential: Optional[HaloPotential] = None,
        self_gravity: bool = True
    ):
        """Initialize simulator.
        
        Args:
            backend: Compute backend
            integrator: Integrator to use (default: Verlet)
            dt: Time step
            halo_potential: Optional analytic halo potential
            self_gravity: If False, disk particles don't attract each other (test particle mode)
        """
        self.backend = backend
        self.integrator = integrator or VerletIntegrator()
        self.dt = dt
        
        self.system = NBodySystem(backend, halo_potential=halo_potential, self_gravity=self_gravity)
        self.time = 0.0
        self.paused = False
        self.step_count = 0
        
        # Profiling: last step timing (ms)
        self._last_forces_ms: Optional[float] = None
        self._last_integrator_ms: Optional[float] = None
        self._profile: bool = False
        
        # Callbacks
        self.on_step_callback: Optional[Callable] = None
        self.on_energy_callback: Optional[Callable] = None
    
    def set_profiling(self, enabled: bool = True):
        """Enable or disable step timing (forces ms, integrator ms)."""
        self._profile = enabled
    
    def get_timing(self) -> dict:
        """Return last step timing in ms: forces_ms, integrator_ms. Arrays remain on device when profiling."""
        return {
            "forces_ms": self._last_forces_ms,
            "integrator_ms": self._last_integrator_ms,
        }
    
    def initialize(self, positions, velocities, masses, virialize: bool = False, target_Q: float = 1.0, particle_types = None):
        """Initialize particle system.
        
        Args:
            positions: Initial positions
            velocities: Initial velocities
            masses: Particle masses
            virialize: If True, rescale velocities to achieve target virial ratio
            target_Q: Target virial ratio (default: 1.0 for equilibrium)
            particle_types: Array of 'disk', 'bulge', 'halo', 'core' for each particle (n,)
        """
        # Center the system: subtract COM position and COM velocity
        positions_np = np.asarray(self.backend.to_numpy(positions))
        velocities_np = np.asarray(self.backend.to_numpy(velocities))
        masses_np = np.asarray(self.backend.to_numpy(masses)).flatten()
        
        # Compute center of mass
        total_mass = np.sum(masses_np)
        COM = np.sum(masses_np[:, np.newaxis] * positions_np, axis=0) / total_mass
        COMv = np.sum(masses_np[:, np.newaxis] * velocities_np, axis=0) / total_mass
        
        # Subtract COM from all particles
        positions_centered = positions_np - COM
        velocities_centered = velocities_np - COMv
        
        # Convert back to backend arrays
        positions_centered = self.backend.array(positions_centered)
        velocities_centered = self.backend.array(velocities_centered)
        
        self.system.initialize(positions_centered, velocities_centered, masses, particle_types=particle_types)
        
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
                # Disk v_circ from same field as simulation: a_central + a_halo + a_bulge (no a_disk unless analytic disk)
                preset = getattr(self, 'preset', None)
                halo_potential = self.system.halo_potential
                G = self.system.G
                central_mass = getattr(preset, 'central_mass', 100.0) if preset else 100.0
                disk_mass = getattr(preset, 'disk_mass', 1000.0) if preset else 1000.0
                disk_scale_radius = getattr(preset, 'disk_scale_radius', 10.0) if preset else 10.0
                positions_np = np.asarray(self.backend.to_numpy(self.system.positions))
                masses_np = np.asarray(self.backend.to_numpy(self.system.masses)).flatten()
                is_bulge = (np.asarray(particle_types) == 'bulge')
                bulge_positions = positions_np[is_bulge] if np.any(is_bulge) else None
                bulge_masses = masses_np[is_bulge] if np.any(is_bulge) else None
                eps = getattr(self.system, 'epsilon', 1e-6)

                new_velocities, Q_final, scale_info = virialize_component_wise(
                    self.system.positions,
                    self.system.velocities,
                    self.system.masses,
                    self.backend,
                    diagnostics,
                    target_Q=target_Q,
                    particle_types=particle_types,
                    halo_potential=halo_potential,
                    G=G,
                    central_mass=central_mass,
                    disk_mass=disk_mass,
                    disk_scale_radius=disk_scale_radius,
                    use_analytic_disk=False,
                    bulge_positions=bulge_positions,
                    bulge_masses=bulge_masses,
                    eps=eps,
                )
                self.system.velocities = new_velocities
                
                # Compute angular momentum (returns magnitude for 2D, full vector for 3D)
                L_initial = self.system.compute_angular_momentum()
                # For 2D, L is scalar (Lz); for 3D, it's a vector
                if isinstance(L_initial, (int, float)) or (hasattr(L_initial, '__len__') and len(L_initial) == 1):
                    Lz_initial = float(L_initial) if not hasattr(L_initial, '__len__') else float(L_initial[0])
                else:
                    Lz_initial = float(L_initial[-1]) if len(L_initial) > 2 else float(L_initial[0])
                
                L_final = self.system.compute_angular_momentum()
                if isinstance(L_final, (int, float)) or (hasattr(L_final, '__len__') and len(L_final) == 1):
                    Lz_final = float(L_final) if not hasattr(L_final, '__len__') else float(L_final[0])
                else:
                    Lz_final = float(L_final[-1]) if len(L_final) > 2 else float(L_final[0])
                
                print(f"Virialization (component-wise): Q_initial = {Q_initial:.4f}, Q_target = {target_Q:.4f}, Q_final = {Q_final:.4f}")
                print(f"  Angular momentum: Lz_initial = {Lz_initial:.2f}, Lz_final = {Lz_final:.2f}")
                if 'f_rot' in scale_info:
                    print(f"  Disk: f_rot = {scale_info['f_rot']:.4f}, sigma_r = {scale_info.get('sigma_r_fraction', 0.05):.4f}*v_circ, sigma_t = {scale_info.get('sigma_t_fraction', 0.03):.4f}*v_circ")
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
        
        # Final center: ensure COM at origin and COMv ~ 0 (virialization may have shifted COMv)
        positions_np = np.asarray(self.backend.to_numpy(self.system.positions))
        velocities_np = np.asarray(self.backend.to_numpy(self.system.velocities))
        masses_np = np.asarray(self.backend.to_numpy(self.system.masses)).flatten()
        total_mass = np.sum(masses_np)
        COM = np.sum(masses_np[:, np.newaxis] * positions_np, axis=0) / total_mass
        COMv = np.sum(masses_np[:, np.newaxis] * velocities_np, axis=0) / total_mass
        self.system.positions = self.backend.array(positions_np - COM)
        self.system.velocities = self.backend.array(velocities_np - COMv)
        
        self.time = 0.0
        self.step_count = 0
    
    def step(self):
        """Perform one simulation step (no host transfer; arrays stay on backend)."""
        if self.paused:
            return
        
        if self._profile:
            t0 = time.perf_counter()
        forces_old = self.system.compute_forces()
        if self._profile:
            t1 = time.perf_counter()
        new_positions, v_half = self.integrator.step(
            self.system.positions,
            self.system.velocities,
            self.system.masses,
            forces_old,
            self.dt,
            self.backend,
        )
        if self._profile:
            t2 = time.perf_counter()
        old_positions = self.system.positions
        self.system.positions = new_positions
        forces_new = self.system.compute_forces()
        if self._profile:
            t3 = time.perf_counter()
        if hasattr(self.integrator, "complete_step"):
            new_velocities = self.integrator.complete_step(
                v_half,
                self.system.masses,
                forces_old,
                forces_new,
                self.dt,
                self.backend,
            )
        else:
            new_velocities = v_half
        if self._profile:
            t4 = time.perf_counter()
            self._last_forces_ms = (t1 - t0 + t3 - t2) * 1000.0
            self._last_integrator_ms = (t2 - t1 + t4 - t3) * 1000.0
        
        self.system.positions = old_positions
        self.system.positions = new_positions
        self.system.velocities = new_velocities
        self.time += self.dt
        self.step_count += 1
        
        if self.on_step_callback:
            self.on_step_callback(self)
        if self.on_energy_callback:
            energy = self.system.compute_total_energy()
            self.on_energy_callback(self, energy)
    
    def run_steps(self, k: int):
        """Run k simulation steps without callbacks (for decoupled render: run K steps, then render)."""
        for _ in range(k):
            if self.paused:
                return
            self.step()
    
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
