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
from galaxy_sim.physics.diagnostics import Diagnostics


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
        self.debug_inner: bool = False
        self.debug_interval: int = 50
        self.debug_fraction: float = 0.1
        self.debug_table: bool = True
        self.debug_table_interval: int = 100
    
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
        
        # Pass r_min_disk for softening policy if preset provides it
        preset = getattr(self, 'preset', None)
        if preset is not None and getattr(preset, 'min_radius_non_core', None) is not None:
            self.system.r_min_disk = float(preset.min_radius_non_core)

        self.system.initialize(positions_centered, velocities_centered, masses, particle_types=particle_types)

        # Configure analytic bulge/disk potentials if requested by preset
        if preset is not None:
            if getattr(preset, 'use_analytic_bulge', False):
                bulge_scale = getattr(preset, 'bulge_radius', 1.0) * 0.3
                self.system.analytic_bulge_potential = HaloPotential(
                    model="plummer",
                    M=getattr(preset, 'bulge_mass', 0.0),
                    a=bulge_scale,
                    enabled=True,
                    G=self.system.G,
                )
            else:
                self.system.analytic_bulge_potential = None
            if getattr(preset, 'use_analytic_disk', False):
                disk_mass_scale = 1.1 if getattr(preset, 'gravity_mode', None) == "test_particles" else 1.0
                self.system.analytic_disk_potential = HaloPotential(
                    model="plummer",
                    M=getattr(preset, 'disk_mass', 0.0) * disk_mass_scale,
                    a=getattr(preset, 'disk_scale_radius', 1.0),
                    enabled=True,
                    G=self.system.G,
                )
            else:
                self.system.analytic_disk_potential = None
        
        # Use consistent diagnostics for virial ratio
        from galaxy_sim.physics.diagnostics import Diagnostics
        
        diagnostics = Diagnostics(
            self.backend,
            G=self.system.G,
            epsilon=self.system.epsilon,
            halo_potential=self.system.halo_potential,
            analytic_bulge_potential=self.system.analytic_bulge_potential,
            analytic_disk_potential=self.system.analytic_disk_potential,
            self_gravity=self.system.self_gravity,
            particle_types=self.system.particle_types,
            eps_cd=self.system.eps_cd,
            eps_bd=self.system.eps_bd,
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
                central_mass = getattr(preset, 'core_mass', getattr(preset, 'central_mass', 100.0)) if preset else 100.0
                disk_mass = getattr(preset, 'disk_mass', 1000.0) if preset else 1000.0
                disk_scale_radius = getattr(preset, 'disk_scale_radius', 10.0) if preset else 10.0
                positions_np = np.asarray(self.backend.to_numpy(self.system.positions))
                masses_np = np.asarray(self.backend.to_numpy(self.system.masses)).flatten()
                is_bulge = (np.asarray(particle_types) == 'bulge')
                bulge_positions = positions_np[is_bulge] if np.any(is_bulge) else None
                bulge_masses = masses_np[is_bulge] if np.any(is_bulge) else None
                eps = getattr(self.system, 'epsilon', 1e-6)

                low_n = self.system.n_particles < NBodySystem.LOW_N_THRESHOLD
                f_rot = 0.93 if low_n else 1.1
                sigma_r_fraction = 0.02 if low_n else 0.05
                sigma_t_fraction = 0.02 if low_n else 0.03
                allow_f_rot_adjust = False if low_n else True
                new_velocities, Q_final, scale_info = virialize_component_wise(
                    self.system.positions,
                    self.system.velocities,
                    self.system.masses,
                    self.backend,
                    diagnostics,
                    target_Q=target_Q,
                    particle_types=particle_types,
                    f_rot=f_rot,
                    sigma_r_fraction=sigma_r_fraction,
                    sigma_t_fraction=sigma_t_fraction,
                    allow_f_rot_adjust=allow_f_rot_adjust,
                    halo_potential=halo_potential,
                    G=G,
                    central_mass=central_mass,
                    disk_mass=disk_mass,
                    disk_scale_radius=disk_scale_radius,
                    use_analytic_disk=True,
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
        self.system.time = self.time
        
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
        
        if self.debug_inner and (self.step_count % self.debug_interval == 0):
            self._log_inner_disk_state()
        if self.debug_table and (self.step_count % self.debug_table_interval == 0):
            self._log_stability_table()
        
        if self.on_step_callback:
            self.on_step_callback(self)
        if self.on_energy_callback:
            energy = self.system.compute_total_energy()
            self.on_energy_callback(self, energy)

    def _log_inner_disk_state(self):
        """Log specific energy for innermost disk particles."""
        if self.system.particle_types is None:
            return
        positions_np = np.asarray(self.backend.to_numpy(self.system.positions))
        velocities_np = np.asarray(self.backend.to_numpy(self.system.velocities))
        masses_np = np.asarray(self.backend.to_numpy(self.system.masses)).flatten()
        types_np = np.asarray(self.system.particle_types)
        is_disk = (types_np == 'disk')
        if not np.any(is_disk):
            return
        disk_idx = np.where(is_disk)[0]
        disk_positions = positions_np[is_disk]
        radii = np.linalg.norm(disk_positions, axis=1)
        if radii.size == 0:
            return
        n_inner = max(1, int(self.debug_fraction * radii.size))
        inner_indices = disk_idx[np.argsort(radii)[:n_inner]]
        energies = []
        for i in inner_indices:
            v_sq = np.sum(velocities_np[i] ** 2)
            phi = 0.0
            for j in range(len(masses_np)):
                if i == j:
                    continue
                r_diff = positions_np[j] - positions_np[i]
                r_sq = np.sum(r_diff ** 2)
                eps_ij = self.system.epsilon
                if self.system.eps_cd is not None:
                    if (types_np[i] == 'disk' and types_np[j] == 'core') or (types_np[i] == 'core' and types_np[j] == 'disk'):
                        eps_ij = self.system.eps_cd
                if self.system.eps_bd is not None:
                    if (types_np[i] == 'disk' and types_np[j] == 'bulge') or (types_np[i] == 'bulge' and types_np[j] == 'disk'):
                        eps_ij = self.system.eps_bd
                phi -= self.system.G * masses_np[j] / np.sqrt(r_sq + eps_ij ** 2)
            e_spec = 0.5 * v_sq + phi
            energies.append(e_spec)
        bound_frac = np.mean(np.array(energies) < 0.0)
        r_min = float(np.min(radii))
        r_med = float(np.median(radii))
        print(f"[InnerDisk] step={self.step_count} bound_frac={bound_frac:.2f} r_min={r_min:.3f} r_med={r_med:.3f}")

    def _log_stability_table(self):
        """Log K, U, E, Lz, bound fraction."""
        diagnostics = Diagnostics(
            self.backend,
            G=self.system.G,
            epsilon=self.system.epsilon,
            halo_potential=self.system.halo_potential,
            analytic_bulge_potential=self.system.analytic_bulge_potential,
            analytic_disk_potential=self.system.analytic_disk_potential,
            self_gravity=self.system.self_gravity,
            particle_types=self.system.particle_types,
            eps_cd=self.system.eps_cd,
            eps_bd=self.system.eps_bd,
        )
        K, U, E = diagnostics.compute_energies(
            self.system.positions,
            self.system.velocities,
            self.system.masses,
        )
        Lz = self.system.compute_angular_momentum()
        bound_frac = diagnostics.compute_bound_fraction(
            self.system.positions,
            self.system.velocities,
            self.system.masses,
        )
        print(f"[Diag] step={self.step_count} K={K:.4f} U={U:.4f} E={E:.4f} Lz={float(Lz):.4f} bound={bound_frac:.2f}")
    
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
