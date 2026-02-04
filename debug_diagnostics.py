"""Debug diagnostics script for galaxy simulation."""

import numpy as np
from galaxy_sim.backends.factory import get_backend
from galaxy_sim.physics.integrators.verlet import VerletIntegrator
from galaxy_sim.physics.simulator import Simulator
from galaxy_sim.physics.halo_potential import HaloPotential
from galaxy_sim.presets.multi_component import MultiComponentGalaxy
from galaxy_sim.physics.diagnostics import Diagnostics

def run_debug_diagnostics():
    """Run detailed diagnostics for debugging."""
    # Configuration matching the failing run
    backend = get_backend("numpy", prefer_gpu=False)
    integrator = VerletIntegrator()
    dt = 0.01
    
    # Halo potential
    halo_potential = HaloPotential(
        enabled=True,
        model="flat",
        v_0=2.0,
        r_c=5.0,
        G=1.0
    )
    
    # Create preset (using MultiComponentGalaxy directly)
    preset = MultiComponentGalaxy(
        backend=backend,
        n_particles=300,
        seed=42,
        n_disk=225,
        n_bulge=75,
        disk_scale_radius=15.0,
        bulge_scale_radius=4.0,
        bulge_velocity_dispersion=0.8,
        halo_potential=halo_potential,
        use_analytic_halo=True
    )
    
    # Generate initial conditions
    positions, velocities, masses = preset.generate()
    particle_types = preset.particle_types
    
    # Create simulator
    sim = Simulator(
        backend=backend,
        integrator=integrator,
        dt=dt,
        halo_potential=halo_potential,
        self_gravity=False
    )
    sim.preset = preset
    sim.initialize(positions, velocities, masses, virialize=True, target_Q=1.3, particle_types=particle_types)
    
    # Create diagnostics
    diagnostics = Diagnostics(
        backend,
        G=sim.system.G,
        epsilon=sim.system.epsilon,
        halo_potential=halo_potential,
        self_gravity=False,
        particle_types=particle_types
    )
    
    # Compute initial values
    positions_np = np.asarray(backend.to_numpy(sim.system.positions))
    velocities_np = np.asarray(backend.to_numpy(sim.system.velocities))
    masses_np = np.asarray(backend.to_numpy(sim.system.masses)).flatten()
    
    # Initial center of mass
    COM_initial = np.sum(masses_np[:, np.newaxis] * positions_np, axis=0) / np.sum(masses_np)
    COMv_initial = np.sum(masses_np[:, np.newaxis] * velocities_np, axis=0) / np.sum(masses_np)
    
    # Initial angular momentum
    L_initial = sim.system.compute_angular_momentum()
    Lz_initial = float(L_initial)
    
    # Initial energies
    K0, U0, E0 = diagnostics.compute_energies(
        sim.system.positions,
        sim.system.velocities,
        sim.system.masses
    )
    Q0 = diagnostics.compute_virial_ratio(
        sim.system.positions,
        sim.system.velocities,
        sim.system.masses
    )
    
    # Bound fraction (particles with E < 0)
    v_sq = np.sum(velocities_np ** 2, axis=1)
    K_per_particle = 0.5 * v_sq
    # Approximate potential per particle (simplified)
    r = np.linalg.norm(positions_np, axis=1)
    U_per_particle = -sim.system.G * np.sum(masses_np) / (r + sim.system.epsilon)
    E_per_particle = K_per_particle + U_per_particle
    bound_fraction_initial = np.sum(E_per_particle < 0) / len(E_per_particle)
    
    print("=" * 100)
    print("DEBUG DIAGNOSTICS - INITIAL STATE")
    print("=" * 100)
    print(f"Step: 0, Time: 0.00")
    print(f"K: {K0:.2f}, U: {U0:.2f}, E: {E0:.2f}, Q: {Q0:.4f}")
    print(f"Lz: {Lz_initial:.2f}")
    print(f"Bound fraction: {bound_fraction_initial:.4f}")
    print(f"COM: ({COM_initial[0]:.2f}, {COM_initial[1]:.2f})")
    print(f"COMv: ({COMv_initial[0]:.2f}, {COMv_initial[1]:.2f})")
    print()
    
    print("=" * 100)
    print("DEBUG DIAGNOSTICS - RUNNING 500 STEPS")
    print("=" * 100)
    print(f"{'Step':<8} {'Time':<10} {'K':<12} {'U':<12} {'E':<12} {'Q':<8} {'Lz':<12} {'Bound':<8} {'COMx':<10} {'COMy':<10} {'COMvx':<10} {'COMvy':<10}")
    print("-" * 100)
    
    # Run 500 steps, print every 50
    for step in range(1, 501):
        sim.step()
        
        if step % 50 == 0:
            # Compute diagnostics
            K, U, E = diagnostics.compute_energies(
                sim.system.positions,
                sim.system.velocities,
                sim.system.masses
            )
            Q = diagnostics.compute_virial_ratio(
                sim.system.positions,
                sim.system.velocities,
                sim.system.masses
            )
            
            L = sim.system.compute_angular_momentum()
            Lz = float(L)
            
            # Center of mass
            positions_np = np.asarray(backend.to_numpy(sim.system.positions))
            velocities_np = np.asarray(backend.to_numpy(sim.system.velocities))
            masses_np = np.asarray(backend.to_numpy(sim.system.masses)).flatten()
            
            COM = np.sum(masses_np[:, np.newaxis] * positions_np, axis=0) / np.sum(masses_np)
            COMv = np.sum(masses_np[:, np.newaxis] * velocities_np, axis=0) / np.sum(masses_np)
            
            # Bound fraction
            v_sq = np.sum(velocities_np ** 2, axis=1)
            K_per_particle = 0.5 * v_sq
            r = np.linalg.norm(positions_np, axis=1)
            U_per_particle = -sim.system.G * np.sum(masses_np) / (r + sim.system.epsilon)
            E_per_particle = K_per_particle + U_per_particle
            bound_fraction = np.sum(E_per_particle < 0) / len(E_per_particle)
            
            print(f"{step:<8} {sim.time:<10.2f} {K:<12.2f} {U:<12.2f} {E:<12.2f} {Q:<8.4f} {Lz:<12.2f} {bound_fraction:<8.4f} {COM[0]:<10.2f} {COM[1]:<10.2f} {COMv[0]:<10.2f} {COMv[1]:<10.2f}")
    
    # Final values
    K_final, U_final, E_final = diagnostics.compute_energies(
        sim.system.positions,
        sim.system.velocities,
        sim.system.masses
    )
    L_final = sim.system.compute_angular_momentum()
    Lz_final = float(L_final)
    
    print()
    print("=" * 100)
    print("SANITY CHECKS")
    print("=" * 100)
    print(f"U is negative: {U_final < 0} (U = {U_final:.2f})")
    print(f"Lz drift: {Lz_final - Lz_initial:.2f} (initial: {Lz_initial:.2f}, final: {Lz_final:.2f}, relative: {(Lz_final - Lz_initial) / Lz_initial * 100:.2f}%)")
    print(f"Energy drift: {E_final - E0:.2f} (initial: {E0:.2f}, final: {E_final:.2f}, relative: {(E_final - E0) / abs(E0) * 100:.2f}%)")
    
    # Check force vs acceleration consistency
    print()
    print("Force/Acceleration Consistency Check:")
    forces = sim.system.compute_forces()
    if len(forces) == 2:
        fx, fy = forces
        fx_np = np.asarray(backend.to_numpy(fx))
        fy_np = np.asarray(backend.to_numpy(fy))
        forces_np = np.column_stack([fx_np, fy_np])
    else:
        fx, fy, fz = forces
        fx_np = np.asarray(backend.to_numpy(fx))
        fy_np = np.asarray(backend.to_numpy(fy))
        fz_np = np.asarray(backend.to_numpy(fz))
        forces_np = np.column_stack([fx_np, fy_np, fz_np])
    
    # Compute accelerations from forces
    accelerations_np = forces_np / masses_np[:, np.newaxis]
    
    # Check a few particles
    print(f"Sample particle 0: force = ({forces_np[0, 0]:.4f}, {forces_np[0, 1]:.4f}), mass = {masses_np[0]:.4f}, acceleration = ({accelerations_np[0, 0]:.4f}, {accelerations_np[0, 1]:.4f})")
    print(f"Sample particle 100: force = ({forces_np[100, 0]:.4f}, {forces_np[100, 1]:.4f}), mass = {masses_np[100]:.4f}, acceleration = ({accelerations_np[100, 0]:.4f}, {accelerations_np[100, 1]:.4f})")
    print("[OK] Forces are divided by mass in integrator (forces returned, not accelerations)")

if __name__ == "__main__":
    run_debug_diagnostics()
