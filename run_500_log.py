"""Run 500 steps and log K, U, E, Lz, bound_fraction. Expect Lz stable and bound_fraction high with visible disk."""

import numpy as np
from galaxy_sim.backends.factory import get_backend
from galaxy_sim.physics.integrators.verlet import VerletIntegrator
from galaxy_sim.physics.simulator import Simulator
from galaxy_sim.physics.halo_potential import HaloPotential
from galaxy_sim.presets.multi_component import MultiComponentGalaxy
from galaxy_sim.physics.diagnostics import Diagnostics


def bound_fraction(positions_np, velocities_np, masses_np, G, eps):
    r = np.linalg.norm(positions_np, axis=1)
    v_sq = np.sum(velocities_np ** 2, axis=1)
    K_per = 0.5 * v_sq
    U_per = -G * np.sum(masses_np) / (r + eps)
    E_per = K_per + U_per
    return np.sum(E_per < 0) / len(E_per)


def main():
    backend = get_backend("numpy", prefer_gpu=False)
    halo = HaloPotential(enabled=True, model="flat", v_0=2.0, r_c=5.0, G=1.0)
    preset = MultiComponentGalaxy(
        backend=backend,
        n_particles=300,
        seed=42,
        n_disk=225,
        n_bulge=75,
        disk_scale_radius=15.0,
        bulge_scale_radius=4.0,
        bulge_velocity_dispersion=0.8,
        halo_potential=halo,
        use_analytic_halo=True,
    )
    positions, velocities, masses = preset.generate()
    particle_types = preset.particle_types

    sim = Simulator(
        backend=backend,
        integrator=VerletIntegrator(),
        dt=0.01,
        halo_potential=halo,
        self_gravity=False,
    )
    sim.preset = preset
    sim.initialize(positions, velocities, masses, virialize=True, target_Q=1.3, particle_types=particle_types)

    diagnostics = Diagnostics(
        backend,
        G=sim.system.G,
        epsilon=sim.system.epsilon,
        halo_potential=halo,
        self_gravity=False,
        particle_types=particle_types,
    )

    positions_np = np.asarray(backend.to_numpy(sim.system.positions))
    velocities_np = np.asarray(backend.to_numpy(sim.system.velocities))
    masses_np = np.asarray(backend.to_numpy(sim.system.masses)).flatten()

    Lz_0 = float(sim.system.compute_angular_momentum())
    K0, U0, E0 = diagnostics.compute_energies(sim.system.positions, sim.system.velocities, sim.system.masses)
    bf_0 = bound_fraction(positions_np, velocities_np, masses_np, sim.system.G, sim.system.epsilon)

    log_lines = []
    header = "step,time,K,U,E,Lz,bound_fraction"
    log_lines.append(header)
    log_lines.append(f"0,{sim.time:.4f},{K0:.4f},{U0:.4f},{E0:.4f},{Lz_0:.4f},{bf_0:.4f}")

    for step in range(1, 501):
        sim.step()
        if step % 50 == 0 or step == 500:
            K, U, E = diagnostics.compute_energies(
                sim.system.positions, sim.system.velocities, sim.system.masses
            )
            Lz = float(sim.system.compute_angular_momentum())
            positions_np = np.asarray(backend.to_numpy(sim.system.positions))
            velocities_np = np.asarray(backend.to_numpy(sim.system.velocities))
            bf = bound_fraction(positions_np, velocities_np, masses_np, sim.system.G, sim.system.epsilon)
            log_lines.append(f"{step},{sim.time:.4f},{K:.4f},{U:.4f},{E:.4f},{Lz:.4f},{bf:.4f}")

    K_f, U_f, E_f = diagnostics.compute_energies(
        sim.system.positions, sim.system.velocities, sim.system.masses
    )
    Lz_f = float(sim.system.compute_angular_momentum())
    positions_np = np.asarray(backend.to_numpy(sim.system.positions))
    velocities_np = np.asarray(backend.to_numpy(sim.system.velocities))
    bf_f = bound_fraction(positions_np, velocities_np, masses_np, sim.system.G, sim.system.epsilon)

    # Print log
    for line in log_lines:
        print(line)

    # Summary
    Lz_drift_pct = 100 * (Lz_f - Lz_0) / Lz_0 if Lz_0 != 0 else 0
    print()
    print("--- Summary after 500 steps ---")
    print(f"K,U,E: {K_f:.2f}, {U_f:.2f}, {E_f:.2f}")
    print(f"Lz: initial={Lz_0:.2f}, final={Lz_f:.2f}, drift={Lz_drift_pct:.4f}%")
    print(f"bound_fraction: initial={bf_0:.4f}, final={bf_f:.4f}")
    print(f"Lz stable (|drift|<1%): {abs(Lz_drift_pct) < 1}")
    print(f"bound_fraction high (final>0.8): {bf_f > 0.8}")
    print("For visible disk run with: --preset spiral_disk or disk_plus_bulge and --render")


if __name__ == "__main__":
    main()
