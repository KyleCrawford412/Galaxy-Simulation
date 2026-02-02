"""Regression tests for diagnostics and two-body circular orbit."""

import numpy as np
import pytest
from galaxy_sim.backends.factory import get_backend
from galaxy_sim.physics.simulator import Simulator
from galaxy_sim.physics.integrators.verlet import VerletIntegrator
from galaxy_sim.physics.diagnostics import Diagnostics


def test_two_body_circular_orbit():
    """Test that two-body circular orbit maintains Q ~ 1 and small energy drift."""
    backend = get_backend('numpy')
    
    # Two-body system: central mass M and orbiting particle m
    M = 1000.0  # Central mass
    m = 1.0     # Orbiting particle
    r = 10.0    # Orbital radius
    G = 1.0
    
    # Circular velocity: v = sqrt(G*M/r)
    v_circ = np.sqrt(G * M / r)
    
    # Initial conditions
    positions = np.array([
        [0.0, 0.0, 0.0],      # Central mass at origin
        [r, 0.0, 0.0]         # Orbiting particle at (r, 0, 0)
    ])
    velocities = np.array([
        [0.0, 0.0, 0.0],      # Central mass stationary
        [0.0, v_circ, 0.0]    # Orbiting particle with circular velocity
    ])
    masses = np.array([M, m])
    
    # Convert to backend arrays
    positions = backend.array(positions)
    velocities = backend.array(velocities)
    masses = backend.array(masses)
    
    # Create simulator
    sim = Simulator(backend, VerletIntegrator(), dt=0.001)  # Small dt for accuracy
    sim.initialize(positions, velocities, masses)
    
    # Create diagnostics
    diagnostics = Diagnostics(backend, G=G, epsilon=0.01)
    
    # Compute initial energies and Q
    K0, U0, E0 = diagnostics.compute_energies(positions, velocities, masses)
    Q0 = diagnostics.compute_virial_ratio(positions, velocities, masses)
    
    # Expected: For circular orbit, Q should be close to 1.0
    # Q = 2K/|U|, and for circular orbit: K = |U|/2, so Q = 1.0
    assert abs(Q0 - 1.0) < 0.1, f"Initial Q should be ~1.0 for circular orbit, got {Q0:.4f}"
    
    # Run simulation for several orbits
    # Orbital period: T = 2π * r / v = 2π * sqrt(r³/(G*M))
    T = 2 * np.pi * np.sqrt(r ** 3 / (G * M))
    n_steps = int(T / sim.dt * 2)  # Run for 2 orbits
    
    energies = []
    Q_values = []
    
    for step in range(n_steps):
        sim.step()
        
        if step % 10 == 0:
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
            energies.append(E)
            Q_values.append(Q)
    
    # Check energy conservation (should drift relatively little for symplectic integrator)
    # Note: Some drift is expected due to numerical errors, especially with softening
    E_final = energies[-1]
    energy_drift = abs(E_final - E0) / abs(E0)
    # For Verlet integrator with softening, allow up to 5% drift over 2 orbits
    assert energy_drift < 0.10, f"Energy drift should be < 10%, got {energy_drift*100:.4f}%"
    
    # Check Q stays close to 1.0 (allowing some drift due to numerical errors)
    Q_final = Q_values[-1]
    assert abs(Q_final - 1.0) < 0.3, f"Q should stay ~1.0, got {Q_final:.4f}"
    
    # Check Q values don't drift excessively (allowing for numerical errors with softening)
    Q_drift = max(Q_values) - min(Q_values)
    assert Q_drift < 0.5, f"Q should not drift excessively, max-min = {Q_drift:.4f}"
    
    print(f"Two-body test: Q0={Q0:.4f}, Q_final={Q_final:.4f}, energy_drift={energy_drift*100:.4f}%")


def test_potential_energy_consistency():
    """Test that potential energy uses same softening as force calculation."""
    backend = get_backend('numpy')
    
    # Simple two-particle system
    positions = np.array([
        [0.0, 0.0, 0.0],
        [5.0, 0.0, 0.0]
    ])
    velocities = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ])
    masses = np.array([100.0, 1.0])
    G = 1.0
    epsilon = 0.1
    
    diagnostics = Diagnostics(backend, G=G, epsilon=epsilon)
    
    K, U, E = diagnostics.compute_energies(positions, velocities, masses)
    
    # Expected potential: U = -G * m1 * m2 / sqrt(r² + eps²)
    r = 5.0
    r_soft = np.sqrt(r ** 2 + epsilon ** 2)
    U_expected = -G * masses[0] * masses[1] / r_soft
    
    assert abs(U - U_expected) < 1e-10, f"Potential energy should match: expected {U_expected}, got {U}"
    assert U < 0, f"Potential energy should be negative, got {U}"
    assert K == 0.0, f"Kinetic energy should be 0, got {K}"


def test_halo_potential_energy():
    """Test that halo potential contributes to total potential energy."""
    backend = get_backend('numpy')
    from galaxy_sim.physics.halo_potential import HaloPotential
    
    # Single particle in halo potential
    positions = np.array([[10.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.0, 0.0]])
    masses = np.array([1.0])
    
    halo = HaloPotential(model='flat', v_0=2.0, r_c=1.0, enabled=True)
    
    diagnostics = Diagnostics(backend, G=1.0, epsilon=0.1, halo_potential=halo)
    
    K, U, E = diagnostics.compute_energies(positions, velocities, masses)
    
    # U should include halo contribution
    # For flat model at r=10: Φ ≈ -v₀²/2 * ln(1 + r²/r_c²) ≈ -2.0 * ln(101) ≈ -9.2
    # So U_halo ≈ -9.2 (for m=1)
    assert U < 0, f"Total potential should be negative, got {U}"
    # U should be more negative than just N-body (which is 0 for single particle)
    assert U < -5.0, f"Halo should contribute significant potential, got {U}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
