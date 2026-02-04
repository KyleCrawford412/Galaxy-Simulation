"""Tests for physics engine."""

import numpy as np
from galaxy_sim.backends.numpy_backend import NumPyBackend
from galaxy_sim.physics.nbody import NBodySystem
from galaxy_sim.physics.simulator import Simulator
from galaxy_sim.physics.integrators.verlet import VerletIntegrator


def test_nbody_initialization():
    """Test N-body system initialization."""
    backend = NumPyBackend()
    system = NBodySystem(backend)
    
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    masses = np.array([1.0, 1.0])
    
    system.initialize(positions, velocities, masses)
    
    assert system.n_particles == 2
    assert np.allclose(backend.to_numpy(system.positions), positions)


def test_force_calculation():
    """Test gravitational force calculation."""
    backend = NumPyBackend()
    system = NBodySystem(backend)
    
    # Two particles separated by distance 1
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    masses = np.array([1.0, 1.0])
    
    system.initialize(positions, velocities, masses)
    forces = system.compute_forces()
    
    # Should have non-zero forces
    fx, fy, fz = forces
    assert not np.allclose(backend.to_numpy(fx), 0)


def test_energy_calculation():
    """Test energy computation."""
    backend = NumPyBackend()
    system = NBodySystem(backend)
    
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    masses = np.array([1.0, 1.0])
    
    system.initialize(positions, velocities, masses)
    
    ke = system.compute_kinetic_energy()
    pe = system.compute_potential_energy()
    total = system.compute_total_energy()
    
    assert ke >= 0  # Kinetic energy should be non-negative
    assert pe < 0  # Potential energy should be negative (attractive)
    assert abs(total - (ke + pe)) < 1e-10  # Should sum correctly


def test_simulator_basic():
    """Test basic simulator operation."""
    backend = NumPyBackend()
    integrator = VerletIntegrator()
    sim = Simulator(backend, integrator, dt=0.01)
    
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    velocities = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    masses = np.array([1.0, 1.0])
    
    sim.initialize(positions, velocities, masses)
    
    initial_energy = sim.get_energy()
    
    # Run a few steps
    for _ in range(10):
        sim.step()
    
    # System should have evolved
    assert sim.time > 0
    assert sim.step_count == 10
    
    # Energy should be approximately conserved (with Verlet)
    final_energy = sim.get_energy()
    energy_error = abs(final_energy - initial_energy) / abs(initial_energy)
    assert energy_error < 0.1  # Within 10% (allowing for numerical errors)


def test_softened_circular_orbits():
    """Test near-circular orbits at r ~ epsilon."""
    backend = NumPyBackend()
    eps = 0.1
    M = 1000.0
    m = 1.0
    dt = 0.001
    integrator = VerletIntegrator()
    sim = Simulator(backend, integrator, dt=dt)
    for r in [2 * eps, 5 * eps, 10 * eps]:
        positions = np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]])
        v_circ = np.sqrt(1.0 * M * r / ((r ** 2 + eps ** 2) ** 1.5))
        velocities = np.array([[0.0, 0.0, 0.0], [0.0, v_circ, 0.0]])
        masses = np.array([M, m])
        sim.system = NBodySystem(backend, epsilon=eps)
        sim.initialize(positions, velocities, masses)
        radii = []
        for _ in range(500):
            sim.step()
            pos = backend.to_numpy(sim.system.positions)
            rel = pos[1] - pos[0]
            radii.append(np.linalg.norm(rel))
        r_med = np.median(radii)
        assert abs(r_med - r) / r < 0.3
