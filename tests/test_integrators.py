"""Tests for numerical integrators."""

import numpy as np
from galaxy_sim.backends.numpy_backend import NumPyBackend
from galaxy_sim.physics.integrators.euler import EulerIntegrator
from galaxy_sim.physics.integrators.verlet import VerletIntegrator
from galaxy_sim.physics.integrators.rk4 import RK4Integrator


def test_euler_integrator():
    """Test Euler integrator."""
    backend = NumPyBackend()
    integrator = EulerIntegrator()
    
    positions = backend.array([[0.0, 0.0], [1.0, 0.0]])
    velocities = backend.array([[0.0, 0.0], [0.0, 0.0]])
    masses = backend.array([1.0, 1.0])
    forces = (backend.array([0.0, 1.0]), backend.array([0.0, 0.0]))
    dt = 0.01
    
    new_pos, new_vel = integrator.step(positions, velocities, masses, forces, dt, backend)
    
    # Should have moved
    assert not np.allclose(backend.to_numpy(new_pos), backend.to_numpy(positions))
    assert integrator.name == "euler"
    assert integrator.order == 1


def test_verlet_integrator():
    """Test Verlet integrator."""
    backend = NumPyBackend()
    integrator = VerletIntegrator()
    
    positions = backend.array([[0.0, 0.0], [1.0, 0.0]])
    velocities = backend.array([[0.0, 0.0], [0.0, 0.0]])
    masses = backend.array([1.0, 1.0])
    forces = (backend.array([0.0, 1.0]), backend.array([0.0, 0.0]))
    dt = 0.01
    
    new_pos, new_vel = integrator.step(positions, velocities, masses, forces, dt, backend)
    
    # Should have moved
    assert not np.allclose(backend.to_numpy(new_pos), backend.to_numpy(positions))
    assert integrator.name == "verlet"
    assert integrator.order == 2


def test_rk4_integrator():
    """Test RK4 integrator."""
    backend = NumPyBackend()
    integrator = RK4Integrator()
    
    positions = backend.array([[0.0, 0.0], [1.0, 0.0]])
    velocities = backend.array([[0.0, 0.0], [0.0, 0.0]])
    masses = backend.array([1.0, 1.0])
    forces = (backend.array([0.0, 1.0]), backend.array([0.0, 0.0]))
    dt = 0.01
    
    new_pos, new_vel = integrator.step(positions, velocities, masses, forces, dt, backend)
    
    # Should have moved
    assert not np.allclose(backend.to_numpy(new_pos), backend.to_numpy(positions))
    assert integrator.name == "rk4"
    assert integrator.order == 4
