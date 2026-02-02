"""Tests for preset scenarios."""

import numpy as np
from galaxy_sim.backends.numpy_backend import NumPyBackend
from galaxy_sim.presets import SpiralGalaxy, CollisionScenario, GlobularCluster, GalaxyCluster


def test_spiral_galaxy():
    """Test spiral galaxy preset."""
    backend = NumPyBackend()
    preset = SpiralGalaxy(backend, n_particles=100, seed=42)
    
    positions, velocities, masses = preset.generate()
    
    assert positions.shape[0] == 100
    assert velocities.shape[0] == 100
    assert masses.shape[0] == 100
    assert preset.name == "spiral"


def test_collision_scenario():
    """Test collision scenario preset."""
    backend = NumPyBackend()
    preset = CollisionScenario(backend, n_particles=100, seed=42)
    
    positions, velocities, masses = preset.generate()
    
    assert positions.shape[0] == 100
    assert preset.name == "collision"


def test_globular_cluster():
    """Test globular cluster preset."""
    backend = NumPyBackend()
    preset = GlobularCluster(backend, n_particles=100, seed=42)
    
    positions, velocities, masses = preset.generate()
    
    assert positions.shape[0] == 100
    assert preset.name == "globular"


def test_galaxy_cluster():
    """Test galaxy cluster preset."""
    backend = NumPyBackend()
    preset = GalaxyCluster(backend, n_particles=100, seed=42)
    
    positions, velocities, masses = preset.generate()
    
    assert positions.shape[0] == 100
    assert preset.name == "cluster"


def test_preset_reproducibility():
    """Test that presets are reproducible with same seed."""
    backend = NumPyBackend()
    
    preset1 = SpiralGalaxy(backend, n_particles=100, seed=42)
    pos1, vel1, mass1 = preset1.generate()
    
    preset2 = SpiralGalaxy(backend, n_particles=100, seed=42)
    pos2, vel2, mass2 = preset2.generate()
    
    # Should be identical with same seed
    assert np.allclose(backend.to_numpy(pos1), backend.to_numpy(pos2))
    assert np.allclose(backend.to_numpy(vel1), backend.to_numpy(vel2))
    assert np.allclose(backend.to_numpy(mass1), backend.to_numpy(mass2))
