"""Tests for I/O functionality."""

import numpy as np
import tempfile
import os
from galaxy_sim.io.state_io import save_state, load_state


def test_save_load_npz():
    """Test saving and loading NPZ format."""
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    velocities = np.array([[0.0, 1.0], [0.0, -1.0]])
    masses = np.array([1.0, 2.0])
    metadata = {"time": 10.0, "steps": 100}
    
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        temp_path = f.name
    
    try:
        save_state(positions, velocities, masses, temp_path, metadata)
        
        loaded_pos, loaded_vel, loaded_mass, loaded_meta = load_state(temp_path)
        
        assert np.allclose(positions, loaded_pos)
        assert np.allclose(velocities, loaded_vel)
        assert np.allclose(masses, loaded_mass)
        assert loaded_meta.get("time") == 10.0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_save_load_json():
    """Test saving and loading JSON format."""
    positions = np.array([[0.0, 0.0], [1.0, 0.0]])
    velocities = np.array([[0.0, 1.0], [0.0, -1.0]])
    masses = np.array([1.0, 2.0])
    metadata = {"time": 10.0, "steps": 100}
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    try:
        save_state(positions, velocities, masses, temp_path, metadata)
        
        loaded_pos, loaded_vel, loaded_mass, loaded_meta = load_state(temp_path)
        
        assert np.allclose(positions, loaded_pos)
        assert np.allclose(velocities, loaded_vel)
        assert np.allclose(masses, loaded_mass)
        assert loaded_meta.get("time") == 10.0
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
