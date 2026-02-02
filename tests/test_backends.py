"""Tests for compute backends."""

import pytest
import numpy as np
from galaxy_sim.backends.factory import get_backend, list_available_backends
from galaxy_sim.backends.numpy_backend import NumPyBackend


def test_numpy_backend_basic():
    """Test basic NumPy backend operations."""
    backend = NumPyBackend()
    
    # Test array creation
    arr = backend.array([1, 2, 3])
    assert backend.to_numpy(arr).shape == (3,)
    
    # Test zeros
    zeros = backend.zeros((3, 3))
    assert backend.to_numpy(zeros).shape == (3, 3)
    assert np.allclose(backend.to_numpy(zeros), 0)
    
    # Test operations
    a = backend.array([1.0, 2.0, 3.0])
    b = backend.array([4.0, 5.0, 6.0])
    
    assert np.allclose(backend.to_numpy(backend.add(a, b)), [5, 7, 9])
    assert np.allclose(backend.to_numpy(backend.multiply(a, b)), [4, 10, 18])
    assert np.allclose(backend.to_numpy(backend.norm(a)), np.linalg.norm([1, 2, 3]))


def test_backend_factory():
    """Test backend factory."""
    # NumPy should always be available
    backends = list_available_backends()
    assert "numpy" in backends
    
    # Should be able to get NumPy backend
    backend = get_backend("numpy")
    assert backend.name == "numpy"
    
    # Auto-select should work
    backend = get_backend()
    assert backend is not None


def test_backend_seed():
    """Test seed setting for reproducibility."""
    backend = NumPyBackend()
    backend.set_seed(42)
    
    r1 = backend.random_uniform((10,), seed=42)
    r2 = backend.random_uniform((10,), seed=42)
    
    assert np.allclose(backend.to_numpy(r1), backend.to_numpy(r2))
