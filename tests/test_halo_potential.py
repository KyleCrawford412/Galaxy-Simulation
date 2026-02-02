"""Unit tests for halo potential."""

import numpy as np
import pytest
from galaxy_sim.backends.factory import get_backend
from galaxy_sim.physics.halo_potential import HaloPotential


def test_halo_potential_disabled():
    """Test that disabled halo returns None."""
    backend = get_backend('numpy')
    halo = HaloPotential(enabled=False)
    
    positions = backend.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    acc = halo.compute_acceleration(positions, backend)
    
    assert acc is None


def test_halo_flat_model_acceleration_inward():
    """Test that flat model acceleration points inward (attractive)."""
    backend = get_backend('numpy')
    halo = HaloPotential(model='flat', v_0=2.0, r_c=1.0, enabled=True)
    
    # Test particle at (3, 0, 0) - should accelerate toward origin
    positions = backend.array([[3.0, 0.0, 0.0]])
    acc = halo.compute_acceleration(positions, backend)
    acc_np = np.asarray(backend.to_numpy(acc))
    
    # Acceleration should point toward origin (negative x direction)
    assert acc_np[0, 0] < 0, "Acceleration should point inward (negative x)"
    assert acc_np[0, 1] == 0.0, "Y component should be zero"
    assert acc_np[0, 2] == 0.0, "Z component should be zero"
    
    # Test particle at (0, 4, 0) - should accelerate toward origin
    positions = backend.array([[0.0, 4.0, 0.0]])
    acc = halo.compute_acceleration(positions, backend)
    acc_np = np.asarray(backend.to_numpy(acc))
    
    # Acceleration should point toward origin (negative y direction)
    assert acc_np[0, 0] == 0.0, "X component should be zero"
    assert acc_np[0, 1] < 0, "Acceleration should point inward (negative y)"
    assert acc_np[0, 2] == 0.0, "Z component should be zero"


def test_halo_plummer_model_acceleration_inward():
    """Test that Plummer model acceleration points inward (attractive)."""
    backend = get_backend('numpy')
    halo = HaloPotential(model='plummer', M=1000.0, a=2.0, G=1.0, enabled=True)
    
    # Test particle at (5, 0, 0) - should accelerate toward origin
    positions = backend.array([[5.0, 0.0, 0.0]])
    acc = halo.compute_acceleration(positions, backend)
    acc_np = np.asarray(backend.to_numpy(acc))
    
    # Acceleration should point toward origin (negative x direction)
    assert acc_np[0, 0] < 0, "Acceleration should point inward (negative x)"
    assert acc_np[0, 1] == 0.0, "Y component should be zero"
    assert acc_np[0, 2] == 0.0, "Z component should be zero"


def test_halo_flat_model_magnitude():
    """Test flat model acceleration magnitude at different radii."""
    backend = get_backend('numpy')
    halo = HaloPotential(model='flat', v_0=2.0, r_c=1.0, enabled=True)
    
    # At large r, acceleration should be approximately -v₀²/r
    positions = backend.array([[10.0, 0.0, 0.0]])
    acc = halo.compute_acceleration(positions, backend)
    acc_np = np.asarray(backend.to_numpy(acc))
    acc_mag = np.linalg.norm(acc_np[0])
    
    # Expected: |a| ≈ v₀²/r = 4/10 = 0.4
    expected = 4.0 / 10.0  # v₀²/r
    assert abs(acc_mag - expected) < 0.1, f"At large r, |a| should be ≈ {expected}, got {acc_mag}"


def test_halo_plummer_model_magnitude():
    """Test Plummer model acceleration magnitude."""
    backend = get_backend('numpy')
    halo = HaloPotential(model='plummer', M=1000.0, a=2.0, G=1.0, enabled=True)
    
    # At r >> a, acceleration should be approximately -GM/r²
    positions = backend.array([[10.0, 0.0, 0.0]])
    acc = halo.compute_acceleration(positions, backend)
    acc_np = np.asarray(backend.to_numpy(acc))
    acc_mag = np.linalg.norm(acc_np[0])
    
    # Expected: |a| ≈ GM/r² = 1000/100 = 10 (for large r)
    # But with softening: |a| = GM*r/(r²+a²)^(3/2) = 1000*10/(104)^(3/2) ≈ 9.4
    expected_approx = 1000.0 * 10.0 / (104.0 ** 1.5)
    assert abs(acc_mag - expected_approx) < 1.0, f"At large r, |a| should be ≈ {expected_approx}, got {acc_mag}"


def test_halo_multiple_particles():
    """Test halo acceleration for multiple particles."""
    backend = get_backend('numpy')
    halo = HaloPotential(model='flat', v_0=1.0, r_c=1.0, enabled=True)
    
    # Multiple particles at different positions
    positions = backend.array([
        [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, 0.0, 3.0],
        [2.0, 2.0, 0.0]
    ])
    acc = halo.compute_acceleration(positions, backend)
    acc_np = np.asarray(backend.to_numpy(acc))
    
    # Should have shape (4, 3)
    assert acc_np.shape == (4, 3), f"Expected shape (4, 3), got {acc_np.shape}"
    
    # All accelerations should point toward origin
    for i in range(4):
        pos = positions[i]
        acc_vec = acc_np[i]
        # Acceleration should be opposite to position vector (pointing inward)
        dot_product = np.dot(pos, acc_vec)
        assert dot_product < 0, f"Particle {i}: acceleration should point inward, dot product = {dot_product}"


def test_halo_at_origin():
    """Test halo acceleration at origin (should be zero or very small)."""
    backend = get_backend('numpy')
    halo = HaloPotential(model='flat', v_0=1.0, r_c=1.0, enabled=True)
    
    # Particle at origin
    positions = backend.array([[0.0, 0.0, 0.0]])
    acc = halo.compute_acceleration(positions, backend)
    acc_np = np.asarray(backend.to_numpy(acc))
    
    # Acceleration should be very small (avoiding singularity)
    acc_mag = np.linalg.norm(acc_np[0])
    assert acc_mag < 1.0, f"At origin, acceleration should be small, got {acc_mag}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
