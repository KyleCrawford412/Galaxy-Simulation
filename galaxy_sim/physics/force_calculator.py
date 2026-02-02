"""Unified force calculation with vectorization and GPU optimization."""

import numpy as np
from typing import Tuple, Literal
from galaxy_sim.backends.base import Backend


class ForceCalculator:
    """Unified force calculation interface with auto-selection of method."""
    
    def __init__(
        self,
        method: Literal["direct", "vectorized"] = "auto",
        use_gpu: bool = True
    ):
        """Initialize force calculator.
        
        Args:
            method: Force calculation method ('direct', 'vectorized', or 'auto')
            use_gpu: Whether to prefer GPU-optimized methods
        """
        self.method = method
        self.use_gpu = use_gpu
    
    def compute_forces(
        self,
        positions,
        masses,
        backend: Backend,
        G: float = 1.0,
        epsilon: float = 1e-3
    ) -> Tuple:
        """Compute gravitational forces on all particles.
        
        Args:
            positions: Particle positions (n, dim)
            masses: Particle masses (n,)
            backend: Compute backend
            G: Gravitational constant
            epsilon: Softening parameter
            
        Returns:
            Tuple of force components (fx, fy, [fz])
        """
        # Check if backend is GPU-capable
        is_gpu = backend.device.startswith('cuda') or backend.device.startswith('gpu')
        
        if is_gpu and self.use_gpu:
            # Use GPU-optimized path
            forces = self._compute_forces_gpu_optimized(positions, masses, backend, G, epsilon)
        else:
            # Convert to numpy for reliable computation
            positions_np = np.asarray(backend.to_numpy(positions))
            masses_np = np.asarray(backend.to_numpy(masses)).flatten()
            forces = self._compute_forces_vectorized(positions_np, masses_np, G, epsilon)
        
        n = forces.shape[0]
        dim = forces.shape[1]
        
        # Return as tuple of components
        if dim == 3:
            return forces[:, 0], forces[:, 1], forces[:, 2]
        else:
            return forces[:, 0], forces[:, 1]
    
    def _compute_forces_vectorized(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
        G: float,
        epsilon: float
    ) -> np.ndarray:
        """Fully vectorized force calculation using broadcasting.
        
        This computes all pairwise forces at once using numpy broadcasting.
        Much faster than loop-based approach, especially on GPU.
        
        Args:
            positions: Particle positions (n, dim)
            masses: Particle masses (n,)
            G: Gravitational constant
            epsilon: Softening parameter
            
        Returns:
            Forces array (n, dim)
        """
        n = positions.shape[0]
        dim = positions.shape[1]
        
        # Compute all pairwise position differences using broadcasting
        # positions[i] - positions[j] for all i, j
        # Shape: (n, 1, dim) - (1, n, dim) = (n, n, dim)
        r_diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        
        # Distance squared: |r_ij|^2
        # Shape: (n, n)
        r_sq = np.sum(r_diff ** 2, axis=2) + epsilon ** 2
        
        # Distance: |r_ij|
        # Shape: (n, n)
        r = np.sqrt(r_sq)
        
        # Force magnitude: G * m_j / r^3
        # masses: (n,) -> (1, n) for broadcasting
        # Shape: (n, n)
        force_magnitude = G * masses[np.newaxis, :] / (r_sq * r)
        
        # Force vectors: force_magnitude * r_ij / r
        # force_magnitude: (n, n) -> (n, n, 1)
        # r_diff: (n, n, dim)
        # r: (n, n) -> (n, n, 1)
        # Shape: (n, n, dim)
        force_vectors = force_magnitude[:, :, np.newaxis] * r_diff / r[:, :, np.newaxis]
        
        # Zero out self-interactions (diagonal)
        # Create identity matrix to mask diagonal
        identity = np.eye(n, dtype=bool)
        force_vectors[identity] = 0.0
        
        # Sum forces from all particles
        # Shape: (n, dim)
        forces = np.sum(force_vectors, axis=1)
        
        return forces
    
    def _compute_forces_gpu_optimized(
        self,
        positions,
        masses,
        backend: Backend,
        G: float,
        epsilon: float
    ) -> np.ndarray:
        """GPU-optimized force calculation using backend operations.
        
        For GPU backends, this uses backend operations directly for better performance.
        
        Args:
            positions: Particle positions (backend array)
            masses: Particle masses (backend array)
            backend: Compute backend
            G: Gravitational constant
            epsilon: Softening parameter
            
        Returns:
            Forces array (n, dim) as numpy array
        """
        # For JAX backend, use JIT-compiled version
        if backend.name == "jax":
            return self._compute_forces_jax(positions, masses, backend, G, epsilon)
        
        # For other GPU backends, convert to numpy and use vectorized
        # (Can be optimized further with backend-specific code)
        positions_np = np.asarray(backend.to_numpy(positions))
        masses_np = np.asarray(backend.to_numpy(masses)).flatten()
        
        return self._compute_forces_vectorized(positions_np, masses_np, G, epsilon)
    
    def _compute_forces_jax(
        self,
        positions,
        masses,
        backend: Backend,
        G: float,
        epsilon: float
    ) -> np.ndarray:
        """JAX-optimized force calculation with JIT compilation."""
        try:
            import jax
            import jax.numpy as jnp
            
            # JIT-compile the force calculation
            @jax.jit
            def compute_forces_jax(positions_jax, masses_jax):
                n = positions_jax.shape[0]
                dim = positions_jax.shape[1]
                
                # Pairwise differences
                r_diff = positions_jax[:, jnp.newaxis, :] - positions_jax[jnp.newaxis, :, :]
                
                # Distance squared
                r_sq = jnp.sum(r_diff ** 2, axis=2) + epsilon ** 2
                r = jnp.sqrt(r_sq)
                
                # Force magnitude
                force_magnitude = G * masses_jax[jnp.newaxis, :] / (r_sq * r)
                
                # Force vectors
                force_vectors = force_magnitude[:, :, jnp.newaxis] * r_diff / r[:, :, jnp.newaxis]
                
                # Zero diagonal
                identity = jnp.eye(n, dtype=bool)
                force_vectors = jnp.where(identity[:, :, jnp.newaxis], 0.0, force_vectors)
                
                # Sum
                forces = jnp.sum(force_vectors, axis=1)
                return forces
            
            # Convert to JAX arrays
            positions_jax = jnp.array(backend.to_numpy(positions))
            masses_jax = jnp.array(backend.to_numpy(masses)).flatten()
            
            # Compute forces
            forces_jax = compute_forces_jax(positions_jax, masses_jax)
            
            # Convert back to numpy
            return np.asarray(forces_jax)
            
        except ImportError:
            # Fallback to numpy if JAX not available
            positions_np = np.asarray(backend.to_numpy(positions))
            masses_np = np.asarray(backend.to_numpy(masses)).flatten()
            return self._compute_forces_vectorized(positions_np, masses_np, G, epsilon)
