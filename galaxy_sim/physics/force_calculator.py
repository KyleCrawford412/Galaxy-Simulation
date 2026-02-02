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
        epsilon: float = 1e-3,
        characteristic_size: float = None
    ) -> Tuple:
        """Compute gravitational forces on all particles.
        
        Args:
            positions: Particle positions (n, dim)
            masses: Particle masses (n,)
            backend: Compute backend
            G: Gravitational constant
            epsilon: Base softening parameter
            characteristic_size: Characteristic size of system (for distance-dependent epsilon)
            
        Returns:
            Tuple of force components (fx, fy, [fz])
        """
        # Check if backend is GPU-capable
        is_gpu = backend.device.startswith('cuda') or backend.device.startswith('gpu')
        
        if is_gpu and self.use_gpu:
            # Use GPU-optimized path
            forces = self._compute_forces_gpu_optimized(positions, masses, backend, G, epsilon, characteristic_size)
        else:
            # Convert to numpy for reliable computation
            positions_np = np.asarray(backend.to_numpy(positions))
            # Ensure positions is 2D (n, dim)
            original_shape = positions_np.shape
            if positions_np.ndim > 2:
                # If 3D or higher, flatten all but last dimension
                # e.g., (n, n, dim) -> (n*n, dim) then take first n rows
                positions_np = positions_np.reshape(-1, positions_np.shape[-1])
                # If we have too many rows, take first n (where n = sqrt of first dim)
                n_expected = int(np.sqrt(original_shape[0])) if len(original_shape) >= 2 else positions_np.shape[0]
                if positions_np.shape[0] > n_expected:
                    positions_np = positions_np[:n_expected, :]
            elif positions_np.ndim == 1:
                # If 1D, reshape to (n, 1)
                positions_np = positions_np.reshape(-1, 1)
            elif positions_np.ndim == 0:
                # Scalar - shouldn't happen
                raise ValueError(f"Positions should be at least 1D, got scalar")
            
            # Final check: ensure 2D
            if positions_np.ndim != 2:
                raise ValueError(f"After processing, positions should be 2D, got shape {positions_np.shape} (original: {original_shape})")
            
            masses_np = np.asarray(backend.to_numpy(masses)).flatten()
            forces = self._compute_forces_vectorized(positions_np, masses_np, G, epsilon, characteristic_size)
        
        # Verify forces shape is (n, dim)
        if forces.ndim != 2:
            raise ValueError(f"Forces should be 2D (n, dim), got shape {forces.shape} with ndim={forces.ndim}")
        
        n = forces.shape[0]
        dim = forces.shape[1]
        
        # Return as tuple of components (1D arrays)
        if dim == 3:
            fx = forces[:, 0]  # Shape: (n,)
            fy = forces[:, 1]  # Shape: (n,)
            fz = forces[:, 2]  # Shape: (n,)
            # Ensure 1D
            if fx.ndim > 1:
                fx = fx.flatten()
            if fy.ndim > 1:
                fy = fy.flatten()
            if fz.ndim > 1:
                fz = fz.flatten()
            return fx, fy, fz
        else:
            fx = forces[:, 0]  # Shape: (n,)
            fy = forces[:, 1]  # Shape: (n,)
            # Ensure 1D
            if fx.ndim > 1:
                fx = fx.flatten()
            if fy.ndim > 1:
                fy = fy.flatten()
            return fx, fy
    
    def _compute_forces_vectorized(
        self,
        positions: np.ndarray,
        masses: np.ndarray,
        G: float,
        epsilon: float,
        characteristic_size: float = None
    ) -> np.ndarray:
        """Fully vectorized force calculation using broadcasting.
        
        This computes all pairwise forces at once using numpy broadcasting.
        Much faster than loop-based approach, especially on GPU.
        
        Args:
            positions: Particle positions (n, dim)
            masses: Particle masses (n,)
            G: Gravitational constant
            epsilon: Base softening parameter
            characteristic_size: Characteristic size of system (for distance-dependent epsilon)
            
        Returns:
            Forces array (n, dim)
        """
        n = positions.shape[0]
        dim = positions.shape[1]
        
        # Ensure positions is 2D (n, dim)
        if positions.ndim != 2:
            raise ValueError(f"positions should be 2D (n, dim), got shape {positions.shape} with ndim={positions.ndim}")
        
        # Compute all pairwise position differences using broadcasting
        # r_diff = r_j - r_i (direction from particle i to particle j)
        # This gives the correct force direction: force on i from j points from i toward j
        # Shape: (1, n, dim) - (n, 1, dim) = (n, n, dim)
        r_diff = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
        
        # Verify r_diff shape
        if r_diff.ndim != 3:
            raise ValueError(f"r_diff should be 3D (n, n, dim), got shape {r_diff.shape} from positions shape {positions.shape}")
        
        # Distance squared: |r_ij|^2 = sum over spatial dimensions
        # Shape: (n, n, dim) -> (n, n) after sum over axis=2
        # Use keepdims=False to ensure we get 2D result
        r_sq = np.sum(r_diff ** 2, axis=2, keepdims=False)
        
        # Ensure r_sq is 2D (n, n)
        if r_sq.ndim != 2:
            # If somehow still 3D, try reshaping
            if r_sq.ndim == 3 and r_sq.shape[2] == 1:
                r_sq = r_sq[:, :, 0]
            else:
                raise ValueError(f"r_sq should be 2D (n, n), got shape {r_sq.shape} from r_diff shape {r_diff.shape}")
        
        # Distance: |r_ij|
        # Shape: (n, n)
        r = np.sqrt(r_sq)
        
        # Distance-dependent epsilon: smaller epsilon for close particles (stronger forces)
        # larger epsilon for distant particles (prevents numerical issues)
        if characteristic_size is not None and characteristic_size > 0:
            # Epsilon scales with distance: epsilon(r) = epsilon_base * (1 + r / (0.1 * char_size))
            # This makes epsilon smaller for close encounters, larger for distant ones
            epsilon_scale = 1.0 + r / (0.1 * characteristic_size)
            epsilon_effective = epsilon * epsilon_scale
        else:
            epsilon_effective = epsilon
        
        # Gravitational softening: use (r^2 + eps^2)^(3/2) in denominator
        # This is the standard Plummer softening
        # Softened distance cubed: (r^2 + eps^2)^(3/2)
        # Ensure r_sq is 2D: (n, n)
        r_soft_cubed = (r_sq + epsilon_effective ** 2) ** 1.5
        
        # Force magnitude: G * m_i * m_j / (r^2 + eps^2)^(3/2)
        # This returns TRUE FORCES (not accelerations), so integrator division by m_i is correct
        # masses: (n,) -> (n, 1) and (1, n) for broadcasting
        # Shape: (n, n)
        force_magnitude = G * masses[:, np.newaxis] * masses[np.newaxis, :] / r_soft_cubed
        
        # Force vectors: force_magnitude * r_diff
        # Since force_magnitude already has (r^2 + eps^2)^(3/2) in denominator,
        # we multiply by r_diff directly (no need to divide by r_soft again)
        # force_magnitude: (n, n) -> (n, n, 1)
        # r_diff: (n, n, dim)
        # Shape: (n, n, dim)
        force_vectors = force_magnitude[:, :, np.newaxis] * r_diff
        
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
        epsilon: float,
        characteristic_size: float = None
    ) -> np.ndarray:
        """GPU-optimized force calculation using backend operations.
        
        For GPU backends, this uses backend operations directly for better performance.
        
        Args:
            positions: Particle positions (backend array)
            masses: Particle masses (backend array)
            backend: Compute backend
            G: Gravitational constant
            epsilon: Base softening parameter
            characteristic_size: Characteristic size of system
            
        Returns:
            Forces array (n, dim) as numpy array
        """
        # For JAX backend, use JIT-compiled version
        if backend.name == "jax":
            return self._compute_forces_jax(positions, masses, backend, G, epsilon, characteristic_size)
        
        # For other GPU backends, convert to numpy and use vectorized
        # (Can be optimized further with backend-specific code)
        positions_np = np.asarray(backend.to_numpy(positions))
        masses_np = np.asarray(backend.to_numpy(masses)).flatten()
        
        return self._compute_forces_vectorized(positions_np, masses_np, G, epsilon, characteristic_size)
    
    def _compute_forces_jax(
        self,
        positions,
        masses,
        backend: Backend,
        G: float,
        epsilon: float,
        characteristic_size: float = None
    ) -> np.ndarray:
        """JAX-optimized force calculation with JIT compilation."""
        try:
            import jax
            import jax.numpy as jnp
            
            # JIT-compile the force calculation
            @jax.jit
            def compute_forces_jax(positions_jax, masses_jax, epsilon_val, char_size):
                n = positions_jax.shape[0]
                dim = positions_jax.shape[1]
                
                # Pairwise differences: r_diff = r_j - r_i (direction from i to j)
                r_diff = positions_jax[jnp.newaxis, :, :] - positions_jax[:, jnp.newaxis, :]
                
                # Distance squared
                r_sq = jnp.sum(r_diff ** 2, axis=2)
                
                # Distance
                r = jnp.sqrt(r_sq)
                
                # Distance-dependent epsilon
                if char_size is not None and char_size > 0:
                    epsilon_scale = 1.0 + r / (0.1 * char_size)
                    epsilon_effective = epsilon_val * epsilon_scale
                else:
                    epsilon_effective = epsilon_val
                
                # Gravitational softening: (r^2 + eps^2)^(3/2)
                r_soft_cubed = (r_sq + epsilon_effective ** 2) ** 1.5
                
                # Force magnitude: G * m_i * m_j / (r^2 + eps^2)^(3/2)
                # Returns TRUE FORCES (not accelerations)
                force_magnitude = G * masses_jax[:, jnp.newaxis] * masses_jax[jnp.newaxis, :] / r_soft_cubed
                
                # Force vectors: force_magnitude * r_diff (no division by r_soft)
                force_vectors = force_magnitude[:, :, jnp.newaxis] * r_diff
                
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
            forces_jax = compute_forces_jax(positions_jax, masses_jax, epsilon, characteristic_size)
            
            # Convert back to numpy
            return np.asarray(forces_jax)
            
        except ImportError:
            # Fallback to numpy if JAX not available
            positions_np = np.asarray(backend.to_numpy(positions))
            masses_np = np.asarray(backend.to_numpy(masses)).flatten()
            return self._compute_forces_vectorized(positions_np, masses_np, G, epsilon)
