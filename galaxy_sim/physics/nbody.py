"""Core N-body physics calculations."""

import numpy as np
from typing import Tuple
from galaxy_sim.backends.base import Backend


class NBodySystem:
    """N-body gravitational system.
    
    Handles force calculations, energy computation, and particle state management.
    """
    
    G = 1.0  # Gravitational constant (normalized units)
    EPSILON = 1e-3  # Softening parameter to prevent singularities
    
    def __init__(self, backend: Backend):
        """Initialize N-body system.
        
        Args:
            backend: Compute backend for array operations
        """
        self.backend = backend
        self.positions = None
        self.velocities = None
        self.masses = None
        self.n_particles = 0
    
    def initialize(self, positions, velocities, masses):
        """Initialize particle state.
        
        Args:
            positions: Array of shape (n, 3) for 3D or (n, 2) for 2D
            velocities: Array of shape (n, 3) or (n, 2)
            masses: Array of shape (n,)
        """
        self.positions = self.backend.array(positions)
        self.velocities = self.backend.array(velocities)
        self.masses = self.backend.array(masses)
        self.n_particles = self.positions.shape[0]
    
    def compute_forces(self) -> Tuple:
        """Compute gravitational forces on all particles.
        
        Returns:
            Tuple of (force_x, force_y, force_z) or (force_x, force_y) for 2D
        """
        n = self.n_particles
        dim = self.positions.shape[1]
        
        # Initialize force arrays
        forces = self.backend.zeros((n, dim))
        
        # Vectorized force calculation
        # For each particle i, compute force from all other particles j
        # Convert positions and masses to numpy for reliable computation
        positions_np = np.asarray(self.backend.to_numpy(self.positions))
        masses_np = np.asarray(self.backend.to_numpy(self.masses)).flatten()[:n]
        
        for i in range(n):
            # Position differences: r_ij = r_j - r_i (using numpy directly)
            r_diff_np = positions_np - positions_np[i]
            
            # Distance squared: |r_ij|^2
            r_sq_np = np.sum(r_diff_np ** 2, axis=1) + self.EPSILON ** 2
            
            # Ensure r_sq_np is 1D with correct length
            r_sq_np = np.asarray(r_sq_np).flatten()[:n]
            
            # Distance: |r_ij|
            r_np = np.sqrt(r_sq_np)
            
            # Ensure r_np is 1D with correct length
            r_np = np.asarray(r_np).flatten()[:n]
            
            # Ensure masses_np is 1D with correct length
            masses_np_1d = np.asarray(masses_np).flatten()[:n]
            
            # r^3 = r_sq * r (both should be shape (n,))
            r_cubed = r_sq_np * r_np
            
            # Verify shapes before division
            assert len(r_cubed) == n, f"r_cubed shape wrong: {r_cubed.shape}, expected ({n},)"
            assert len(masses_np_1d) == n, f"masses shape wrong: {masses_np_1d.shape}, expected ({n},)"
            
            # G * m_j / r^3
            force_magnitude_np = (self.G * masses_np_1d) / r_cubed
            
            # Force vector: force_magnitude * r_ij (using numpy)
            # force_magnitude_np is (n,), r_diff_np is (n, dim)
            # We need to broadcast: (n, 1) * (n, dim) = (n, dim)
            force_vectors_np = force_magnitude_np[:, np.newaxis] * r_diff_np
            
            # Zero out self-interaction (i == j) using numpy
            force_vectors_np[i] = 0.0
            
            # Sum forces from all particles (should give shape (dim,))
            force_total_np = np.sum(force_vectors_np, axis=0)
            
            # Ensure it's 1D with correct length (dim)
            force_total_np = np.asarray(force_total_np).flatten()[:dim]
            
            # Convert to backend array and store (forces[i] should be shape (dim,))
            forces[i, :] = self.backend.array(force_total_np)
        
        # Return as tuple of components for compatibility
        if dim == 3:
            return forces[:, 0], forces[:, 1], forces[:, 2]
        else:
            return forces[:, 0], forces[:, 1]
    
    def compute_kinetic_energy(self) -> float:
        """Compute total kinetic energy.
        
        Returns:
            Total kinetic energy: 0.5 * sum(m_i * v_i^2)
        """
        # Compute velocity squared: sum of squares of velocity components
        v_sq = self.backend.sum(self.backend.square(self.velocities), axis=1)
        
        # Convert to numpy to ensure correct shapes
        masses_np = self.backend.to_numpy(self.masses).flatten()
        v_sq_np = self.backend.to_numpy(v_sq).flatten()
        
        # Ensure they have the same length
        n = min(len(masses_np), len(v_sq_np))
        if n != len(masses_np) or n != len(v_sq_np):
            # This shouldn't happen, but handle it gracefully
            masses_np = masses_np[:n]
            v_sq_np = v_sq_np[:n]
        
        # Compute kinetic energy: 0.5 * sum(m * v^2)
        ke = 0.5 * np.sum(masses_np * v_sq_np)
        return float(ke)
    
    def compute_potential_energy(self) -> float:
        """Compute total potential energy.
        
        Returns:
            Total potential energy: -G * sum_i sum_j>i (m_i * m_j / r_ij)
        """
        pe = 0.0
        n = self.n_particles
        
        # Convert to numpy for easier computation
        positions_np = self.backend.to_numpy(self.positions)
        masses_np = self.backend.to_numpy(self.masses).flatten()
        
        for i in range(n):
            for j in range(i + 1, n):
                r_diff = positions_np[j] - positions_np[i]
                r = np.linalg.norm(r_diff)
                r = max(r, self.EPSILON)  # Softening
                pe -= self.G * masses_np[i] * masses_np[j] / r
        
        return pe
    
    def compute_total_energy(self) -> float:
        """Compute total energy (kinetic + potential).
        
        Returns:
            Total energy
        """
        return self.compute_kinetic_energy() + self.compute_potential_energy()
    
    def get_state(self):
        """Get current state (positions, velocities, masses).
        
        Returns:
            Tuple of (positions, velocities, masses) as numpy arrays
        """
        return (
            self.backend.to_numpy(self.positions),
            self.backend.to_numpy(self.velocities),
            self.backend.to_numpy(self.masses)
        )
    
    def set_state(self, positions, velocities, masses):
        """Set particle state.
        
        Args:
            positions: Array of positions
            velocities: Array of velocities
            masses: Array of masses
        """
        self.positions = self.backend.array(positions)
        self.velocities = self.backend.array(velocities)
        self.masses = self.backend.array(masses)
        self.n_particles = self.positions.shape[0]
