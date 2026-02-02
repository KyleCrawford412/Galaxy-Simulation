"""Core N-body physics calculations."""

import numpy as np
from typing import Tuple
from galaxy_sim.backends.base import Backend
from galaxy_sim.physics.force_calculator import ForceCalculator


class NBodySystem:
    """N-body gravitational system.
    
    Handles force calculations, energy computation, and particle state management.
    """
    
    G = 1.0  # Gravitational constant (normalized units)
    EPSILON_DEFAULT = 1e-3  # Default softening parameter
    
    def __init__(self, backend: Backend, use_vectorized_forces: bool = True, epsilon: float = None):
        """Initialize N-body system.
        
        Args:
            backend: Compute backend for array operations
            use_vectorized_forces: Use vectorized force calculation (faster)
            epsilon: Softening parameter (auto-calculated if None)
        """
        self.backend = backend
        self.positions = None
        self.velocities = None
        self.masses = None
        self.n_particles = 0
        self.epsilon = epsilon  # Will be set adaptively in initialize() if None
        self.force_calculator = ForceCalculator(method="vectorized") if use_vectorized_forces else None
    
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
        
        # Calculate adaptive epsilon if not set
        if self.epsilon is None:
            self.epsilon = self._calculate_adaptive_epsilon()
    
    def _calculate_adaptive_epsilon(self) -> float:
        """Calculate adaptive softening parameter based on particle distribution.
        
        Epsilon is set to ~10% of the typical inter-particle spacing
        to prevent excessive forces when particles are close.
        
        Returns:
            Adaptive epsilon value
        """
        if self.n_particles < 2:
            return self.EPSILON_DEFAULT
        
        # Convert to numpy for calculation
        positions_np = np.asarray(self.backend.to_numpy(self.positions))
        
        # Estimate typical spacing from particle distribution
        # Use the characteristic size of the system
        if positions_np.shape[1] == 3:
            # 3D: use max distance from center
            center = np.mean(positions_np, axis=0)
            distances = np.linalg.norm(positions_np - center, axis=1)
            characteristic_size = np.max(distances) if len(distances) > 0 else 1.0
        else:
            # 2D: use max distance from center
            center = np.mean(positions_np, axis=0)
            distances = np.linalg.norm(positions_np - center, axis=1)
            characteristic_size = np.max(distances) if len(distances) > 0 else 1.0
        
        # Estimate average spacing: (volume / n)^(1/dim)
        # For 3D: spacing ~ (4/3 * pi * r^3 / n)^(1/3) ~ r / n^(1/3)
        # For 2D: spacing ~ (pi * r^2 / n)^(1/2) ~ r / sqrt(n)
        dim = positions_np.shape[1]
        if dim == 3:
            avg_spacing = characteristic_size / (self.n_particles ** (1/3))
        else:
            avg_spacing = characteristic_size / (self.n_particles ** 0.5)
        
        # Epsilon should be ~10-20% of average spacing
        # But not smaller than default or larger than characteristic_size/10
        epsilon = max(self.EPSILON_DEFAULT, min(avg_spacing * 0.15, characteristic_size * 0.1))
        
        return float(epsilon)
    
    def compute_forces(self) -> Tuple:
        """Compute gravitational forces on all particles.
        
        Uses vectorized calculation for better performance.
        
        Returns:
            Tuple of (force_x, force_y, force_z) or (force_x, force_y) for 2D
        """
        if self.force_calculator is not None:
            # Use vectorized force calculator
            return self.force_calculator.compute_forces(
                self.positions,
                self.masses,
                self.backend,
                G=self.G,
                epsilon=self.epsilon
            )
        else:
            # Fallback to loop-based (slower, but more compatible)
            return self._compute_forces_loop()
    
    def _compute_forces_loop(self) -> Tuple:
        """Loop-based force calculation (fallback method)."""
        n = self.n_particles
        dim = self.positions.shape[1]
        
        # Initialize force arrays
        forces = self.backend.zeros((n, dim))
        
        # Convert to numpy for reliable computation
        positions_np = np.asarray(self.backend.to_numpy(self.positions))
        masses_np = np.asarray(self.backend.to_numpy(self.masses)).flatten()[:n]
        
        for i in range(n):
            # Position differences: r_ij = r_j - r_i
            r_diff_np = positions_np - positions_np[i]
            
            # Distance squared: |r_ij|^2
            r_sq_np = np.sum(r_diff_np ** 2, axis=1) + self.epsilon ** 2
            r_sq_np = np.asarray(r_sq_np).flatten()[:n]
            
            # Distance: |r_ij|
            r_np = np.sqrt(r_sq_np)
            r_np = np.asarray(r_np).flatten()[:n]
            
            # r^3 = r_sq * r
            r_cubed = r_sq_np * r_np
            
            # G * m_j / r^3
            force_magnitude_np = (self.G * masses_np) / r_cubed
            
            # Force vector: force_magnitude * r_ij
            force_vectors_np = force_magnitude_np[:, np.newaxis] * r_diff_np
            
            # Zero out self-interaction
            force_vectors_np[i] = 0.0
            
            # Sum forces
            force_total_np = np.sum(force_vectors_np, axis=0)
            force_total_np = np.asarray(force_total_np).flatten()[:dim]
            
            # Store
            forces[i, :] = self.backend.array(force_total_np)
        
        # Return as tuple of components
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
                r = max(r, self.epsilon)  # Softening
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
