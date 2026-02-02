"""Core N-body physics calculations."""

import numpy as np
from typing import Tuple, Optional
from galaxy_sim.backends.base import Backend
from galaxy_sim.physics.force_calculator import ForceCalculator
from galaxy_sim.physics.halo_potential import HaloPotential


class NBodySystem:
    """N-body gravitational system.
    
    Handles force calculations, energy computation, and particle state management.
    """
    
    G = 1.0  # Gravitational constant (normalized units)
    EPSILON_DEFAULT = 1e-3  # Default softening parameter
    
    def __init__(
        self,
        backend: Backend,
        use_vectorized_forces: bool = True,
        epsilon: float = None,
        halo_potential: Optional[HaloPotential] = None
    ):
        """Initialize N-body system.
        
        Args:
            backend: Compute backend for array operations
            use_vectorized_forces: Use vectorized force calculation (faster)
            epsilon: Softening parameter (auto-calculated if None)
            halo_potential: Optional analytic halo potential for flat rotation curve
        """
        self.backend = backend
        self.positions = None
        self.velocities = None
        self.masses = None
        self.n_particles = 0
        self.epsilon = epsilon  # Will be set adaptively in initialize() if None
        self.characteristic_size = None  # Will be calculated in initialize()
        self.force_calculator = ForceCalculator(method="vectorized") if use_vectorized_forces else None
        self.halo_potential = halo_potential
    
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
        
        # Calculate adaptive epsilon and characteristic size if not set
        if self.epsilon is None or self.characteristic_size is None:
            self.epsilon, self.characteristic_size = self._calculate_adaptive_epsilon()
    
    def _calculate_adaptive_epsilon(self) -> Tuple[float, float]:
        """Calculate adaptive softening parameter based on particle distribution.
        
        Epsilon is set to ~5% of the typical inter-particle spacing
        to prevent excessive forces when particles are close.
        
        Returns:
            Tuple of (epsilon, characteristic_size)
        """
        if self.n_particles < 2:
            return self.EPSILON_DEFAULT, 1.0
        
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
        
        # Epsilon should be ~5-10% of average spacing for stronger forces
        # But not smaller than default or larger than characteristic_size/20
        epsilon = max(self.EPSILON_DEFAULT, min(avg_spacing * 0.05, characteristic_size * 0.05))
        
        return float(epsilon), float(characteristic_size)
    
    def compute_forces(self) -> Tuple:
        """Compute gravitational forces on all particles.
        
        Uses vectorized calculation for better performance.
        Optionally adds analytic halo potential acceleration.
        
        Returns:
            Tuple of (force_x, force_y, force_z) or (force_x, force_y) for 2D
        """
        if self.force_calculator is not None:
            # Use vectorized force calculator with distance-dependent epsilon
            forces = self.force_calculator.compute_forces(
                self.positions,
                self.masses,
                self.backend,
                G=self.G,
                epsilon=self.epsilon,
                characteristic_size=self.characteristic_size
            )
        else:
            # Fallback to loop-based (slower, but more compatible)
            forces = self._compute_forces_loop()
        
        # Add halo potential acceleration if enabled
        if self.halo_potential is not None and self.halo_potential.enabled:
            halo_acc = self.halo_potential.compute_acceleration(self.positions, self.backend)
            if halo_acc is not None:
                # Convert halo acceleration to forces (multiply by masses)
                # Halo forces = m * a_halo
                masses_np = np.asarray(self.backend.to_numpy(self.masses)).flatten()
                halo_acc_np = np.asarray(self.backend.to_numpy(halo_acc))
                
                # Multiply each row by corresponding mass
                halo_forces_np = halo_acc_np * masses_np[:, np.newaxis]
                
                # Add to N-body forces
                if len(forces) == 3:
                    fx, fy, fz = forces
                    fx_np = np.asarray(self.backend.to_numpy(fx)).flatten()
                    fy_np = np.asarray(self.backend.to_numpy(fy)).flatten()
                    fz_np = np.asarray(self.backend.to_numpy(fz)).flatten()
                    fx = self.backend.array(fx_np + halo_forces_np[:, 0])
                    fy = self.backend.array(fy_np + halo_forces_np[:, 1])
                    fz = self.backend.array(fz_np + halo_forces_np[:, 2])
                    forces = (fx, fy, fz)
                else:
                    fx, fy = forces
                    fx_np = np.asarray(self.backend.to_numpy(fx)).flatten()
                    fy_np = np.asarray(self.backend.to_numpy(fy)).flatten()
                    fx = self.backend.array(fx_np + halo_forces_np[:, 0])
                    fy = self.backend.array(fy_np + halo_forces_np[:, 1])
                    forces = (fx, fy)
        
        return forces
    
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
        """Compute total potential energy using Plummer softening.
        
        Uses the same softening as force calculation:
        U = -G * sum_i sum_j>i (m_i * m_j / sqrt(r_ij^2 + eps^2))
        
        Also includes halo potential contribution if present.
        
        Returns:
            Total potential energy (including halo)
        """
        from galaxy_sim.physics.diagnostics import Diagnostics
        
        diagnostics = Diagnostics(
            self.backend,
            G=self.G,
            epsilon=self.epsilon,
            halo_potential=self.halo_potential
        )
        
        _, U, _ = diagnostics.compute_energies(
            self.positions,
            self.velocities,
            self.masses
        )
        
        return U
    
    def compute_total_energy(self) -> float:
        """Compute total energy (kinetic + potential).
        
        Returns:
            Total energy
        """
        return self.compute_kinetic_energy() + self.compute_potential_energy()
    
    def compute_virial_ratio(self) -> float:
        """Compute virial ratio Q = 2K / |U|.
        
        Q = 1.0 means virial equilibrium (stable)
        Q < 1.0 means sub-virial (will collapse)
        Q > 1.0 means super-virial (will expand)
        
        Returns:
            Virial ratio Q
        """
        K = self.compute_kinetic_energy()
        U = self.compute_potential_energy()
        
        if abs(U) < 1e-10:
            return float('inf')  # Avoid division by zero
        
        Q = 2.0 * K / abs(U)
        return float(Q)
    
    def compute_angular_momentum(self) -> float:
        """Compute total angular momentum magnitude.
        
        Returns:
            Total angular momentum magnitude
        """
        # Convert to numpy
        positions_np = self.backend.to_numpy(self.positions)
        velocities_np = self.backend.to_numpy(self.velocities)
        masses_np = self.backend.to_numpy(self.masses).flatten()
        
        # Angular momentum: L = sum(m_i * r_i × v_i)
        # For 2D: L_z = sum(m_i * (x_i * v_y_i - y_i * v_x_i))
        # For 3D: L = |sum(m_i * r_i × v_i)|
        n = positions_np.shape[0]
        dim = positions_np.shape[1]
        
        if dim == 2:
            # 2D: L_z only
            L_z = np.sum(masses_np * (positions_np[:, 0] * velocities_np[:, 1] - 
                                      positions_np[:, 1] * velocities_np[:, 0]))
            return float(np.abs(L_z))
        else:
            # 3D: full angular momentum vector
            L = np.zeros(3)
            for i in range(n):
                r = positions_np[i]
                v = velocities_np[i]
                L += masses_np[i] * np.cross(r, v)
            return float(np.linalg.norm(L))
    
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
