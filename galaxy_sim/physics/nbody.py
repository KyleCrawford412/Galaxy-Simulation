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
        eta: float = 0.7,
        halo_potential: Optional[HaloPotential] = None,
        self_gravity: bool = True,
        particle_types: Optional[np.ndarray] = None
    ):
        """Initialize N-body system.
        
        Args:
            backend: Compute backend for array operations
            use_vectorized_forces: Use vectorized force calculation (faster)
            epsilon: Softening parameter (auto-calculated if None via eta * median_nearest_neighbor_distance)
            eta: Scaling for adaptive epsilon (default 0.7, range 0.5-1.2); only used when epsilon is None
            halo_potential: Optional analytic halo potential for flat rotation curve
            self_gravity: If False, disk particles don't attract each other (test particle mode)
            particle_types: Array of 'disk', 'bulge', 'halo', 'core' for each particle (n,)
        """
        self.backend = backend
        self.positions = None
        self.velocities = None
        self.masses = None
        self.n_particles = 0
        self.epsilon = epsilon  # Will be set once in initialize() if None (eps0 = eta * median_nn_distance)
        self._eta = float(np.clip(eta, 0.5, 1.2))
        self.force_calculator = ForceCalculator(method="vectorized") if use_vectorized_forces else None
        self.halo_potential = halo_potential
        self.self_gravity = self_gravity
        self.particle_types = particle_types  # Will be set in initialize() if provided
    
    def initialize(self, positions, velocities, masses, particle_types: Optional[np.ndarray] = None):
        """Initialize particle state.
        
        Args:
            positions: Array of shape (n, 3) for 3D or (n, 2) for 2D
            velocities: Array of shape (n, 3) or (n, 2)
            masses: Array of shape (n,)
            particle_types: Array of 'disk', 'bulge', 'halo', 'core' for each particle (n,)
        """
        self.positions = self.backend.array(positions)
        self.velocities = self.backend.array(velocities)
        self.masses = self.backend.array(masses)
        self.n_particles = self.positions.shape[0]
        
        # Store particle types if provided
        if particle_types is not None:
            self.particle_types = particle_types
        elif self.particle_types is None:
            # Default: all particles are 'bulge' (full N-body)
            self.particle_types = np.array(['bulge'] * self.n_particles, dtype=object)
        
        # Compute constant eps0 once at initialization; never recompute during run
        if self.epsilon is None:
            self.epsilon = self._calculate_adaptive_epsilon(eta=self._eta)
    
    def _calculate_adaptive_epsilon(self, eta: float) -> float:
        """Calculate constant softening parameter based on median nearest neighbor distance.
        
        eps0 = eta * median_nearest_neighbor_distance (eta default 0.7, range 0.5-1.2).
        Stored as self.epsilon and used for all particle pairs and all timesteps (conservative force).
        
        Args:
            eta: Scaling factor for epsilon (clamped to 0.5-1.2)
        
        Returns:
            Constant softening eps0 (float)
        """
        if self.n_particles < 2:
            return self.EPSILON_DEFAULT
        
        # Convert to numpy for calculation
        positions_np = np.asarray(self.backend.to_numpy(self.positions))
        
        # Compute median nearest neighbor distance
        try:
            r_diff = positions_np[np.newaxis, :, :] - positions_np[:, np.newaxis, :]
            distances_sq = np.sum(r_diff ** 2, axis=2)
            np.fill_diagonal(distances_sq, np.inf)
            nearest_neighbor_distances_sq = np.min(distances_sq, axis=1)
            nearest_neighbor_distances = np.sqrt(nearest_neighbor_distances_sq)
            median_nn_distance = np.median(nearest_neighbor_distances)
        except Exception:
            center = np.mean(positions_np, axis=0)
            distances_from_center = np.linalg.norm(positions_np - center, axis=1)
            char_size = np.max(distances_from_center) if len(distances_from_center) > 0 else 1.0
            dim = positions_np.shape[1]
            if dim == 3:
                avg_spacing = char_size / (self.n_particles ** (1/3))
            else:
                avg_spacing = char_size / (self.n_particles ** 0.5)
            median_nn_distance = avg_spacing
        
        eta_clamped = np.clip(eta, 0.5, 1.2)
        epsilon = eta_clamped * median_nn_distance
        epsilon = max(self.EPSILON_DEFAULT, epsilon)
        return float(epsilon)
    
    def compute_forces(self) -> Tuple:
        """Compute gravitational forces on all particles.
        
        Uses vectorized calculation for better performance.
        Optionally adds analytic halo potential acceleration.
        
        Returns:
            Tuple of (force_x, force_y, force_z) or (force_x, force_y) for 2D
        """
        if self.force_calculator is not None:
            # Use vectorized force calculator with constant eps0 (Plummer softening)
            forces = self.force_calculator.compute_forces(
                self.positions,
                self.masses,
                self.backend,
                G=self.G,
                epsilon=self.epsilon,
                self_gravity=self.self_gravity,
                particle_types=self.particle_types
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
        
        # Get particle types
        particle_types_np = None
        if self.particle_types is not None:
            particle_types_np = np.asarray(self.particle_types)
            is_disk = (particle_types_np == 'disk')
        
        for i in range(n):
            # Position differences: r_ij = r_j - r_i
            r_diff_np = positions_np - positions_np[i]
            
            # Distance squared: |r_ij|^2
            r_sq_np = np.sum(r_diff_np ** 2, axis=1)
            r_sq_np = np.asarray(r_sq_np).flatten()[:n]
            
            # Plummer softening: (r^2 + eps0^2)^(3/2)
            r_soft_cubed = (r_sq_np + self.epsilon ** 2) ** 1.5
            
            # G * m_i * m_j / (r^2 + eps0^2)^(3/2)
            force_magnitude_np = (self.G * masses_np[i] * masses_np) / r_soft_cubed
            
            # If self_gravity is False, zero out disk-disk interactions
            if not self.self_gravity and particle_types_np is not None:
                # Zero out force if both i and j are disk particles
                if is_disk[i]:
                    force_magnitude_np[is_disk] = 0.0
            
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
