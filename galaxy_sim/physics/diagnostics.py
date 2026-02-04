"""Diagnostics for N-body simulations."""

import numpy as np
from typing import Tuple
from galaxy_sim.backends.base import Backend
from galaxy_sim.physics.halo_potential import HaloPotential


class Diagnostics:
    """Compute consistent energy diagnostics matching the force law."""
    
    def __init__(
        self,
        backend: Backend,
        G: float = 1.0,
        epsilon: float = 0.1,
        halo_potential: HaloPotential = None,
        self_gravity: bool = True,
        particle_types = None
    ):
        """Initialize diagnostics.
        
        Args:
            backend: Compute backend
            G: Gravitational constant
            epsilon: Softening parameter (must match force calculation)
            halo_potential: Optional halo potential (for potential energy contribution)
            self_gravity: If False, disk particles don't attract each other
            particle_types: Array of 'disk', 'bulge', 'halo', 'core' for each particle (n,)
        """
        self.backend = backend
        self.G = G
        self.epsilon = epsilon
        self.halo_potential = halo_potential
        self.self_gravity = self_gravity
        self.particle_types = particle_types
    
    def compute_energies(
        self,
        positions,
        velocities,
        masses
    ) -> Tuple[float, float, float]:
        """Compute kinetic, potential, and total energy using consistent potential.
        
        Potential uses the same Plummer softening as force law:
        U = -G * Σ_{i<j} m_i * m_j / sqrt(r_ij^2 + eps^2)
        
        Plus halo potential contribution if present.
        
        Args:
            positions: Particle positions (n, dim)
            velocities: Particle velocities (n, dim)
            masses: Particle masses (n,)
            
        Returns:
            Tuple of (kinetic_energy, potential_energy, total_energy)
        """
        # Convert to numpy
        positions_np = np.asarray(self.backend.to_numpy(positions))
        velocities_np = np.asarray(self.backend.to_numpy(velocities))
        masses_np = np.asarray(self.backend.to_numpy(masses)).flatten()
        
        n = len(masses_np)
        
        # Kinetic energy: K = 0.5 * Σ m_i * v_i^2
        v_sq = np.sum(velocities_np ** 2, axis=1)
        K = 0.5 * np.sum(masses_np * v_sq)
        
        # Potential energy: U = -G * Σ_{i<j} m_i * m_j / sqrt(r_ij^2 + eps^2)
        # This matches the Plummer softening used in force calculation
        # If self_gravity is False, skip disk-disk interactions
        U_nbody = 0.0
        particle_types_np = None
        if hasattr(self, 'particle_types') and self.particle_types is not None:
            particle_types_np = np.asarray(self.particle_types)
            is_disk = (particle_types_np == 'disk')
        self_gravity = getattr(self, 'self_gravity', True)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Skip disk-disk interactions if self_gravity is False
                if not self_gravity and particle_types_np is not None:
                    if is_disk[i] and is_disk[j]:
                        continue
                
                r_diff = positions_np[j] - positions_np[i]
                r_sq = np.sum(r_diff ** 2)
                r_soft = np.sqrt(r_sq + self.epsilon ** 2)  # Plummer softening with constant eps0
                U_nbody -= self.G * masses_np[i] * masses_np[j] / r_soft
        
        # Halo potential contribution (if present)
        U_halo = 0.0
        if self.halo_potential is not None and self.halo_potential.enabled:
            # Compute halo potential energy for each particle
            # For flat model: Φ(r) = v₀² * ln(r/r_c) (approximately)
            # For Plummer: Φ(r) = -GM / sqrt(r² + a²)
            for i in range(n):
                r = np.linalg.norm(positions_np[i])
                r_safe = max(r, 1e-6)
                
                if self.halo_potential.model == "flat":
                    # Flat rotation curve potential: Φ ≈ v₀² * ln(r/r_c) for r >> r_c
                    # More accurate: integrate a = -v₀²*r/(r²+r_c²) to get potential
                    # Φ(r) = -v₀²/2 * ln(1 + r²/r_c²) (up to constant)
                    phi_halo = -0.5 * self.halo_potential.v_0 ** 2 * np.log(1 + (r_safe / self.halo_potential.r_c) ** 2)
                elif self.halo_potential.model == "plummer":
                    # Plummer potential: Φ = -GM / sqrt(r² + a²)
                    r_soft_halo = np.sqrt(r_safe ** 2 + self.halo_potential.a ** 2)
                    phi_halo = -self.halo_potential.G * self.halo_potential.M / r_soft_halo
                else:
                    phi_halo = 0.0
                
                U_halo += masses_np[i] * phi_halo
        
        U_total = U_nbody + U_halo
        
        # Total energy
        E_total = K + U_total
        
        return float(K), float(U_total), float(E_total)
    
    def compute_virial_ratio(
        self,
        positions,
        velocities,
        masses
    ) -> float:
        """Compute virial ratio Q = 2K / |U| using consistent energies.
        
        Args:
            positions: Particle positions (n, dim)
            velocities: Particle velocities (n, dim)
            masses: Particle masses (n,)
            
        Returns:
            Virial ratio Q
        """
        K, U, E = self.compute_energies(positions, velocities, masses)
        
        if abs(U) < 1e-10:
            return float('inf')
        
        Q = 2.0 * K / abs(U)
        return float(Q)
