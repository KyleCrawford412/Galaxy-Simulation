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
        analytic_bulge_potential: HaloPotential = None,
        analytic_disk_potential: HaloPotential = None,
        self_gravity: bool = True,
        particle_types = None,
        eps_cd: float = None,
        eps_bd: float = None
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
        self.analytic_bulge_potential = analytic_bulge_potential
        self.analytic_disk_potential = analytic_disk_potential
        self.self_gravity = self_gravity
        self.particle_types = particle_types
        self.eps_cd = eps_cd
        self.eps_bd = eps_bd
    
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
        is_disk = None
        is_bulge = None
        is_core = None
        if hasattr(self, 'particle_types') and self.particle_types is not None:
            particle_types_np = np.asarray(self.particle_types)
            is_disk = (particle_types_np == 'disk')
            is_bulge = (particle_types_np == 'bulge')
            is_core = (particle_types_np == 'core')
        self_gravity = getattr(self, 'self_gravity', True)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Skip disk-disk interactions if self_gravity is False
                if not self_gravity and particle_types_np is not None:
                    if is_disk[i] and is_disk[j]:
                        continue
                
                r_diff = positions_np[j] - positions_np[i]
                r_sq = np.sum(r_diff ** 2)
                eps_ij = self.epsilon
                if self.eps_cd is not None and is_disk is not None and is_core is not None:
                    if (is_disk[i] and is_core[j]) or (is_core[i] and is_disk[j]):
                        eps_ij = self.eps_cd
                if self.eps_bd is not None and is_disk is not None and is_bulge is not None:
                    if (is_disk[i] and is_bulge[j]) or (is_bulge[i] and is_disk[j]):
                        eps_ij = self.eps_bd
                r_soft = np.sqrt(r_sq + eps_ij ** 2)
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

        # Analytic bulge/disk potential contributions (if present)
        U_analytic = 0.0
        for potential in (self.analytic_bulge_potential, self.analytic_disk_potential):
            if potential is None or not potential.enabled:
                continue
            for i in range(n):
                r = np.linalg.norm(positions_np[i])
                r_safe = max(r, 1e-6)
                # Plummer potential: Φ = -GM / sqrt(r² + a²)
                r_soft = np.sqrt(r_safe ** 2 + potential.a ** 2)
                phi = -potential.G * potential.M / r_soft
                U_analytic += masses_np[i] * phi
        
        U_total = U_nbody + U_halo + U_analytic
        
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

    def compute_bound_fraction(self, positions, velocities, masses) -> float:
        """Compute fraction of particles with negative specific energy."""
        positions_np = np.asarray(self.backend.to_numpy(positions))
        velocities_np = np.asarray(self.backend.to_numpy(velocities))
        masses_np = np.asarray(self.backend.to_numpy(masses)).flatten()
        n = len(masses_np)
        types_np = np.asarray(self.particle_types) if self.particle_types is not None else None
        energies = []
        for i in range(n):
            v_sq = np.sum(velocities_np[i] ** 2)
            phi = 0.0
            for j in range(n):
                if i == j:
                    continue
                r_diff = positions_np[j] - positions_np[i]
                r_sq = np.sum(r_diff ** 2)
                eps_ij = self.epsilon
                if self.eps_cd is not None and types_np is not None:
                    if (types_np[i] == 'disk' and types_np[j] == 'core') or (types_np[i] == 'core' and types_np[j] == 'disk'):
                        eps_ij = self.eps_cd
                if self.eps_bd is not None and types_np is not None:
                    if (types_np[i] == 'disk' and types_np[j] == 'bulge') or (types_np[i] == 'bulge' and types_np[j] == 'disk'):
                        eps_ij = self.eps_bd
                phi -= self.G * masses_np[j] / np.sqrt(r_sq + eps_ij ** 2)
            e_spec = 0.5 * v_sq + phi
            energies.append(e_spec)
        bound_frac = np.mean(np.array(energies) < 0.0)
        return float(bound_frac)

    def compute_radial_profile(self, positions, bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Compute radial histogram profile."""
        positions_np = np.asarray(self.backend.to_numpy(positions))
        radii = np.linalg.norm(positions_np, axis=1)
        hist, bin_edges = np.histogram(radii, bins=bins)
        return hist, bin_edges
