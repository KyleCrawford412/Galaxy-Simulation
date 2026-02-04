"""Analytic halo potential for flat rotation curve."""

import numpy as np
from typing import Optional, Literal
from galaxy_sim.backends.base import Backend


class HaloPotential:
    """Analytic dark matter halo potential.
    
    Supports two models:
    1. Flat rotation curve (isothermal-like): a(r) = -v₀² * r / (r² + r_c²)
       Produces v ≈ constant for large r
    2. Plummer sphere: Φ = -GM / sqrt(r² + a²) ⇒ a = -GM * r / (r² + a²)^(3/2)
       Classic softened potential
    """
    
    def __init__(
        self,
        model: Literal["flat", "plummer"] = "flat",
        v_0: float = 1.0,  # For flat model
        r_c: float = 1.0,  # Core radius (for flat model) or scale radius (for Plummer)
        M: float = 1000.0,  # Mass (for Plummer model)
        a: float = 1.0,  # Scale radius (for Plummer model, alias for r_c)
        enabled: bool = False,
        G: float = 1.0  # Gravitational constant
    ):
        """Initialize halo potential.
        
        Args:
            model: "flat" for flat rotation curve, "plummer" for Plummer sphere
            v_0: Asymptotic circular velocity (for flat model)
            r_c: Core radius (for flat model) or scale radius (for Plummer model)
            M: Total mass (for Plummer model)
            a: Scale radius (for Plummer model, overrides r_c if specified)
            enabled: Whether halo potential is active
            G: Gravitational constant
        """
        self.model = model
        self.v_0 = v_0
        self.r_c = r_c
        self.M = M
        self.a = a if a != r_c else r_c  # Use a if different from r_c, else use r_c
        self.enabled = enabled
        self.G = G
    
    def compute_acceleration(
        self,
        positions,
        backend: Backend
    ):
        """Compute halo acceleration for all particles (backend-native, no to_numpy).
        
        Args:
            positions: Particle positions (n, dim) as backend array
            backend: Compute backend
            
        Returns:
            Acceleration array (n, dim) as backend array, or None if disabled
        """
        if not self.enabled:
            return None
        
        # Distance from origin (backend ops only)
        r = backend.norm(positions, axis=1)
        r_safe = backend.maximum(r, 1e-6)
        r_sq = backend.square(r_safe)
        
        if self.model == "flat":
            # a(r) = v₀² * r / (r² + r_c²); direction inward => -r_unit * a_mag
            a_mag = backend.divide(
                backend.multiply(self.v_0 ** 2, r_safe),
                backend.add(r_sq, self.r_c ** 2),
            )
        elif self.model == "plummer":
            r_soft_sq = backend.add(r_sq, self.a ** 2)
            r_soft_cubed = backend.power(r_soft_sq, 1.5)
            a_mag = backend.divide(
                backend.multiply(self.G * self.M, r_safe),
                r_soft_cubed,
            )
        else:
            raise ValueError(f"Unknown halo model: {self.model}")
        
        # Unit vector inward: -positions / r_safe
        r_unit = backend.divide(
            backend.multiply(positions, -1.0),
            backend.expand_dims(r_safe, 1),
        )
        accelerations = backend.multiply(backend.expand_dims(a_mag, 1), r_unit)
        return accelerations
    
    def get_circular_velocity(self, r: np.ndarray) -> np.ndarray:
        """Get circular velocity at radius r.
        
        For this potential: v² = v₀² * r² / (r² + r_c²)
        For r >> r_c: v ≈ v₀ (flat rotation curve)
        
        Args:
            r: Radial distances
            
        Returns:
            Circular velocities
        """
        r_safe = np.maximum(r, self.r_c * 0.01)
        r_sq = r_safe ** 2
        v_sq = self.v_0 ** 2 * r_sq / (r_sq + self.r_c ** 2)
        return np.sqrt(v_sq)
