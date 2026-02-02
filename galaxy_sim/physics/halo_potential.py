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
    ) -> Optional[np.ndarray]:
        """Compute halo acceleration for all particles.
        
        Args:
            positions: Particle positions (n, dim) as backend array
            backend: Compute backend
            
        Returns:
            Acceleration array (n, dim) or None if disabled
        """
        if not self.enabled:
            return None
        
        # Convert to numpy for computation
        positions_np = np.asarray(backend.to_numpy(positions))
        n = positions_np.shape[0]
        dim = positions_np.shape[1]
        
        # Distance from origin
        r = np.linalg.norm(positions_np, axis=1)  # Shape: (n,)
        
        # Avoid division by zero
        r_safe = np.maximum(r, 1e-6)
        
        if self.model == "flat":
            # Flat rotation curve model: a(r) = -v₀² * r / (r² + r_c²)
            # For r >> r_c: a ≈ -v₀²/r (gives flat rotation curve)
            # For r << r_c: a ≈ -v₀²*r/r_c² (linear, avoids singularity)
            r_sq = r_safe ** 2
            a_mag = self.v_0 ** 2 * r_safe / (r_sq + self.r_c ** 2)  # Positive magnitude
        elif self.model == "plummer":
            # Plummer sphere: Φ = -GM / sqrt(r² + a²)
            # Acceleration: a = -GM * r / (r² + a²)^(3/2)
            r_sq = r_safe ** 2
            r_soft_sq = r_sq + self.a ** 2
            r_soft_cubed = r_soft_sq ** 1.5
            a_mag = self.G * self.M * r_safe / r_soft_cubed  # Positive magnitude
        else:
            raise ValueError(f"Unknown halo model: {self.model}")
        
        # Acceleration direction: radial (toward origin, attractive)
        # Unit vector: -r / |r| (points inward, negative)
        r_unit = -positions_np / r_safe[:, np.newaxis]  # Shape: (n, dim), points inward
        
        # Acceleration vectors: positive magnitude * negative unit vector = negative (inward)
        accelerations = a_mag[:, np.newaxis] * r_unit  # Shape: (n, dim)
        
        # Convert back to backend array
        return backend.array(accelerations)
    
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
