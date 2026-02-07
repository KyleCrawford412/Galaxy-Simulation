"""Rotating spiral potential (density-wave) for spiral arms."""

from typing import Optional
import math
from galaxy_sim.backends.base import Backend


class SpiralPotential:
    """External rotating spiral potential with analytic acceleration in x-y."""

    def __init__(
        self,
        m: int = 2,
        pitch_angle_deg: float = 18.0,
        pattern_speed: float = 0.35,
        amplitude: float = 0.05,
        r_ref: float = 5.0,
        r_sigma: float = 2.0,
        r_taper_inner: Optional[float] = None,
        r_taper_outer: Optional[float] = None,
        enabled: bool = True,
    ):
        self.m = int(m)
        self.pitch_angle_deg = float(pitch_angle_deg)
        self.pattern_speed = float(pattern_speed)
        self.amplitude = float(amplitude)
        self.r_ref = float(r_ref)
        self.r_sigma = float(r_sigma)
        self.r_taper_inner = float(r_taper_inner) if r_taper_inner is not None else self.r_ref * 0.3
        self.r_taper_outer = float(r_taper_outer) if r_taper_outer is not None else (self.r_ref + 3.0 * self.r_sigma)
        self.enabled = bool(enabled)

    def compute_acceleration(self, positions, backend: Backend, t: float = 0.0):
        """Compute spiral potential acceleration in the x-y plane.

        Args:
            positions: (n, dim) array
            backend: compute backend
            t: simulation time
        """
        if not self.enabled:
            return None

        x = positions[:, 0]
        y = positions[:, 1]
        r_sq = backend.add(backend.square(x), backend.square(y))
        r = backend.sqrt(r_sq)
        r_safe = backend.maximum(r, 1e-6)
        theta = backend.atan2(y, x)

        pitch_rad = math.radians(self.pitch_angle_deg)
        cot_pitch = 1.0 / math.tan(pitch_rad)
        m = float(self.m)

        log_term = backend.log(backend.divide(r_safe, self.r_ref))
        phi_spiral = backend.add(
            backend.multiply(m, backend.subtract(theta, self.pattern_speed * t)),
            backend.multiply(m * cot_pitch, log_term),
        )
        cos_phi = backend.cos(phi_spiral)
        sin_phi = backend.sin(phi_spiral)

        # Gaussian radial envelope
        dr = backend.subtract(r_safe, self.r_ref)
        g = backend.exp(backend.divide(backend.multiply(-1.0, backend.square(dr)), 2.0 * self.r_sigma ** 2))
        dg_dr = backend.multiply(g, backend.divide(backend.multiply(-1.0, dr), self.r_sigma ** 2))

        # Taper: inner * outer
        r_core = self.r_taper_inner
        r_outer = self.r_taper_outer
        inner = backend.divide(r_sq, backend.add(r_sq, r_core ** 2))
        d_inner = backend.divide(
            backend.multiply(2.0 * r_safe * (r_core ** 2), 1.0),
            backend.square(backend.add(r_sq, r_core ** 2)),
        )
        outer = backend.exp(backend.multiply(-1.0, backend.square(backend.divide(r_safe, r_outer))))
        d_outer = backend.multiply(outer, backend.multiply(-2.0 * r_safe, 1.0 / (r_outer ** 2)))
        taper = backend.multiply(inner, outer)
        d_taper = backend.add(backend.multiply(d_inner, outer), backend.multiply(inner, d_outer))

        # Phase derivatives
        dphi_dr = backend.divide(m * cot_pitch, r_safe)
        dphi_dtheta = m

        # Potential
        A = self.amplitude
        # dPhi/dr
        term1 = backend.multiply(dg_dr, backend.multiply(cos_phi, taper))
        term2 = backend.multiply(g, backend.multiply(cos_phi, d_taper))
        term3 = backend.multiply(g, backend.multiply(backend.multiply(-1.0, sin_phi), backend.multiply(dphi_dr, taper)))
        dPhi_dr = backend.multiply(A, backend.add(backend.add(term1, term2), term3))

        # dPhi/dtheta
        dPhi_dtheta = backend.multiply(A, backend.multiply(g, backend.multiply(backend.multiply(-1.0, sin_phi), dphi_dtheta)))
        dPhi_dtheta = backend.multiply(dPhi_dtheta, taper)

        # Polar to Cartesian
        a_r = backend.multiply(-1.0, dPhi_dr)
        a_theta = backend.multiply(-1.0, backend.divide(dPhi_dtheta, r_safe))
        cos_t = backend.cos(theta)
        sin_t = backend.sin(theta)

        a_x = backend.subtract(backend.multiply(a_r, cos_t), backend.multiply(a_theta, sin_t))
        a_y = backend.add(backend.multiply(a_r, sin_t), backend.multiply(a_theta, cos_t))

        if positions.shape[1] == 3:
            a_z = backend.zeros_like(a_x)
            return backend.stack([a_x, a_y, a_z], axis=1)
        return backend.stack([a_x, a_y], axis=1)
