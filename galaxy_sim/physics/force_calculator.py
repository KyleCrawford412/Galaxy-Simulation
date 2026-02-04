"""Unified force calculation with vectorization and GPU optimization.

No to_numpy() in the simulation loop: all physics stays on backend (JAX/NumPy/CuPy/PyTorch).
Conversions only at rendering/output boundaries. For N > 20k, Barnes-Hut approximation
is available (single conversion at force boundary).
"""

from typing import Tuple, Literal, Optional, Any
from galaxy_sim.backends.base import Backend

try:
    from galaxy_sim.physics.barnes_hut import (
        compute_forces_barnes_hut,
        BARNES_HUT_N_THRESHOLD,
    )
except ImportError:
    compute_forces_barnes_hut = None
    BARNES_HUT_N_THRESHOLD = 20_000

# JAX JIT-compiled force kernel (module-level so it compiles once)
_jax_force_jit = None


def _get_jax_force_jit():
    """Lazy compile JAX force kernel (no host transfer)."""
    global _jax_force_jit
    if _jax_force_jit is not None:
        return _jax_force_jit
    try:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def _forces_jax(positions: Any, masses: Any, G: float, eps: float) -> Any:
            # positions (n, dim), masses (n,) - all JAX arrays on device
            n = positions.shape[0]
            dim = positions.shape[1]
            pos_i = jnp.reshape(positions, (n, 1, dim))
            pos_j = jnp.reshape(positions, (1, n, dim))
            r_diff = pos_j - pos_i
            r_sq = jnp.sum(r_diff ** 2, axis=2)
            r_soft_cubed = (r_sq + eps ** 2) ** 1.5
            m_i = jnp.reshape(masses, (n, 1))
            m_j = jnp.reshape(masses, (1, n))
            force_mag = G * m_i * m_j / r_soft_cubed
            force_mag = jnp.where(jnp.eye(n, dtype=bool), 0.0, force_mag)
            force_vectors = jnp.expand_dims(force_mag, axis=2) * r_diff
            return jnp.sum(force_vectors, axis=1)

        _jax_force_jit = _forces_jax
        return _jax_force_jit
    except ImportError:
        return None


class ForceCalculator:
    """Unified force calculation interface; backend-native, no host transfer in hot path."""

    def __init__(
        self,
        method: Literal["direct", "vectorized"] = "auto",
        use_gpu: bool = True,
    ):
        self.method = method
        self.use_gpu = use_gpu

    def compute_forces(
        self,
        positions: Any,
        masses: Any,
        backend: Backend,
        G: float = 1.0,
        epsilon: float = 1e-3,
        self_gravity: bool = True,
        particle_types: Optional[Any] = None,
        is_disk: Optional[Any] = None,
        is_bulge: Optional[Any] = None,
        is_core: Optional[Any] = None,
        eps_cd: Optional[float] = None,
        eps_bd: Optional[float] = None,
    ) -> Tuple:
        """Compute gravitational forces on all particles.

        All arrays remain on backend (no to_numpy). Returns tuple (fx, fy, [fz]) as backend arrays.

        Args:
            positions: (n, dim) backend array
            masses: (n,) backend array
            backend: Compute backend
            G: Gravitational constant
            epsilon: Constant softening (eps0)
            self_gravity: If False, disk-disk forces are zeroed (requires is_disk)
            particle_types: Optional type labels (used only if is_disk not provided)
            is_disk: Optional (n,) backend array 1.0/0.0 for disk/other (no self_gravity path)

        Returns:
            (fx, fy) or (fx, fy, fz) as backend arrays
        """
        n = positions.shape[0]
        dim = positions.shape[1]

        threshold = getattr(self, "barnes_hut_threshold", BARNES_HUT_N_THRESHOLD)
        use_bh = getattr(self, "use_barnes_hut_for_large_n", True)
        # Large N: use Barnes-Hut 2D approximation (O(N log N)); conversion only at this boundary
        if (
            n > threshold
            and use_bh
            and compute_forces_barnes_hut is not None
            and self_gravity
            and is_disk is None
        ):
            result = compute_forces_barnes_hut(
                positions, masses, backend, G=G, epsilon=epsilon
            )
            return result  # (fx, fy) or (fx, fy, fz)

        # JAX path: keep on device, JIT-compiled (only when no disk-disk masking)
        if backend.name == "jax" and eps_cd is None and eps_bd is None:
            jit_fn = _get_jax_force_jit()
            if jit_fn is not None and (self_gravity or is_disk is None):
                masses_1d = backend.reshape(masses, (n,))
                forces = jit_fn(positions, masses_1d, G, epsilon)
                return _forces_to_components(forces, dim, backend)
            # fallback to backend vectorized (e.g. self_gravity=False with is_disk)

        # Backend-native vectorized path (NumPy, CuPy, PyTorch, or JAX fallback)
        forces = self._compute_forces_backend_native(
            positions,
            masses,
            backend,
            G,
            epsilon,
            self_gravity,
            is_disk,
            is_bulge,
            is_core,
            eps_cd,
            eps_bd,
        )
        return _forces_to_components(forces, dim, backend)

    def _compute_forces_backend_native(
        self,
        positions: Any,
        masses: Any,
        backend: Backend,
        G: float,
        epsilon: float,
        self_gravity: bool,
        is_disk: Optional[Any],
        is_bulge: Optional[Any],
        is_core: Optional[Any],
        eps_cd: Optional[float],
        eps_bd: Optional[float],
    ) -> Any:
        """Vectorized force calculation using only backend ops (no to_numpy)."""
        n = positions.shape[0]
        dim = positions.shape[1]
        # r_diff: (1,n,dim) - (n,1,dim) -> (n,n,dim)
        pos_i = backend.reshape(positions, (n, 1, dim))
        pos_j = backend.reshape(positions, (1, n, dim))
        r_diff = backend.subtract(pos_j, pos_i)
        r_sq = backend.sum(backend.square(r_diff), axis=2)
        # Interaction-specific softening
        eps_matrix = None
        if eps_cd is not None or eps_bd is not None:
            eps_matrix = backend.multiply(backend.ones((n, n)), float(epsilon))
            if eps_cd is not None and is_disk is not None and is_core is not None:
                disk_i = backend.expand_dims(is_disk, 1)
                disk_j = backend.expand_dims(is_disk, 0)
                core_i = backend.expand_dims(is_core, 1)
                core_j = backend.expand_dims(is_core, 0)
                mask_cd = backend.add(
                    backend.multiply(disk_i, core_j),
                    backend.multiply(core_i, disk_j),
                )
                eps_matrix = backend.where(mask_cd, float(eps_cd), eps_matrix)
            if eps_bd is not None and is_disk is not None and is_bulge is not None:
                disk_i = backend.expand_dims(is_disk, 1)
                disk_j = backend.expand_dims(is_disk, 0)
                bulge_i = backend.expand_dims(is_bulge, 1)
                bulge_j = backend.expand_dims(is_bulge, 0)
                mask_bd = backend.add(
                    backend.multiply(disk_i, bulge_j),
                    backend.multiply(bulge_i, disk_j),
                )
                eps_matrix = backend.where(mask_bd, float(eps_bd), eps_matrix)
        if eps_matrix is None:
            r_soft_cubed = backend.power(
                backend.add(r_sq, epsilon ** 2), 1.5
            )
        else:
            r_soft_cubed = backend.power(
                backend.add(r_sq, backend.square(eps_matrix)), 1.5
            )
        m_i = backend.expand_dims(masses, 1)
        m_j = backend.expand_dims(masses, 0)
        force_magnitude = backend.multiply(
            G, backend.divide(backend.multiply(m_i, m_j), r_soft_cubed)
        )
        # Zero diagonal: force_magnitude * (1 - eye)
        identity = backend.eye(n)
        force_magnitude = backend.multiply(
            force_magnitude,
            backend.subtract(1.0, identity),
        )
        # force_vectors (n,n,dim)
        force_mag_exp = backend.expand_dims(force_magnitude, 2)
        force_vectors = backend.multiply(force_mag_exp, r_diff)
        if not self_gravity and is_disk is not None:
            disk_disk = backend.multiply(
                backend.expand_dims(is_disk, 1),
                backend.expand_dims(is_disk, 0),
            )
            mask = backend.subtract(1.0, backend.expand_dims(disk_disk, 2))
            force_vectors = backend.multiply(force_vectors, mask)
        forces = backend.sum(force_vectors, axis=1)
        return forces


def _forces_to_components(forces: Any, dim: int, backend: Backend) -> Tuple:
    """Split (n, dim) forces into (fx, fy) or (fx, fy, fz)."""
    if dim == 3:
        return forces[:, 0], forces[:, 1], forces[:, 2]
    return forces[:, 0], forces[:, 1]
