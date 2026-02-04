"""Barnes-Hut 2D quadtree approximation for O(N log N) forces when N is large (>20k).

Use when N > threshold to avoid O(N^2) memory and time. Conversion to NumPy
happens only at the boundary of this module; caller passes backend arrays and
receives backend array forces.
"""

import numpy as np
from typing import Tuple, Optional, Any
from galaxy_sim.backends.base import Backend


# Default threshold above which to use Barnes-Hut (pure O(N^2) does not scale on GPU)
BARNES_HUT_N_THRESHOLD = 20_000


class _QuadNode:
    """Single node of the quadtree (2D)."""

    __slots__ = ("center", "size", "mass", "com", "particle_idx", "children")

    def __init__(self, center: np.ndarray, size: float):
        self.center = center  # (2,)
        self.size = size
        self.mass = 0.0
        self.com = np.zeros(2)
        self.particle_idx: Optional[int] = None
        self.children: Optional[Tuple["_QuadNode", "_QuadNode", "_QuadNode", "_QuadNode"]] = None


def _build_tree(
    positions: np.ndarray,
    masses: np.ndarray,
    center: np.ndarray,
    size: float,
    indices: np.ndarray,
) -> Optional[_QuadNode]:
    """Build quadtree for particles in indices. positions (n, 2), masses (n,)."""
    if len(indices) == 0:
        return None
    node = _QuadNode(center, size)
    if len(indices) == 1:
        i = indices[0]
        node.mass = masses[i]
        node.com = positions[i].copy()
        node.particle_idx = i
        return node
    # Subdivide
    half = size / 2
    cx, cy = center[0], center[1]
    quads = [
        np.where((positions[indices, 0] < cx) & (positions[indices, 1] < cy))[0],
        np.where((positions[indices, 0] >= cx) & (positions[indices, 1] < cy))[0],
        np.where((positions[indices, 0] < cx) & (positions[indices, 1] >= cy))[0],
        np.where((positions[indices, 0] >= cx) & (positions[indices, 1] >= cy))[0],
    ]
    children = []
    centers = [
        np.array([cx - half / 2, cy - half / 2]),
        np.array([cx + half / 2, cy - half / 2]),
        np.array([cx - half / 2, cy + half / 2]),
        np.array([cx + half / 2, cy + half / 2]),
    ]
    for q, c in zip(quads, centers):
        sub = indices[q]
        child = _build_tree(positions, masses, c, half, sub)
        if child is not None:
            node.mass += child.mass
            node.com += child.mass * child.com
            children.append(child)
        else:
            children.append(None)
    if node.mass > 0:
        node.com /= node.mass
    node.children = tuple(children)
    return node


def _force_on_particle(
    i: int,
    pos: np.ndarray,
    tree: Optional[_QuadNode],
    G: float,
    eps: float,
    theta: float,
) -> np.ndarray:
    """Compute force on particle i from tree (theta = opening angle threshold)."""
    if tree is None:
        return np.zeros(2)
    r = pos - tree.com
    r_norm_sq = np.sum(r ** 2) + eps ** 2
    r_norm = np.sqrt(r_norm_sq)
    if tree.particle_idx is not None:
        if tree.particle_idx == i:
            return np.zeros(2)
        # Leaf: direct force
        f_mag = G * tree.mass / (r_norm_sq ** 1.5)
        return f_mag * r
    # Cell: open if s/d > theta
    s = tree.size
    if s / (r_norm + 1e-10) < theta:
        f_mag = G * tree.mass / (r_norm_sq ** 1.5)
        return f_mag * r
    # Recurse
    out = np.zeros(2)
    for ch in tree.children:
        if ch is not None:
            out += _force_on_particle(i, pos, ch, G, eps, theta)
    return out


def compute_forces_barnes_hut_2d(
    positions: np.ndarray,
    masses: np.ndarray,
    G: float = 1.0,
    epsilon: float = 1e-3,
    theta: float = 0.7,
) -> np.ndarray:
    """Compute forces (n, 2) using 2D Barnes-Hut. positions (n, 2), masses (n,)."""
    n = positions.shape[0]
    if n == 0:
        return np.zeros((0, 2))
    pos_2d = positions[:, :2] if positions.shape[1] >= 2 else positions
    center = np.mean(pos_2d, axis=0)
    size = float(np.max(np.ptp(pos_2d, axis=0)) * 1.01 or 1.0)
    indices = np.arange(n)
    tree = _build_tree(pos_2d, masses, center, size, indices)
    forces = np.zeros((n, 2))
    for i in range(n):
        forces[i] = _force_on_particle(i, pos_2d[i], tree, G, epsilon, theta)
    return forces


def compute_forces_barnes_hut(
    positions: Any,
    masses: Any,
    backend: Backend,
    G: float = 1.0,
    epsilon: float = 1e-3,
    theta: float = 0.7,
) -> Tuple[Any, Any]:
    """Barnes-Hut 2D; accepts backend arrays, returns (fx, fy) as backend arrays.
    Conversion only at boundary (to_numpy for tree, backend.array for result).
    """
    positions_np = np.asarray(backend.to_numpy(positions))
    masses_np = np.asarray(backend.to_numpy(masses)).flatten()
    n = positions_np.shape[0]
    dim = positions_np.shape[1]
    forces_2d = compute_forces_barnes_hut_2d(positions_np, masses_np, G, epsilon, theta)
    fx = backend.array(forces_2d[:, 0])
    fy = backend.array(forces_2d[:, 1])
    if dim == 3:
        fz = backend.zeros(n)
        return fx, fy, fz
    return fx, fy
