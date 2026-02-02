"""
Galaxy Simulator - A senior-level N-body galaxy simulation framework.

Features:
- Multiple compute backends (NumPy, JAX, PyTorch, CuPy)
- Multiple integrators (Euler, Verlet, RK4)
- 2D and 3D rendering
- Preset scenarios (spiral, collision, globular, cluster)
- Export to MP4/GIF
- CLI and GUI interfaces
"""

__version__ = "0.1.0"

from galaxy_sim.physics.simulator import Simulator
from galaxy_sim.backends.factory import get_backend, list_available_backends

__all__ = [
    "Simulator",
    "get_backend",
    "list_available_backends",
]
