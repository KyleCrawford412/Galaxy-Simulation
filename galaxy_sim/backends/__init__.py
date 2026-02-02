"""Compute backend abstractions for galaxy simulation."""

from galaxy_sim.backends.base import Backend
from galaxy_sim.backends.factory import get_backend, list_available_backends

__all__ = ["Backend", "get_backend", "list_available_backends"]
