"""Backend factory for creating and managing compute backends."""

from typing import List, Optional
from galaxy_sim.backends.base import Backend
from galaxy_sim.backends.numpy_backend import NumPyBackend

# Optional backends - will be imported if available
_jax_backend = None
_pytorch_backend = None
_cupy_backend = None

try:
    from galaxy_sim.backends.jax_backend import JAXBackend
    _jax_backend = JAXBackend
except ImportError:
    pass

try:
    from galaxy_sim.backends.pytorch_backend import PyTorchBackend
    _pytorch_backend = PyTorchBackend
except ImportError:
    pass

try:
    from galaxy_sim.backends.cupy_backend import CuPyBackend
    _cupy_backend = CuPyBackend
except ImportError:
    pass


def list_available_backends() -> List[str]:
    """List all available backends.
    
    Returns:
        List of backend names that can be instantiated
    """
    backends = ["numpy"]  # Always available
    
    if _jax_backend is not None:
        backends.append("jax")
    
    if _pytorch_backend is not None:
        backends.append("pytorch")
    
    if _cupy_backend is not None:
        backends.append("cupy")
    
    return backends


def get_backend(name: Optional[str] = None, prefer_gpu: bool = True) -> Backend:
    """Get a backend instance.
    
    Args:
        name: Backend name ('numpy', 'jax', 'pytorch', 'cupy'). If None, auto-selects.
        prefer_gpu: If True and name is None, prefer GPU backends over CPU.
        
    Returns:
        Backend instance
        
    Raises:
        ValueError: If requested backend is not available
    """
    if name is None:
        # Auto-select: prefer GPU backends if available
        if prefer_gpu:
            if _cupy_backend is not None:
                try:
                    return _cupy_backend()
                except Exception:
                    pass
            if _jax_backend is not None:
                try:
                    return _jax_backend()
                except Exception:
                    pass
            if _pytorch_backend is not None:
                try:
                    return _pytorch_backend()
                except Exception:
                    pass
        
        # Fallback to NumPy
        return NumPyBackend()
    
    name_lower = name.lower()
    
    if name_lower == "numpy":
        return NumPyBackend()
    elif name_lower == "jax":
        if _jax_backend is None:
            raise ValueError("JAX backend not available. Install with: pip install jax jaxlib")
        return _jax_backend()
    elif name_lower == "pytorch":
        if _pytorch_backend is None:
            raise ValueError("PyTorch backend not available. Install with: pip install torch")
        return _pytorch_backend()
    elif name_lower == "cupy":
        if _cupy_backend is None:
            raise ValueError("CuPy backend not available. Install with: pip install cupy")
        return _cupy_backend()
    else:
        available = list_available_backends()
        raise ValueError(f"Unknown backend '{name}'. Available: {available}")
