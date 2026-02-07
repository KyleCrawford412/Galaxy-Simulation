"""JAX backend implementation (optional, GPU support)."""

from typing import Any, Tuple, Union
import numpy as np
from galaxy_sim.backends.base import Backend

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class JAXBackend(Backend):
    """JAX-based backend with GPU support."""
    
    def __init__(self, device: str = None, use_float32: bool = True):
        """Initialize JAX backend.
        
        Args:
            device: Device string (e.g., 'cpu', 'gpu:0'). Auto-selects if None.
            use_float32: Use float32 for arrays (faster on GPU, less memory).
        """
        if not JAX_AVAILABLE:
            raise ImportError("JAX not available. Install with: pip install jax jaxlib")
        
        self._device = device or jax.devices()[0]
        self._seed = None
        self._float32 = use_float32
        self._dtype = jnp.float32 if use_float32 else jnp.float64
    
    @property
    def name(self) -> str:
        return "jax"
    
    @property
    def device(self) -> str:
        return str(self._device)
    
    def array(self, data: Any, dtype=None) -> Any:
        return jnp.array(data, dtype=dtype)
    
    def zeros(self, shape: Tuple[int, ...], dtype=None) -> Any:
        return jnp.zeros(shape, dtype=dtype or jnp.float64)
    
    def ones(self, shape: Tuple[int, ...], dtype=None) -> Any:
        return jnp.ones(shape, dtype=dtype or jnp.float64)
    
    def zeros_like(self, array: Any) -> Any:
        return jnp.zeros_like(array)
    
    def norm(self, vec: Any, axis: int = -1, keepdims: bool = False) -> Any:
        return jnp.linalg.norm(vec, axis=axis, keepdims=keepdims)
    
    def dot(self, a: Any, b: Any) -> Any:
        return jnp.dot(a, b)
    
    def sum(self, array: Any, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> Any:
        return jnp.sum(array, axis=axis, keepdims=keepdims)
    
    def mean(self, array: Any, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> Any:
        return jnp.mean(array, axis=axis, keepdims=keepdims)
    
    def sqrt(self, array: Any) -> Any:
        return jnp.sqrt(array)
    
    def square(self, array: Any) -> Any:
        return jnp.square(array)
    
    def add(self, a: Any, b: Any) -> Any:
        return jnp.add(a, b)
    
    def subtract(self, a: Any, b: Any) -> Any:
        return jnp.subtract(a, b)
    
    def multiply(self, a: Any, b: Any) -> Any:
        return jnp.multiply(a, b)
    
    def divide(self, a: Any, b: Any) -> Any:
        return jnp.divide(a, b)
    
    def power(self, base: Any, exponent: Any) -> Any:
        return jnp.power(base, exponent)
    
    def maximum(self, a: Any, b: Any) -> Any:
        return jnp.maximum(a, b)
    
    def minimum(self, a: Any, b: Any) -> Any:
        return jnp.minimum(a, b)
    
    def clip(self, array: Any, min_val: float, max_val: float) -> Any:
        return jnp.clip(array, min_val, max_val)
    
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        return jnp.where(condition, x, y)
    
    def stack(self, arrays, axis: int = 0) -> Any:
        return jnp.stack(arrays, axis=axis)

    def reshape(self, array: Any, newshape: Tuple[int, ...]) -> Any:
        return jnp.reshape(array, newshape)

    def expand_dims(self, array: Any, axis: int) -> Any:
        return jnp.expand_dims(array, axis=axis)

    def eye(self, n: int, dtype=None) -> Any:
        return jnp.eye(n, dtype=dtype or self._dtype)

    def sin(self, array: Any) -> Any:
        return jnp.sin(array)

    def cos(self, array: Any) -> Any:
        return jnp.cos(array)

    def tan(self, array: Any) -> Any:
        return jnp.tan(array)

    def atan2(self, y: Any, x: Any) -> Any:
        return jnp.arctan2(y, x)

    def exp(self, array: Any) -> Any:
        return jnp.exp(array)

    def log(self, array: Any) -> Any:
        return jnp.log(array)

    def to_numpy(self, array: Any) -> np.ndarray:
        return np.asarray(array)
    
    def random_normal(self, shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0, seed: int = None) -> Any:
        key = jax.random.PRNGKey(seed if seed is not None else (self._seed or 0))
        return jax.random.normal(key, shape) * std + mean
    
    def random_uniform(self, shape: Tuple[int, ...], low: float = 0.0, high: float = 1.0, seed: int = None) -> Any:
        key = jax.random.PRNGKey(seed if seed is not None else (self._seed or 0))
        return jax.random.uniform(key, shape, minval=low, maxval=high)
    
    def set_seed(self, seed: int) -> None:
        self._seed = seed
