"""CuPy backend implementation (optional, CUDA-only)."""

from typing import Any, Tuple, Union
import numpy as np
from galaxy_sim.backends.base import Backend

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class CuPyBackend(Backend):
    """CuPy-based backend (CUDA GPU only)."""
    
    def __init__(self, device: int = 0):
        """Initialize CuPy backend.
        
        Args:
            device: CUDA device ID
        """
        if not CUPY_AVAILABLE:
            raise ImportError("CuPy not available. Install with: pip install cupy")
        
        self._device = device
        cp.cuda.Device(device).use()
        self._seed = None
    
    @property
    def name(self) -> str:
        return "cupy"
    
    @property
    def device(self) -> str:
        return f"cuda:{self._device}"
    
    def array(self, data: Any, dtype=None) -> Any:
        return cp.array(data, dtype=dtype)
    
    def zeros(self, shape: Tuple[int, ...], dtype=None) -> Any:
        return cp.zeros(shape, dtype=dtype or cp.float64)
    
    def ones(self, shape: Tuple[int, ...], dtype=None) -> Any:
        return cp.ones(shape, dtype=dtype or cp.float64)
    
    def zeros_like(self, array: Any) -> Any:
        return cp.zeros_like(array)
    
    def norm(self, vec: Any, axis: int = -1, keepdims: bool = False) -> Any:
        return cp.linalg.norm(vec, axis=axis, keepdims=keepdims)
    
    def dot(self, a: Any, b: Any) -> Any:
        return cp.dot(a, b)
    
    def sum(self, array: Any, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> Any:
        return cp.sum(array, axis=axis, keepdims=keepdims)
    
    def mean(self, array: Any, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> Any:
        return cp.mean(array, axis=axis, keepdims=keepdims)
    
    def sqrt(self, array: Any) -> Any:
        return cp.sqrt(array)
    
    def square(self, array: Any) -> Any:
        return cp.square(array)
    
    def add(self, a: Any, b: Any) -> Any:
        return cp.add(a, b)
    
    def subtract(self, a: Any, b: Any) -> Any:
        return cp.subtract(a, b)
    
    def multiply(self, a: Any, b: Any) -> Any:
        return cp.multiply(a, b)
    
    def divide(self, a: Any, b: Any) -> Any:
        return cp.divide(a, b)
    
    def power(self, base: Any, exponent: Any) -> Any:
        return cp.power(base, exponent)
    
    def maximum(self, a: Any, b: Any) -> Any:
        return cp.maximum(a, b)
    
    def minimum(self, a: Any, b: Any) -> Any:
        return cp.minimum(a, b)
    
    def clip(self, array: Any, min_val: float, max_val: float) -> Any:
        return cp.clip(array, min_val, max_val)
    
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        return cp.where(condition, x, y)
    
    def to_numpy(self, array: Any) -> np.ndarray:
        return cp.asnumpy(array)
    
    def random_normal(self, shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0, seed: int = None) -> Any:
        rng = cp.random.RandomState(seed if seed is not None else self._seed)
        return rng.normal(mean, std, shape)
    
    def random_uniform(self, shape: Tuple[int, ...], low: float = 0.0, high: float = 1.0, seed: int = None) -> Any:
        rng = cp.random.RandomState(seed if seed is not None else (self._seed or 0))
        return rng.uniform(low, high, shape)
    
    def set_seed(self, seed: int) -> None:
        self._seed = seed
        cp.random.seed(seed)
