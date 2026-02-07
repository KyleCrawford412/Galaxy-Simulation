"""NumPy backend implementation."""

from typing import Any, Tuple, Union
import numpy as np
from galaxy_sim.backends.base import Backend


class NumPyBackend(Backend):
    """NumPy-based backend (baseline, always available)."""
    
    def __init__(self):
        self._seed = None
    
    @property
    def name(self) -> str:
        return "numpy"
    
    @property
    def device(self) -> str:
        return "cpu"
    
    def array(self, data: Any, dtype=None) -> np.ndarray:
        return np.array(data, dtype=dtype)
    
    def zeros(self, shape: Tuple[int, ...], dtype=None) -> np.ndarray:
        return np.zeros(shape, dtype=dtype or np.float64)
    
    def ones(self, shape: Tuple[int, ...], dtype=None) -> np.ndarray:
        return np.ones(shape, dtype=dtype or np.float64)
    
    def zeros_like(self, array: Any) -> np.ndarray:
        return np.zeros_like(array)
    
    def norm(self, vec: Any, axis: int = -1, keepdims: bool = False) -> np.ndarray:
        return np.linalg.norm(vec, axis=axis, keepdims=keepdims)
    
    def dot(self, a: Any, b: Any) -> np.ndarray:
        return np.dot(a, b)
    
    def sum(self, array: Any, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> np.ndarray:
        return np.sum(array, axis=axis, keepdims=keepdims)
    
    def mean(self, array: Any, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> np.ndarray:
        return np.mean(array, axis=axis, keepdims=keepdims)
    
    def sqrt(self, array: Any) -> np.ndarray:
        return np.sqrt(array)
    
    def square(self, array: Any) -> np.ndarray:
        return np.square(array)
    
    def add(self, a: Any, b: Any) -> np.ndarray:
        return np.add(a, b)
    
    def subtract(self, a: Any, b: Any) -> np.ndarray:
        return np.subtract(a, b)
    
    def multiply(self, a: Any, b: Any) -> np.ndarray:
        return np.multiply(a, b)
    
    def divide(self, a: Any, b: Any) -> np.ndarray:
        return np.divide(a, b)
    
    def power(self, base: Any, exponent: Any) -> np.ndarray:
        return np.power(base, exponent)
    
    def maximum(self, a: Any, b: Any) -> np.ndarray:
        return np.maximum(a, b)
    
    def minimum(self, a: Any, b: Any) -> np.ndarray:
        return np.minimum(a, b)
    
    def clip(self, array: Any, min_val: float, max_val: float) -> np.ndarray:
        return np.clip(array, min_val, max_val)
    
    def where(self, condition: Any, x: Any, y: Any) -> np.ndarray:
        return np.where(condition, x, y)
    
    def stack(self, arrays, axis: int = 0) -> np.ndarray:
        return np.stack(arrays, axis=axis)

    def reshape(self, array: Any, newshape: Tuple[int, ...]) -> np.ndarray:
        return np.reshape(array, newshape)

    def expand_dims(self, array: Any, axis: int) -> np.ndarray:
        return np.expand_dims(array, axis=axis)

    def eye(self, n: int, dtype=None) -> np.ndarray:
        return np.eye(n, dtype=dtype or np.float64)

    def sin(self, array: Any) -> np.ndarray:
        return np.sin(array)

    def cos(self, array: Any) -> np.ndarray:
        return np.cos(array)

    def tan(self, array: Any) -> np.ndarray:
        return np.tan(array)

    def atan2(self, y: Any, x: Any) -> np.ndarray:
        return np.arctan2(y, x)

    def exp(self, array: Any) -> np.ndarray:
        return np.exp(array)

    def log(self, array: Any) -> np.ndarray:
        return np.log(array)

    def to_numpy(self, array: Any) -> np.ndarray:
        return np.asarray(array)
    
    def random_normal(self, shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0, seed: int = None) -> np.ndarray:
        if seed is not None:
            rng = np.random.default_rng(seed)
            return rng.normal(mean, std, shape)
        elif self._seed is not None:
            rng = np.random.default_rng(self._seed)
            return rng.normal(mean, std, shape)
        return np.random.normal(mean, std, shape)
    
    def random_uniform(self, shape: Tuple[int, ...], low: float = 0.0, high: float = 1.0, seed: int = None) -> np.ndarray:
        if seed is not None:
            rng = np.random.default_rng(seed)
            return rng.uniform(low, high, shape)
        elif self._seed is not None:
            rng = np.random.default_rng(self._seed)
            return rng.uniform(low, high, shape)
        return np.random.uniform(low, high, shape)
    
    def set_seed(self, seed: int) -> None:
        self._seed = seed
        np.random.seed(seed)
