"""PyTorch backend implementation (optional, GPU support)."""

from typing import Any, Tuple, Union
import numpy as np
from galaxy_sim.backends.base import Backend

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class PyTorchBackend(Backend):
    """PyTorch-based backend with GPU support."""
    
    def __init__(self, device: str = None):
        """Initialize PyTorch backend.
        
        Args:
            device: Device string (e.g., 'cpu', 'cuda:0'). Auto-selects if None.
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        
        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = torch.device(device)
        
        self._seed = None
    
    @property
    def name(self) -> str:
        return "pytorch"
    
    @property
    def device(self) -> str:
        return str(self._device)
    
    def array(self, data: Any, dtype=None) -> Any:
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data)
        else:
            tensor = torch.tensor(data, dtype=dtype)
        return tensor.to(self._device)
    
    def zeros(self, shape: Tuple[int, ...], dtype=None) -> Any:
        return torch.zeros(shape, dtype=dtype or torch.float64, device=self._device)
    
    def ones(self, shape: Tuple[int, ...], dtype=None) -> Any:
        return torch.ones(shape, dtype=dtype or torch.float64, device=self._device)
    
    def zeros_like(self, array: Any) -> Any:
        return torch.zeros_like(array)
    
    def norm(self, vec: Any, axis: int = -1, keepdims: bool = False) -> Any:
        return torch.norm(vec, dim=axis, keepdim=keepdims)
    
    def dot(self, a: Any, b: Any) -> Any:
        return torch.dot(a, b)
    
    def sum(self, array: Any, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> Any:
        if axis is None:
            return torch.sum(array)
        return torch.sum(array, dim=axis, keepdim=keepdims)
    
    def mean(self, array: Any, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> Any:
        if axis is None:
            return torch.mean(array)
        return torch.mean(array, dim=axis, keepdim=keepdims)
    
    def sqrt(self, array: Any) -> Any:
        return torch.sqrt(array)
    
    def square(self, array: Any) -> Any:
        return torch.square(array)
    
    def add(self, a: Any, b: Any) -> Any:
        return torch.add(a, b)
    
    def subtract(self, a: Any, b: Any) -> Any:
        return torch.subtract(a, b)
    
    def multiply(self, a: Any, b: Any) -> Any:
        return torch.multiply(a, b)
    
    def divide(self, a: Any, b: Any) -> Any:
        return torch.divide(a, b)
    
    def power(self, base: Any, exponent: Any) -> Any:
        return torch.pow(base, exponent)
    
    def maximum(self, a: Any, b: Any) -> Any:
        return torch.maximum(a, b)
    
    def minimum(self, a: Any, b: Any) -> Any:
        return torch.minimum(a, b)
    
    def clip(self, array: Any, min_val: float, max_val: float) -> Any:
        return torch.clamp(array, min_val, max_val)
    
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        return torch.where(condition, x, y)
    
    def stack(self, arrays, axis: int = 0) -> Any:
        return torch.stack(arrays, dim=axis)

    def reshape(self, array: Any, newshape: Tuple[int, ...]) -> Any:
        return torch.reshape(array, newshape)

    def expand_dims(self, array: Any, axis: int) -> Any:
        return torch.unsqueeze(array, dim=axis)

    def eye(self, n: int, dtype=None) -> Any:
        return torch.eye(n, dtype=dtype or torch.float64, device=self._device)

    def to_numpy(self, array: Any) -> np.ndarray:
        if isinstance(array, torch.Tensor):
            return array.cpu().numpy()
        return np.asarray(array)
    
    def random_normal(self, shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0, seed: int = None) -> Any:
        generator = torch.Generator(device=self._device)
        if seed is not None:
            generator.manual_seed(seed)
        elif self._seed is not None:
            generator.manual_seed(self._seed)
        return torch.normal(mean, std, size=shape, generator=generator, device=self._device)
    
    def random_uniform(self, shape: Tuple[int, ...], low: float = 0.0, high: float = 1.0, seed: int = None) -> Any:
        generator = torch.Generator(device=self._device)
        if seed is not None:
            generator.manual_seed(seed)
        elif self._seed is not None:
            generator.manual_seed(self._seed)
        return torch.empty(shape, device=self._device).uniform_(low, high)
    
    def set_seed(self, seed: int) -> None:
        self._seed = seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
