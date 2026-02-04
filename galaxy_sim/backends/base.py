"""Abstract base class for compute backends."""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Union
import numpy as np


class Backend(ABC):
    """Abstract interface for array computation backends.
    
    This allows the same simulation code to run on different execution
    engines (NumPy, JAX, PyTorch, CuPy) with a unified API.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this backend."""
        pass
    
    @property
    @abstractmethod
    def device(self) -> str:
        """Return the device type (e.g., 'cpu', 'cuda:0')."""
        pass
    
    @abstractmethod
    def array(self, data: Any, dtype=None) -> Any:
        """Create an array from data.
        
        Args:
            data: Input data (list, numpy array, etc.)
            dtype: Optional data type
            
        Returns:
            Backend array object
        """
        pass
    
    @abstractmethod
    def zeros(self, shape: Tuple[int, ...], dtype=None) -> Any:
        """Create an array of zeros.
        
        Args:
            shape: Array shape
            dtype: Optional data type
            
        Returns:
            Zero-filled array
        """
        pass
    
    @abstractmethod
    def ones(self, shape: Tuple[int, ...], dtype=None) -> Any:
        """Create an array of ones."""
        pass
    
    @abstractmethod
    def zeros_like(self, array: Any) -> Any:
        """Create an array of zeros with same shape as input."""
        pass
    
    @abstractmethod
    def norm(self, vec: Any, axis: int = -1, keepdims: bool = False) -> Any:
        """Compute L2 norm along specified axis.
        
        Args:
            vec: Input vector/array
            axis: Axis along which to compute norm
            keepdims: Whether to keep dimensions
            
        Returns:
            Norm values
        """
        pass
    
    @abstractmethod
    def dot(self, a: Any, b: Any) -> Any:
        """Compute dot product."""
        pass
    
    @abstractmethod
    def sum(self, array: Any, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> Any:
        """Sum array elements along axis."""
        pass
    
    @abstractmethod
    def mean(self, array: Any, axis: Union[int, Tuple[int, ...]] = None, keepdims: bool = False) -> Any:
        """Compute mean along axis."""
        pass
    
    @abstractmethod
    def sqrt(self, array: Any) -> Any:
        """Compute square root."""
        pass
    
    @abstractmethod
    def square(self, array: Any) -> Any:
        """Compute square."""
        pass
    
    @abstractmethod
    def add(self, a: Any, b: Any) -> Any:
        """Element-wise addition."""
        pass
    
    @abstractmethod
    def subtract(self, a: Any, b: Any) -> Any:
        """Element-wise subtraction."""
        pass
    
    @abstractmethod
    def multiply(self, a: Any, b: Any) -> Any:
        """Element-wise multiplication."""
        pass
    
    @abstractmethod
    def divide(self, a: Any, b: Any) -> Any:
        """Element-wise division."""
        pass
    
    @abstractmethod
    def power(self, base: Any, exponent: Any) -> Any:
        """Element-wise power."""
        pass
    
    @abstractmethod
    def maximum(self, a: Any, b: Any) -> Any:
        """Element-wise maximum."""
        pass
    
    @abstractmethod
    def minimum(self, a: Any, b: Any) -> Any:
        """Element-wise minimum."""
        pass
    
    @abstractmethod
    def clip(self, array: Any, min_val: float, max_val: float) -> Any:
        """Clip array values to range."""
        pass
    
    @abstractmethod
    def where(self, condition: Any, x: Any, y: Any) -> Any:
        """Conditional selection."""
        pass
    
    @abstractmethod
    def stack(self, arrays, axis: int = 0) -> Any:
        """Stack arrays along a new axis. E.g. stack([ax, ay, az], axis=1) -> (n, 3)."""
        pass

    @abstractmethod
    def reshape(self, array: Any, newshape: Tuple[int, ...]) -> Any:
        """Reshape array to newshape (for broadcasting, etc.)."""
        pass

    @abstractmethod
    def expand_dims(self, array: Any, axis: int) -> Any:
        """Expand the shape by inserting a new axis at axis (e.g. (n,) -> (n, 1))."""
        pass

    @abstractmethod
    def eye(self, n: int, dtype=None) -> Any:
        """Identity matrix (n, n), for masking diagonal."""
        pass

    @abstractmethod
    def to_numpy(self, array: Any) -> np.ndarray:
        """Convert backend array to NumPy array.
        
        This is needed for rendering and I/O operations.
        """
        pass
    
    @abstractmethod
    def random_normal(self, shape: Tuple[int, ...], mean: float = 0.0, std: float = 1.0, seed: int = None) -> Any:
        """Generate random normal distribution."""
        pass
    
    @abstractmethod
    def random_uniform(self, shape: Tuple[int, ...], low: float = 0.0, high: float = 1.0, seed: int = None) -> Any:
        """Generate random uniform distribution."""
        pass
    
    @abstractmethod
    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        pass
