"""
Tensor implementation for the Neurenix framework.
"""

import numpy as np
from typing import List, Tuple, Union, Optional, Sequence, Any
from enum import Enum

from neurenix.device import Device, DeviceType

class DType(Enum):
    """Data types supported by Neurenix tensors."""
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"
    
    @classmethod
    def from_numpy(cls, dtype: np.dtype) -> "DType":
        """Convert a NumPy dtype to a Neurenix DType."""
        if dtype == np.float32:
            return cls.FLOAT32
        elif dtype == np.float64:
            return cls.FLOAT64
        elif dtype == np.int32:
            return cls.INT32
        elif dtype == np.int64:
            return cls.INT64
        elif dtype == np.bool_:
            return cls.BOOL
        else:
            raise ValueError(f"Unsupported NumPy dtype: {dtype}")
    
    def to_numpy(self) -> np.dtype:
        """Convert a Neurenix DType to a NumPy dtype."""
        if self == DType.FLOAT32:
            return np.float32
        elif self == DType.FLOAT64:
            return np.float64
        elif self == DType.INT32:
            return np.int32
        elif self == DType.INT64:
            return np.int64
        elif self == DType.BOOL:
            return np.bool_
        else:
            raise ValueError(f"Unsupported Neurenix dtype: {self}")

class Tensor:
    """
    A multi-dimensional array with support for various hardware devices.
    
    This is the main data structure in Neurenix, similar to tensors in other
    frameworks like PyTorch or TensorFlow.
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, List, Tuple, "Tensor", None] = None,
        shape: Optional[Sequence[int]] = None,
        dtype: Optional[Union[DType, str]] = None,
        device: Optional[Device] = None,
        requires_grad: bool = False,
    ):
        """
        Create a new tensor.
        
        Args:
            data: The data to initialize the tensor with. Can be a NumPy array,
                a list, a tuple, another Tensor, or None (for uninitialized tensor).
            shape: The shape of the tensor. If None, inferred from data.
            dtype: The data type of the tensor. If None, inferred from data.
            device: The device to store the tensor on. If None, uses the default device.
            requires_grad: Whether to track gradients for this tensor.
        """
        from neurenix.core import get_config
        
        # Set device
        if device is None:
            device_str = get_config().get("device", "cpu")
            if device_str == "cpu":
                self._device = Device(DeviceType.CPU)
            elif device_str.startswith("cuda"):
                # Extract device index if specified (e.g., "cuda:0")
                parts = device_str.split(":")
                index = int(parts[1]) if len(parts) > 1 else 0
                self._device = Device(DeviceType.CUDA, index)
            else:
                raise ValueError(f"Unsupported device: {device_str}")
        else:
            self._device = device
        
        # Handle different input types
        if data is None:
            if shape is None:
                raise ValueError("Either data or shape must be provided")
            
            # Create uninitialized tensor with the given shape
            if dtype is None:
                dtype = DType.FLOAT32
            elif isinstance(dtype, str):
                dtype = DType(dtype)
            
            self._shape = tuple(shape)
            self._dtype = dtype
            self._data = None  # Will be allocated on the device
            
            # TODO: Allocate memory on the device when Phynexus bindings are available
            # For now, use NumPy as a fallback
            self._numpy_data = np.zeros(shape, dtype=dtype.to_numpy())
            
        elif isinstance(data, Tensor):
            # Create a new tensor from an existing tensor
            self._shape = data.shape
            self._dtype = data.dtype if dtype is None else dtype
            self._data = None  # Will be a copy of the source tensor's data
            
            # TODO: Copy data from the source tensor when Phynexus bindings are available
            # For now, use NumPy as a fallback
            self._numpy_data = data._numpy_data.copy()
            
        elif isinstance(data, (list, tuple)):
            # Create a tensor from a Python list or tuple
            numpy_data = np.array(data)
            self._shape = numpy_data.shape
            
            if dtype is None:
                self._dtype = DType.from_numpy(numpy_data.dtype)
            elif isinstance(dtype, str):
                self._dtype = DType(dtype)
                numpy_data = numpy_data.astype(self._dtype.to_numpy())
            else:
                self._dtype = dtype
                numpy_data = numpy_data.astype(self._dtype.to_numpy())
            
            self._data = None  # Will be allocated on the device
            self._numpy_data = numpy_data
            
        elif isinstance(data, np.ndarray):
            # Create a tensor from a NumPy array
            self._shape = data.shape
            
            if dtype is None:
                self._dtype = DType.from_numpy(data.dtype)
            elif isinstance(dtype, str):
                self._dtype = DType(dtype)
                data = data.astype(self._dtype.to_numpy())
            else:
                self._dtype = dtype
                data = data.astype(self._dtype.to_numpy())
            
            self._data = None  # Will be allocated on the device
            self._numpy_data = data
            
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        
        # Set up gradient tracking
        self._requires_grad = requires_grad
        self._grad = None if requires_grad else None
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the tensor."""
        return self._shape
    
    @property
    def ndim(self) -> int:
        """Get the number of dimensions of the tensor."""
        return len(self._shape)
    
    @property
    def size(self) -> int:
        """Get the total number of elements in the tensor."""
        return np.prod(self._shape)
    
    @property
    def dtype(self) -> DType:
        """Get the data type of the tensor."""
        return self._dtype
    
    @property
    def device(self) -> Device:
        """Get the device where the tensor is stored."""
        return self._device
    
    @property
    def requires_grad(self) -> bool:
        """Check if the tensor requires gradients."""
        return self._requires_grad
    
    @property
    def grad(self) -> Optional["Tensor"]:
        """Get the gradient of the tensor."""
        return self._grad
    
    def numpy(self) -> np.ndarray:
        """
        Convert the tensor to a NumPy array.
        
        This operation will copy the tensor data from the device to the CPU if necessary.
        
        Returns:
            A NumPy array with the tensor data.
        """
        # TODO: Copy data from the device when Phynexus bindings are available
        # For now, just return the NumPy data
        return self._numpy_data
    
    def to(self, device: Device) -> "Tensor":
        """
        Move the tensor to the specified device.
        
        Args:
            device: The target device.
            
        Returns:
            A new tensor on the target device.
        """
        if device == self._device:
            return self
        
        # Create a new tensor on the target device
        result = Tensor(
            shape=self._shape,
            dtype=self._dtype,
            device=device,
            requires_grad=self._requires_grad,
        )
        
        # TODO: Copy data to the new device when Phynexus bindings are available
        # For now, just copy the NumPy data
        result._numpy_data = self._numpy_data.copy()
        
        return result
    
    def __repr__(self) -> str:
        """Get a string representation of the tensor."""
        return f"Tensor(shape={self._shape}, dtype={self._dtype}, device={self._device})"
