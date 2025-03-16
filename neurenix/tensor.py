"""
Tensor implementation for the Neurenix framework.
"""

import numpy as np
from typing import List, Tuple, Union, Optional, Sequence, Any
from enum import Enum
from contextlib import contextmanager

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
    
    def to_numpy(self):
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
            elif device_str.startswith("tpu"):
                # Extract device index if specified (e.g., "tpu:0")
                parts = device_str.split(":")
                index = int(parts[1]) if len(parts) > 1 else 0
                self._device = Device(DeviceType.TPU, index)
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
        return int(np.prod(self._shape))
    
    @property
    def dtype(self) -> DType:
        """Get the data type of the tensor."""
        if isinstance(self._dtype, str):
            return DType(self._dtype)
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
        
    def clone(self) -> "Tensor":
        """
        Create a clone of this tensor.
        
        Returns:
            A new tensor with the same data and attributes.
        """
        # Create a new tensor with the same data and attributes
        result = Tensor(
            self._numpy_data.copy(),
            dtype=self._dtype,
            device=self._device,
            requires_grad=self._requires_grad,
        )
        
        # Copy gradient if it exists
        if self._grad is not None:
            result._grad = self._grad.clone()
        
        return result
    
    def __repr__(self) -> str:
        """Get a string representation of the tensor."""
        return f"Tensor(shape={self._shape}, dtype={self._dtype}, device={self._device})"
        
    def __add__(self, other: Union["Tensor", float, int]) -> "Tensor":
        """
        Add another tensor or scalar to this tensor.
        
        Args:
            other: The tensor or scalar to add.
            
        Returns:
            A new tensor with the result of the addition.
        """
        # Handle scalar addition
        if isinstance(other, (int, float)):
            return Tensor(self._numpy_data + other, device=self._device)
        
        # Handle tensor addition
        if isinstance(other, Tensor):
            # TODO: Use Phynexus bindings when available
            # For now, use NumPy as a fallback
            return Tensor(self._numpy_data + other._numpy_data, device=self._device)
        
        raise TypeError(f"Unsupported operand type for +: {type(other)}")
    
    def __sub__(self, other: Union["Tensor", float, int]) -> "Tensor":
        """
        Subtract another tensor or scalar from this tensor.
        
        Args:
            other: The tensor or scalar to subtract.
            
        Returns:
            A new tensor with the result of the subtraction.
        """
        # Handle scalar subtraction
        if isinstance(other, (int, float)):
            return Tensor(self._numpy_data - other, device=self._device)
        
        # Handle tensor subtraction
        if isinstance(other, Tensor):
            # TODO: Use Phynexus bindings when available
            # For now, use NumPy as a fallback
            return Tensor(self._numpy_data - other._numpy_data, device=self._device)
        
        raise TypeError(f"Unsupported operand type for -: {type(other)}")
    
    def __mul__(self, other: Union["Tensor", float, int]) -> "Tensor":
        """
        Multiply this tensor by another tensor or scalar.
        
        Args:
            other: The tensor or scalar to multiply by.
            
        Returns:
            A new tensor with the result of the multiplication.
        """
        # Handle scalar multiplication
        if isinstance(other, (int, float)):
            return Tensor(self._numpy_data * other, device=self._device)
        
        # Handle tensor multiplication
        if isinstance(other, Tensor):
            # TODO: Use Phynexus bindings when available
            # For now, use NumPy as a fallback
            return Tensor(self._numpy_data * other._numpy_data, device=self._device)
        
        raise TypeError(f"Unsupported operand type for *: {type(other)}")
    
    def __truediv__(self, other: Union["Tensor", float, int]) -> "Tensor":
        """
        Divide this tensor by another tensor or scalar.
        
        Args:
            other: The tensor or scalar to divide by.
            
        Returns:
            A new tensor with the result of the division.
        """
        # Handle scalar division
        if isinstance(other, (int, float)):
            return Tensor(self._numpy_data / other, device=self._device)
        
        # Handle tensor division
        if isinstance(other, Tensor):
            # TODO: Use Phynexus bindings when available
            # For now, use NumPy as a fallback
            return Tensor(self._numpy_data / other._numpy_data, device=self._device)
        
        raise TypeError(f"Unsupported operand type for /: {type(other)}")
        
    def __getitem__(self, index) -> "Tensor":
        """
        Get a slice of the tensor.
        
        Args:
            index: The index or slice to get.
            
        Returns:
            A new tensor with the selected elements.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        result = self._numpy_data[index]
        
        # If the result is a scalar, wrap it in a 0-dimensional tensor
        if np.isscalar(result):
            result = np.array(result)
        
        return Tensor(result, device=self._device)
        
    def reshape(self, *shape) -> "Tensor":
        """
        Reshape the tensor to the given shape.
        
        Args:
            shape: The new shape of the tensor.
            
        Returns:
            A new tensor with the given shape.
        """
        # Handle the case where shape is passed as a tuple or list
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        result = Tensor(
            self._numpy_data.reshape(shape),
            device=self._device
        )
        return result
        
    def transpose(self, dim0: int = 0, dim1: int = 1) -> "Tensor":
        """
        Transpose the tensor along the given dimensions.
        
        Args:
            dim0: First dimension to transpose.
            dim1: Second dimension to transpose.
            
        Returns:
            A new tensor with the transposed dimensions.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        
        # Create a list of dimensions
        dims = list(range(self.ndim))
        
        # Swap the specified dimensions
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        
        result = Tensor(
            np.transpose(self._numpy_data, dims),
            device=self._device
        )
        return result
        
    def gather(self, dim: int, index: "Tensor") -> "Tensor":
        """
        Gather values along a dimension using indices.
        
        Args:
            dim: Dimension along which to gather.
            index: Tensor containing the indices to gather.
            
        Returns:
            A new tensor containing the gathered values.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        # Convert indices to integer type for take_along_axis
        index_array = index._numpy_data.astype(np.int64)
        gathered = np.take_along_axis(self._numpy_data, index_array, axis=dim)
        return Tensor(gathered, device=self.device)
    
    # Activation functions
    
    def relu(self, inplace: bool = False) -> "Tensor":
        """
        Apply the rectified linear unit function element-wise.
        
        Args:
            inplace: If True, do the operation in-place.
            
        Returns:
            A new tensor with the ReLU activation applied.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        if inplace:
            self._numpy_data = np.maximum(self._numpy_data, 0)
            return self
        else:
            result = Tensor(np.maximum(self._numpy_data, 0), device=self._device)
            return result
    
    def sigmoid(self) -> "Tensor":
        """
        Apply the sigmoid function element-wise.
        
        Returns:
            A new tensor with the sigmoid activation applied.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        result = Tensor(1 / (1 + np.exp(-self._numpy_data)), device=self._device)
        return result
    
    def tanh(self) -> "Tensor":
        """
        Apply the hyperbolic tangent function element-wise.
        
        Returns:
            A new tensor with the tanh activation applied.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        result = Tensor(np.tanh(self._numpy_data), device=self._device)
        return result
    
    def softmax(self, dim: int = -1) -> "Tensor":
        """
        Apply the softmax function along the specified dimension.
        
        Args:
            dim: Dimension along which to apply softmax.
            
        Returns:
            A new tensor with the softmax activation applied.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        # Compute softmax values along the specified dimension
        exp_x = np.exp(self._numpy_data - np.max(self._numpy_data, axis=dim, keepdims=True))
        softmax_values = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
        result = Tensor(softmax_values, device=self._device)
        return result
    
    def log_softmax(self, dim: int = -1) -> "Tensor":
        """
        Apply the log softmax function along the specified dimension.
        
        Args:
            dim: Dimension along which to apply log softmax.
            
        Returns:
            A new tensor with the log softmax activation applied.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        # Compute log softmax values along the specified dimension
        max_val = np.max(self._numpy_data, axis=dim, keepdims=True)
        exp_x = np.exp(self._numpy_data - max_val)
        sum_exp_x = np.sum(exp_x, axis=dim, keepdims=True)
        log_softmax_values = self._numpy_data - max_val - np.log(sum_exp_x)
        result = Tensor(log_softmax_values, device=self._device)
        return result
    
    def leaky_relu(self, negative_slope: float = 0.01, inplace: bool = False) -> "Tensor":
        """
        Apply the leaky rectified linear unit function element-wise.
        
        Args:
            negative_slope: Controls the angle of the negative slope.
            inplace: If True, do the operation in-place.
            
        Returns:
            A new tensor with the leaky ReLU activation applied.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        if inplace:
            self._numpy_data = np.where(self._numpy_data > 0, self._numpy_data, self._numpy_data * negative_slope)
            return self
        else:
            result = Tensor(
                np.where(self._numpy_data > 0, self._numpy_data, self._numpy_data * negative_slope),
                device=self._device
            )
            return result
    
    def elu(self, alpha: float = 1.0, inplace: bool = False) -> "Tensor":
        """
        Apply the exponential linear unit function element-wise.
        
        Args:
            alpha: Controls the value to which an ELU saturates for negative inputs.
            inplace: If True, do the operation in-place.
            
        Returns:
            A new tensor with the ELU activation applied.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        if inplace:
            self._numpy_data = np.where(
                self._numpy_data > 0,
                self._numpy_data,
                alpha * (np.exp(self._numpy_data) - 1)
            )
            return self
        else:
            result = Tensor(
                np.where(
                    self._numpy_data > 0,
                    self._numpy_data,
                    alpha * (np.exp(self._numpy_data) - 1)
                ),
                device=self._device
            )
            return result
    
    def selu(self, inplace: bool = False) -> "Tensor":
        """
        Apply the scaled exponential linear unit function element-wise.
        
        Args:
            inplace: If True, do the operation in-place.
            
        Returns:
            A new tensor with the SELU activation applied.
        """
        # SELU parameters
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        if inplace:
            self._numpy_data = scale * np.where(
                self._numpy_data > 0,
                self._numpy_data,
                alpha * (np.exp(self._numpy_data) - 1)
            )
            return self
        else:
            result = Tensor(
                scale * np.where(
                    self._numpy_data > 0,
                    self._numpy_data,
                    alpha * (np.exp(self._numpy_data) - 1)
                ),
                device=self._device
            )
            return result
    
    def gelu(self, approximate: bool = False) -> "Tensor":
        """
        Apply the Gaussian error linear unit function element-wise.
        
        Args:
            approximate: If True, use an approximation of the GELU function.
            
        Returns:
            A new tensor with the GELU activation applied.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        if approximate:
            # Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            sqrt_2_over_pi = np.sqrt(2 / np.pi)
            result = Tensor(
                0.5 * self._numpy_data * (
                    1 + np.tanh(
                        sqrt_2_over_pi * (self._numpy_data + 0.044715 * np.power(self._numpy_data, 3))
                    )
                ),
                device=self._device
            )
        else:
            # Exact formula: 0.5 * x * (1 + erf(x / sqrt(2)))
            from scipy import special
            result = Tensor(
                0.5 * self._numpy_data * (1 + special.erf(self._numpy_data / np.sqrt(2))),
                device=self._device
            )
        return result
    
    # Tensor operations
    
    def matmul(self, other: "Tensor") -> "Tensor":
        """
        Matrix multiplication with another tensor.
        
        Args:
            other: The tensor to multiply with.
            
        Returns:
            A new tensor with the result of the matrix multiplication.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        result = Tensor(
            np.matmul(self._numpy_data, other._numpy_data),
            device=self._device
        )
        return result
    
    def mean(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """
        Compute the mean along the specified dimension.
        
        Args:
            dim: Dimension along which to compute the mean. If None, compute the mean of all elements.
            keepdim: Whether to keep the reduced dimension.
            
        Returns:
            A new tensor with the mean values.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        result = Tensor(
            np.mean(self._numpy_data, axis=dim, keepdims=keepdim),
            device=self._device
        )
        return result
    
    def sum(self, dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """
        Compute the sum along the specified dimension.
        
        Args:
            dim: Dimension along which to compute the sum. If None, compute the sum of all elements.
            keepdim: Whether to keep the reduced dimension.
            
        Returns:
            A new tensor with the sum values.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        result = Tensor(
            np.sum(self._numpy_data, axis=dim, keepdims=keepdim),
            device=self._device
        )
        return result
    
    def abs(self) -> "Tensor":
        """
        Compute the absolute value element-wise.
        
        Returns:
            A new tensor with the absolute values.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        result = Tensor(
            np.abs(self._numpy_data),
            device=self._device
        )
        return result
    
    def clamp(self, min: Optional[float] = None, max: Optional[float] = None) -> "Tensor":
        """
        Clamp all elements in the tensor to be between min and max.
        
        Args:
            min: Lower bound of the range to be clamped to.
            max: Upper bound of the range to be clamped to.
            
        Returns:
            A new tensor with clamped values.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        result = Tensor(
            np.clip(self._numpy_data, min, max),
            device=self._device
        )
        return result
    
    def log1p(self) -> "Tensor":
        """
        Compute log(1 + x) element-wise.
        
        Returns:
            A new tensor with the result of log(1 + x).
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        result = Tensor(
            np.log1p(self._numpy_data),
            device=self._device
        )
        return result
    
    # Static methods for tensor creation
    
    @staticmethod
    def stack(tensors: List["Tensor"], dim: int = 0) -> "Tensor":
        """
        Stack tensors along a new dimension.
        
        Args:
            tensors: List of tensors to stack.
            dim: Dimension along which to stack.
            
        Returns:
            A new tensor containing the stacked tensors.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        numpy_tensors = [t._numpy_data for t in tensors]
        stacked = np.stack(numpy_tensors, axis=dim)
        return Tensor(stacked, device=tensors[0].device)
    
    @staticmethod
    def cat(tensors: List["Tensor"], dim: int = 0) -> "Tensor":
        """
        Concatenate tensors along an existing dimension.
        
        Args:
            tensors: List of tensors to concatenate.
            dim: Dimension along which to concatenate.
            
        Returns:
            A new tensor containing the concatenated tensors.
        """
        # TODO: Use Phynexus bindings when available
        # For now, use NumPy as a fallback
        numpy_tensors = [t._numpy_data for t in tensors]
        concatenated = np.concatenate(numpy_tensors, axis=dim)
        return Tensor(concatenated, device=tensors[0].device)
    
    @staticmethod
    def zeros(shape: Sequence[int], dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> "Tensor":
        """
        Create a tensor filled with zeros.
        
        Args:
            shape: Shape of the tensor.
            dtype: Data type of the tensor. If None, defaults to FLOAT32.
            device: Device to store the tensor on. If None, uses the default device.
            requires_grad: Whether the tensor requires gradients. Default: False.
            
        Returns:
            A new tensor filled with zeros.
        """
        if dtype is None:
            dtype = DType.FLOAT32
        
        return Tensor(
            np.zeros(shape, dtype=dtype.to_numpy()),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad
        )
    
    @staticmethod
    def ones(shape: Sequence[int], dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> "Tensor":
        """
        Create a tensor filled with ones.
        
        Args:
            shape: Shape of the tensor.
            dtype: Data type of the tensor. If None, defaults to FLOAT32.
            device: Device to store the tensor on. If None, uses the default device.
            requires_grad: Whether the tensor requires gradients. Default: False.
            
        Returns:
            A new tensor filled with ones.
        """
        if dtype is None:
            dtype = DType.FLOAT32
        
        return Tensor(
            np.ones(shape, dtype=dtype.to_numpy()),
            dtype=dtype,
            device=device,
            requires_grad=requires_grad
        )
    
    @staticmethod
    def randn(shape: Sequence[int], dtype: Optional[DType] = None, device: Optional[Device] = None, requires_grad: bool = False) -> "Tensor":
        """
        Create a tensor filled with random numbers from a normal distribution.
        
        Args:
            shape: Shape of the tensor.
            dtype: Data type of the tensor. If None, defaults to FLOAT32.
            device: Device to store the tensor on. If None, uses the default device.
            requires_grad: Whether the tensor requires gradients. Default: False.
            
        Returns:
            A new tensor filled with random numbers.
        """
        if dtype is None:
            dtype = DType.FLOAT32
        
        # Generate random values and convert to the right dtype
        random_data = np.random.randn(*shape)
        if dtype is not None:
            random_data = np.asarray(random_data, dtype=dtype.to_numpy())
        
        return Tensor(
            random_data,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad
        )
    
    def backward(self):
        """
        Compute gradients through the computation graph.
        
        This method computes gradients for all tensors in the computation
        graph that require gradients. The gradients are stored in the
        grad attribute of each tensor.
        """
        if not self._requires_grad:
            return
        
        # Initialize gradient for scalar tensors
        if self._grad is None and self._numpy_data.size == 1:
            self._grad = Tensor(np.ones_like(self._numpy_data), device=self.device)
        
        # TODO: Implement proper backward pass when Phynexus bindings are available
        # For now, just a placeholder
    
    @staticmethod
    def exp(x: "Tensor") -> "Tensor":
        """
        Compute the exponential of the input tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with exponential of each element
        """
        return Tensor(np.exp(x._numpy_data), device=x.device)
    
    @staticmethod
    def randn_like(x: "Tensor") -> "Tensor":
        """
        Create a tensor with the same shape as the input tensor,
        filled with random numbers from a normal distribution.
        
        Args:
            x: Input tensor
            
        Returns:
            A new tensor with the same shape as x
        """
        return Tensor(np.random.randn(*x.shape), device=x.device)
    
    @staticmethod
    def sum(x: "Tensor", dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """
        Sum of tensor elements along a dimension.
        
        Args:
            x: Input tensor
            dim: Dimension to reduce. If None, all dimensions are reduced.
            keepdim: Whether to keep the reduced dimension
            
        Returns:
            Sum of elements
        """
        return Tensor(
            np.sum(x._numpy_data, axis=dim, keepdims=keepdim),
            device=x.device
        )
    def backward(self):
        """
        Compute gradients through the computation graph.
        
        This method computes gradients for all tensors in the computation
        graph that require gradients. The gradients are stored in the
        grad attribute of each tensor.
        """
        if not self._requires_grad:
            return
        
        # Initialize gradient for scalar tensors
        if self._grad is None and self._numpy_data.size == 1:
            self._grad = Tensor(np.ones_like(self._numpy_data), device=self.device)
        
        # TODO: Implement proper backward pass when Phynexus bindings are available
        # For now, just a placeholder
    
    @staticmethod
    def exp(x: "Tensor") -> "Tensor":
        """
        Compute the exponential of the input tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with exponential of each element
        """
        return Tensor(np.exp(x._numpy_data), device=x.device)
    
    @staticmethod
    def randn_like(x: "Tensor") -> "Tensor":
        """
        Create a tensor with the same shape as the input tensor,
        filled with random numbers from a normal distribution.
        
        Args:
            x: Input tensor
            
        Returns:
            A new tensor with the same shape as x
        """
        return Tensor(np.random.randn(*x.shape), device=x.device)
    
    @staticmethod
    def sum(x: "Tensor", dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """
        Sum of tensor elements along a dimension.
        
        Args:
            x: Input tensor
            dim: Dimension to reduce. If None, all dimensions are reduced.
            keepdim: Whether to keep the reduced dimension
            
        Returns:
            Sum of elements
        """
        return Tensor(
            np.sum(x._numpy_data, axis=dim, keepdims=keepdim),
            device=x.device
        )
    
    @staticmethod
    def exp(x: "Tensor") -> "Tensor":
        """
        Compute the exponential of the input tensor.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with exponential of each element
        """
        return Tensor(np.exp(x._numpy_data), device=x.device)
    
    @staticmethod
    def randn_like(x: "Tensor") -> "Tensor":
        """
        Create a tensor with the same shape as the input tensor,
        filled with random numbers from a normal distribution.
        
        Args:
            x: Input tensor
            
        Returns:
            A new tensor with the same shape as x
        """
        return Tensor(np.random.randn(*x.shape), device=x.device)
    
    @staticmethod
    def sum(x: "Tensor", dim: Optional[int] = None, keepdim: bool = False) -> "Tensor":
        """
        Sum of tensor elements along a dimension.
        
        Args:
            x: Input tensor
            dim: Dimension to reduce. If None, all dimensions are reduced.
            keepdim: Whether to keep the reduced dimension
            
        Returns:
            Sum of elements
        """
        return Tensor(
            np.sum(x._numpy_data, axis=dim, keepdims=keepdim),
            device=x.device
        )
    
    @staticmethod
    @contextmanager
    def no_grad():
        """
        Context manager to disable gradient computation.
        
        This context manager temporarily disables gradient computation,
        which is useful for inference or when you want to ensure that
        no gradients are computed for certain operations.
        
        Example:
            >>> x = Tensor([1, 2, 3], requires_grad=True)
            >>> with Tensor.no_grad():
            ...     y = x * 2  # y will have requires_grad=False
        """
        # TODO: Implement proper no_grad when Phynexus bindings are available
        # For now, just a placeholder
        try:
            yield
        finally:
            pass
