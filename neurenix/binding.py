"""
Python bindings for the Phynexus engine.

This module provides Python bindings for the Rust implementation of the Phynexus engine.
"""

import os
import sys
import platform
import logging
from typing import List, Tuple, Dict, Any, Optional, Union, Sequence

import numpy as np

logger = logging.getLogger("neurenix")

# Try to import the Rust extension module
try:
    from neurenix._phynexus import *
    _HAS_PHYNEXUS = True
    logger.info("Phynexus Rust extension loaded successfully")
except ImportError:
    # If the extension module is not available, use a fallback implementation
    _HAS_PHYNEXUS = False
    logger.warning("Phynexus Rust extension not found. Using Python fallback implementation.")
    
    # Define fallback classes and functions
    class Tensor:
        """
        Fallback implementation of the Tensor class.
        """
        
        def __init__(self, data, device=None):
            """
            Initialize a tensor.
            
            Args:
                data: Tensor data (numpy array or Python sequence)
                device: Device to store the tensor on
            """
            if isinstance(data, np.ndarray):
                self.data = data.astype(np.float32)
            else:
                self.data = np.array(data, dtype=np.float32)
            
            self.device = device or "cpu"
            self.requires_grad = False
            self.grad = None
        
        def __repr__(self):
            return f"Tensor(shape={self.data.shape}, device={self.device})"
        
        @property
        def shape(self):
            return self.data.shape
        
        def to_numpy(self):
            return self.data
        
        @classmethod
        def zeros(cls, shape, device=None):
            return cls(np.zeros(shape, dtype=np.float32), device)
        
        @classmethod
        def ones(cls, shape, device=None):
            return cls(np.ones(shape, dtype=np.float32), device)
        
        @classmethod
        def randn(cls, shape, device=None):
            return cls(np.random.randn(*shape).astype(np.float32), device)
    
    class Device:
        """
        Fallback implementation of the Device class.
        """
        
        def __init__(self, device_type, device_index=0):
            """
            Initialize a device.
            
            Args:
                device_type: Device type (cpu, cuda, rocm, webgpu)
                device_index: Device index
            """
            self.device_type = device_type
            self.device_index = device_index
        
        def __repr__(self):
            return f"Device({self.device_type}:{self.device_index})"
    
    # Define device types
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    WEBGPU = "webgpu"
    TPU = "tpu"
    
    def get_device_count(device_type):
        """
        Get the number of devices of the specified type.
        
        Args:
            device_type: Device type (cpu, cuda, rocm, webgpu)
            
        Returns:
            Number of devices
        """
        if device_type == CPU:
            return 1
        elif device_type == CUDA:
            # Try to get CUDA device count
            try:
                import torch
                return torch.cuda.device_count()
            except (ImportError, AttributeError):
                try:
                    import subprocess
                    result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE)
                    if result.returncode == 0:
                        return len([line for line in result.stdout.decode('utf-8').split('\n') if 'GPU' in line])
                    return 0
                except:
                    return 0
        elif device_type == ROCM:
            try:
                import subprocess
                result = subprocess.run(['rocm-smi', '--showallgpus'], stdout=subprocess.PIPE)
                if result.returncode == 0:
                    return len([line for line in result.stdout.decode('utf-8').split('\n') if 'GPU' in line])
                return 0
            except:
                return 0
        elif device_type == WEBGPU:
            return 0
        elif device_type == TPU:
            try:
                import os
                if os.environ.get('TPU_NAME') is not None:
                    return 1
                return 0
            except:
                return 0
        else:
            raise ValueError(f"Unknown device type: {device_type}")
    
    def is_device_available(device_type):
        """
        Check if a device type is available.
        
        Args:
            device_type: Device type (cpu, cuda, rocm, webgpu)
            
        Returns:
            True if the device type is available, False otherwise
        """
        return get_device_count(device_type) > 0
    
    def is_cuda_available():
        """
        Check if CUDA is available.
        
        Returns:
            True if CUDA is available, False otherwise
        """
        return is_device_available(CUDA)
    
    def is_rocm_available():
        """
        Check if ROCm is available.
        
        Returns:
            True if ROCm is available, False otherwise
        """
        return is_device_available(ROCM)
    
    def is_webgpu_available():
        """
        Check if WebGPU is available.
        
        Returns:
            True if WebGPU is available, False otherwise
        """
        return is_device_available(WEBGPU)
    
    def is_tpu_available():
        """
        Check if TPU is available.
        
        Returns:
            True if TPU is available, False otherwise
        """
        return get_device_count(TPU) > 0
    
    def init():
        """
        Initialize the Phynexus engine.
        """
        pass
    
    def shutdown():
        """
        Shutdown the Phynexus engine.
        """
        pass
    
    def version():
        """
        Get the version of the Phynexus engine.
        
        Returns:
            Version string
        """
        return "0.1.0 (Python fallback)"

# Define a function to get the appropriate device
def get_device(device_str=None):
    """
    Get a device object from a device string.
    
    Args:
        device_str: Device string (e.g., "cpu", "cuda:0", "rocm:1", "webgpu")
        
    Returns:
        Device object
    """
    if device_str is None:
        # Use CPU by default
        return Device(CPU, 0)
    
    # Parse device string
    if ":" in device_str:
        device_type, device_index = device_str.split(":", 1)
        device_index = int(device_index)
    else:
        device_type = device_str
        device_index = 0
    
    # Check if the device is available
    if not is_device_available(device_type):
        print(f"Warning: Device {device_type} is not available. Using CPU instead.")
        return Device(CPU, 0)
    
    return Device(device_type, device_index)

def copy_tensor_to_device(src_tensor, dst_tensor, non_blocking=False):
    """
    Copy a tensor to a device.
    
    Args:
        src_tensor: Source tensor
        dst_tensor: Destination tensor
        non_blocking: Whether to perform the copy asynchronously
    """
    if not _HAS_PHYNEXUS:
        dst_tensor._numpy_data = src_tensor._numpy_data.copy()

def hot_swap_tensor_device(tensor, device):
    """
    Hot-swap a tensor's device.
    
    Args:
        tensor: Tensor to hot-swap
        device: Target device
    """
    if not _HAS_PHYNEXUS:
        tensor.device = device

def synchronize(device):
    """
    Synchronize a device.
    
    Args:
        device: Device to synchronize
    """
    if not _HAS_PHYNEXUS:
        pass

def get_available_devices():
    """
    Get a list of all available devices.
    
    Returns:
        List of available devices
    """
    devices = []
    
    devices.append(get_device("cpu"))
    
    cuda_count = get_device_count(CUDA)
    for i in range(cuda_count):
        devices.append(get_device(f"cuda:{i}"))
    
    rocm_count = get_device_count(ROCM)
    for i in range(rocm_count):
        devices.append(get_device(f"rocm:{i}"))
    
    tpu_count = get_device_count(TPU)
    for i in range(tpu_count):
        devices.append(get_device(f"tpu:{i}"))
    
    return devices

def get_optimal_device():
    """
    Get the optimal device for computation.
    
    Returns:
        Optimal device
    """
    devices = get_available_devices()
    
    for device in devices:
        if device.device_type == CUDA:
            return device
    
    for device in devices:
        if device.device_type == ROCM:
            return device
    
    for device in devices:
        if device.device_type == TPU:
            return device
    
    return get_device("cpu")

def matmul(a, b):
    """
    Perform matrix multiplication.
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        Result tensor
    """
    if not _HAS_PHYNEXUS:
        import numpy as np
        result = np.matmul(a._numpy_data, b._numpy_data)
        return Tensor(result, device=a.device)

def add(a, b):
    """
    Add two tensors.
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        Result tensor
    """
    if not _HAS_PHYNEXUS:
        result = a._numpy_data + b._numpy_data
        return Tensor(result, device=a.device)

def subtract(a, b):
    """
    Subtract two tensors.
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        Result tensor
    """
    if not _HAS_PHYNEXUS:
        result = a._numpy_data - b._numpy_data
        return Tensor(result, device=a.device)

def multiply(a, b):
    """
    Multiply two tensors element-wise.
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        Result tensor
    """
    if not _HAS_PHYNEXUS:
        result = a._numpy_data * b._numpy_data
        return Tensor(result, device=a.device)

def divide(a, b):
    """
    Divide two tensors element-wise.
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        Result tensor
    """
    if not _HAS_PHYNEXUS:
        result = a._numpy_data / b._numpy_data
        return Tensor(result, device=a.device)

def reshape(a, shape):
    """
    Reshape a tensor.
    
    Args:
        a: Tensor to reshape
        shape: New shape
        
    Returns:
        Reshaped tensor
    """
    if not _HAS_PHYNEXUS:
        result = a._numpy_data.reshape(shape)
        return Tensor(result, device=a.device)

def transpose(a, dim0, dim1):
    """
    Transpose a tensor.
    
    Args:
        a: Tensor to transpose
        dim0: First dimension to swap
        dim1: Second dimension to swap
        
    Returns:
        Transposed tensor
    """
    if not _HAS_PHYNEXUS:
        dims = list(range(len(a._numpy_data.shape)))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        result = np.transpose(a._numpy_data, dims)
        return Tensor(result, device=a.device)

def relu(x, inplace=False):
    """
    Apply ReLU activation function.
    
    Args:
        x: Input tensor
        inplace: Whether to perform the operation in-place
        
    Returns:
        Result tensor
    """
    if not _HAS_PHYNEXUS:
        if inplace:
            x._numpy_data = np.maximum(x._numpy_data, 0)
            return x
        else:
            result = np.maximum(x._numpy_data, 0)
            return Tensor(result, device=x.device)

def sigmoid(x):
    """
    Apply sigmoid activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Result tensor
    """
    if not _HAS_PHYNEXUS:
        result = 1 / (1 + np.exp(-x._numpy_data))
        return Tensor(result, device=x.device)

def tanh(x):
    """
    Apply tanh activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Result tensor
    """
    if not _HAS_PHYNEXUS:
        result = np.tanh(x._numpy_data)
        return Tensor(result, device=x.device)

def softmax(x, dim=1):
    """
    Apply softmax activation function.
    
    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax
        
    Returns:
        Result tensor
    """
    if not _HAS_PHYNEXUS:
        exp_x = np.exp(x._numpy_data - np.max(x._numpy_data, axis=dim, keepdims=True))
        result = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
        return Tensor(result, device=x.device)

def log_softmax(x, dim=1):
    """
    Apply log softmax activation function.
    
    Args:
        x: Input tensor
        dim: Dimension along which to apply log softmax
        
    Returns:
        Result tensor
    """
    if not _HAS_PHYNEXUS:
        max_val = np.max(x._numpy_data, axis=dim, keepdims=True)
        exp_x = np.exp(x._numpy_data - max_val)
        sum_exp_x = np.sum(exp_x, axis=dim, keepdims=True)
        result = x._numpy_data - max_val - np.log(sum_exp_x)
        return Tensor(result, device=x.device)

def leaky_relu(x, negative_slope=0.01, inplace=False):
    """
    Apply leaky ReLU activation function.
    
    Args:
        x: Input tensor
        negative_slope: Controls the angle of the negative slope
        inplace: Whether to perform the operation in-place
        
    Returns:
        Result tensor
    """
    if not _HAS_PHYNEXUS:
        if inplace:
            x._numpy_data = np.where(x._numpy_data > 0, x._numpy_data, x._numpy_data * negative_slope)
            return x
        else:
            result = np.where(x._numpy_data > 0, x._numpy_data, x._numpy_data * negative_slope)
            return Tensor(result, device=x.device)

def elu(x, alpha=1.0, inplace=False):
    """
    Apply ELU activation function.
    
    Args:
        x: Input tensor
        alpha: Controls the value to which the function saturates for negative inputs
        inplace: Whether to perform the operation in-place
        
    Returns:
        Result tensor
    """
    if not _HAS_PHYNEXUS:
        if inplace:
            x._numpy_data = np.where(
                x._numpy_data > 0,
                x._numpy_data,
                alpha * (np.exp(x._numpy_data) - 1)
            )
            return x
        else:
            result = np.where(
                x._numpy_data > 0,
                x._numpy_data,
                alpha * (np.exp(x._numpy_data) - 1)
            )
            return Tensor(result, device=x.device)

def selu(x, inplace=False):
    """
    Apply SELU activation function.
    
    Args:
        x: Input tensor
        inplace: Whether to perform the operation in-place
        
    Returns:
        Result tensor
    """
    if not _HAS_PHYNEXUS:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        
        if inplace:
            x._numpy_data = scale * np.where(
                x._numpy_data > 0,
                x._numpy_data,
                alpha * (np.exp(x._numpy_data) - 1)
            )
            return x
        else:
            result = scale * np.where(
                x._numpy_data > 0,
                x._numpy_data,
                alpha * (np.exp(x._numpy_data) - 1)
            )
            return Tensor(result, device=x.device)

def gelu(x, approximate=False):
    """
    Apply GELU activation function.
    
    Args:
        x: Input tensor
        approximate: Whether to use an approximation of the GELU function
        
    Returns:
        Result tensor
    """
    if not _HAS_PHYNEXUS:
        if approximate:
            sqrt_2_over_pi = np.sqrt(2 / np.pi)
            result = 0.5 * x._numpy_data * (1 + np.tanh(sqrt_2_over_pi * (x._numpy_data + 0.044715 * np.power(x._numpy_data, 3))))
        else:
            result = 0.5 * x._numpy_data * (1 + np.math.erf(x._numpy_data / np.sqrt(2)))
        
        return Tensor(result, device=x.device)

# Initialize the Phynexus engine
if _HAS_PHYNEXUS:
    py_init()
else:
    init()

# Register shutdown function
import atexit
if _HAS_PHYNEXUS:
    atexit.register(shutdown)
else:
    pass
