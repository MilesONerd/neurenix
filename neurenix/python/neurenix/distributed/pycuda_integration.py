"""
PyCUDA integration for distributed computing in Neurenix.

This module provides integration with PyCUDA for GPU computing,
enabling high-performance tensor operations on NVIDIA GPUs.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    from pycuda.compiler import SourceModule
    PYCUDA_AVAILABLE = True
except ImportError:
    PYCUDA_AVAILABLE = False


class CudaContext:
    """
    CUDA context for GPU computing.
    
    This class manages a CUDA context for GPU computing,
    enabling high-performance tensor operations on NVIDIA GPUs.
    """
    
    def __init__(
        self,
        device_id: int = 0,
        enable_profiling: bool = False,
    ):
        """
        Initialize CUDA context.
        
        Args:
            device_id: GPU device ID
            enable_profiling: Whether to enable profiling
        """
        if not PYCUDA_AVAILABLE:
            raise ImportError(
                "PyCUDA is not available. Please install it with 'pip install pycuda'."
            )
        
        self.device_id = device_id
        self.enable_profiling = enable_profiling
        self.context = None
        self.device = None
        
        # Kernel cache
        self._kernel_cache = {}
    
    def __enter__(self):
        """Initialize the CUDA context."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up the CUDA context."""
        self.stop()
    
    def start(self):
        """Start the CUDA context."""
        if self.context is not None:
            return
        
        # Initialize CUDA
        cuda.init()
        
        # Get device
        self.device = cuda.Device(self.device_id)
        
        # Create context
        flags = 0
        if self.enable_profiling:
            flags |= cuda.ctx_flags.SCHED_AUTO | cuda.ctx_flags.MAP_HOST
        
        self.context = self.device.make_context(flags=flags)
        
        # Print device info
        print(f"Using CUDA device: {self.device.name()} (ID: {self.device_id})")
        print(f"  Compute capability: {self.device.compute_capability()}")
        print(f"  Total memory: {self.device.total_memory() // (1024 * 1024)} MB")
    
    def stop(self):
        """Stop the CUDA context."""
        if self.context is None:
            return
        
        # Pop context
        self.context.pop()
        
        # Clean up
        self.context = None
        self.device = None
    
    @property
    def is_running(self) -> bool:
        """Check if the CUDA context is running."""
        return self.context is not None
    
    def get_kernel(self, name: str, source: str, function_name: str) -> Any:
        """
        Get a CUDA kernel.
        
        Args:
            name: Kernel name
            source: Kernel source code
            function_name: Function name in the kernel
            
        Returns:
            CUDA kernel
        """
        if not self.is_running:
            raise RuntimeError("CUDA context is not running")
        
        # Check if kernel is in cache
        if name in self._kernel_cache:
            return self._kernel_cache[name]
        
        # Compile kernel
        module = SourceModule(source)
        kernel = module.get_function(function_name)
        
        # Cache kernel
        self._kernel_cache[name] = kernel
        
        return kernel
    
    def allocate(self, shape: Tuple[int, ...], dtype: np.dtype) -> "gpuarray.GPUArray":
        """
        Allocate memory on the GPU.
        
        Args:
            shape: Array shape
            dtype: Array data type
            
        Returns:
            GPU array
        """
        if not self.is_running:
            raise RuntimeError("CUDA context is not running")
        
        return gpuarray.zeros(shape, dtype)
    
    def to_gpu(self, array: np.ndarray) -> "gpuarray.GPUArray":
        """
        Copy array to GPU.
        
        Args:
            array: NumPy array
            
        Returns:
            GPU array
        """
        if not self.is_running:
            raise RuntimeError("CUDA context is not running")
        
        return gpuarray.to_gpu(array)
    
    def from_gpu(self, array: Any) -> np.ndarray:
        """
        Copy array from GPU.
        
        Args:
            array: GPU array
            
        Returns:
            NumPy array
        """
        if not self.is_running:
            raise RuntimeError("CUDA context is not running")
        
        return array.get()
    
    def synchronize(self):
        """Synchronize the CUDA context."""
        if not self.is_running:
            raise RuntimeError("CUDA context is not running")
        
        self.context.synchronize()


def tensor_to_gpu(tensor):
    """
    Convert a Neurenix tensor to a GPU array.
    
    Args:
        tensor: Neurenix tensor
        
    Returns:
        GPU array
    """
    if not PYCUDA_AVAILABLE:
        raise ImportError(
            "PyCUDA is not available. Please install it with 'pip install pycuda'."
        )
    
    # Convert to numpy array
    array = tensor.numpy()
    
    # Convert to GPU array
    return gpuarray.to_gpu(array)


def gpu_to_tensor(gpu_array):
    """
    Convert a GPU array to a Neurenix tensor.
    
    Args:
        gpu_array: GPU array
        
    Returns:
        Neurenix tensor
    """
    if not PYCUDA_AVAILABLE:
        raise ImportError(
            "PyCUDA is not available. Please install it with 'pip install pycuda'."
        )
    
    from neurenix.tensor import Tensor
    
    # Convert to numpy array
    array = gpu_array.get()
    
    # Convert to Neurenix tensor
    return Tensor(array)


# Example CUDA kernels
ELEMENT_WISE_ADD_KERNEL = """
__global__ void element_wise_add(float *a, float *b, float *c, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}
"""

ELEMENT_WISE_MUL_KERNEL = """
__global__ void element_wise_mul(float *a, float *b, float *c, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        c[idx] = a[idx] * b[idx];
    }
}
"""

MATRIX_MUL_KERNEL = """
__global__ void matrix_mul(float *a, float *b, float *c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < k)
    {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}
"""
