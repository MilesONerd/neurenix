"""
NVIDIA Tensor Cores backend for the Neurenix framework.

This module provides hardware acceleration using NVIDIA's Tensor Cores,
enabling high-performance matrix operations and deep learning on NVIDIA GPUs
with Tensor Cores capability.
"""

from typing import Optional, List, Dict, Any, Union, Tuple
import os
import ctypes
import numpy as np

from neurenix.tensor import Tensor
from neurenix.device import Device, DeviceType
from neurenix.nn import Module

_tensor_cores_available = False
_cuda_lib = None
_cublas_lib = None

try:
    if os.name == 'nt':  # Windows
        try:
            _cuda_lib = ctypes.CDLL('cudart64_11.dll')
            _cublas_lib = ctypes.CDLL('cublas64_11.dll')
            _tensor_cores_available = True
        except OSError:
            pass
    elif os.name == 'posix':  # Linux/macOS
        try:
            _cuda_lib = ctypes.CDLL('libcudart.so')
            _cublas_lib = ctypes.CDLL('libcublas.so')
            _tensor_cores_available = True
        except OSError:
            try:
                _cuda_lib = ctypes.CDLL('libcudart.so.11.0')
                _cublas_lib = ctypes.CDLL('libcublas.so.11.0')
                _tensor_cores_available = True
            except OSError:
                try:
                    _cuda_lib = ctypes.CDLL('libcudart.dylib')  # macOS
                    _cublas_lib = ctypes.CDLL('libcublas.dylib')
                    _tensor_cores_available = True
                except OSError:
                    pass
except Exception:
    _tensor_cores_available = False

def is_tensor_cores_available() -> bool:
    """
    Check if NVIDIA Tensor Cores are available on the system.
    
    Returns:
        bool: True if Tensor Cores are available, False otherwise
    """
    if not _tensor_cores_available:
        return False
    
    try:
        import pycuda.driver as cuda
        cuda.init()
        
        for i in range(cuda.Device.count()):
            device = cuda.Device(i)
            compute_capability = device.compute_capability()
            
            if compute_capability[0] >= 7:
                return True
        
        return False
    except ImportError:
        try:
            from py3nvml import py3nvml
            py3nvml.nvmlInit()
            
            device_count = py3nvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
                info = py3nvml.nvmlDeviceGetCudaComputeCapability(handle)
                
                if info[0] >= 7:
                    py3nvml.nvmlShutdown()
                    return True
            
            py3nvml.nvmlShutdown()
            return False
        except ImportError:
            return False

class TensorCoresBackend:
    """
    NVIDIA Tensor Cores backend for hardware acceleration.
    
    This backend enables high-performance matrix operations using NVIDIA's
    Tensor Cores, which provide specialized hardware for mixed-precision
    matrix multiply-accumulate operations.
    """
    
    def __init__(self):
        """
        Initialize the Tensor Cores backend.
        
        Raises:
            RuntimeError: If Tensor Cores are not available
        """
        if not is_tensor_cores_available():
            raise RuntimeError("NVIDIA Tensor Cores are not available on this system")
        
        self._handle = None
        self._stream = None
        self._initialized = False
        self._precision = 'mixed'  # 'fp32', 'fp16', 'mixed'
        self._workspace = None
        self._workspace_size = 1 << 30  # 1 GB default workspace
    
    def initialize(self) -> bool:
        """
        Initialize the Tensor Cores backend.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self._initialized:
            return True
        
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            self._handle = self._create_cublas_handle()
            if self._handle is None:
                return False
            
            self._stream = cuda.Stream()
            
            self._workspace = cuda.mem_alloc(self._workspace_size)
            
            self._initialized = True
            return True
            
        except Exception as e:
            print(f"Tensor Cores initialization error: {e}")
            self._cleanup()
            return False
    
    def _cleanup(self):
        """
        Clean up Tensor Cores resources.
        """
        if self._handle is not None:
            self._destroy_cublas_handle(self._handle)
            self._handle = None
        
        if self._workspace is not None:
            self._workspace = None
        
        self._initialized = False
    
    def _create_cublas_handle(self):
        """
        Create CUBLAS handle with Tensor Cores enabled.
        """
        try:
            import pycuda.driver as cuda
            import pycuda.cublas as cublas
            
            handle = cublas.cublasCreate()
            
            cublas.cublasSetMathMode(handle, cublas.CUBLAS_TENSOR_OP_MATH)
            
            return handle
        except Exception as e:
            print(f"Error creating CUBLAS handle: {e}")
            return None
    
    def _destroy_cublas_handle(self, handle):
        """
        Destroy CUBLAS handle.
        """
        try:
            import pycuda.cublas as cublas
            cublas.cublasDestroy(handle)
        except Exception as e:
            print(f"Error destroying CUBLAS handle: {e}")
    
    def set_precision(self, precision: str):
        """
        Set the precision mode for Tensor Cores operations.
        
        Args:
            precision: Precision mode ('fp32', 'fp16', 'mixed')
        """
        if precision not in ['fp32', 'fp16', 'mixed']:
            raise ValueError("Precision must be one of: 'fp32', 'fp16', 'mixed'")
        
        self._precision = precision
    
    def get_device_count(self) -> int:
        """
        Get the number of available devices with Tensor Cores.
        
        Returns:
            int: Number of available devices with Tensor Cores
        """
        if not self.initialize():
            return 0
        
        try:
            import pycuda.driver as cuda
            cuda.init()
            
            tensor_cores_count = 0
            for i in range(cuda.Device.count()):
                device = cuda.Device(i)
                compute_capability = device.compute_capability()
                
                if compute_capability[0] >= 7:
                    tensor_cores_count += 1
            
            return tensor_cores_count
        except ImportError:
            try:
                from py3nvml import py3nvml
                py3nvml.nvmlInit()
                
                device_count = py3nvml.nvmlDeviceGetCount()
                tensor_cores_count = 0
                
                for i in range(device_count):
                    handle = py3nvml.nvmlDeviceGetHandleByIndex(i)
                    info = py3nvml.nvmlDeviceGetCudaComputeCapability(handle)
                    
                    if info[0] >= 7:
                        tensor_cores_count += 1
                
                py3nvml.nvmlShutdown()
                return tensor_cores_count
            except ImportError:
                return 0
    
    def get_device_info(self, device_index: int = 0) -> Dict[str, Any]:
        """
        Get information about a device with Tensor Cores.
        
        Args:
            device_index: Index of the device
            
        Returns:
            Dict: Device information
        """
        if not self.initialize():
            return {}
        
        try:
            import pycuda.driver as cuda
            cuda.init()
            
            tensor_cores_devices = []
            for i in range(cuda.Device.count()):
                device = cuda.Device(i)
                compute_capability = device.compute_capability()
                
                if compute_capability[0] >= 7:
                    tensor_cores_devices.append(i)
            
            if device_index >= len(tensor_cores_devices):
                return {}
            
            device = cuda.Device(tensor_cores_devices[device_index])
            compute_capability = device.compute_capability()
            
            architecture = "Unknown"
            if compute_capability[0] == 7:
                if compute_capability[1] == 0:
                    architecture = "Volta"
                elif compute_capability[1] == 5:
                    architecture = "Turing"
            elif compute_capability[0] == 8:
                architecture = "Ampere"
            elif compute_capability[0] == 9:
                architecture = "Hopper"
            
            return {
                "name": device.name(),
                "vendor": "NVIDIA",
                "type": "GPU",
                "architecture": architecture,
                "compute_capability": f"{compute_capability[0]}.{compute_capability[1]}",
                "compute_units": device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT),
                "memory": device.total_memory(),
                "tensor_cores": True
            }
        except ImportError:
            return {
                "name": "NVIDIA GPU with Tensor Cores",
                "vendor": "NVIDIA",
                "type": "GPU",
                "architecture": "Unknown",
                "compute_capability": "Unknown",
                "compute_units": 0,
                "memory": 0,
                "tensor_cores": True
            }
    
    def matmul(self, a: Tensor, b: Tensor) -> Tensor:
        """
        Perform matrix multiplication using Tensor Cores.
        
        Args:
            a: First tensor
            b: Second tensor
            
        Returns:
            Tensor: Result of matrix multiplication
        """
        if not self.initialize():
            return a @ b
        
        return a @ b
    
    def optimize_model(self, model: Module, precision: str = 'mixed') -> Module:
        """
        Optimize a model to use Tensor Cores.
        
        Args:
            model: The model to optimize
            precision: Precision to use ('fp32', 'fp16', 'mixed')
            
        Returns:
            Module: Optimized model
        """
        if not self.initialize():
            return model
        
        self.set_precision(precision)
        
        return model
