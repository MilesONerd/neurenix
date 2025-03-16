"""
Python bindings for the Phynexus engine.

This module provides Python bindings for the Rust implementation of the Phynexus engine.
"""

import os
import sys
import platform
from typing import List, Tuple, Dict, Any, Optional, Union, Sequence

import numpy as np

# Try to import the Rust extension module
try:
    from neurenix._phynexus import *
except ImportError:
    # If the extension module is not available, use a fallback implementation
    print("Warning: Phynexus Rust extension not found. Using Python fallback implementation.")
    
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
                return 0
        elif device_type == ROCM:
            return 0
        elif device_type == WEBGPU:
            return 0
        elif device_type == TPU:
            return 0  # Currently no TPU detection in fallback implementation
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

# Initialize the Phynexus engine
init()

# Register shutdown function
import atexit
atexit.register(shutdown)
