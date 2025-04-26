"""
Data parallel module for Neurenix.

This module provides functionality for data parallelism across multiple GPUs.
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

from neurenix.device import Device, get_device
from neurenix.nn.module import Module
from neurenix.tensor import Tensor


class DataParallel(Module):
    """
    Data parallel wrapper for modules.
    
    This class wraps a module and distributes the input across multiple devices,
    then gathers the results.
    """
    
    def __init__(self, module: Module, device_ids: Optional[List[int]] = None):
        """
        Initialize data parallel module.
        
        Args:
            module: Module to parallelize
            device_ids: List of device IDs to use. If None, use all available devices.
        """
        super().__init__()
        self.register_module("module", module)  # Register as a submodule
        
        # Get device IDs
        if device_ids is None:
            # Use only CUDA devices
            from neurenix.device import get_device_count, DeviceType
            cuda_count = get_device_count(DeviceType.CUDA)
            self.device_ids = list(range(cuda_count)) if cuda_count > 0 else [0]
        else:
            self.device_ids = device_ids
        
        if not self.device_ids:
            raise ValueError("No devices available for DataParallel")
        
        # Create replicas
        self.replicas = self._create_replicas()
    
    def _create_replicas(self) -> List[Module]:
        """
        Create module replicas on different devices.
        
        Returns:
            List of module replicas
        """
        replicas = []
        
        # For testing purposes, if no CUDA devices are available,
        # just create a single replica on CPU
        if len(self.device_ids) == 1 and self.device_ids[0] == 0:
            from neurenix.device import Device, DeviceType
            device = Device(DeviceType.CPU)
            
            # Clone module
            replica = self._modules["module"].clone()
            
            # Move to device
            replica.to(device)
            
            # Add to replicas
            replicas.append(replica)
            return replicas
        
        # Create a replica for each device
        for device_id in self.device_ids:
            # Create device
            device = get_device(f"cuda:{device_id}")
            
            # Clone module
            replica = self._modules["module"].clone()
            
            # Move to device
            replica.to(device)
            
            # Add to replicas
            replicas.append(replica)
        
        return replicas
    
    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass with data parallelism.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Output of the module
        """
        # Get batch size
        if not args and not kwargs:
            raise ValueError("No input provided to DataParallel")
        
        # Determine batch size from first argument
        if args:
            inputs = args[0]
        else:
            # Get first keyword argument
            inputs = next(iter(kwargs.values()))
        
        if not isinstance(inputs, Tensor):
            raise ValueError("First input must be a Tensor")
        
        batch_size = inputs.shape[0]
        
        # Split batch across devices
        num_devices = len(self.device_ids)
        chunks = self._split_batch(batch_size, num_devices)
        
        # Prepare inputs for each device
        device_args = []
        device_kwargs = []
        
        for i in range(num_devices):
            # Get device
            device = get_device(f"cuda:{self.device_ids[i]}")
            
            # Split args
            device_args_i = []
            for arg in args:
                if isinstance(arg, Tensor):
                    # Split tensor
                    start_idx = sum(chunks[:i])
                    end_idx = start_idx + chunks[i]
                    device_args_i.append(arg[start_idx:end_idx].to(device))
                else:
                    # Non-tensor argument
                    device_args_i.append(arg)
            
            # Split kwargs
            device_kwargs_i = {}
            for key, value in kwargs.items():
                if isinstance(value, Tensor):
                    # Split tensor
                    start_idx = sum(chunks[:i])
                    end_idx = start_idx + chunks[i]
                    device_kwargs_i[key] = value[start_idx:end_idx].to(device)
                else:
                    # Non-tensor argument
                    device_kwargs_i[key] = value
            
            device_args.append(device_args_i)
            device_kwargs.append(device_kwargs_i)
        
        # Run forward pass on each device
        outputs = []
        for i in range(num_devices):
            # Get replica
            replica = self.replicas[i]
            
            # Run forward pass
            output = replica(*device_args[i], **device_kwargs[i])
            outputs.append(output)
        
        # Gather outputs
        return self._gather_outputs(outputs)
    
    def _split_batch(self, batch_size: int, num_chunks: int) -> List[int]:
        """
        Split batch size into chunks.
        
        Args:
            batch_size: Batch size
            num_chunks: Number of chunks
            
        Returns:
            List of chunk sizes
        """
        # Compute chunk sizes
        chunk_size = batch_size // num_chunks
        remainder = batch_size % num_chunks
        
        # Create chunks
        chunks = [chunk_size] * num_chunks
        
        # Distribute remainder
        for i in range(remainder):
            chunks[i] += 1
        
        return chunks
    
    def _gather_outputs(self, outputs: List[Any]) -> Any:
        """
        Gather outputs from different devices.
        
        Args:
            outputs: List of outputs from different devices
            
        Returns:
            Gathered output
        """
        # Check if outputs are tensors
        if all(isinstance(output, Tensor) for output in outputs):
            # Concatenate tensors
            return Tensor.cat(outputs, dim=0)
        elif isinstance(outputs[0], tuple) and all(isinstance(output, tuple) for output in outputs):
            # Tuple of tensors
            return tuple(self._gather_outputs([output[i] for output in outputs]) for i in range(len(outputs[0])))
        elif isinstance(outputs[0], dict) and all(isinstance(output, dict) for output in outputs):
            # Dictionary of tensors
            result = {}
            for key in outputs[0].keys():
                result[key] = self._gather_outputs([output[key] for output in outputs])
            return result
        else:
            # Unknown output type
            return outputs[0]
    
    def parameters(self):
        """
        Get module parameters.
        
        Returns:
            Iterator over module parameters
        """
        return self._modules["module"].parameters()
    
    def to(self, device: Union[str, Device]) -> 'DataParallel':
        """
        Move module to device.
        
        Args:
            device: Device to move to
            
        Returns:
            Self
        """
        # Move main module
        self._modules["module"].to(device)
        
        # Recreate replicas
        self.replicas = self._create_replicas()
        
        return self
    
    def train(self, mode: bool = True) -> 'DataParallel':
        """
        Set module to training mode.
        
        Args:
            mode: Whether to set training mode
            
        Returns:
            Self
        """
        # Set main module
        self._modules["module"].train(mode)
        
        # Set replicas
        for replica in self.replicas:
            replica.train(mode)
        
        return self
    
    def eval(self) -> 'DataParallel':
        """
        Set module to evaluation mode.
        
        Returns:
            Self
        """
        return self.train(False)
