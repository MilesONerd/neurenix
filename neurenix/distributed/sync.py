"""
Synchronization primitives for distributed training.

This module provides synchronization primitives for distributed training,
such as synchronized batch normalization.
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor
from neurenix.distributed.distributed import get_world_size, get_rank


class SyncBatchNorm(Module):
    """
    Synchronized Batch Normalization.
    
    This module synchronizes batch normalization statistics across processes
    in distributed training.
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        """
        Initialize synchronized batch normalization.
        
        Args:
            num_features: Number of features
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics
            affine: Whether to use learnable affine parameters
            track_running_stats: Whether to track running statistics
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        # Running statistics
        if track_running_stats:
            self.register_buffer("running_mean", Tensor.zeros(num_features))
            self.register_buffer("running_var", Tensor.ones(num_features))
            self.register_buffer("num_batches_tracked", Tensor.tensor(0, dtype="int64"))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)
        
        # Learnable parameters
        if affine:
            self.weight = Tensor.ones(num_features)
            self.bias = Tensor.zeros(num_features)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, num_features, ...)
            
        Returns:
            Normalized tensor
        """
        # Check if we're in distributed mode
        world_size = get_world_size()
        
        if world_size == 1:
            # Single process mode, use regular batch normalization
            return self._forward_single_process(x)
        else:
            # Multi-process mode, synchronize statistics
            return self._forward_multi_process(x)
    
    def _forward_single_process(self, x: Tensor) -> Tensor:
        """
        Forward pass for single process.
        
        Args:
            x: Input tensor of shape (batch_size, num_features, ...)
            
        Returns:
            Normalized tensor
        """
        # Get input shape
        input_shape = x.shape
        
        # Reshape to (batch_size, num_features, -1)
        x_reshaped = x.reshape(input_shape[0], self.num_features, -1)
        
        # Compute statistics
        batch_mean = x_reshaped.mean(dim=(0, 2))
        batch_var = x_reshaped.var(dim=(0, 2), unbiased=False)
        
        # Update running statistics
        if self.track_running_stats and self.training:
            self.num_batches_tracked += 1
            if self.momentum is None:
                # Use cumulative moving average
                momentum = 1.0 / float(self.num_batches_tracked)
            else:
                # Use exponential moving average
                momentum = self.momentum
            
            self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
            self.running_var = (1 - momentum) * self.running_var + momentum * batch_var
        
        # Use running statistics in evaluation mode
        if not self.training and self.track_running_stats:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = batch_mean
            var = batch_var
        
        # Normalize
        x_reshaped = (x_reshaped - mean.reshape(1, -1, 1)) / Tensor.sqrt(var.reshape(1, -1, 1) + self.eps)
        
        # Apply affine transformation
        if self.affine:
            x_reshaped = x_reshaped * self.weight.reshape(1, -1, 1) + self.bias.reshape(1, -1, 1)
        
        # Reshape back to original shape
        return x_reshaped.reshape(input_shape)
    
    def _forward_multi_process(self, x: Tensor) -> Tensor:
        """
        Forward pass for multiple processes.
        
        Args:
            x: Input tensor of shape (batch_size, num_features, ...)
            
        Returns:
            Normalized tensor
        """
        # Get input shape
        input_shape = x.shape
        
        # Reshape to (batch_size, num_features, -1)
        x_reshaped = x.reshape(input_shape[0], self.num_features, -1)
        
        # Compute local statistics
        batch_size = x_reshaped.shape[0]
        num_elements = batch_size * x_reshaped.shape[2]
        
        sum_x = x_reshaped.sum(dim=(0, 2))
        sum_x2 = (x_reshaped ** 2).sum(dim=(0, 2))
        
        # Synchronize statistics across processes
        # In a real implementation, this would use all_reduce
        # For now, we'll simulate it
        world_size = get_world_size()
        
        # Simulate all_reduce for sum_x, sum_x2, and num_elements
        # In a real implementation, this would use a communication backend
        global_sum_x = sum_x * world_size  # Simulated all_reduce
        global_sum_x2 = sum_x2 * world_size  # Simulated all_reduce
        global_num_elements = num_elements * world_size  # Simulated all_reduce
        
        # Compute global mean and variance
        global_mean = global_sum_x / global_num_elements
        global_var = (global_sum_x2 / global_num_elements) - (global_mean ** 2)
        
        # Update running statistics
        if self.track_running_stats and self.training:
            self.num_batches_tracked += 1
            if self.momentum is None:
                # Use cumulative moving average
                momentum = 1.0 / float(self.num_batches_tracked)
            else:
                # Use exponential moving average
                momentum = self.momentum
            
            self.running_mean = (1 - momentum) * self.running_mean + momentum * global_mean
            self.running_var = (1 - momentum) * self.running_var + momentum * global_var
        
        # Use running statistics in evaluation mode
        if not self.training and self.track_running_stats:
            mean = self.running_mean
            var = self.running_var
        else:
            mean = global_mean
            var = global_var
        
        # Normalize
        x_reshaped = (x_reshaped - mean.reshape(1, -1, 1)) / Tensor.sqrt(var.reshape(1, -1, 1) + self.eps)
        
        # Apply affine transformation
        if self.affine:
            x_reshaped = x_reshaped * self.weight.reshape(1, -1, 1) + self.bias.reshape(1, -1, 1)
        
        # Reshape back to original shape
        return x_reshaped.reshape(input_shape)
    
    def extra_repr(self) -> str:
        """
        Extra representation.
        
        Returns:
            String representation
        """
        return (
            f"{self.num_features}, "
            f"eps={self.eps}, "
            f"momentum={self.momentum}, "
            f"affine={self.affine}, "
            f"track_running_stats={self.track_running_stats}"
        )
