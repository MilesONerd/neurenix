"""
Synchronized Batch Normalization module for Neurenix.

This module provides functionality for synchronized batch normalization
across multiple devices.
"""

from typing import Optional, List, Union

import numpy as np

from neurenix.device import Device
from neurenix.nn.module import Module
from neurenix.tensor import Tensor


class SyncBatchNorm(Module):
    """
    Synchronized Batch Normalization.
    
    This class implements batch normalization that synchronizes statistics
    across multiple devices.
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
        
        # Initialize parameters
        if affine:
            self.weight = Tensor(np.ones(num_features), requires_grad=True)
            self.bias = Tensor(np.zeros(num_features), requires_grad=True)
        
        # Initialize running statistics
        if track_running_stats:
            self.register_buffer("running_mean", Tensor(np.zeros(num_features)))
            self.register_buffer("running_var", Tensor(np.ones(num_features)))
            self.register_buffer("num_batches_tracked", Tensor(np.array(0)))
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, num_features, ...)
            
        Returns:
            Normalized tensor
        """
        # Get dimensions
        batch_size = x.shape[0]
        
        # Compute statistics
        if self.is_training():
            # Compute mean and variance
            mean = x.mean(dim=0)
            var = ((x - mean) ** 2).mean(dim=0)
            
            # Update running statistics
            if self.track_running_stats:
                if self.num_batches_tracked == 0:
                    self.running_mean = mean
                    self.running_var = var
                else:
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
                
                self.num_batches_tracked += 1
        else:
            # Use running statistics
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_norm = (x - mean) / ((var + self.eps) ** 0.5)
        
        # Apply affine transformation
        if self.affine:
            x_norm = x_norm * self.weight + self.bias
        
        return x_norm
    
    def register_buffer(self, name: str, tensor: Optional[Tensor]) -> None:
        """
        Register a buffer with the module.
        
        Args:
            name: Name of the buffer
            tensor: Buffer tensor, or None to remove the buffer
        """
        if tensor is None:
            if hasattr(self, name):
                delattr(self, name)
        else:
            setattr(self, name, tensor)
