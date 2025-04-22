"""
Normalization layers for neural networks.
"""

from typing import Tuple, Union, List, Optional
import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor

class BatchNorm1d(Module):
    """
    Applies Batch Normalization over a 2D or 3D input.
    
    Normalizes the input by subtracting the mean and dividing by the standard deviation.
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        """
        Initialize a BatchNorm1d layer.
        
        Args:
            num_features: Number of features (C from an input of size (N, C) or (N, C, L))
            eps: Value added to the denominator for numerical stability
            momentum: Value used for the running_mean and running_var computation
            affine: If True, this module has learnable affine parameters
            track_running_stats: If True, this module tracks the running mean and variance
        """
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = Tensor.ones(num_features)
            self.bias = Tensor.zeros(num_features)
        else:
            self.register_buffer('weight', None)
            self.register_buffer('bias', None)
        
        if self.track_running_stats:
            self.register_buffer('running_mean', Tensor.zeros(num_features))
            self.register_buffer('running_var', Tensor.ones(num_features))
            self.register_buffer('num_batches_tracked', Tensor.tensor(0, dtype=np.int64))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BatchNorm1d layer.
        
        Args:
            x: Input tensor of shape (batch_size, num_features) or (batch_size, num_features, length)
            
        Returns:
            Normalized tensor of the same shape as input
        """
        if x.dim() != 2 and x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input (got {x.dim()}D input)")
        
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
        
        if x.dim() == 2:
            if self.training or not self.track_running_stats:
                mean = x.mean(dim=0)
                var = x.var(dim=0, unbiased=False)
                
                if self.track_running_stats:
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            else:
                mean = self.running_mean
                var = self.running_var
            
            x_normalized = (x - mean) / Tensor.sqrt(var + self.eps)
            
            if self.affine:
                x_normalized = x_normalized * self.weight + self.bias
            
            return x_normalized
        else:
            if self.training or not self.track_running_stats:
                mean = x.mean(dim=(0, 2))
                var = x.var(dim=(0, 2), unbiased=False)
                
                if self.track_running_stats:
                    self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                    self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
            else:
                mean = self.running_mean
                var = self.running_var
            
            mean = mean.reshape(1, -1, 1)
            var = var.reshape(1, -1, 1)
            
            x_normalized = (x - mean) / Tensor.sqrt(var + self.eps)
            
            if self.affine:
                weight = self.weight.reshape(1, -1, 1)
                bias = self.bias.reshape(1, -1, 1)
                x_normalized = x_normalized * weight + bias
            
            return x_normalized
    
    def __repr__(self):
        return f"BatchNorm1d(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats})"


class BatchNorm2d(Module):
    """
    Applies Batch Normalization over a 4D input.
    
    Normalizes the input by subtracting the mean and dividing by the standard deviation.
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ):
        """
        Initialize a BatchNorm2d layer.
        
        Args:
            num_features: Number of features (C from an input of size (N, C, H, W))
            eps: Value added to the denominator for numerical stability
            momentum: Value used for the running_mean and running_var computation
            affine: If True, this module has learnable affine parameters
            track_running_stats: If True, this module tracks the running mean and variance
        """
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = Tensor.ones(num_features)
            self.bias = Tensor.zeros(num_features)
        else:
            self.register_buffer('weight', None)
            self.register_buffer('bias', None)
        
        if self.track_running_stats:
            self.register_buffer('running_mean', Tensor.zeros(num_features))
            self.register_buffer('running_var', Tensor.ones(num_features))
            self.register_buffer('num_batches_tracked', Tensor.tensor(0, dtype=np.int64))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the BatchNorm2d layer.
        
        Args:
            x: Input tensor of shape (batch_size, num_features, height, width)
            
        Returns:
            Normalized tensor of the same shape as input
        """
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (got {x.dim()}D input)")
        
        if self.training and self.track_running_stats:
            self.num_batches_tracked += 1
        
        if self.training or not self.track_running_stats:
            mean = x.mean(dim=(0, 2, 3))
            var = x.var(dim=(0, 2, 3), unbiased=False)
            
            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        
        mean = mean.reshape(1, -1, 1, 1)
        var = var.reshape(1, -1, 1, 1)
        
        x_normalized = (x - mean) / Tensor.sqrt(var + self.eps)
        
        if self.affine:
            weight = self.weight.reshape(1, -1, 1, 1)
            bias = self.bias.reshape(1, -1, 1, 1)
            x_normalized = x_normalized * weight + bias
        
        return x_normalized
    
    def __repr__(self):
        return f"BatchNorm2d(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats})"
