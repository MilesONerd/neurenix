"""
Normalization layers for the Neurenix framework.

This module provides implementations of various normalization layers
such as BatchNorm1d, BatchNorm2d, LayerNorm, InstanceNorm, etc.
"""

from typing import Optional, Tuple, Union
import numpy as np

from ..core import PhynexusExtension
from .module import Module
from .parameter import Parameter

class BatchNorm1d(Module):
    """
    Batch normalization for 1D inputs.
    
    Applies Batch Normalization over a 2D or 3D input as described in
    the paper: "Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift" [1].
    
    [1] https://arxiv.org/abs/1502.03167
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        """
        Initialize BatchNorm1d.
        
        Args:
            num_features: Number of features or channels
            eps: Small constant for numerical stability
            momentum: Value used for the running_mean and running_var computation
            affine: Whether to use learnable affine parameters
            track_running_stats: Whether to track running statistics
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = Parameter(np.ones(num_features))
            self.bias = Parameter(np.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        if self.track_running_stats:
            self.register_buffer('running_mean', np.zeros(num_features))
            self.register_buffer('running_var', np.ones(num_features))
            self.register_buffer('num_batches_tracked', 0)
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
            
        self.training = True
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, num_features) or
               (batch_size, num_features, sequence_length)
            
        Returns:
            Normalized tensor of the same shape as input
        """
        if PhynexusExtension.is_available():
            return PhynexusExtension.batch_norm_1d(
                x, self.running_mean, self.running_var,
                self.weight, self.bias, self.training,
                self.momentum, self.eps
            )
            
        if self.training:
            if len(x.shape) == 2:
                batch_mean = np.mean(x, axis=0)
                batch_var = np.var(x, axis=0)
            else:
                batch_mean = np.mean(x, axis=(0, 2))
                batch_var = np.var(x, axis=(0, 2))
                
            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                self.num_batches_tracked += 1
                
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var
            
        if len(x.shape) == 2:
            x_norm = (x - mean) / np.sqrt(var + self.eps)
        else:
            x_norm = (x - mean[:, np.newaxis]) / np.sqrt(var[:, np.newaxis] + self.eps)
            
        if self.affine:
            if len(x.shape) == 2:
                return self.weight * x_norm + self.bias
            else:
                return self.weight[:, np.newaxis] * x_norm + self.bias[:, np.newaxis]
        else:
            return x_norm
            
    def __repr__(self):
        return f"BatchNorm1d(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats})"


class BatchNorm2d(Module):
    """
    Batch normalization for 2D inputs.
    
    Applies Batch Normalization over a 4D input as described in
    the paper: "Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift" [1].
    
    [1] https://arxiv.org/abs/1502.03167
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        """
        Initialize BatchNorm2d.
        
        Args:
            num_features: Number of features or channels
            eps: Small constant for numerical stability
            momentum: Value used for the running_mean and running_var computation
            affine: Whether to use learnable affine parameters
            track_running_stats: Whether to track running statistics
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = Parameter(np.ones(num_features))
            self.bias = Parameter(np.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        if self.track_running_stats:
            self.register_buffer('running_mean', np.zeros(num_features))
            self.register_buffer('running_var', np.ones(num_features))
            self.register_buffer('num_batches_tracked', 0)
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
            
        self.training = True
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, num_features, height, width)
            
        Returns:
            Normalized tensor of the same shape as input
        """
        if PhynexusExtension.is_available():
            return PhynexusExtension.batch_norm_2d(
                x, self.running_mean, self.running_var,
                self.weight, self.bias, self.training,
                self.momentum, self.eps
            )
            
        if self.training:
            batch_mean = np.mean(x, axis=(0, 2, 3))
            batch_var = np.var(x, axis=(0, 2, 3))
                
            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                self.num_batches_tracked += 1
                
            mean = batch_mean
            var = batch_var
        else:
            mean = self.running_mean
            var = self.running_var
            
        x_norm = (x - mean[:, np.newaxis, np.newaxis]) / np.sqrt(var[:, np.newaxis, np.newaxis] + self.eps)
            
        if self.affine:
            return self.weight[:, np.newaxis, np.newaxis] * x_norm + self.bias[:, np.newaxis, np.newaxis]
        else:
            return x_norm
            
    def __repr__(self):
        return f"BatchNorm2d(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats})"


class LayerNorm(Module):
    """
    Layer normalization.
    
    Applies Layer Normalization over a mini-batch of inputs as described in
    the paper: "Layer Normalization" [1].
    
    [1] https://arxiv.org/abs/1607.06450
    """
    
    def __init__(self, normalized_shape: Union[int, Tuple[int, ...]], eps: float = 1e-5,
                 elementwise_affine: bool = True):
        """
        Initialize LayerNorm.
        
        Args:
            normalized_shape: Input shape from an expected input
            eps: Small constant for numerical stability
            elementwise_affine: Whether to use learnable affine parameters
        """
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
            
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = Parameter(np.ones(normalized_shape))
            self.bias = Parameter(np.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor of the same shape as input
        """
        if PhynexusExtension.is_available():
            return PhynexusExtension.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
            
        shape = x.shape
        ndim = len(shape)
        normalized_ndim = len(self.normalized_shape)
        
        if ndim < normalized_ndim:
            raise ValueError(f"Input tensor has fewer dimensions ({ndim}) than normalized_shape ({normalized_ndim})")
            
        axes = tuple(range(ndim - normalized_ndim, ndim))
        
        mean = np.mean(x, axis=axes, keepdims=True)
        var = np.var(x, axis=axes, keepdims=True)
        
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        
        if self.elementwise_affine:
            weight_shape = tuple([1] * (ndim - normalized_ndim)) + self.normalized_shape
            bias_shape = tuple([1] * (ndim - normalized_ndim)) + self.normalized_shape
            
            return self.weight.reshape(weight_shape) * x_norm + self.bias.reshape(bias_shape)
        else:
            return x_norm
            
    def __repr__(self):
        return f"LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine})"


class InstanceNorm1d(Module):
    """
    Instance normalization for 1D inputs.
    
    Applies Instance Normalization over a 3D input as described in
    the paper: "Instance Normalization: The Missing Ingredient for Fast Stylization" [1].
    
    [1] https://arxiv.org/abs/1607.08022
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = False, track_running_stats: bool = False):
        """
        Initialize InstanceNorm1d.
        
        Args:
            num_features: Number of features or channels
            eps: Small constant for numerical stability
            momentum: Value used for the running_mean and running_var computation
            affine: Whether to use learnable affine parameters
            track_running_stats: Whether to track running statistics
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = Parameter(np.ones(num_features))
            self.bias = Parameter(np.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        if self.track_running_stats:
            self.register_buffer('running_mean', np.zeros(num_features))
            self.register_buffer('running_var', np.ones(num_features))
            self.register_buffer('num_batches_tracked', 0)
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
            
        self.training = True
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, num_features, sequence_length)
            
        Returns:
            Normalized tensor of the same shape as input
        """
        if PhynexusExtension.is_available():
            return PhynexusExtension.instance_norm_1d(
                x, self.running_mean, self.running_var,
                self.weight, self.bias, self.training,
                self.momentum, self.eps
            )
            
        if len(x.shape) != 3:
            raise ValueError(f"Expected 3D input (batch_size, num_features, sequence_length), got {len(x.shape)}D")
            
        batch_size, num_features, seq_len = x.shape
        
        if self.training or not self.track_running_stats:
            instance_mean = np.mean(x, axis=2)
            instance_var = np.var(x, axis=2)
            
            if self.track_running_stats:
                batch_mean = np.mean(instance_mean, axis=0)
                batch_var = np.mean(instance_var, axis=0)
                
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                self.num_batches_tracked += 1
        else:
            instance_mean = np.tile(self.running_mean, (batch_size, 1))
            instance_var = np.tile(self.running_var, (batch_size, 1))
            
        instance_mean = instance_mean.reshape(batch_size, num_features, 1)
        instance_var = instance_var.reshape(batch_size, num_features, 1)
        
        x_norm = (x - instance_mean) / np.sqrt(instance_var + self.eps)
        
        if self.affine:
            weight = self.weight.reshape(1, num_features, 1)
            bias = self.bias.reshape(1, num_features, 1)
            return weight * x_norm + bias
        else:
            return x_norm
            
    def __repr__(self):
        return f"InstanceNorm1d(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats})"


class InstanceNorm2d(Module):
    """
    Instance normalization for 2D inputs.
    
    Applies Instance Normalization over a 4D input as described in
    the paper: "Instance Normalization: The Missing Ingredient for Fast Stylization" [1].
    
    [1] https://arxiv.org/abs/1607.08022
    """
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1,
                 affine: bool = False, track_running_stats: bool = False):
        """
        Initialize InstanceNorm2d.
        
        Args:
            num_features: Number of features or channels
            eps: Small constant for numerical stability
            momentum: Value used for the running_mean and running_var computation
            affine: Whether to use learnable affine parameters
            track_running_stats: Whether to track running statistics
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = Parameter(np.ones(num_features))
            self.bias = Parameter(np.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
        if self.track_running_stats:
            self.register_buffer('running_mean', np.zeros(num_features))
            self.register_buffer('running_var', np.ones(num_features))
            self.register_buffer('num_batches_tracked', 0)
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
            
        self.training = True
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, num_features, height, width)
            
        Returns:
            Normalized tensor of the same shape as input
        """
        if PhynexusExtension.is_available():
            return PhynexusExtension.instance_norm_2d(
                x, self.running_mean, self.running_var,
                self.weight, self.bias, self.training,
                self.momentum, self.eps
            )
            
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input (batch_size, num_features, height, width), got {len(x.shape)}D")
            
        batch_size, num_features, height, width = x.shape
        
        if self.training or not self.track_running_stats:
            instance_mean = np.mean(x, axis=(2, 3))
            instance_var = np.var(x, axis=(2, 3))
            
            if self.track_running_stats:
                batch_mean = np.mean(instance_mean, axis=0)
                batch_var = np.mean(instance_var, axis=0)
                
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
                self.num_batches_tracked += 1
        else:
            instance_mean = np.tile(self.running_mean, (batch_size, 1))
            instance_var = np.tile(self.running_var, (batch_size, 1))
            
        instance_mean = instance_mean.reshape(batch_size, num_features, 1, 1)
        instance_var = instance_var.reshape(batch_size, num_features, 1, 1)
        
        x_norm = (x - instance_mean) / np.sqrt(instance_var + self.eps)
        
        if self.affine:
            weight = self.weight.reshape(1, num_features, 1, 1)
            bias = self.bias.reshape(1, num_features, 1, 1)
            return weight * x_norm + bias
        else:
            return x_norm
            
    def __repr__(self):
        return f"InstanceNorm2d(num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats})"
