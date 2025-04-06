"""
Multi-Scale Pooling operations for Neurenix.

This module provides pooling operations that work across multiple scales
or resolutions, enabling more effective feature extraction.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable

from neurenix.nn import Module, Conv2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d
from neurenix.tensor import Tensor


class MultiScalePooling(Module):
    """
    Base class for multi-scale pooling operations.
    
    Multi-scale pooling extracts features at multiple scales or resolutions,
    allowing models to capture both fine-grained details and global context.
    """
    
    def __init__(self, 
                 output_size: Union[int, Tuple[int, int]],
                 pool_type: str = 'avg'):
        """
        Initialize a multi-scale pooling module.
        
        Args:
            output_size: Size of the output features after pooling
            pool_type: Type of pooling to use ('avg' or 'max')
        """
        super().__init__()
        self.output_size = output_size if isinstance(output_size, tuple) else (output_size, output_size)
        self.pool_type = pool_type
        
        if pool_type == 'avg':
            self.pool = AdaptiveAvgPool2d(self.output_size)
        elif pool_type == 'max':
            self.pool = AdaptiveMaxPool2d(self.output_size)
        else:
            raise ValueError(f"Unsupported pool_type: {pool_type}. Use 'avg' or 'max'.")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the multi-scale pooling module.
        
        Args:
            x: Input tensor
            
        Returns:
            Pooled tensor
        """
        return self.pool(x)


class PyramidPooling(MultiScalePooling):
    """
    Pyramid Pooling Module (PPM) as used in PSPNet.
    
    Performs pooling at multiple scales and concatenates the results
    to capture information at different scales.
    """
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 pool_sizes: List[int] = [1, 2, 3, 6],
                 pool_type: str = 'avg'):
        """
        Initialize a pyramid pooling module.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels per pyramid level
            pool_sizes: List of output sizes for each pyramid level
            pool_type: Type of pooling to use ('avg' or 'max')
        """
        super().__init__(output_size=(1, 1), pool_type=pool_type)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pool_sizes = pool_sizes
        
        self.pyramid_levels = []
        
        for pool_size in pool_sizes:
            level = Module([
                AdaptiveAvgPool2d((pool_size, pool_size)) if pool_type == 'avg' else AdaptiveMaxPool2d((pool_size, pool_size)),
                Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            ])
            self.pyramid_levels.append(level)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the pyramid pooling module.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Concatenated multi-scale features
        """
        h, w = x.shape[2:]
        features = [x]  # Start with the original features
        
        for level in self.pyramid_levels:
            pooled = level(x)
            upsampled = pooled.resize((h, w))
            features.append(upsampled)
        
        return Tensor.cat(features, dim=1)


class SpatialPyramidPooling(Module):
    """
    Spatial Pyramid Pooling (SPP) as introduced in SPPNet.
    
    Allows processing inputs of variable sizes by pooling them into
    fixed-length representations.
    """
    
    def __init__(self, 
                 output_sizes: List[int] = [1, 2, 4],
                 pool_type: str = 'max'):
        """
        Initialize a spatial pyramid pooling module.
        
        Args:
            output_sizes: List of output grid sizes for each pyramid level
            pool_type: Type of pooling to use ('avg' or 'max')
        """
        super().__init__()
        self.output_sizes = output_sizes
        self.pool_type = pool_type
        
        self.pools = []
        for size in output_sizes:
            if pool_type == 'avg':
                pool = AdaptiveAvgPool2d((size, size))
            elif pool_type == 'max':
                pool = AdaptiveMaxPool2d((size, size))
            else:
                raise ValueError(f"Unsupported pool_type: {pool_type}. Use 'avg' or 'max'.")
            
            self.pools.append(pool)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the spatial pyramid pooling module.
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Flattened fixed-length representation
        """
        batch_size = x.shape[0]
        features = []
        
        for pool in self.pools:
            pooled = pool(x)
            flat = pooled.view(batch_size, -1)
            features.append(flat)
        
        return Tensor.cat(features, dim=1)
