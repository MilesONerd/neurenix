"""
Multi-Scale Transforms for Neurenix.

This module provides transformations for converting data between different scales
or resolutions, enabling more effective multi-scale model architectures.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable

from neurenix.nn import Module, Conv2d, ConvTranspose2d, AvgPool2d, Upsample
from neurenix.tensor import Tensor


class MultiScaleTransform(Module):
    """
    Base class for multi-scale transformations.
    
    Multi-scale transforms convert data between different scales or resolutions,
    enabling models to work with multi-scale representations.
    """
    
    def __init__(self):
        """
        Initialize a multi-scale transform module.
        """
        super().__init__()
    
    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Forward pass through the multi-scale transform module.
        
        Args:
            x: Input tensor
            
        Returns:
            List of tensors at different scales
        """
        raise NotImplementedError("Subclasses must implement forward")


class Rescale(MultiScaleTransform):
    """
    Rescale transform for generating multi-scale representations.
    
    Rescales an input tensor to multiple scales using interpolation.
    """
    
    def __init__(self, 
                 scales: List[float],
                 mode: str = 'bilinear',
                 align_corners: bool = False):
        """
        Initialize a rescale transform.
        
        Args:
            scales: List of scale factors to apply
            mode: Interpolation mode ('nearest', 'bilinear', 'bicubic')
            align_corners: Whether to align corners in interpolation
        """
        super().__init__()
        self.scales = scales
        self.mode = mode
        self.align_corners = align_corners
        
        self.upsamplers = []
        for scale in scales:
            if scale != 1.0:  # No need for upsampler if scale is 1.0
                self.upsamplers.append(
                    Upsample(scale_factor=scale, mode=mode, align_corners=align_corners if mode != 'nearest' else None)
                )
            else:
                self.upsamplers.append(None)
    
    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Forward pass through the rescale transform.
        
        Args:
            x: Input tensor
            
        Returns:
            List of tensors at different scales
        """
        outputs = []
        
        for i, scale in enumerate(self.scales):
            if scale == 1.0:
                outputs.append(x)
            else:
                outputs.append(self.upsamplers[i](x))
        
        return outputs


class PyramidDownsample(MultiScaleTransform):
    """
    Pyramid downsampling transform for generating multi-scale representations.
    
    Creates a pyramid of increasingly downsampled versions of the input tensor.
    """
    
    def __init__(self, 
                 num_levels: int = 3,
                 downsample_factor: float = 0.5,
                 mode: str = 'pool'):
        """
        Initialize a pyramid downsampling transform.
        
        Args:
            num_levels: Number of pyramid levels to generate
            downsample_factor: Factor by which to reduce dimensions at each level
            mode: Downsampling mode ('pool' or 'conv')
        """
        super().__init__()
        self.num_levels = num_levels
        self.downsample_factor = downsample_factor
        self.mode = mode
        
        self.downsamplers = []
        
        for i in range(num_levels - 1):  # No downsampler needed for the original resolution
            if mode == 'pool':
                kernel_size = int(1 / downsample_factor)
                self.downsamplers.append(AvgPool2d(kernel_size=kernel_size, stride=kernel_size))
            elif mode == 'conv':
                stride = int(1 / downsample_factor)
                self.downsamplers.append(
                    Conv2d(in_channels=-1, out_channels=-1, kernel_size=3, stride=stride, padding=1)
                )
            else:
                raise ValueError(f"Unsupported mode: {mode}. Use 'pool' or 'conv'.")
    
    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Forward pass through the pyramid downsampling transform.
        
        Args:
            x: Input tensor
            
        Returns:
            List of tensors at different scales, from original to most downsampled
        """
        outputs = [x]  # Start with the original resolution
        current = x
        
        for i in range(self.num_levels - 1):
            if self.mode == 'conv':
                in_channels = current.shape[1]
                self.downsamplers[i].in_channels = in_channels
                self.downsamplers[i].out_channels = in_channels
            
            current = self.downsamplers[i](current)
            outputs.append(current)
        
        return outputs


class GaussianPyramid(MultiScaleTransform):
    """
    Gaussian pyramid transform for generating multi-scale representations.
    
    Creates a Gaussian pyramid by repeatedly smoothing and downsampling the input.
    """
    
    def __init__(self, 
                 num_levels: int = 3,
                 sigma: float = 1.0,
                 kernel_size: int = 5):
        """
        Initialize a Gaussian pyramid transform.
        
        Args:
            num_levels: Number of pyramid levels to generate
            sigma: Standard deviation for Gaussian smoothing
            kernel_size: Size of the Gaussian kernel
        """
        super().__init__()
        self.num_levels = num_levels
        self.sigma = sigma
        self.kernel_size = kernel_size
        
        self.smooth_down = []
        
        for i in range(num_levels - 1):
            smooth_down = Module([
                Conv2d(in_channels=-1, out_channels=-1, kernel_size=kernel_size, padding=kernel_size//2),
                AvgPool2d(kernel_size=2, stride=2)
            ])
            self.smooth_down.append(smooth_down)
    
    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Forward pass through the Gaussian pyramid transform.
        
        Args:
            x: Input tensor
            
        Returns:
            List of tensors at different scales, from original to most downsampled
        """
        outputs = [x]  # Start with the original resolution
        current = x
        
        for i in range(self.num_levels - 1):
            in_channels = current.shape[1]
            self.smooth_down[i][0].in_channels = in_channels
            self.smooth_down[i][0].out_channels = in_channels
            
            current = self.smooth_down[i](current)
            outputs.append(current)
        
        return outputs


class LaplacianPyramid(MultiScaleTransform):
    """
    Laplacian pyramid transform for generating multi-scale representations.
    
    Creates a Laplacian pyramid by computing differences between Gaussian pyramid levels.
    """
    
    def __init__(self, 
                 num_levels: int = 3,
                 sigma: float = 1.0,
                 kernel_size: int = 5):
        """
        Initialize a Laplacian pyramid transform.
        
        Args:
            num_levels: Number of pyramid levels to generate
            sigma: Standard deviation for Gaussian smoothing
            kernel_size: Size of the Gaussian kernel
        """
        super().__init__()
        self.num_levels = num_levels
        
        self.gaussian_pyramid = GaussianPyramid(num_levels, sigma, kernel_size)
        
        self.upsample = []
        
        for i in range(num_levels - 1):
            upsample = Module([
                Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                Conv2d(in_channels=-1, out_channels=-1, kernel_size=kernel_size, padding=kernel_size//2)
            ])
            self.upsample.append(upsample)
    
    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Forward pass through the Laplacian pyramid transform.
        
        Args:
            x: Input tensor
            
        Returns:
            List of tensors representing Laplacian pyramid levels
        """
        gaussian_levels = self.gaussian_pyramid(x)
        
        laplacian_levels = []
        
        for i in range(self.num_levels - 1):
            in_channels = gaussian_levels[i+1].shape[1]
            self.upsample[i][1].in_channels = in_channels
            self.upsample[i][1].out_channels = in_channels
            
            upsampled = self.upsample[i](gaussian_levels[i+1])
            
            laplacian = gaussian_levels[i] - upsampled
            laplacian_levels.append(laplacian)
        
        laplacian_levels.append(gaussian_levels[-1])
        
        return laplacian_levels
