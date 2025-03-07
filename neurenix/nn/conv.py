"""
Convolutional layers for the Neurenix framework.

This module provides convolutional neural network layers for 1D, 2D, and 3D data.
"""

from typing import Union, Tuple, List, Optional, Sequence

import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor
from neurenix.device import Device

# Type aliases
_size_1_t = Union[int, Tuple[int]]
_size_2_t = Union[int, Tuple[int, int]]
_size_3_t = Union[int, Tuple[int, int, int]]

def _pair(x: _size_1_t) -> Tuple[int, int]:
    """Convert a single value or a tuple of length 1 to a tuple of length 2."""
    if isinstance(x, tuple):
        if len(x) == 1:
            return (x[0], x[0])
        elif len(x) == 2:
            return x
        else:
            raise ValueError(f"Expected tuple of length 1 or 2, got {len(x)}")
    else:
        return (x, x)

def _triple(x: _size_1_t) -> Tuple[int, int, int]:
    """Convert a single value or a tuple of length 1 to a tuple of length 3."""
    if isinstance(x, tuple):
        if len(x) == 1:
            return (x[0], x[0], x[0])
        elif len(x) == 3:
            return x
        else:
            raise ValueError(f"Expected tuple of length 1 or 3, got {len(x)}")
    else:
        return (x, x, x)

class Conv1d(Module):
    """
    1D convolution layer.
    
    Applies a 1D convolution over an input signal composed of several input channels.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution (default: 1)
        padding: Zero-padding added to both sides of the input (default: 0)
        dilation: Spacing between kernel elements (default: 1)
        groups: Number of blocked connections from input to output channels (default: 1)
        bias: If True, adds a learnable bias to the output (default: True)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,)
        self.stride = stride if isinstance(stride, tuple) else (stride,)
        self.padding = padding if isinstance(padding, tuple) else (padding,)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation,)
        self.groups = groups
        
        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups})")
        
        # Initialize weights
        self.weight = Tensor.randn(
            (out_channels, in_channels // groups, *self.kernel_size),
            requires_grad=True
        )
        
        if bias:
            self.bias = Tensor.zeros((out_channels,), requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the 1D convolution layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, length)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, output_length)
        """
        return x.conv1d(
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
    
    def __repr__(self) -> str:
        return (
            f"Conv1d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None})"
        )


class Conv2d(Module):
    """
    2D convolution layer.
    
    Applies a 2D convolution over an input image composed of several input channels.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution (default: 1)
        padding: Zero-padding added to both sides of the input (default: 0)
        dilation: Spacing between kernel elements (default: 1)
        groups: Number of blocked connections from input to output channels (default: 1)
        bias: If True, adds a learnable bias to the output (default: True)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        
        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups})")
        
        # Initialize weights
        self.weight = Tensor.randn(
            (out_channels, in_channels // groups, *self.kernel_size),
            requires_grad=True
        )
        
        if bias:
            self.bias = Tensor.zeros((out_channels,), requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the 2D convolution layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, output_height, output_width)
        """
        return x.conv2d(
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
    
    def __repr__(self) -> str:
        return (
            f"Conv2d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None})"
        )


class Conv3d(Module):
    """
    3D convolution layer.
    
    Applies a 3D convolution over an input volume composed of several input channels.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution (default: 1)
        padding: Zero-padding added to both sides of the input (default: 0)
        dilation: Spacing between kernel elements (default: 1)
        groups: Number of blocked connections from input to output channels (default: 1)
        bias: If True, adds a learnable bias to the output (default: True)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)
        self.groups = groups
        
        if in_channels % groups != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by groups ({groups})")
        if out_channels % groups != 0:
            raise ValueError(f"out_channels ({out_channels}) must be divisible by groups ({groups})")
        
        # Initialize weights
        self.weight = Tensor.randn(
            (out_channels, in_channels // groups, *self.kernel_size),
            requires_grad=True
        )
        
        if bias:
            self.bias = Tensor.zeros((out_channels,), requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the 3D convolution layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, output_depth, output_height, output_width)
        """
        return x.conv3d(
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )
    
    def __repr__(self) -> str:
        return (
            f"Conv3d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None})"
        )
