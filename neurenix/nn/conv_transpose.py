"""
Transposed convolution layers for the Neurenix framework.
"""

import numpy as np
from typing import Tuple, Union, Optional

from neurenix.nn.module import Module
from neurenix.tensor import Tensor
import neurenix.binding as binding

class ConvTranspose2d(Module):
    """
    Applies a 2D transposed convolution over an input signal composed of several input planes.
    
    This module can be seen as the gradient of Conv2d with respect to its input.
    It is also known as a fractionally-strided convolution or a deconvolution.
    
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        output_padding (int or tuple, optional): Additional size added to one side of the output. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input to output channels. Default: 1
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        output_padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True
    ):
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = bias
        
        weight_shape = (in_channels, out_channels // groups, kernel_size[0], kernel_size[1])
        self.weight = Tensor(np.random.randn(*weight_shape) * 0.01)
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels))
        else:
            self.bias = None
            
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters using Kaiming initialization."""
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        std = np.sqrt(2.0 / fan_in)
        self.weight.data = np.random.normal(0, std, self.weight.shape)
        
        if self.bias is not None:
            self.bias.data = np.zeros(self.out_channels)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the transposed convolution operation.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Tensor: Output tensor of shape (batch_size, out_channels, output_height, output_width)
        """
        try:
            return binding.conv_transpose2d(
                x, self.weight, self.bias,
                self.stride, self.padding, self.output_padding, self.dilation, self.groups
            )
        except (ImportError, AttributeError):
            return self._python_forward(x)
    
    def _python_forward(self, x: Tensor) -> Tensor:
        """
        Pure Python implementation of transposed convolution as a fallback.
        
        Args:
            x (Tensor): Input tensor
            
        Returns:
            Tensor: Output tensor
        """
        batch_size, in_channels, in_height, in_width = x.shape
        
        out_height = (in_height - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1
        out_width = (in_width - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (self.kernel_size[1] - 1) + self.output_padding[1] + 1
        
        output = Tensor(np.zeros((batch_size, self.out_channels, out_height, out_width)))
        
        for b in range(batch_size):
            for c_out in range(self.out_channels):
                for c_in in range(self.in_channels):
                    for h_in in range(in_height):
                        for w_in in range(in_width):
                            for kh in range(self.kernel_size[0]):
                                for kw in range(self.kernel_size[1]):
                                    h_out = h_in * self.stride[0] + kh * self.dilation[0] - self.padding[0]
                                    w_out = w_in * self.stride[1] + kw * self.dilation[1] - self.padding[1]
                                    
                                    if 0 <= h_out < out_height and 0 <= w_out < out_width:
                                        output.data[b, c_out, h_out, w_out] += (
                                            x.data[b, c_in, h_in, w_in] * 
                                            self.weight.data[c_in, c_out // self.groups, kh, kw]
                                        )
        
        if self.bias is not None:
            for c_out in range(self.out_channels):
                output.data[:, c_out, :, :] += self.bias.data[c_out]
                
        return output
    
    def __repr__(self):
        return (
            f"ConvTranspose2d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, output_padding={self.output_padding}, "
            f"dilation={self.dilation}, groups={self.groups}, bias={self.use_bias})"
        )
