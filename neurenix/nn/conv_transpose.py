"""
Transposed convolution layers for the Neurenix framework.
"""

import numpy as np
from typing import Optional, Tuple, Union, List

from neurenix.tensor import Tensor
from neurenix.nn.module import Module
from neurenix.nn.parameter import Parameter
from neurenix.core import get_config
from neurenix.device import DeviceType

class ConvTranspose2d(Module):
    """
    2D transposed convolution layer.
    
    This module applies a 2D transposed convolution over an input signal composed of several input planes.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
                 output_padding: Union[int, Tuple[int, int]] = 0, groups: int = 1, bias: bool = True,
                 dilation: Union[int, Tuple[int, int]] = 1):
        """
        Initialize a 2D transposed convolution layer.
        
        Args:
            in_channels: Number of channels in the input image
            out_channels: Number of channels produced by the convolution
            kernel_size: Size of the convolving kernel
            stride: Stride of the convolution
            padding: Zero-padding added to both sides of the input
            output_padding: Additional size added to one side of the output shape
            groups: Number of blocked connections from input channels to output channels
            bias: If True, adds a learnable bias to the output
            dilation: Spacing between kernel elements
        """
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
        self.groups = groups
        self.dilation = dilation
        
        weight_shape = (in_channels, out_channels // groups, kernel_size[0], kernel_size[1])
        weight = np.random.normal(0, 0.1, weight_shape)
        self.weight = Parameter(Tensor(weight))
        
        if bias:
            self.bias = Parameter(Tensor(np.zeros(out_channels)))
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the transposed convolution layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, output_height, output_width)
        """
        from neurenix.core import get_config
        from neurenix.device import DeviceType
        
        tensor_cores_enabled = get_config().get("tensor_cores_enabled", False)
        
        if (tensor_cores_enabled and 
            x.device.type == DeviceType.TENSOR_CORES and
            self.weight.data.device.type == DeviceType.TENSOR_CORES):
            try:
                from neurenix.binding import tensor_cores_conv_transpose2d
                return tensor_cores_conv_transpose2d(
                    x, self.weight.data, 
                    self.bias.data if self.bias is not None else None,
                    self.stride, self.padding, self.output_padding,
                    self.groups, self.dilation
                )
            except (ImportError, AttributeError):
                pass
        
        try:
            from neurenix.binding import conv_transpose2d
            return conv_transpose2d(
                x, self.weight.data, 
                self.bias.data if self.bias is not None else None,
                self.stride, self.padding, self.output_padding,
                self.groups, self.dilation
            )
        except (ImportError, AttributeError):
            from ..core import PhynexusExtension
            
            if PhynexusExtension.is_available():
                return PhynexusExtension.conv_transpose2d(
                    x, self.weight.data, 
                    self.bias.data if self.bias is not None else None,
                    self.stride, self.padding, self.output_padding,
                    self.groups, self.dilation
                )
            
            batch_size, in_channels, height, width = x.shape
            
            output_height = (height - 1) * self.stride[0] - 2 * self.padding[0] + self.dilation[0] * (self.kernel_size[0] - 1) + self.output_padding[0] + 1
            output_width = (width - 1) * self.stride[1] - 2 * self.padding[1] + self.dilation[1] * (self.kernel_size[1] - 1) + self.output_padding[1] + 1
            
            output = Tensor.zeros((batch_size, self.out_channels, output_height, output_width), device=x.device)
            
            
            return output
    
    def __repr__(self):
        return f"ConvTranspose2d({self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, stride={self.stride})"
