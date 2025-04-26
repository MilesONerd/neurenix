"""
Pooling layers for neural networks.
"""

from typing import Tuple, Union, List, Optional
import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor

class MaxPool2d(Module):
    """
    2D max pooling layer.
    
    Applies a 2D max pooling over an input signal composed of several input planes.
    """
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], 
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 return_indices: bool = False,
                 ceil_mode: bool = False):
        """
        Initialize a 2D max pooling layer.
        
        Args:
            kernel_size: Size of the pooling window
            stride: Stride of the pooling window. Default: kernel_size
            padding: Padding added to both sides of the input. Default: 0
            dilation: Controls the spacing between kernel elements. Default: 1
            return_indices: If True, will return the indices along with the outputs
            ceil_mode: If True, will use ceil instead of floor to compute the output shape
        """
        super().__init__()
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        
        self.register_buffer('indices', None)
    
    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass of the max pooling layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, channels, output_height, output_width)
            If return_indices is True, also returns the indices of the max values
        """
        batch_size, channels, height, width = x.shape
        
        if self.ceil_mode:
            output_height = int(np.ceil((height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
            output_width = int(np.ceil((width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        else:
            output_height = int(np.floor((height + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) / self.stride[0] + 1))
            output_width = int(np.floor((width + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) / self.stride[1] + 1))
        
        output = Tensor.zeros((batch_size, channels, output_height, output_width), device=x.device)
        
        if self.return_indices:
            indices = Tensor.zeros((batch_size, channels, output_height, output_width), device=x.device)
        
        x_np = x.to_numpy()
        output_np = np.zeros((batch_size, channels, output_height, output_width), dtype=np.float32)
        indices_np = np.zeros((batch_size, channels, output_height, output_width), dtype=np.int64)
        
        if self.padding[0] > 0 or self.padding[1] > 0:
            x_padded = np.pad(x_np, ((0, 0), (0, 0), (self.padding[0], self.padding[0]), (self.padding[1], self.padding[1])), mode='constant')
        else:
            x_padded = x_np
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(output_height):
                    for w in range(output_width):
                        h_start = h * self.stride[0]
                        h_end = h_start + self.kernel_size[0]
                        w_start = w * self.stride[0]
                        w_end = w_start + self.kernel_size[1]
                        
                        window = x_padded[b, c, h_start:h_end, w_start:w_end]
                        
                        max_val = np.max(window)
                        max_idx = np.argmax(window)
                        
                        output_np[b, c, h, w] = max_val
                        indices_np[b, c, h, w] = max_idx
        
        output = Tensor(output_np, device=x.device)
        
        if self.return_indices:
            indices = Tensor(indices_np, device=x.device)
            self.indices = indices
            return output, indices
        else:
            return output
    
    def __repr__(self):
        return f"MaxPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
