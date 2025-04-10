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
        from neurenix.core import get_config
        from neurenix.device import DeviceType
        
        tensor_cores_enabled = get_config().get("tensor_cores_enabled", False)
        
        if (tensor_cores_enabled and 
            x.device.type == DeviceType.TENSOR_CORES):
            try:
                from neurenix.binding import tensor_cores_max_pool2d
                if self.return_indices:
                    output, indices = tensor_cores_max_pool2d(
                        x, self.kernel_size, self.stride, self.padding, 
                        self.dilation, self.return_indices, self.ceil_mode
                    )
                    self.indices = indices
                    return output, indices
                else:
                    return tensor_cores_max_pool2d(
                        x, self.kernel_size, self.stride, self.padding, 
                        self.dilation, self.return_indices, self.ceil_mode
                    )
            except (ImportError, AttributeError):
                pass
        
        try:
            from neurenix.binding import max_pool2d
            if self.return_indices:
                output, indices = max_pool2d(
                    x, self.kernel_size, self.stride, self.padding, 
                    self.dilation, self.return_indices, self.ceil_mode
                )
                self.indices = indices
                return output, indices
            else:
                return max_pool2d(
                    x, self.kernel_size, self.stride, self.padding, 
                    self.dilation, self.return_indices, self.ceil_mode
                )
        except (ImportError, AttributeError):
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


class AvgPool2d(Module):
    """
    2D average pooling layer.
    
    Applies a 2D average pooling over an input signal composed of several input planes.
    """
    
    def __init__(self, kernel_size: Union[int, Tuple[int, int]], 
                 stride: Optional[Union[int, Tuple[int, int]]] = None,
                 padding: Union[int, Tuple[int, int]] = 0,
                 ceil_mode: bool = False,
                 count_include_pad: bool = True,
                 divisor_override: Optional[int] = None):
        """
        Initialize a 2D average pooling layer.
        
        Args:
            kernel_size: Size of the pooling window
            stride: Stride of the pooling window. Default: kernel_size
            padding: Padding added to both sides of the input. Default: 0
            ceil_mode: If True, will use ceil instead of floor to compute the output shape
            count_include_pad: If True, include padding in the averaging calculation
            divisor_override: If specified, it will be used as the divisor, otherwise size of the pooling region will be used
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
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the average pooling layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, channels, output_height, output_width)
        """
        from neurenix.core import get_config
        from neurenix.device import DeviceType
        
        tensor_cores_enabled = get_config().get("tensor_cores_enabled", False)
        
        if (tensor_cores_enabled and 
            x.device.type == DeviceType.TENSOR_CORES):
            try:
                from neurenix.binding import tensor_cores_avg_pool2d
                return tensor_cores_avg_pool2d(
                    x, self.kernel_size, self.stride, self.padding,
                    self.ceil_mode, self.count_include_pad, self.divisor_override
                )
            except (ImportError, AttributeError):
                pass
        
        try:
            from neurenix.binding import avg_pool2d
            return avg_pool2d(
                x, self.kernel_size, self.stride, self.padding,
                self.ceil_mode, self.count_include_pad, self.divisor_override
            )
        except (ImportError, AttributeError):
            from ..core import PhynexusExtension
            
            if PhynexusExtension.is_available():
                return PhynexusExtension.avg_pool2d(
                    x, self.kernel_size, self.stride, self.padding,
                    self.ceil_mode, self.count_include_pad, self.divisor_override
                )
            
            batch_size, channels, height, width = x.shape
            
            if self.ceil_mode:
                output_height = int(np.ceil((height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1))
                output_width = int(np.ceil((width + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1))
            else:
                output_height = int(np.floor((height + 2 * self.padding[0] - self.kernel_size[0]) / self.stride[0] + 1))
                output_width = int(np.floor((width + 2 * self.padding[1] - self.kernel_size[1]) / self.stride[1] + 1))
            
            output = Tensor.zeros((batch_size, channels, output_height, output_width), device=x.device)
            
            x_np = x.to_numpy()
            output_np = np.zeros((batch_size, channels, output_height, output_width), dtype=np.float32)
            
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
                            w_start = w * self.stride[1]
                            w_end = w_start + self.kernel_size[1]
                            
                            window = x_padded[b, c, h_start:h_end, w_start:w_end]
                            
                            if self.count_include_pad:
                                divisor = self.kernel_size[0] * self.kernel_size[1]
                            else:
                                h_start_orig = max(0, h_start - self.padding[0])
                                h_end_orig = min(height, h_end - self.padding[0])
                                w_start_orig = max(0, w_start - self.padding[1])
                                w_end_orig = min(width, w_end - self.padding[1])
                                divisor = (h_end_orig - h_start_orig) * (w_end_orig - w_start_orig)
                            
                            if self.divisor_override is not None:
                                divisor = self.divisor_override
                            
                            avg_val = np.sum(window) / divisor
                            output_np[b, c, h, w] = avg_val
            
            output = Tensor(output_np, device=x.device)
            return output
    
    def __repr__(self):
        return f"AvgPool2d(kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"


class AdaptiveMaxPool2d(Module):
    """
    2D adaptive max pooling layer.
    
    Applies a 2D adaptive max pooling over an input signal composed of several input planes.
    The output size is specified, and the stride and kernel size are calculated automatically.
    """
    
    def __init__(self, output_size: Union[int, Tuple[int, int]], return_indices: bool = False):
        """
        Initialize a 2D adaptive max pooling layer.
        
        Args:
            output_size: Size of the output. Can be a single integer for square output,
                         or a tuple of two integers for height x width output.
            return_indices: If True, will return the indices along with the outputs
        """
        super().__init__()
        
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
            
        self.output_size = output_size
        self.return_indices = return_indices
        
        self.register_buffer('indices', None)
    
    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass of the adaptive max pooling layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, channels, output_height, output_width)
            If return_indices is True, also returns the indices of the max values
        """
        from neurenix.core import get_config
        from neurenix.device import DeviceType
        
        tensor_cores_enabled = get_config().get("tensor_cores_enabled", False)
        
        if (tensor_cores_enabled and 
            x.device.type == DeviceType.TENSOR_CORES):
            try:
                from neurenix.binding import tensor_cores_adaptive_max_pool2d
                if self.return_indices:
                    output, indices = tensor_cores_adaptive_max_pool2d(x, self.output_size, self.return_indices)
                    self.indices = indices
                    return output, indices
                else:
                    return tensor_cores_adaptive_max_pool2d(x, self.output_size, self.return_indices)
            except (ImportError, AttributeError):
                pass
        
        try:
            from neurenix.binding import adaptive_max_pool2d
            if self.return_indices:
                output, indices = adaptive_max_pool2d(x, self.output_size, self.return_indices)
                self.indices = indices
                return output, indices
            else:
                return adaptive_max_pool2d(x, self.output_size, self.return_indices)
        except (ImportError, AttributeError):
            from ..core import PhynexusExtension
            
            if PhynexusExtension.is_available():
                if self.return_indices:
                    output, indices = PhynexusExtension.adaptive_max_pool2d(x, self.output_size, self.return_indices)
                    self.indices = indices
                    return output, indices
                else:
                    return PhynexusExtension.adaptive_max_pool2d(x, self.output_size, self.return_indices)
            
            batch_size, channels, height, width = x.shape
            output_height, output_width = self.output_size
            
            stride_h = height // output_height
            stride_w = width // output_width
            
            kernel_h = height - (output_height - 1) * stride_h
            kernel_w = width - (output_width - 1) * stride_w
            
            output = Tensor.zeros((batch_size, channels, output_height, output_width), device=x.device)
            
            if self.return_indices:
                indices = Tensor.zeros((batch_size, channels, output_height, output_width), device=x.device, dtype=Tensor.int64)
            
            x_np = x.to_numpy()
            output_np = np.zeros((batch_size, channels, output_height, output_width), dtype=np.float32)
            indices_np = np.zeros((batch_size, channels, output_height, output_width), dtype=np.int64)
            
            for b in range(batch_size):
                for c in range(channels):
                    for h in range(output_height):
                        for w in range(output_width):
                            h_start = h * stride_h
                            h_end = min(h_start + kernel_h, height)
                            w_start = w * stride_w
                            w_end = min(w_start + kernel_w, width)
                            
                            window = x_np[b, c, h_start:h_end, w_start:w_end]
                            
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
        return f"AdaptiveMaxPool2d(output_size={self.output_size})"


class AdaptiveAvgPool2d(Module):
    """
    2D adaptive average pooling layer.
    
    Applies a 2D adaptive average pooling over an input signal composed of several input planes.
    The output size is specified, and the stride and kernel size are calculated automatically.
    """
    
    def __init__(self, output_size: Union[int, Tuple[int, int]]):
        """
        Initialize a 2D adaptive average pooling layer.
        
        Args:
            output_size: Size of the output. Can be a single integer for square output,
                         or a tuple of two integers for height x width output.
        """
        super().__init__()
        
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
            
        self.output_size = output_size
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the adaptive average pooling layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, channels, output_height, output_width)
        """
        from neurenix.core import get_config
        from neurenix.device import DeviceType
        
        tensor_cores_enabled = get_config().get("tensor_cores_enabled", False)
        
        if (tensor_cores_enabled and 
            x.device.type == DeviceType.TENSOR_CORES):
            try:
                from neurenix.binding import tensor_cores_adaptive_avg_pool2d
                return tensor_cores_adaptive_avg_pool2d(x, self.output_size)
            except (ImportError, AttributeError):
                pass
        
        try:
            from neurenix.binding import adaptive_avg_pool2d
            return adaptive_avg_pool2d(x, self.output_size)
        except (ImportError, AttributeError):
            from ..core import PhynexusExtension
            
            if PhynexusExtension.is_available():
                return PhynexusExtension.adaptive_avg_pool2d(x, self.output_size)
            
            batch_size, channels, height, width = x.shape
            output_height, output_width = self.output_size
            
            stride_h = height // output_height
            stride_w = width // output_width
            
            kernel_h = height - (output_height - 1) * stride_h
            kernel_w = width - (output_width - 1) * stride_w
            
            output = Tensor.zeros((batch_size, channels, output_height, output_width), device=x.device)
            
            x_np = x.to_numpy()
            output_np = np.zeros((batch_size, channels, output_height, output_width), dtype=np.float32)
            
            for b in range(batch_size):
                for c in range(channels):
                    for h in range(output_height):
                        for w in range(output_width):
                            h_start = h * stride_h
                            h_end = min(h_start + kernel_h, height)
                            w_start = w * stride_w
                            w_end = min(w_start + kernel_w, width)
                            
                            window = x_np[b, c, h_start:h_end, w_start:w_end]
                            
                            avg_val = np.mean(window)
                            output_np[b, c, h, w] = avg_val
            
            output = Tensor(output_np, device=x.device)
            return output
    
    def __repr__(self):
        return f"AdaptiveAvgPool2d(output_size={self.output_size})"
