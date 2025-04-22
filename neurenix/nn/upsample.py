"""
Upsampling layers for neural networks.
"""

from typing import Optional, Union, Tuple
import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor
import neurenix.binding as binding

class Upsample(Module):
    """
    Upsamples a given multi-channel 1D (temporal), 2D (spatial) or 3D (volumetric) data.
    
    The input data is assumed to be of the form minibatch x channels x [optional depth] x [optional height] x width.
    The modes available for upsampling are: nearest, linear, bilinear, bicubic, and trilinear.
    """
    
    def __init__(
        self,
        size: Optional[Union[int, Tuple[int, ...], list]] = None,
        scale_factor: Optional[Union[float, Tuple[float, ...], list]] = None,
        mode: str = 'nearest',
        align_corners: Optional[bool] = None
    ):
        """
        Initialize an Upsample layer.
        
        Args:
            size: Output spatial size. If size is an int, it is treated as a 1D upsampling.
                 If it has 2 elements, it is treated as 2D (height, width).
                 If it has 3 elements, it is treated as 3D (depth, height, width).
            scale_factor: Multiplier for spatial size. If scale_factor is a tuple,
                         it is treated as different scales for different dimensions.
            mode: Algorithm used for upsampling: 'nearest', 'linear', 'bilinear',
                 'bicubic', or 'trilinear'. Default: 'nearest'
            align_corners: If True, the corner pixels of the input and output tensors are aligned,
                          and thus preserving the values at the corner pixels. Default: None
        """
        super().__init__()
        
        if size is None and scale_factor is None:
            raise ValueError("Either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("Only one of size or scale_factor should be defined")
        
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
        valid_modes = ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']
        if mode not in valid_modes:
            raise ValueError(f"Mode '{mode}' is not supported. Valid modes are: {valid_modes}")
        
        if align_corners is not None and mode == 'nearest':
            raise ValueError("align_corners option can only be set with the interpolating modes: "
                            "linear, bilinear, bicubic, or trilinear")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the upsampling layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, [depth], [height], width)
            
        Returns:
            Upsampled tensor
        """
        try:
            return binding.upsample(
                x, self.size, self.scale_factor,
                self.mode, self.align_corners
            )
        except (ImportError, AttributeError):
            return self._python_forward(x)
    
    def _python_forward(self, x: Tensor) -> Tensor:
        """
        Pure Python implementation of upsampling as a fallback.
        
        Args:
            x: Input tensor of shape (batch_size, channels, [depth], [height], width)
            
        Returns:
            Upsampled tensor
        """
        input_dim = x.dim() - 2  # Subtract batch and channel dimensions
        
        if self.size is not None:
            output_size = self.size
            if isinstance(output_size, int):
                output_size = [output_size] * input_dim
        else:  # self.scale_factor is not None
            scale_factor = self.scale_factor
            if isinstance(scale_factor, (int, float)):
                scale_factor = [scale_factor] * input_dim
            
            input_size = x.shape[2:] if input_dim > 0 else []
            output_size = [int(s * f) for s, f in zip(input_size, scale_factor)]
        
        if input_dim == 1:
            return self._upsample_1d(x, output_size[0])
        elif input_dim == 2:
            return self._upsample_2d(x, output_size[0], output_size[1])
        elif input_dim == 3:
            return self._upsample_3d(x, output_size[0], output_size[1], output_size[2])
        else:
            raise ValueError(f"Unsupported input dimension: {input_dim}")
    
    def _upsample_1d(self, x: Tensor, output_width: int) -> Tensor:
        """
        Upsample a 1D tensor.
        
        Args:
            x: Input tensor of shape (batch_size, channels, width)
            output_width: Target width
            
        Returns:
            Upsampled tensor of shape (batch_size, channels, output_width)
        """
        batch_size, channels, width = x.shape
        x_np = x.to_numpy()
        
        if self.mode == 'nearest':
            scale = output_width / width
            output = np.zeros((batch_size, channels, output_width), dtype=np.float32)
            
            for b in range(batch_size):
                for c in range(channels):
                    for w in range(output_width):
                        src_w = min(int(w / scale), width - 1)
                        output[b, c, w] = x_np[b, c, src_w]
        else:
            scale = (width - 1) / (output_width - 1) if self.align_corners else width / output_width
            output = np.zeros((batch_size, channels, output_width), dtype=np.float32)
            
            for b in range(batch_size):
                for c in range(channels):
                    for w in range(output_width):
                        if self.align_corners:
                            src_w = scale * w
                        else:
                            src_w = scale * (w + 0.5) - 0.5
                        
                        src_w = max(0, min(width - 1, src_w))
                        w0 = int(src_w)
                        w1 = min(w0 + 1, width - 1)
                        dw = src_w - w0
                        
                        output[b, c, w] = (1 - dw) * x_np[b, c, w0] + dw * x_np[b, c, w1]
        
        return Tensor(output, device=x.device)
    
    def _upsample_2d(self, x: Tensor, output_height: int, output_width: int) -> Tensor:
        """
        Upsample a 2D tensor.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            output_height: Target height
            output_width: Target width
            
        Returns:
            Upsampled tensor of shape (batch_size, channels, output_height, output_width)
        """
        batch_size, channels, height, width = x.shape
        x_np = x.to_numpy()
        
        if self.mode == 'nearest':
            scale_h = output_height / height
            scale_w = output_width / width
            output = np.zeros((batch_size, channels, output_height, output_width), dtype=np.float32)
            
            for b in range(batch_size):
                for c in range(channels):
                    for h in range(output_height):
                        src_h = min(int(h / scale_h), height - 1)
                        for w in range(output_width):
                            src_w = min(int(w / scale_w), width - 1)
                            output[b, c, h, w] = x_np[b, c, src_h, src_w]
        elif self.mode in ['linear', 'bilinear']:
            if self.align_corners:
                scale_h = (height - 1) / (output_height - 1)
                scale_w = (width - 1) / (output_width - 1)
            else:
                scale_h = height / output_height
                scale_w = width / output_width
            
            output = np.zeros((batch_size, channels, output_height, output_width), dtype=np.float32)
            
            for b in range(batch_size):
                for c in range(channels):
                    for h in range(output_height):
                        if self.align_corners:
                            src_h = scale_h * h
                        else:
                            src_h = scale_h * (h + 0.5) - 0.5
                        
                        src_h = max(0, min(height - 1, src_h))
                        h0 = int(src_h)
                        h1 = min(h0 + 1, height - 1)
                        dh = src_h - h0
                        
                        for w in range(output_width):
                            if self.align_corners:
                                src_w = scale_w * w
                            else:
                                src_w = scale_w * (w + 0.5) - 0.5
                            
                            src_w = max(0, min(width - 1, src_w))
                            w0 = int(src_w)
                            w1 = min(w0 + 1, width - 1)
                            dw = src_w - w0
                            
                            output[b, c, h, w] = (1 - dh) * (1 - dw) * x_np[b, c, h0, w0] + \
                                                (1 - dh) * dw * x_np[b, c, h0, w1] + \
                                                dh * (1 - dw) * x_np[b, c, h1, w0] + \
                                                dh * dw * x_np[b, c, h1, w1]
        else:
            if self.align_corners:
                scale_h = (height - 1) / (output_height - 1)
                scale_w = (width - 1) / (output_width - 1)
            else:
                scale_h = height / output_height
                scale_w = width / output_width
            
            output = np.zeros((batch_size, channels, output_height, output_width), dtype=np.float32)
            
            for b in range(batch_size):
                for c in range(channels):
                    for h in range(output_height):
                        if self.align_corners:
                            src_h = scale_h * h
                        else:
                            src_h = scale_h * (h + 0.5) - 0.5
                        
                        src_h = max(0, min(height - 1, src_h))
                        h0 = int(src_h)
                        h1 = min(h0 + 1, height - 1)
                        dh = src_h - h0
                        
                        for w in range(output_width):
                            if self.align_corners:
                                src_w = scale_w * w
                            else:
                                src_w = scale_w * (w + 0.5) - 0.5
                            
                            src_w = max(0, min(width - 1, src_w))
                            w0 = int(src_w)
                            w1 = min(w0 + 1, width - 1)
                            dw = src_w - w0
                            
                            output[b, c, h, w] = (1 - dh) * (1 - dw) * x_np[b, c, h0, w0] + \
                                                (1 - dh) * dw * x_np[b, c, h0, w1] + \
                                                dh * (1 - dw) * x_np[b, c, h1, w0] + \
                                                dh * dw * x_np[b, c, h1, w1]
        
        return Tensor(output, device=x.device)
    
    def _upsample_3d(self, x: Tensor, output_depth: int, output_height: int, output_width: int) -> Tensor:
        """
        Upsample a 3D tensor.
        
        Args:
            x: Input tensor of shape (batch_size, channels, depth, height, width)
            output_depth: Target depth
            output_height: Target height
            output_width: Target width
            
        Returns:
            Upsampled tensor of shape (batch_size, channels, output_depth, output_height, output_width)
        """
        batch_size, channels, depth, height, width = x.shape
        x_np = x.to_numpy()
        
        if self.mode == 'nearest':
            scale_d = output_depth / depth
            scale_h = output_height / height
            scale_w = output_width / width
            output = np.zeros((batch_size, channels, output_depth, output_height, output_width), dtype=np.float32)
            
            for b in range(batch_size):
                for c in range(channels):
                    for d in range(output_depth):
                        src_d = min(int(d / scale_d), depth - 1)
                        for h in range(output_height):
                            src_h = min(int(h / scale_h), height - 1)
                            for w in range(output_width):
                                src_w = min(int(w / scale_w), width - 1)
                                output[b, c, d, h, w] = x_np[b, c, src_d, src_h, src_w]
        elif self.mode in ['linear', 'trilinear']:
            if self.align_corners:
                scale_d = (depth - 1) / (output_depth - 1)
                scale_h = (height - 1) / (output_height - 1)
                scale_w = (width - 1) / (output_width - 1)
            else:
                scale_d = depth / output_depth
                scale_h = height / output_height
                scale_w = width / output_width
            
            output = np.zeros((batch_size, channels, output_depth, output_height, output_width), dtype=np.float32)
            
            for b in range(batch_size):
                for c in range(channels):
                    for d in range(output_depth):
                        if self.align_corners:
                            src_d = scale_d * d
                        else:
                            src_d = scale_d * (d + 0.5) - 0.5
                        
                        src_d = max(0, min(depth - 1, src_d))
                        d0 = int(src_d)
                        d1 = min(d0 + 1, depth - 1)
                        dd = src_d - d0
                        
                        for h in range(output_height):
                            if self.align_corners:
                                src_h = scale_h * h
                            else:
                                src_h = scale_h * (h + 0.5) - 0.5
                            
                            src_h = max(0, min(height - 1, src_h))
                            h0 = int(src_h)
                            h1 = min(h0 + 1, height - 1)
                            dh = src_h - h0
                            
                            for w in range(output_width):
                                if self.align_corners:
                                    src_w = scale_w * w
                                else:
                                    src_w = scale_w * (w + 0.5) - 0.5
                                
                                src_w = max(0, min(width - 1, src_w))
                                w0 = int(src_w)
                                w1 = min(w0 + 1, width - 1)
                                dw = src_w - w0
                                
                                c000 = x_np[b, c, d0, h0, w0]
                                c001 = x_np[b, c, d0, h0, w1]
                                c010 = x_np[b, c, d0, h1, w0]
                                c011 = x_np[b, c, d0, h1, w1]
                                c100 = x_np[b, c, d1, h0, w0]
                                c101 = x_np[b, c, d1, h0, w1]
                                c110 = x_np[b, c, d1, h1, w0]
                                c111 = x_np[b, c, d1, h1, w1]
                                
                                c00 = c000 * (1 - dw) + c001 * dw
                                c01 = c010 * (1 - dw) + c011 * dw
                                c10 = c100 * (1 - dw) + c101 * dw
                                c11 = c110 * (1 - dw) + c111 * dw
                                
                                c0 = c00 * (1 - dh) + c01 * dh
                                c1 = c10 * (1 - dh) + c11 * dh
                                
                                output[b, c, d, h, w] = c0 * (1 - dd) + c1 * dd
        
        return Tensor(output, device=x.device)
    
    def extra_repr(self) -> str:
        """Return a string containing extra information about the module."""
        if self.scale_factor is not None:
            info = f"scale_factor={self.scale_factor}"
        else:
            info = f"size={self.size}"
        
        info += f", mode='{self.mode}'"
        if self.align_corners is not None:
            info += f", align_corners={self.align_corners}"
        
        return info
