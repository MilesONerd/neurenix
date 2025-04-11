"""
Upsampling layers for the Neurenix framework.
"""

import numpy as np
from typing import Optional, Tuple, Union, List

from neurenix.tensor import Tensor
from neurenix.nn.module import Module
from neurenix.core import get_config
from neurenix.device import DeviceType

class Upsample(Module):
    """
    Upsamples a given multi-channel 2D (spatial) input.
    
    The input data is assumed to be of the form minibatch x channels x height x width.
    """
    
    def __init__(self, size: Optional[Union[int, Tuple[int, int]]] = None, 
                 scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
                 mode: str = 'nearest', align_corners: Optional[bool] = None):
        """
        Initialize an upsampling layer.
        
        Args:
            size: Output spatial size
            scale_factor: Multiplier for spatial size
            mode: Upsampling algorithm: 'nearest', 'linear', 'bilinear', 'bicubic'
            align_corners: If True, the corner pixels of the input and output tensors are aligned
        """
        super().__init__()
        
        if size is None and scale_factor is None:
            raise ValueError("either size or scale_factor should be defined")
        if size is not None and scale_factor is not None:
            raise ValueError("only one of size or scale_factor should be defined")
            
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the upsampling layer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Upsampled tensor
        """
        from neurenix.core import get_config
        from neurenix.device import DeviceType
        
        tensor_cores_enabled = get_config().get("tensor_cores_enabled", False)
        
        if (tensor_cores_enabled and 
            x.device.type == DeviceType.TENSOR_CORES):
            try:
                from neurenix.binding import tensor_cores_upsample
                return tensor_cores_upsample(x, self.size, self.scale_factor, self.mode, self.align_corners)
            except (ImportError, AttributeError):
                pass
        
        try:
            from neurenix.binding import upsample
            return upsample(x, self.size, self.scale_factor, self.mode, self.align_corners)
        except (ImportError, AttributeError):
            from ..core import PhynexusExtension
            
            if PhynexusExtension.is_available():
                return PhynexusExtension.upsample(x, self.size, self.scale_factor, self.mode, self.align_corners)
            
            x_np = x.to_numpy()
            batch_size, channels, height, width = x_np.shape
            
            if self.size is not None:
                if isinstance(self.size, int):
                    out_h = self.size
                    out_w = self.size
                else:
                    out_h, out_w = self.size
            else:
                if isinstance(self.scale_factor, (int, float)):
                    out_h = int(height * self.scale_factor)
                    out_w = int(width * self.scale_factor)
                else:
                    out_h = int(height * self.scale_factor[0])
                    out_w = int(width * self.scale_factor[1])
            
            if self.mode == 'nearest':
                output = np.zeros((batch_size, channels, out_h, out_w), dtype=x_np.dtype)
                
                h_ratio = height / out_h
                w_ratio = width / out_w
                
                for b in range(batch_size):
                    for c in range(channels):
                        for h in range(out_h):
                            for w in range(out_w):
                                in_h = min(height - 1, int(h * h_ratio))
                                in_w = min(width - 1, int(w * w_ratio))
                                output[b, c, h, w] = x_np[b, c, in_h, in_w]
                                
                return Tensor(output, device=x.device)
            
            elif self.mode == 'bilinear' or self.mode == 'linear':
                output = np.zeros((batch_size, channels, out_h, out_w), dtype=x_np.dtype)
                
                h_ratio = (height - 1) / (out_h - 1) if self.align_corners else height / out_h
                w_ratio = (width - 1) / (out_w - 1) if self.align_corners else width / out_w
                
                for b in range(batch_size):
                    for c in range(channels):
                        for h in range(out_h):
                            for w in range(out_w):
                                if self.align_corners and out_h > 1:
                                    in_h = h * h_ratio
                                else:
                                    in_h = (h + 0.5) * h_ratio - 0.5
                                    
                                if self.align_corners and out_w > 1:
                                    in_w = w * w_ratio
                                else:
                                    in_w = (w + 0.5) * w_ratio - 0.5
                                
                                in_h_int = int(in_h)
                                in_w_int = int(in_w)
                                
                                if in_h_int >= height - 1:
                                    output[b, c, h, w] = x_np[b, c, height - 1, in_w_int]
                                    continue
                                    
                                if in_w_int >= width - 1:
                                    output[b, c, h, w] = x_np[b, c, in_h_int, width - 1]
                                    continue
                                
                                h_weight = in_h - in_h_int
                                w_weight = in_w - in_w_int
                                
                                output[b, c, h, w] = (
                                    x_np[b, c, in_h_int, in_w_int] * (1 - h_weight) * (1 - w_weight) +
                                    x_np[b, c, in_h_int + 1, in_w_int] * h_weight * (1 - w_weight) +
                                    x_np[b, c, in_h_int, in_w_int + 1] * (1 - h_weight) * w_weight +
                                    x_np[b, c, in_h_int + 1, in_w_int + 1] * h_weight * w_weight
                                )
                                
                return Tensor(output, device=x.device)
            
            else:
                raise ValueError(f"Unsupported upsampling mode: {self.mode}")
    
    def __repr__(self):
        if self.size is not None:
            return f"Upsample(size={self.size}, mode='{self.mode}')"
        else:
            return f"Upsample(scale_factor={self.scale_factor}, mode='{self.mode}')"
