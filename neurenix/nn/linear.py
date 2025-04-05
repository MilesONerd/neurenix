"""
Linear (fully connected) layer implementation.
"""

from typing import Optional, Tuple, Union

import numpy as np

from neurenix.tensor import Tensor, DType
from neurenix.device import Device
from neurenix.nn.module import Module

class Linear(Module):
    """
    Linear transformation: y = xW^T + b
    
    This is equivalent to a fully connected layer in a neural network.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: Optional[DType] = None,
        device: Optional[Device] = None,
    ):
        """
        Initialize a linear layer.
        
        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If True, adds a learnable bias to the output.
            dtype: Data type of the parameters.
            device: Device to store the parameters on.
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using Kaiming initialization
        weight_data = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        self.weight = Tensor(weight_data, dtype=dtype, device=device, requires_grad=True)
        
        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Tensor(bias_data, dtype=dtype, device=device, requires_grad=True)
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the linear layer.
        
        Args:
            x: Input tensor of shape (..., in_features).
            
        Returns:
            Output tensor of shape (..., out_features).
        """
        try:
            from neurenix.binding import linear
            return linear(x, self.weight, self.bias)
        except (ImportError, AttributeError):
            x_np = x.numpy()
            weight_np = self.weight.numpy()
            
            # Reshape input for matrix multiplication
            orig_shape = x_np.shape
            x_reshaped = x_np.reshape(-1, self.in_features)
            
            # Perform the linear transformation
            output = np.matmul(x_reshaped, weight_np.T)
            
            # Add bias if present
            if self.bias is not None:
                output += self.bias.numpy()
            
            # Reshape output to match input shape
            output = output.reshape(*orig_shape[:-1], self.out_features)
            
            # Create a new tensor from the output
            return Tensor(output, device=x.device)
    
    def __repr__(self) -> str:
        """Get a string representation of the linear layer."""
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
