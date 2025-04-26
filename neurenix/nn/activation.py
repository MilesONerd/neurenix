"""
Activation functions for the Neurenix framework.

This module provides activation functions for neural networks.
"""

from typing import Optional

import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor

class Activation(Module):
    """
    Base class for activation functions.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the activation function.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")


class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU) activation function.
    
    Applies the rectified linear unit function element-wise:
    ReLU(x) = max(0, x)
    """
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        return x.relu(inplace=self.inplace)
    
    def __repr__(self) -> str:
        return f"ReLU(inplace={self.inplace})"


class Sigmoid(Activation):
    """
    Sigmoid activation function.
    
    Applies the sigmoid function element-wise:
    Sigmoid(x) = 1 / (1 + exp(-x))
    """
    
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()
    
    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Activation):
    """
    Hyperbolic Tangent (Tanh) activation function.
    
    Applies the hyperbolic tangent function element-wise:
    Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    """
    
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()
    
    def __repr__(self) -> str:
        return "Tanh()"


class LeakyReLU(Activation):
    """
    Leaky Rectified Linear Unit (Leaky ReLU) activation function.
    
    Applies the leaky rectified linear unit function element-wise:
    LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
    
    Args:
        negative_slope: Controls the angle of the negative slope (default: 0.01)
        inplace: If True, does the operation in-place (default: False)
    """
    
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        return x.leaky_relu(self.negative_slope, inplace=self.inplace)
    
    def __repr__(self) -> str:
        return f"LeakyReLU(negative_slope={self.negative_slope}, inplace={self.inplace})"


class ELU(Activation):
    """
    Exponential Linear Unit (ELU) activation function.
    
    Applies the exponential linear unit function element-wise:
    ELU(x) = max(0, x) + min(0, alpha * (exp(x) - 1))
    
    Args:
        alpha: Controls the value to which an ELU saturates for negative inputs (default: 1.0)
        inplace: If True, does the operation in-place (default: False)
    """
    
    def __init__(self, alpha: float = 1.0, inplace: bool = False):
        super().__init__()
        self.alpha = alpha
        self.inplace = inplace
    
    def forward(self, x: Tensor) -> Tensor:
        return x.elu(self.alpha, inplace=self.inplace)
    
    def __repr__(self) -> str:
        return f"ELU(alpha={self.alpha}, inplace={self.inplace})"


class SELU(Activation):
    """
    Scaled Exponential Linear Unit (SELU) activation function.
    
    Applies the scaled exponential linear unit function element-wise:
    SELU(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
    
    The values of scale and alpha are fixed to ensure self-normalizing properties.
    
    Args:
        inplace: If True, does the operation in-place (default: False)
    """
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.scale = 1.0507009873554804934193349852946
        self.alpha = 1.6732632423543772848170429916717
    
    def forward(self, x: Tensor) -> Tensor:
        return x.selu(inplace=self.inplace)
    
    def __repr__(self) -> str:
        return f"SELU(inplace={self.inplace})"


class Softmax(Activation):
    """
    Softmax activation function.
    
    Applies the softmax function element-wise:
    Softmax(x_i) = exp(x_i) / sum_j(exp(x_j))
    
    Args:
        dim: Dimension along which to apply softmax (default: -1)
    """
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        return x.softmax(dim=self.dim)
    
    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"


class LogSoftmax(Activation):
    """
    LogSoftmax activation function.
    
    Applies the log softmax function element-wise:
    LogSoftmax(x_i) = log(exp(x_i) / sum_j(exp(x_j)))
                     = x_i - log(sum_j(exp(x_j)))
    
    Args:
        dim: Dimension along which to apply log softmax (default: -1)
    """
    
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: Tensor) -> Tensor:
        return x.log_softmax(dim=self.dim)
    
    def __repr__(self) -> str:
        return f"LogSoftmax(dim={self.dim})"


class GELU(Activation):
    """
    Gaussian Error Linear Unit (GELU) activation function.
    
    Applies the GELU function element-wise:
    GELU(x) = x * Φ(x)
    
    where Φ(x) is the cumulative distribution function of the standard normal distribution.
    
    Args:
        approximate: If True, use an approximation of the GELU function (default: False)
    """
    
    def __init__(self, approximate: bool = False):
        super().__init__()
        self.approximate = approximate
    
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu(approximate=self.approximate)
    
    def __repr__(self) -> str:
        return f"GELU(approximate={self.approximate})"
