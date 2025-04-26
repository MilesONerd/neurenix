"""
Dropout layers for neural networks.
"""

from typing import Optional
import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor

class Dropout(Module):
    """
    Dropout layer.
    
    During training, randomly zeroes some of the elements of the input tensor with probability p.
    """
    
    def __init__(self, p: float = 0.5, inplace: bool = False):
        """
        Initialize a dropout layer.
        
        Args:
            p: Probability of an element to be zeroed. Default: 0.5
            inplace: If True, will do this operation in-place. Default: False
        """
        super().__init__()
        
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability has to be between 0 and 1, got {p}")
        
        self.p = p
        self.inplace = inplace
        self.mask = None
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the dropout layer.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        if not self.is_training() or self.p == 0:
            return x
        
        mask = np.random.binomial(1, 1 - self.p, x.shape).astype(np.float32) / (1 - self.p)
        self.mask = Tensor(mask, device=x.device)
        
        if self.inplace:
            x.data = x.data * self.mask.data
            return x
        else:
            return Tensor(x.data * self.mask.data, device=x.device)
    
    def __repr__(self):
        return f"Dropout(p={self.p}, inplace={self.inplace})"
