"""
Parameter class for neural network modules.
"""

from typing import Optional, Union, Tuple, List
import numpy as np

from neurenix.tensor import Tensor

class Parameter(Tensor):
    """
    A kind of Tensor that is to be considered a module parameter.
    
    Parameters are Tensor subclasses, that have a very special property when used with Module s - 
    when they're assigned as Module attributes they are automatically added to the list of its parameters.
    """
    
    def __init__(self, data, requires_grad=True, device=None):
        """
        Initialize a parameter.
        
        Args:
            data: Parameter data (tensor, numpy array, or Python sequence)
            requires_grad: Whether the parameter requires gradient
            device: Device to store the parameter on
        """
        if isinstance(data, Tensor):
            super().__init__(data.to_numpy(), device=device or data.device)
        else:
            super().__init__(data, device=device)
        
        self.requires_grad = requires_grad
        self.grad = None if requires_grad else None
    
    def __repr__(self):
        return f"Parameter(shape={self.shape}, requires_grad={self.requires_grad})"
