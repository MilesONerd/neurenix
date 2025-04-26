"""
Base optimizer class for the Neurenix framework.
"""

from typing import List, Dict, Any, Optional, Union, Callable, Iterable

from neurenix.tensor import Tensor

class Optimizer:
    """
    Base class for all optimizers.
    
    This is similar to optimizers in other frameworks like PyTorch,
    providing a way to update model parameters based on gradients.
    """
    
    def __init__(self, params: Iterable[Tensor], defaults: Dict[str, Any]):
        """
        Initialize a new optimizer.
        
        Args:
            params: An iterable of tensors to optimize.
            defaults: Default hyperparameters for the optimizer.
        """
        self.defaults = defaults
        self._parameter_groups = [{"params": list(params), **defaults}]
        self.state: Dict[Tensor, Dict[str, Any]] = {}
    
    def zero_grad(self) -> None:
        """
        Reset the gradients of all optimized tensors.
        
        This should be called before computing gradients for a new batch.
        """
        for group in self._parameter_groups:
            for param in group["params"]:
                if param.grad is not None:
                    try:
                        from neurenix.binding import zero_grad
                        zero_grad(param)
                    except (ImportError, AttributeError):
                        param._grad._numpy_data.fill(0)
    
    def step(self) -> None:
        """
        Update the parameters based on the current gradients.
        
        This should be called after computing gradients.
        """
        raise NotImplementedError("Subclasses must implement step()")
    
    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """
        Add a parameter group to the optimizer.
        
        This is useful for adding parameters with different hyperparameters.
        
        Args:
            param_group: A dictionary containing parameters and hyperparameters.
        """
        assert "params" in param_group, "param_group must contain 'params' key"
        assert isinstance(param_group["params"], Iterable), "params must be an iterable"
        
        # Add default values for missing hyperparameters
        for key, value in self.defaults.items():
            if key not in param_group:
                param_group[key] = value
        
        # Convert params to a list if it's not already
        if not isinstance(param_group["params"], list):
            param_group["params"] = list(param_group["params"])
        
        self._parameter_groups.append(param_group)
