"""
Base module for neural network components.
"""

from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import uuid

from neurenix.tensor import Tensor

class Module:
    """
    Base class for all neural network modules.
    
    This is similar to nn.Module in PyTorch, providing a way to organize
    parameters and submodules in a hierarchical structure.
    """
    
    def __init__(self):
        """Initialize a new module."""
        self._parameters: Dict[str, Tensor] = {}
        self._modules: Dict[str, "Module"] = {}
        self._training = True
        self._id = str(uuid.uuid4())
    
    def __call__(self, *args, **kwargs) -> Any:
        """Call the module as a function."""
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass of the module.
        
        This method should be overridden by all subclasses.
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def register_parameter(self, name: str, param: Optional[Tensor]) -> None:
        """
        Register a parameter with the module.
        
        Args:
            name: The name of the parameter.
            param: The parameter tensor, or None to remove the parameter.
        """
        if param is None:
            self._parameters.pop(name, None)
        else:
            self._parameters[name] = param
    
    def register_module(self, name: str, module: Optional["Module"]) -> None:
        """
        Register a submodule with the module.
        
        Args:
            name: The name of the submodule.
            module: The submodule, or None to remove the submodule.
        """
        if module is None:
            self._modules.pop(name, None)
        else:
            self._modules[name] = module
    
    def parameters(self) -> List[Tensor]:
        """
        Get all parameters of the module and its submodules.
        
        Returns:
            A list of all parameter tensors.
        """
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params
    
    def train(self, mode: bool = True) -> "Module":
        """
        Set the module in training mode.
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False).
            
        Returns:
            The module itself.
        """
        self._training = mode
        for module in self._modules.values():
            module.train(mode)
        return self
    
    def eval(self) -> "Module":
        """
        Set the module in evaluation mode.
        
        Returns:
            The module itself.
        """
        return self.train(False)
    
    def is_training(self) -> bool:
        """
        Check if the module is in training mode.
        
        Returns:
            True if the module is in training mode, False otherwise.
        """
        return self._training
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Set an attribute of the module."""
        if isinstance(value, Tensor):
            self.register_parameter(name, value)
        elif isinstance(value, Module):
            self.register_module(name, value)
        else:
            object.__setattr__(self, name, value)
    
    def __repr__(self) -> str:
        """Get a string representation of the module."""
        return f"{self.__class__.__name__}()"
