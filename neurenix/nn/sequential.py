"""
Sequential container for the Neurenix framework.

This module provides a sequential container for neural network modules.
"""

from typing import List, Tuple, Dict, Any, Optional, Union, Sequence

from neurenix.nn.module import Module
from neurenix.tensor import Tensor

class Sequential(Module):
    """
    A sequential container for neural network modules.
    
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can be passed in.
    
    Example:
        # Using Sequential with a list of modules
        model = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 30)
        )
        
        # Using Sequential with named modules
        model = Sequential({
            'fc1': Linear(10, 20),
            'relu': ReLU(),
            'fc2': Linear(20, 30)
        })
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._modules = {}
        
        if len(args) == 1 and isinstance(args[0], dict):
            # Initialize from an ordered dict
            for name, module in args[0].items():
                self.add_module(name, module)
        else:
            # Initialize from a list of modules
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        
        # Add any keyword arguments as named modules
        for name, module in kwargs.items():
            self.add_module(name, module)
            
    def add_module(self, name: str, module: Module) -> None:
        """
        Add a module to the sequential container.
        
        Args:
            name: Name of the module
            module: Module to add
        """
        self._modules[name] = module
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the sequential container.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        for module in self._modules.values():
            x = module(x)
        
        return x
    
    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, 'Sequential']:
        """
        Get a module or a slice of modules.
        
        Args:
            idx: Index or slice
            
        Returns:
            Module or Sequential container with the selected modules
        """
        if isinstance(idx, slice):
            # Get a slice of modules
            modules = list(self._modules.values())[idx]
            return Sequential(*modules)
        else:
            # Get a single module
            if isinstance(idx, int):
                if idx < 0:
                    idx = len(self) + idx
                
                if idx < 0 or idx >= len(self):
                    raise IndexError(f"Index {idx} out of range for Sequential with {len(self)} modules")
                
                return list(self._modules.values())[idx]
            else:
                # Get a module by name
                return self._modules[idx]
    
    def __len__(self) -> int:
        """
        Get the number of modules in the sequential container.
        
        Returns:
            Number of modules
        """
        return len(self._modules)
    
    def __iter__(self):
        """
        Iterate over the modules in the sequential container.
        
        Returns:
            Iterator over modules
        """
        return iter(self._modules.values())
    
    def __repr__(self) -> str:
        """
        Get a string representation of the sequential container.
        
        Returns:
            String representation
        """
        module_str = ",\n  ".join(str(module).replace("\n", "\n  ") for module in self)
        return f"Sequential(\n  {module_str}\n)"
