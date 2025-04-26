"""
Transfer learning model implementation for the Neurenix framework.
"""

from typing import List, Dict, Any, Optional, Union, Callable

import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor
from neurenix.device import Device

class TransferModel(Module):
    """
    A model for transfer learning that combines a pre-trained base model with new task-specific layers.
    
    This class makes it easy to perform transfer learning by:
    1. Loading a pre-trained model
    2. Optionally freezing some or all of its layers
    3. Adding new layers for the target task
    4. Training only the new layers or fine-tuning the entire model
    """
    
    def __init__(
        self,
        base_model: Module,
        new_layers: Module,
        freeze_base: bool = True,
        fine_tune_layers: Optional[List[str]] = None,
    ):
        """
        Initialize a transfer learning model.
        
        Args:
            base_model: The pre-trained model to use as a feature extractor
            new_layers: New layers to add on top of the base model for the target task
            freeze_base: Whether to freeze the parameters of the base model
            fine_tune_layers: List of layer names in the base model to fine-tune (unfreeze)
                              Only used if freeze_base is True
        """
        super().__init__()
        
        self.base_model = base_model
        self.new_layers = new_layers
        
        # Freeze the base model if requested
        if freeze_base:
            self._freeze_base_model()
            
            # Unfreeze specific layers if requested
            if fine_tune_layers is not None:
                self._unfreeze_layers(fine_tune_layers)
    
    def _freeze_base_model(self):
        """Freeze all parameters in the base model."""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def _unfreeze_layers(self, layer_names: List[str]):
        """
        Unfreeze specific layers in the base model.
        
        Args:
            layer_names: List of layer names to unfreeze
        """
        # This is a simplified implementation that assumes the base model
        # has a flat structure with named modules
        for name, module in self.base_model._modules.items():
            if name in layer_names:
                for param in module.parameters():
                    param.requires_grad = True
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the transfer learning model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor after passing through the base model and new layers
        """
        # Pass input through the base model
        features = self.base_model(x)
        
        # Pass features through the new layers
        return self.new_layers(features)
    
    def get_base_model(self) -> Module:
        """Get the base model."""
        return self.base_model
    
    def get_new_layers(self) -> Module:
        """Get the new layers."""
        return self.new_layers
    
    def freeze_base_model(self):
        """Freeze all parameters in the base model."""
        self._freeze_base_model()
    
    def unfreeze_base_model(self):
        """Unfreeze all parameters in the base model."""
        for param in self.base_model.parameters():
            param.requires_grad = True
    
    def unfreeze_layers(self, layer_names: List[str]):
        """
        Unfreeze specific layers in the base model.
        
        Args:
            layer_names: List of layer names to unfreeze
        """
        self._unfreeze_layers(layer_names)
