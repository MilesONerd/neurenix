"""
Activation visualization module for Neurenix.

This module provides tools for visualizing activations in neural networks,
helping to understand what features the network is detecting.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
import logging

from neurenix.tensor import Tensor
from neurenix.nn.module import Module

logger = logging.getLogger(__name__)

class ActivationVisualization:
    """
    Activation visualization for neural networks.
    
    Visualizes activations of neurons in different layers of a neural network,
    helping to understand what features the network is detecting.
    """
    
    def __init__(self, 
                 model: Module, 
                 layer_names: Optional[List[str]] = None):
        """
        Initialize an activation visualizer.
        
        Args:
            model: The model to visualize
            layer_names: Names of layers to visualize (if None, all layers will be used)
        """
        self.model = model
        self.layer_names = layer_names
        self.hooks = []
        self.activations = {}
        
        try:
            from neurenix.binding import phynexus
            self._has_phynexus = True
            self._phynexus = phynexus
        except ImportError:
            logger.warning("Phynexus extension not found. Using pure Python implementation.")
            self._has_phynexus = False
    
    def _hook_fn(self, name: str):
        """Create a hook function for the given layer name."""
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def register_hooks(self):
        """Register hooks for all layers in the model."""
        self.remove_hooks()
        
        for name, module in self.model.named_modules():
            if self.layer_names is None or name in self.layer_names:
                hook = module.register_forward_hook(self._hook_fn(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.activations = {}
    
    def visualize(self, 
                  input_data: Union[Tensor, np.ndarray], 
                  **kwargs) -> Dict[str, Any]:
        """
        Visualize activations for the given input.
        
        Args:
            input_data: Input data to visualize activations for
            **kwargs: Additional arguments for the visualization
            
        Returns:
            Dictionary containing activation information
        """
        input_tensor = self._convert_to_tensor(input_data)
        
        if self._has_phynexus:
            return self._visualize_native(input_tensor, **kwargs)
        else:
            return self._visualize_python(input_tensor, **kwargs)
    
    def _convert_to_tensor(self, data: Union[Tensor, np.ndarray]) -> Tensor:
        """Convert input data to Tensor if needed."""
        if isinstance(data, np.ndarray):
            return Tensor(data)
        return data
    
    def _visualize_native(self, 
                          input_data: Tensor, 
                          **kwargs) -> Dict[str, Any]:
        """Use native Phynexus implementation for activation visualization."""
        if not hasattr(self._phynexus, "activation_visualization"):
            logger.warning("Native activation_visualization not available. Using Python implementation.")
            return self._visualize_python(input_data, **kwargs)
        
        
        return self._visualize_python(input_data, **kwargs)
    
    def _visualize_python(self, 
                          input_data: Tensor, 
                          **kwargs) -> Dict[str, Any]:
        """Pure Python implementation of activation visualization."""
        self.register_hooks()
        
        self.model.eval()  # Set model to evaluation mode
        with Tensor.no_grad():
            output = self.model(input_data)
        
        processed_activations = {}
        
        for name, activation in self.activations.items():
            activation_np = activation.numpy()
            
            stats = {
                "mean": np.mean(activation_np),
                "std": np.std(activation_np),
                "min": np.min(activation_np),
                "max": np.max(activation_np),
                "shape": activation_np.shape
            }
            
            processed_activations[name] = {
                "activation": activation,
                "stats": stats
            }
        
        self.remove_hooks()
        
        return {
            "activations": processed_activations,
            "output": output
        }
    
    def plot(self, 
             activation_result: Dict[str, Any], 
             **kwargs):
        """
        Plot activation visualizations.
        
        Args:
            activation_result: Dictionary containing activation information from visualize()
            **kwargs: Additional arguments for the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            activations = activation_result.get("activations", {})
            
            if not activations:
                logger.warning("No activations to plot.")
                return
            
            n_layers = len(activations)
            
            fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))
            if n_layers == 1:
                axes = [axes]
            
            for i, (name, data) in enumerate(activations.items()):
                ax = axes[i]
                
                activation = data["activation"]
                activation_np = activation.numpy()
                
                if len(activation_np.shape) == 4:
                    mean_activation = np.mean(activation_np[0], axis=0)
                    im = ax.imshow(mean_activation, cmap='viridis')
                    plt.colorbar(im, ax=ax)
                    ax.set_title(f"{name}\nMean Channel Activation")
                    ax.axis('off')
                elif len(activation_np.shape) == 2:
                    ax.bar(range(min(20, activation_np.shape[1])), 
                           activation_np[0][:20])
                    ax.set_title(f"{name}\nFirst 20 Neurons")
                    ax.set_xlabel("Neuron")
                    ax.set_ylabel("Activation")
                else:
                    ax.hist(activation_np.flatten(), bins=50)
                    ax.set_title(f"{name}\nActivation Distribution")
                    ax.set_xlabel("Activation Value")
                    ax.set_ylabel("Frequency")
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not installed. Cannot plot activations.")
