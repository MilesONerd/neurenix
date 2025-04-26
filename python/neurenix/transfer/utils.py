"""
Utility functions for transfer learning in the Neurenix framework.
"""

from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor
from neurenix.device import Device

def get_layer_outputs(model: Module, input_tensor: Tensor, layer_names: List[str]) -> Dict[str, Tensor]:
    """
    Get the outputs of specific layers in a model for a given input.
    
    This is useful for visualizing activations or extracting features from
    intermediate layers of a model.
    
    Args:
        model: The model to extract layer outputs from
        input_tensor: Input tensor to pass through the model
        layer_names: Names of layers to extract outputs from
        
    Returns:
        Dictionary mapping layer names to their output tensors
    """
    layer_outputs = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            layer_outputs[name] = output
        return hook
    
    hooks = []
    for name in layer_names:
        if hasattr(model, name):
            layer = getattr(model, name)
            hook = layer.register_forward_hook(hook_fn(name))
            hooks.append(hook)
        elif '.' in name:
            parts = name.split('.')
            current = model
            for part in parts[:-1]:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    break
            if hasattr(current, parts[-1]):
                layer = getattr(current, parts[-1])
                hook = layer.register_forward_hook(hook_fn(name))
                hooks.append(hook)
    
    model(input_tensor)
    
    for hook in hooks:
        hook.remove()
    
    missing_layers = set(layer_names) - set(layer_outputs.keys())
    if missing_layers:
        print(f"Warning: Could not find layers: {missing_layers}")
    
    return layer_outputs

def visualize_layer_activations(model: Module, input_tensor: Tensor, layer_name: str):
    """
    Visualize the activations of a specific layer in a model.
    
    Args:
        model: The model to visualize activations for
        input_tensor: Input tensor to pass through the model
        layer_name: Name of the layer to visualize
    """
    outputs = get_layer_outputs(model, input_tensor, [layer_name])
    
    if layer_name not in outputs:
        print(f"Warning: Layer {layer_name} not found in model")
        return
    
    activation = outputs[layer_name]
    
    print(f"Visualizing activations for layer: {layer_name}")
    print(f"Activation shape: {activation.shape}")
    print(f"Activation statistics:")
    print(f"  - Min: {activation.min()}")
    print(f"  - Max: {activation.max()}")
    print(f"  - Mean: {activation.mean()}")
    print(f"  - Std: {activation.std()}")
    
    
    return activation

def get_model_feature_extractor(model: Module, output_layer: str) -> Module:
    """
    Create a feature extractor from a model by truncating it at a specific layer.
    
    Args:
        model: The model to create a feature extractor from
        output_layer: Name of the layer to use as the output
        
    Returns:
        A new model that outputs the activations of the specified layer
    """
    class FeatureExtractor(Module):
        def __init__(self, base_model: Module, output_layer: str):
            super().__init__()
            self.base_model = base_model
            self.output_layer = output_layer
            self.activation = None
            
            self.target_layer = None
            if hasattr(base_model, output_layer):
                self.target_layer = getattr(base_model, output_layer)
            elif '.' in output_layer:
                parts = output_layer.split('.')
                current = base_model
                for part in parts[:-1]:
                    if hasattr(current, part):
                        current = getattr(current, part)
                    else:
                        break
                if hasattr(current, parts[-1]):
                    self.target_layer = getattr(current, parts[-1])
            
            if self.target_layer is None:
                raise ValueError(f"Layer {output_layer} not found in model")
            
            self.hook = self.target_layer.register_forward_hook(self._hook_fn)
        
        def _hook_fn(self, module, input, output):
            self.activation = output
        
        def forward(self, x: Tensor) -> Tensor:
            self.activation = None
            
            self.base_model(x)
            
            if self.activation is None:
                raise RuntimeError(f"Failed to capture output from layer {self.output_layer}")
            
            return self.activation
    
    return FeatureExtractor(model, output_layer)

def compare_model_features(model1: Module, model2: Module, input_tensor: Tensor, layer_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], float]:
    """
    Compare the features extracted by two models at specific layer pairs.
    
    Args:
        model1: First model
        model2: Second model
        input_tensor: Input tensor to pass through both models
        layer_pairs: List of (layer_name1, layer_name2) tuples to compare
        
    Returns:
        Dictionary mapping layer pairs to similarity scores
    """
    model1_layers = [pair[0] for pair in layer_pairs]
    model2_layers = [pair[1] for pair in layer_pairs]
    
    model1_outputs = get_layer_outputs(model1, input_tensor, model1_layers)
    model2_outputs = get_layer_outputs(model2, input_tensor, model2_layers)
    
    similarity_scores = {}
    
    for layer1, layer2 in layer_pairs:
        if layer1 not in model1_outputs or layer2 not in model2_outputs:
            print(f"Warning: Could not compare {layer1} and {layer2}, one or both layers not found")
            continue
        
        act1 = model1_outputs[layer1]
        act2 = model2_outputs[layer2]
        
        act1_flat = act1.reshape(-1)
        act2_flat = act2.reshape(-1)
        
        min_length = min(len(act1_flat), len(act2_flat))
        act1_flat = act1_flat[:min_length]
        act2_flat = act2_flat[:min_length]
        
        norm1 = np.sqrt(np.sum(act1_flat * act1_flat))
        norm2 = np.sqrt(np.sum(act2_flat * act2_flat))
        
        if norm1 == 0 or norm2 == 0:
            similarity = 0.0
        else:
            dot_product = np.sum(act1_flat * act2_flat)
            similarity = dot_product / (norm1 * norm2)
        
        similarity_scores[(layer1, layer2)] = float(similarity)
    
    return similarity_scores
