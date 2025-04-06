"""
Federated Learning utilities module for Neurenix.

This module provides utility functions and classes for federated learning,
including client selection, model compression, and gradient compression.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import random
import math

import neurenix as nx


class ClientSelector:
    """Base class for client selection strategies."""
    
    def select(self, client_ids: List[str], num_clients: int) -> List[str]:
        """
        Select clients for federated learning.
        
        Args:
            client_ids: List of all client IDs
            num_clients: Number of clients to select
            
        Returns:
            List of selected client IDs
        """
        raise NotImplementedError("Subclasses must implement select method")


class RandomClientSelector(ClientSelector):
    """Random client selection strategy."""
    
    def select(self, client_ids: List[str], num_clients: int) -> List[str]:
        """
        Select clients randomly.
        
        Args:
            client_ids: List of all client IDs
            num_clients: Number of clients to select
            
        Returns:
            List of selected client IDs
        """
        if num_clients >= len(client_ids):
            return client_ids
        
        return random.sample(client_ids, num_clients)


class PowerOfChoiceSelector(ClientSelector):
    """Power of choice client selection strategy."""
    
    def __init__(self, metric_fn: callable, d: int = 2, maximize: bool = False):
        """
        Initialize power of choice client selector.
        
        Args:
            metric_fn: Function that takes a client ID and returns a metric value
            d: Number of random clients to sample
            maximize: Whether to maximize or minimize the metric
        """
        self.metric_fn = metric_fn
        self.d = d
        self.maximize = maximize
    
    def select(self, client_ids: List[str], num_clients: int) -> List[str]:
        """
        Select clients using power of choice.
        
        Args:
            client_ids: List of all client IDs
            num_clients: Number of clients to select
            
        Returns:
            List of selected client IDs
        """
        if num_clients >= len(client_ids):
            return client_ids
        
        selected_clients = []
        remaining_clients = client_ids.copy()
        
        for _ in range(num_clients):
            if not remaining_clients:
                break
            
            d = min(self.d, len(remaining_clients))
            candidates = random.sample(remaining_clients, d)
            
            metrics = {client_id: self.metric_fn(client_id) for client_id in candidates}
            
            if self.maximize:
                best_client = max(metrics.items(), key=lambda x: x[1])[0]
            else:
                best_client = min(metrics.items(), key=lambda x: x[1])[0]
            
            selected_clients.append(best_client)
            remaining_clients.remove(best_client)
        
        return selected_clients


class ModelCompressor:
    """Model compressor for federated learning."""
    
    def __init__(self, ratio: float = 0.1, method: str = 'topk'):
        """
        Initialize model compressor.
        
        Args:
            ratio: Compression ratio (0 < ratio <= 1)
            method: Compression method ('topk', 'random', or 'threshold')
        """
        if ratio <= 0 or ratio > 1:
            raise ValueError("Compression ratio must be in (0, 1]")
        
        self.ratio = ratio
        self.method = method
        self.indices = {}
        self.values = {}
    
    def compress(self, model_params: Dict[str, nx.Tensor]) -> Dict[str, nx.Tensor]:
        """
        Compress model parameters.
        
        Args:
            model_params: Model parameters
            
        Returns:
            Compressed model parameters
        """
        compressed_params = {}
        
        for name, param in model_params.items():
            if self.method == 'topk':
                k = max(1, int(param.numel() * self.ratio))
                values, indices = nx.topk(nx.abs(param.flatten()), k)
                
                compressed_params[f"{name}_shape"] = nx.tensor(param.shape)
                compressed_params[f"{name}_indices"] = indices
                compressed_params[f"{name}_values"] = param.flatten()[indices]
                
                self.indices[name] = indices
                self.values[name] = param.flatten()[indices]
                
            elif self.method == 'random':
                mask = nx.zeros(param.shape).flatten()
                k = max(1, int(param.numel() * self.ratio))
                indices = nx.randperm(param.numel())[:k]
                mask[indices] = 1
                mask = mask.reshape(param.shape)
                
                compressed_params[f"{name}_shape"] = nx.tensor(param.shape)
                compressed_params[f"{name}_mask"] = mask
                compressed_params[f"{name}_values"] = param * mask
                
                self.indices[name] = indices
                self.values[name] = param.flatten()[indices]
                
            elif self.method == 'threshold':
                threshold = nx.quantile(nx.abs(param.flatten()), 1 - self.ratio)
                mask = nx.abs(param) >= threshold
                
                compressed_params[f"{name}_shape"] = nx.tensor(param.shape)
                compressed_params[f"{name}_mask"] = mask
                compressed_params[f"{name}_values"] = param * mask
                
                self.indices[name] = mask.flatten().nonzero().squeeze()
                self.values[name] = param.flatten()[self.indices[name]]
                
            else:
                raise ValueError(f"Unknown compression method: {self.method}")
        
        return compressed_params
    
    def decompress(self, compressed_params: Dict[str, nx.Tensor]) -> Dict[str, nx.Tensor]:
        """
        Decompress model parameters.
        
        Args:
            compressed_params: Compressed model parameters
            
        Returns:
            Decompressed model parameters
        """
        decompressed_params = {}
        
        for name in list(compressed_params.keys()):
            if name.endswith("_shape"):
                param_name = name[:-6]
                shape = compressed_params[name].tolist()
                
                if self.method == 'topk':
                    indices = compressed_params[f"{param_name}_indices"]
                    values = compressed_params[f"{param_name}_values"]
                    
                    decompressed = nx.zeros(math.prod(shape), device=values.device)
                    decompressed[indices] = values
                    decompressed = decompressed.reshape(shape)
                    
                    decompressed_params[param_name] = decompressed
                    
                elif self.method in ['random', 'threshold']:
                    mask = compressed_params[f"{param_name}_mask"]
                    values = compressed_params[f"{param_name}_values"]
                    
                    decompressed_params[param_name] = values
        
        return decompressed_params


class GradientCompressor:
    """Gradient compressor for federated learning."""
    
    def __init__(self, ratio: float = 0.1, method: str = 'topk'):
        """
        Initialize gradient compressor.
        
        Args:
            ratio: Compression ratio (0 < ratio <= 1)
            method: Compression method ('topk', 'random', 'threshold', or 'signsgd')
        """
        if ratio <= 0 or ratio > 1:
            raise ValueError("Compression ratio must be in (0, 1]")
        
        self.ratio = ratio
        self.method = method
    
    def compress(self, gradients: Dict[str, nx.Tensor]) -> Dict[str, nx.Tensor]:
        """
        Compress gradients.
        
        Args:
            gradients: Gradients
            
        Returns:
            Compressed gradients
        """
        compressed_grads = {}
        
        for name, grad in gradients.items():
            if grad is None:
                compressed_grads[name] = None
                continue
            
            if self.method == 'topk':
                k = max(1, int(grad.numel() * self.ratio))
                values, indices = nx.topk(nx.abs(grad.flatten()), k)
                
                compressed_grads[f"{name}_shape"] = nx.tensor(grad.shape)
                compressed_grads[f"{name}_indices"] = indices
                compressed_grads[f"{name}_values"] = grad.flatten()[indices]
                
            elif self.method == 'random':
                mask = nx.zeros(grad.shape).flatten()
                k = max(1, int(grad.numel() * self.ratio))
                indices = nx.randperm(grad.numel())[:k]
                mask[indices] = 1
                mask = mask.reshape(grad.shape)
                
                compressed_grads[f"{name}_shape"] = nx.tensor(grad.shape)
                compressed_grads[f"{name}_mask"] = mask
                compressed_grads[f"{name}_values"] = grad * mask
                
            elif self.method == 'threshold':
                threshold = nx.quantile(nx.abs(grad.flatten()), 1 - self.ratio)
                mask = nx.abs(grad) >= threshold
                
                compressed_grads[f"{name}_shape"] = nx.tensor(grad.shape)
                compressed_grads[f"{name}_mask"] = mask
                compressed_grads[f"{name}_values"] = grad * mask
                
            elif self.method == 'signsgd':
                compressed_grads[f"{name}_shape"] = nx.tensor(grad.shape)
                compressed_grads[f"{name}_sign"] = nx.sign(grad)
                
            else:
                raise ValueError(f"Unknown compression method: {self.method}")
        
        return compressed_grads
    
    def decompress(self, compressed_grads: Dict[str, nx.Tensor]) -> Dict[str, nx.Tensor]:
        """
        Decompress gradients.
        
        Args:
            compressed_grads: Compressed gradients
            
        Returns:
            Decompressed gradients
        """
        decompressed_grads = {}
        
        for name in list(compressed_grads.keys()):
            if name.endswith("_shape"):
                param_name = name[:-6]
                shape = compressed_grads[name].tolist()
                
                if self.method == 'topk':
                    indices = compressed_grads[f"{param_name}_indices"]
                    values = compressed_grads[f"{param_name}_values"]
                    
                    decompressed = nx.zeros(math.prod(shape), device=values.device)
                    decompressed[indices] = values
                    decompressed = decompressed.reshape(shape)
                    
                    decompressed_grads[param_name] = decompressed
                    
                elif self.method in ['random', 'threshold']:
                    mask = compressed_grads[f"{param_name}_mask"]
                    values = compressed_grads[f"{param_name}_values"]
                    
                    decompressed_grads[param_name] = values
                    
                elif self.method == 'signsgd':
                    sign = compressed_grads[f"{param_name}_sign"]
                    
                    decompressed_grads[param_name] = sign * 0.01
        
        return decompressed_grads


def cosine_similarity(a: nx.Tensor, b: nx.Tensor) -> float:
    """
    Compute cosine similarity between two tensors.
    
    Args:
        a: First tensor
        b: Second tensor
        
    Returns:
        Cosine similarity
    """
    a_flat = a.flatten()
    b_flat = b.flatten()
    
    dot_product = nx.sum(a_flat * b_flat)
    norm_a = nx.norm(a_flat)
    norm_b = nx.norm(b_flat)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)


def compute_model_similarity(model_a: Dict[str, nx.Tensor], 
                            model_b: Dict[str, nx.Tensor]) -> float:
    """
    Compute similarity between two models.
    
    Args:
        model_a: First model parameters
        model_b: Second model parameters
        
    Returns:
        Model similarity
    """
    similarities = []
    
    for name, param_a in model_a.items():
        if name in model_b:
            param_b = model_b[name]
            
            if param_a.shape == param_b.shape:
                similarity = cosine_similarity(param_a, param_b)
                similarities.append(similarity)
    
    if not similarities:
        return 0.0
    
    return sum(similarities) / len(similarities)


def compute_update_norm(model_update: Dict[str, nx.Tensor]) -> float:
    """
    Compute the norm of a model update.
    
    Args:
        model_update: Model update
        
    Returns:
        Update norm
    """
    squared_sum = 0.0
    
    for name, param in model_update.items():
        squared_sum += nx.sum(param ** 2).item()
    
    return math.sqrt(squared_sum)


def clip_update(model_update: Dict[str, nx.Tensor], max_norm: float) -> Dict[str, nx.Tensor]:
    """
    Clip a model update to a maximum norm.
    
    Args:
        model_update: Model update
        max_norm: Maximum norm
        
    Returns:
        Clipped model update
    """
    update_norm = compute_update_norm(model_update)
    
    if update_norm <= max_norm:
        return model_update
    
    scale = max_norm / update_norm
    
    clipped_update = {}
    
    for name, param in model_update.items():
        clipped_update[name] = param * scale
    
    return clipped_update


def add_updates(update_a: Dict[str, nx.Tensor], 
               update_b: Dict[str, nx.Tensor],
               weight_a: float = 1.0,
               weight_b: float = 1.0) -> Dict[str, nx.Tensor]:
    """
    Add two model updates.
    
    Args:
        update_a: First model update
        update_b: Second model update
        weight_a: Weight for the first update
        weight_b: Weight for the second update
        
    Returns:
        Combined model update
    """
    combined_update = {}
    
    for name, param_a in update_a.items():
        if name in update_b:
            param_b = update_b[name]
            
            if param_a.shape == param_b.shape:
                combined_update[name] = weight_a * param_a + weight_b * param_b
    
    return combined_update
