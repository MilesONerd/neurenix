"""
Federated Learning strategies module for Neurenix.

This module provides implementations of various federated learning
aggregation strategies for distributed training.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import copy

import neurenix as nx


class AggregationStrategy:
    """Base class for federated learning aggregation strategies."""
    
    def aggregate(self, global_model: Dict[str, nx.Tensor],
                 client_models: Dict[str, Dict[str, nx.Tensor]],
                 client_weights: Dict[str, float]) -> Dict[str, nx.Tensor]:
        """
        Aggregate client models into a global model.
        
        Args:
            global_model: Global model parameters
            client_models: Dictionary mapping client IDs to model parameters
            client_weights: Dictionary mapping client IDs to weights
            
        Returns:
            Aggregated model parameters
        """
        raise NotImplementedError("Subclasses must implement aggregate method")


class FedAvg(AggregationStrategy):
    """Federated Averaging (FedAvg) aggregation strategy."""
    
    def aggregate(self, global_model: Dict[str, nx.Tensor],
                 client_models: Dict[str, Dict[str, nx.Tensor]],
                 client_weights: Dict[str, float]) -> Dict[str, nx.Tensor]:
        """
        Aggregate client models using FedAvg.
        
        Args:
            global_model: Global model parameters
            client_models: Dictionary mapping client IDs to model parameters
            client_weights: Dictionary mapping client IDs to weights
            
        Returns:
            Aggregated model parameters
        """
        if not client_models:
            return global_model
        
        total_weight = sum(client_weights.values())
        
        if total_weight == 0:
            return global_model
        
        aggregated_model = {}
        
        for name, param in global_model.items():
            aggregated_model[name] = nx.zeros_like(param)
        
        for client_id, model in client_models.items():
            weight = client_weights.get(client_id, 0.0)
            
            if weight == 0:
                continue
            
            for name, param in model.items():
                if name in aggregated_model:
                    aggregated_model[name] += (param * (weight / total_weight))
        
        return aggregated_model


class FedProx(AggregationStrategy):
    """Federated Proximal (FedProx) aggregation strategy."""
    
    def __init__(self, mu: float = 0.01):
        """
        Initialize FedProx aggregation strategy.
        
        Args:
            mu: Proximal term coefficient
        """
        self.mu = mu
    
    def aggregate(self, global_model: Dict[str, nx.Tensor],
                 client_models: Dict[str, Dict[str, nx.Tensor]],
                 client_weights: Dict[str, float]) -> Dict[str, nx.Tensor]:
        """
        Aggregate client models using FedProx.
        
        Args:
            global_model: Global model parameters
            client_models: Dictionary mapping client IDs to model parameters
            client_weights: Dictionary mapping client IDs to weights
            
        Returns:
            Aggregated model parameters
        """
        fed_avg = FedAvg()
        return fed_avg.aggregate(global_model, client_models, client_weights)


class FedNova(AggregationStrategy):
    """Federated Normalized Averaging (FedNova) aggregation strategy."""
    
    def aggregate(self, global_model: Dict[str, nx.Tensor],
                 client_models: Dict[str, Dict[str, nx.Tensor]],
                 client_weights: Dict[str, float]) -> Dict[str, nx.Tensor]:
        """
        Aggregate client models using FedNova.
        
        Args:
            global_model: Global model parameters
            client_models: Dictionary mapping client IDs to model parameters
            client_weights: Dictionary mapping client IDs to weights
            
        Returns:
            Aggregated model parameters
        """
        if not client_models:
            return global_model
        
        total_weight = sum(client_weights.values())
        
        if total_weight == 0:
            return global_model
        
        aggregated_model = {}
        
        for name, param in global_model.items():
            aggregated_model[name] = nx.zeros_like(param)
        
        normalized_weights = {
            client_id: weight / total_weight
            for client_id, weight in client_weights.items()
        }
        
        model_updates = {}
        
        for client_id, model in client_models.items():
            model_updates[client_id] = {}
            
            for name, param in model.items():
                if name in global_model:
                    model_updates[client_id][name] = param - global_model[name]
        
        for name, param in global_model.items():
            for client_id, updates in model_updates.items():
                if name in updates:
                    weight = normalized_weights.get(client_id, 0.0)
                    
                    if weight > 0:
                        aggregated_model[name] += updates[name] * weight
            
            aggregated_model[name] += global_model[name]
        
        return aggregated_model


class FedOpt(AggregationStrategy):
    """Base class for FedOpt-based aggregation strategies."""
    
    def __init__(self, learning_rate: float = 0.01, beta1: float = 0.9, beta2: float = 0.99,
                 epsilon: float = 1e-8):
        """
        Initialize FedOpt aggregation strategy.
        
        Args:
            learning_rate: Learning rate
            beta1: Exponential decay rate for first moment
            beta2: Exponential decay rate for second moment
            epsilon: Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
    
    def aggregate(self, global_model: Dict[str, nx.Tensor],
                 client_models: Dict[str, Dict[str, nx.Tensor]],
                 client_weights: Dict[str, float]) -> Dict[str, nx.Tensor]:
        """
        Aggregate client models using FedOpt.
        
        Args:
            global_model: Global model parameters
            client_models: Dictionary mapping client IDs to model parameters
            client_weights: Dictionary mapping client IDs to weights
            
        Returns:
            Aggregated model parameters
        """
        if not client_models:
            return global_model
        
        total_weight = sum(client_weights.values())
        
        if total_weight == 0:
            return global_model
        
        pseudo_gradient = {}
        
        for name, param in global_model.items():
            pseudo_gradient[name] = nx.zeros_like(param)
            
            for client_id, model in client_models.items():
                if name in model:
                    weight = client_weights.get(client_id, 0.0)
                    
                    if weight > 0:
                        pseudo_gradient[name] += (model[name] - param) * (weight / total_weight)
        
        if not self.m:
            for name, param in global_model.items():
                self.m[name] = nx.zeros_like(param)
                self.v[name] = nx.zeros_like(param)
        
        self.t += 1
        
        aggregated_model = self._update_global_model(global_model, pseudo_gradient)
        
        return aggregated_model
    
    def _update_global_model(self, global_model: Dict[str, nx.Tensor],
                            pseudo_gradient: Dict[str, nx.Tensor]) -> Dict[str, nx.Tensor]:
        """
        Update global model using optimizer-specific update rule.
        
        Args:
            global_model: Global model parameters
            pseudo_gradient: Pseudo-gradient
            
        Returns:
            Updated global model parameters
        """
        raise NotImplementedError("Subclasses must implement _update_global_model method")


class FedAdagrad(FedOpt):
    """Federated Adagrad (FedAdagrad) aggregation strategy."""
    
    def _update_global_model(self, global_model: Dict[str, nx.Tensor],
                            pseudo_gradient: Dict[str, nx.Tensor]) -> Dict[str, nx.Tensor]:
        """
        Update global model using Adagrad.
        
        Args:
            global_model: Global model parameters
            pseudo_gradient: Pseudo-gradient
            
        Returns:
            Updated global model parameters
        """
        aggregated_model = copy.deepcopy(global_model)
        
        for name, param in global_model.items():
            if name in pseudo_gradient:
                self.v[name] += pseudo_gradient[name] ** 2
                
                aggregated_model[name] = param + self.learning_rate * pseudo_gradient[name] / (
                    nx.sqrt(self.v[name]) + self.epsilon
                )
        
        return aggregated_model


class FedAdam(FedOpt):
    """Federated Adam (FedAdam) aggregation strategy."""
    
    def _update_global_model(self, global_model: Dict[str, nx.Tensor],
                            pseudo_gradient: Dict[str, nx.Tensor]) -> Dict[str, nx.Tensor]:
        """
        Update global model using Adam.
        
        Args:
            global_model: Global model parameters
            pseudo_gradient: Pseudo-gradient
            
        Returns:
            Updated global model parameters
        """
        aggregated_model = copy.deepcopy(global_model)
        
        for name, param in global_model.items():
            if name in pseudo_gradient:
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * pseudo_gradient[name]
                
                self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (pseudo_gradient[name] ** 2)
                
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                
                aggregated_model[name] = param + self.learning_rate * m_hat / (nx.sqrt(v_hat) + self.epsilon)
        
        return aggregated_model


class FedYogi(FedOpt):
    """Federated Yogi (FedYogi) aggregation strategy."""
    
    def _update_global_model(self, global_model: Dict[str, nx.Tensor],
                            pseudo_gradient: Dict[str, nx.Tensor]) -> Dict[str, nx.Tensor]:
        """
        Update global model using Yogi.
        
        Args:
            global_model: Global model parameters
            pseudo_gradient: Pseudo-gradient
            
        Returns:
            Updated global model parameters
        """
        aggregated_model = copy.deepcopy(global_model)
        
        for name, param in global_model.items():
            if name in pseudo_gradient:
                self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * pseudo_gradient[name]
                
                v_update = (pseudo_gradient[name] ** 2) * nx.sign(
                    self.v[name] - (pseudo_gradient[name] ** 2)
                )
                self.v[name] = self.v[name] - (1 - self.beta2) * v_update
                
                m_hat = self.m[name] / (1 - self.beta1 ** self.t)
                v_hat = self.v[name] / (1 - self.beta2 ** self.t)
                
                aggregated_model[name] = param + self.learning_rate * m_hat / (nx.sqrt(v_hat) + self.epsilon)
        
        return aggregated_model
