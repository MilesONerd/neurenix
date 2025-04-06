"""
Regularization-based methods for continual learning.

This module provides regularization-based approaches to prevent
catastrophic forgetting in continual learning scenarios.
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import neurenix
from neurenix.nn import Module
from neurenix.tensor import Tensor
from neurenix.optim import Optimizer

class L2Regularization:
    """
    L2 regularization for continual learning.
    
    Adds L2 regularization to prevent the model from deviating too much
    from its previous state after learning a task.
    
    Attributes:
        model: The neural network model
        params_old: Parameter values after learning previous tasks
        lambda_reg: Regularization strength
    """
    
    def __init__(self, model: Module, lambda_reg: float = 1.0):
        """
        Initialize L2 regularization.
        
        Args:
            model: Neural network model
            lambda_reg: Regularization strength
        """
        self.model = model
        self.lambda_reg = lambda_reg
        self.params_old: Dict[str, Tensor] = {}
        self._initialized = False
        
        try:
            from neurenix.binding import get_phynexus
            self._phynexus = get_phynexus()
            self._has_phynexus = True
        except ImportError:
            self._has_phynexus = False
    
    def register_task(self):
        """
        Register a task by storing current parameter values.
        """
        self.params_old = {}
        for name, param in self.model.named_parameters():
            self.params_old[name] = param.clone().detach()
        
        self._initialized = True
    
    def penalty(self) -> Tensor:
        """
        Compute L2 penalty term to be added to the loss.
        
        Returns:
            Tensor containing the L2 penalty
        """
        if not self._initialized:
            return neurenix.tensor(0.0)
        
        if self._has_phynexus:
            return self._phynexus.continual.compute_l2_penalty(
                self.model, self.params_old, self.lambda_reg
            )
        
        penalty = neurenix.tensor(0.0)
        
        for name, param in self.model.named_parameters():
            if name in self.params_old:
                delta = param - self.params_old[name]
                penalty += (delta.pow(2)).sum()
        
        return 0.5 * self.lambda_reg * penalty


class WeightFreezing:
    """
    Weight freezing for continual learning.
    
    Freezes important weights after learning a task to prevent
    catastrophic forgetting.
    
    Attributes:
        model: The neural network model
        importance_threshold: Threshold for determining important weights
        mask: Binary mask indicating which weights are frozen
    """
    
    def __init__(
        self, 
        model: Module, 
        importance_threshold: float = 0.1,
        importance_method: str = "magnitude"
    ):
        """
        Initialize weight freezing.
        
        Args:
            model: Neural network model
            importance_threshold: Threshold for determining important weights
            importance_method: Method for computing weight importance
                              ("magnitude", "gradient", "fisher")
        """
        self.model = model
        self.importance_threshold = importance_threshold
        self.importance_method = importance_method
        self.mask: Dict[str, Tensor] = {}
        self._initialized = False
        
        try:
            from neurenix.binding import get_phynexus
            self._phynexus = get_phynexus()
            self._has_phynexus = True
        except ImportError:
            self._has_phynexus = False
    
    def register_task(
        self, 
        dataloader: Optional[Any] = None, 
        loss_fn: Optional[Any] = None
    ):
        """
        Register a task by computing weight importance and creating masks.
        
        Args:
            dataloader: Data loader for the task (required for gradient-based methods)
            loss_fn: Loss function (required for gradient-based methods)
        """
        if self._has_phynexus:
            self.mask = self._phynexus.continual.compute_weight_importance(
                self.model, self.importance_method, self.importance_threshold,
                dataloader, loss_fn
            )
            self._initialized = True
            return
        
        self.mask = {}
        
        if self.importance_method == "magnitude":
            for name, param in self.model.named_parameters():
                importance = param.abs()
                self.mask[name] = (importance > self.importance_threshold).float()
        
        elif self.importance_method in ["gradient", "fisher"]:
            if dataloader is None or loss_fn is None:
                raise ValueError(
                    f"Dataloader and loss_fn are required for {self.importance_method} importance"
                )
            
            importance = {}
            for name, param in self.model.named_parameters():
                importance[name] = neurenix.zeros_like(param)
            
            self.model.eval()
            
            for inputs, targets in dataloader:
                self.model.zero_grad()
                outputs = self.model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if self.importance_method == "gradient":
                            importance[name] += param.grad.abs()
                        else:  # fisher
                            importance[name] += param.grad.pow(2)
            
            for name, imp in importance.items():
                self.mask[name] = (imp > self.importance_threshold).float()
        
        self._initialized = True
    
    def apply_mask(self):
        """
        Apply masks to gradients during training to freeze important weights.
        """
        if not self._initialized:
            return
        
        for name, param in self.model.named_parameters():
            if name in self.mask and param.grad is not None:
                param.grad *= (1 - self.mask[name])
