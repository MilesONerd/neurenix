"""
Elastic Weight Consolidation (EWC) implementation for continual learning.

EWC is a regularization method that slows down learning on weights important
for previously learned tasks, preventing catastrophic forgetting.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

import neurenix
from neurenix.nn import Module
from neurenix.tensor import Tensor
from neurenix.optim import Optimizer

class EWC:
    """
    Elastic Weight Consolidation (EWC) for continual learning.
    
    EWC adds a regularization term to the loss function that penalizes changes
    to parameters that were important for previous tasks, as measured by the
    Fisher information matrix.
    
    Attributes:
        model: The neural network model
        importance: Importance of each parameter for previous tasks
        params_old: Parameter values after learning previous tasks
        lambda_reg: Regularization strength
    """
    
    def __init__(
        self, 
        model: Module, 
        lambda_reg: float = 1.0,
        use_online: bool = False
    ):
        """
        Initialize EWC.
        
        Args:
            model: Neural network model
            lambda_reg: Regularization strength
            use_online: Whether to use online EWC (accumulate importance)
        """
        self.model = model
        self.lambda_reg = lambda_reg
        self.use_online = use_online
        self.importance: Dict[str, Tensor] = {}
        self.params_old: Dict[str, Tensor] = {}
        self._initialized = False
        
        try:
            from neurenix.binding import get_phynexus
            self._phynexus = get_phynexus()
            self._has_phynexus = True
        except ImportError:
            self._has_phynexus = False
    
    def register_task(
        self, 
        dataloader: Any, 
        loss_fn: Any, 
        optimizer: Optimizer,
        num_samples: Optional[int] = None
    ):
        """
        Register a task by computing parameter importance.
        
        Args:
            dataloader: Data loader for the task
            loss_fn: Loss function
            optimizer: Optimizer
            num_samples: Number of samples to use for importance estimation
        """
        new_params_old = {}
        for name, param in self.model.named_parameters():
            new_params_old[name] = param.clone().detach()
        
        new_importance = self._compute_importance(dataloader, loss_fn, optimizer, num_samples)
        
        if self._initialized and self.use_online:
            for name, importance in new_importance.items():
                if name in self.importance:
                    self.importance[name] += importance
                else:
                    self.importance[name] = importance
        else:
            self.importance = new_importance
        
        self.params_old = new_params_old
        self._initialized = True
    
    def _compute_importance(
        self, 
        dataloader: Any, 
        loss_fn: Any, 
        optimizer: Optimizer,
        num_samples: Optional[int] = None
    ) -> Dict[str, Tensor]:
        """
        Compute parameter importance using Fisher information.
        
        Args:
            dataloader: Data loader for the task
            loss_fn: Loss function
            optimizer: Optimizer
            num_samples: Number of samples to use for importance estimation
            
        Returns:
            Dictionary mapping parameter names to importance tensors
        """
        if self._has_phynexus:
            return self._phynexus.continual.compute_ewc_importance(
                self.model, dataloader, loss_fn, optimizer, num_samples
            )
        
        importance = {}
        for name, param in self.model.named_parameters():
            importance[name] = neurenix.zeros_like(param)
        
        self.model.eval()
        
        sample_count = 0
        max_samples = float('inf') if num_samples is None else num_samples
        
        for inputs, _ in dataloader:
            if sample_count >= max_samples:
                break
                
            batch_size = inputs.shape[0]
            sample_count += batch_size
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            log_probs = neurenix.log_softmax(outputs, dim=1)
            
            probs = neurenix.exp(log_probs).detach()
            samples = neurenix.multinomial(probs, 1).flatten()
            
            selected_log_probs = log_probs[neurenix.arange(batch_size), samples]
            loss = -selected_log_probs.mean()
            loss.backward()
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    importance[name] += param.grad.pow(2) * batch_size
        
        for name in importance:
            importance[name] /= max(1, sample_count)
        
        return importance
    
    def penalty(self) -> Tensor:
        """
        Compute EWC penalty term to be added to the loss.
        
        Returns:
            Tensor containing the EWC penalty
        """
        if not self._initialized:
            return neurenix.tensor(0.0)
        
        if self._has_phynexus:
            return self._phynexus.continual.compute_ewc_penalty(
                self.model, self.importance, self.params_old, self.lambda_reg
            )
        
        penalty = neurenix.tensor(0.0)
        
        for name, param in self.model.named_parameters():
            if name in self.importance and name in self.params_old:
                delta = param - self.params_old[name]
                penalty += (self.importance[name] * delta.pow(2)).sum()
        
        return 0.5 * self.lambda_reg * penalty
