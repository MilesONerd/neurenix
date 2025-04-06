"""
Synaptic Intelligence implementation for continual learning.

Synaptic Intelligence is a regularization-based approach that estimates
the importance of each parameter to previous tasks and penalizes changes
to important parameters when learning new tasks.
"""

from typing import Dict, List, Optional, Tuple, Union, Any

import neurenix
from neurenix.nn import Module
from neurenix.tensor import Tensor
from neurenix.optim import Optimizer

class SynapticIntelligence:
    """
    Synaptic Intelligence for continual learning.
    
    Estimates parameter importance during training and uses this information
    to protect important parameters when learning new tasks.
    
    Attributes:
        model: The neural network model
        importance: Importance of each parameter for previous tasks
        params_old: Parameter values after learning previous tasks
        lambda_reg: Regularization strength
        damping: Damping factor to avoid division by zero
    """
    
    def __init__(
        self, 
        model: Module, 
        lambda_reg: float = 1.0,
        damping: float = 1e-3
    ):
        """
        Initialize Synaptic Intelligence.
        
        Args:
            model: Neural network model
            lambda_reg: Regularization strength
            damping: Damping factor to avoid division by zero
        """
        self.model = model
        self.lambda_reg = lambda_reg
        self.damping = damping
        self.importance: Dict[str, Tensor] = {}
        self.params_old: Dict[str, Tensor] = {}
        self._initialized = False
        
        self.omega: Dict[str, Tensor] = {}
        self.delta_theta: Dict[str, Tensor] = {}
        
        try:
            from neurenix.binding import get_phynexus
            self._phynexus = get_phynexus()
            self._has_phynexus = True
        except ImportError:
            self._has_phynexus = False
    
    def register_task(self):
        """
        Register a task by storing current parameter values and importance.
        """
        new_params_old = {}
        for name, param in self.model.named_parameters():
            new_params_old[name] = param.clone().detach()
        
        if self._initialized:
            for name, param in self.model.named_parameters():
                if name in self.omega and name in self.params_old:
                    delta = param.detach() - self.params_old[name]
                    if name in self.importance:
                        self.importance[name] += self.omega[name] / (delta.pow(2) + self.damping)
                    else:
                        self.importance[name] = self.omega[name] / (delta.pow(2) + self.damping)
        
        self.omega = {}
        self.delta_theta = {}
        for name, param in self.model.named_parameters():
            self.omega[name] = neurenix.zeros_like(param)
            self.delta_theta[name] = neurenix.zeros_like(param)
        
        self.params_old = new_params_old
        self._initialized = True
    
    def update_importance(self, loss: Tensor):
        """
        Update parameter importance based on gradients.
        
        Args:
            loss: Loss tensor
        """
        if self._has_phynexus:
            self._phynexus.continual.update_synaptic_importance(
                self.model, loss, self.omega, self.delta_theta
            )
            return
        
        loss.backward(retain_graph=True)
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().clone()
                
                if param.grad is not None:
                    self.delta_theta[name] = param.detach() - self.params_old[name]
                
                self.omega[name] -= grad * self.delta_theta[name]
    
    def penalty(self) -> Tensor:
        """
        Compute Synaptic Intelligence penalty term to be added to the loss.
        
        Returns:
            Tensor containing the penalty
        """
        if not self._initialized:
            return neurenix.tensor(0.0)
        
        if self._has_phynexus:
            return self._phynexus.continual.compute_synaptic_penalty(
                self.model, self.importance, self.params_old, self.lambda_reg
            )
        
        penalty = neurenix.tensor(0.0)
        
        for name, param in self.model.named_parameters():
            if name in self.importance and name in self.params_old:
                delta = param - self.params_old[name]
                penalty += (self.importance[name] * delta.pow(2)).sum()
        
        return 0.5 * self.lambda_reg * penalty
    
    def train_step(
        self, 
        loss: Tensor, 
        optimizer: Optimizer
    ):
        """
        Perform a training step with Synaptic Intelligence.
        
        Args:
            loss: Loss tensor
            optimizer: Optimizer
        """
        self.update_importance(loss)
        
        total_loss = loss + self.penalty()
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss
