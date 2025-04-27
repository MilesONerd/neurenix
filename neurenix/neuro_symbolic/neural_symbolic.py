"""
Neural-symbolic integration components for hybrid neuro-symbolic models.

This module provides implementations of neural-symbolic integration components
that combine neural networks with symbolic reasoning systems for improved
interpretability, data efficiency, and reasoning capabilities.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
from collections import defaultdict

from neurenix.tensor import Tensor
from neurenix.nn.module import Module
from neurenix.nn.functional import sigmoid, tanh, relu, softmax

from .symbolic import SymbolicReasoner, LogicProgram, RuleSet, SymbolicKnowledgeBase


class NeuralSymbolicModel(Module):
    """A hybrid neural-symbolic model that combines neural networks with symbolic reasoning."""
    
    def __init__(self, neural_model: Module, symbolic_reasoner: SymbolicReasoner,
                integration_mode: str = 'sequential'):
        """Initialize a neural-symbolic model.
        
        Args:
            neural_model: The neural network component
            symbolic_reasoner: The symbolic reasoning component
            integration_mode: How to integrate neural and symbolic components
                ('sequential', 'parallel', or 'interactive')
        """
        super().__init__()
        self.neural_model = neural_model
        self.symbolic_reasoner = symbolic_reasoner
        self.integration_mode = integration_mode
        
        if integration_mode not in ['sequential', 'parallel', 'interactive']:
            raise ValueError(f"Invalid integration mode: {integration_mode}")
        
    def forward(self, x: Tensor, symbolic_queries: Optional[List[str]] = None) -> Tensor:
        """Forward pass through the neural-symbolic model.
        
        Args:
            x: Input tensor for the neural network
            symbolic_queries: Optional list of symbolic queries to evaluate
            
        Returns:
            Output tensor from the neural-symbolic model
        """
        if self.integration_mode == 'sequential':
            neural_output = self.neural_model(x)
            
            if symbolic_queries is not None:
                symbolic_output = self.symbolic_reasoner.to_tensor(symbolic_queries)
                return Tensor.cat([neural_output, symbolic_output.unsqueeze(1)], dim=1)
            else:
                return neural_output
                
        elif self.integration_mode == 'parallel':
            neural_output = self.neural_model(x)
            
            if symbolic_queries is not None:
                symbolic_output = self.symbolic_reasoner.to_tensor(symbolic_queries)
                
                combined_output = neural_output * (1.0 + symbolic_output.unsqueeze(1))
                return combined_output
            else:
                return neural_output
                
        elif self.integration_mode == 'interactive':
            neural_output = self.neural_model(x)
            
            if symbolic_queries is not None:
                informed_queries = self._inform_symbolic_queries(neural_output, symbolic_queries)
                symbolic_output = self.symbolic_reasoner.to_tensor(informed_queries)
                
                refined_output = self._refine_neural_output(neural_output, symbolic_output)
                return refined_output
            else:
                return neural_output
    
    def _inform_symbolic_queries(self, neural_output: Tensor, 
                               symbolic_queries: List[str]) -> List[str]:
        """Use neural output to inform symbolic queries.
        
        Args:
            neural_output: Output tensor from the neural network
            symbolic_queries: List of symbolic queries to evaluate
            
        Returns:
            List of informed symbolic queries
        """
        return symbolic_queries
    
    def _refine_neural_output(self, neural_output: Tensor, 
                            symbolic_output: Tensor) -> Tensor:
        """Use symbolic output to refine neural output.
        
        Args:
            neural_output: Output tensor from the neural network
            symbolic_output: Output tensor from symbolic reasoning
            
        Returns:
            Refined output tensor
        """
        return neural_output * (1.0 + symbolic_output.unsqueeze(1))


class NeuralSymbolicLoss(Module):
    """Loss function for training neural-symbolic models."""
    
    def __init__(self, neural_loss_fn: Callable[[Tensor, Tensor], Tensor],
                symbolic_weight: float = 0.5):
        """Initialize a neural-symbolic loss function.
        
        Args:
            neural_loss_fn: Loss function for the neural network component
            symbolic_weight: Weight for the symbolic consistency loss
        """
        super().__init__()
        self.neural_loss_fn = neural_loss_fn
        self.symbolic_weight = symbolic_weight
        
    def forward(self, y_pred: Tensor, y_true: Tensor, 
               symbolic_constraints: Optional[List[Tuple[str, bool]]] = None) -> Tensor:
        """Compute the neural-symbolic loss.
        
        Args:
            y_pred: Predicted output from the neural-symbolic model
            y_true: Ground truth output
            symbolic_constraints: Optional list of (constraint, expected_value) tuples
            
        Returns:
            Loss tensor
        """
        neural_loss = self.neural_loss_fn(y_pred, y_true)
        
        if symbolic_constraints is not None:
            symbolic_loss = self._compute_symbolic_loss(y_pred, symbolic_constraints)
            total_loss = neural_loss + self.symbolic_weight * symbolic_loss
            return total_loss
        else:
            return neural_loss
    
    def _compute_symbolic_loss(self, y_pred: Tensor, 
                             symbolic_constraints: List[Tuple[str, bool]]) -> Tensor:
        """Compute the symbolic consistency loss.
        
        Args:
            y_pred: Predicted output from the neural-symbolic model
            symbolic_constraints: List of (constraint, expected_value) tuples
            
        Returns:
            Symbolic consistency loss tensor
        """
        constraint_loss = Tensor([0.0])
        
        for constraint, expected_value in symbolic_constraints:
            if constraint.startswith("output["):
                parts = constraint.split("]")
                if len(parts) >= 2:
                    idx_str = parts[0].replace("output[", "").strip()
                    try:
                        idx = int(idx_str)
                        if idx < y_pred.shape[1]:
                            predicted_value = y_pred[0, idx]
                        else:
                            predicted_value = Tensor([0.5])
                    except ValueError:
                        predicted_value = Tensor([0.5])
                else:
                    predicted_value = Tensor([0.5])
            else:
                predicted_value = Tensor([0.5])
                
            constraint_loss = constraint_loss + (predicted_value - float(expected_value)) ** 2
            
        return constraint_loss / len(symbolic_constraints) if symbolic_constraints else Tensor([0.0])


class NeuralSymbolicTrainer:
    """Trainer for neural-symbolic models."""
    
    def __init__(self, model: NeuralSymbolicModel, loss_fn: NeuralSymbolicLoss,
                optimizer: Any, device: str = 'cpu'):
        """Initialize a neural-symbolic trainer.
        
        Args:
            model: The neural-symbolic model to train
            loss_fn: The loss function to use
            optimizer: The optimizer to use
            device: The device to use for training ('cpu' or 'cuda')
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        
    def train_step(self, x: Tensor, y: Tensor, 
                  symbolic_queries: Optional[List[str]] = None,
                  symbolic_constraints: Optional[List[Tuple[str, bool]]] = None) -> float:
        """Perform a single training step.
        
        Args:
            x: Input tensor
            y: Target tensor
            symbolic_queries: Optional list of symbolic queries to evaluate
            symbolic_constraints: Optional list of (constraint, expected_value) tuples
            
        Returns:
            Loss value
        """
        y_pred = self.model(x, symbolic_queries)
        
        loss = self.loss_fn(y_pred, y, symbolic_constraints)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, dataloader: Any, epochs: int, 
             symbolic_queries: Optional[List[str]] = None,
             symbolic_constraints: Optional[List[Tuple[str, bool]]] = None,
             validation_dataloader: Optional[Any] = None,
             callbacks: Optional[List[Callable]] = None) -> Dict[str, List[float]]:
        """Train the neural-symbolic model.
        
        Args:
            dataloader: DataLoader for training data
            epochs: Number of epochs to train for
            symbolic_queries: Optional list of symbolic queries to evaluate
            symbolic_constraints: Optional list of (constraint, expected_value) tuples
            validation_dataloader: Optional DataLoader for validation data
            callbacks: Optional list of callback functions
            
        Returns:
            Dictionary of training metrics
        """
        metrics = {
            'train_loss': [],
            'val_loss': []
        }
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in dataloader:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                
                loss = self.train_step(x, y, symbolic_queries, symbolic_constraints)
                epoch_loss += loss
                num_batches += 1
            
            avg_train_loss = epoch_loss / num_batches
            metrics['train_loss'].append(avg_train_loss)
            
            if validation_dataloader is not None:
                val_loss = 0.0
                num_val_batches = 0
                
                for batch in validation_dataloader:
                    x, y = batch
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    with Tensor.no_grad():
                        y_pred = self.model(x, symbolic_queries)
                        loss = self.loss_fn(y_pred, y, symbolic_constraints)
                        
                    val_loss += loss.item()
                    num_val_batches += 1
                
                avg_val_loss = val_loss / num_val_batches
                metrics['val_loss'].append(avg_val_loss)
            
            if callbacks is not None:
                for callback in callbacks:
                    callback(epoch, metrics)
        
        return metrics


class NeuralSymbolicInference:
    """Inference for neural-symbolic models."""
    
    def __init__(self, model: NeuralSymbolicModel, device: str = 'cpu'):
        """Initialize a neural-symbolic inference engine.
        
        Args:
            model: The neural-symbolic model to use for inference
            device: The device to use for inference ('cpu' or 'cuda')
        """
        self.model = model
        self.device = device
        
    def predict(self, x: Tensor, symbolic_queries: Optional[List[str]] = None) -> Tensor:
        """Perform inference with the neural-symbolic model.
        
        Args:
            x: Input tensor
            symbolic_queries: Optional list of symbolic queries to evaluate
            
        Returns:
            Predicted output tensor
        """
        x = x.to(self.device)
        
        with Tensor.no_grad():
            y_pred = self.model(x, symbolic_queries)
            
        return y_pred
    
    def explain(self, x: Tensor, symbolic_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """Explain the prediction of the neural-symbolic model.
        
        Args:
            x: Input tensor
            symbolic_queries: Optional list of symbolic queries to evaluate
            
        Returns:
            Dictionary of explanation components
        """
        y_pred = self.predict(x, symbolic_queries)
        
        neural_explanation = self._explain_neural(x, y_pred)
        
        symbolic_explanation = self._explain_symbolic(x, symbolic_queries)
        
        return {
            'prediction': y_pred,
            'neural_explanation': neural_explanation,
            'symbolic_explanation': symbolic_explanation
        }
    
    def _explain_neural(self, x: Tensor, y_pred: Tensor) -> Dict[str, Any]:
        """Explain the neural component of the prediction.
        
        Args:
            x: Input tensor
            y_pred: Predicted output tensor
            
        Returns:
            Dictionary of neural explanation components
        """
        return {
            'input': x.cpu().numpy(),
            'output': y_pred.cpu().numpy()
        }
    
    def _explain_symbolic(self, x: Tensor, symbolic_queries: Optional[List[str]]) -> Dict[str, Any]:
        """Explain the symbolic component of the prediction.
        
        Args:
            x: Input tensor
            symbolic_queries: Optional list of symbolic queries to evaluate
            
        Returns:
            Dictionary of symbolic explanation components
        """
        if symbolic_queries is not None:
            return {
                'queries': symbolic_queries,
                'results': [self.model.symbolic_reasoner.reason(query) for query in symbolic_queries]
            }
        else:
            return {}
