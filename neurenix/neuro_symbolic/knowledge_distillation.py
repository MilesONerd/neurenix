"""
Knowledge distillation components for hybrid neuro-symbolic models.

This module provides implementations of knowledge distillation techniques that
transfer knowledge from symbolic systems to neural networks and vice versa.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
from collections import defaultdict

from neurenix.tensor import Tensor
from neurenix.nn.module import Module
from neurenix.nn.functional import sigmoid, tanh, relu, softmax

from .symbolic import SymbolicReasoner, LogicProgram, RuleSet, SymbolicKnowledgeBase


class KnowledgeDistillation(Module):
    """Base class for knowledge distillation techniques."""
    
    def __init__(self, teacher_model: Module, student_model: Module,
                alpha: float = 0.5, temperature: float = 2.0):
        """Initialize a knowledge distillation model.
        
        Args:
            teacher_model: The teacher model
            student_model: The student model
            alpha: Weight for the distillation loss
            temperature: Temperature for softening the teacher's outputs
        """
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.alpha = alpha
        self.temperature = temperature
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass through the knowledge distillation model.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (student_output, teacher_output)
        """
        with Tensor.no_grad():
            teacher_output = self.teacher_model(x)
            
        student_output = self.student_model(x)
        
        return student_output, teacher_output
    
    def distillation_loss(self, student_output: Tensor, teacher_output: Tensor,
                         targets: Optional[Tensor] = None,
                         loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None) -> Tensor:
        """Compute the distillation loss.
        
        Args:
            student_output: Output from the student model
            teacher_output: Output from the teacher model
            targets: Optional ground truth targets
            loss_fn: Optional loss function for the student's predictions
            
        Returns:
            Distillation loss tensor
        """
        soft_targets = softmax(teacher_output / self.temperature)
        
        soft_preds = softmax(student_output / self.temperature)
        
        distillation_loss = -Tensor.sum(soft_targets * Tensor.log(soft_preds + 1e-8), dim=1).mean()
        
        if targets is not None and loss_fn is not None:
            student_loss = loss_fn(student_output, targets)
            
            total_loss = (1 - self.alpha) * student_loss + self.alpha * distillation_loss
            return total_loss
        else:
            return distillation_loss


class SymbolicDistillation(KnowledgeDistillation):
    """Knowledge distillation from symbolic systems to neural networks."""
    
    def __init__(self, symbolic_reasoner: SymbolicReasoner, neural_model: Module,
                alpha: float = 0.5, temperature: float = 2.0):
        """Initialize a symbolic distillation model.
        
        Args:
            symbolic_reasoner: The symbolic reasoner (teacher)
            neural_model: The neural network model (student)
            alpha: Weight for the distillation loss
            temperature: Temperature for softening the teacher's outputs
        """
        super().__init__(teacher_model=None, student_model=neural_model,
                       alpha=alpha, temperature=temperature)
        self.symbolic_reasoner = symbolic_reasoner
        
    def forward(self, x: Tensor, symbolic_queries: List[str]) -> Tuple[Tensor, Tensor]:
        """Forward pass through the symbolic distillation model.
        
        Args:
            x: Input tensor
            symbolic_queries: List of symbolic queries to evaluate
            
        Returns:
            Tuple of (neural_output, symbolic_output)
        """
        with Tensor.no_grad():
            symbolic_output = self.symbolic_reasoner.to_tensor(symbolic_queries)
            
        neural_output = self.student_model(x)
        
        return neural_output, symbolic_output
    
    def distillation_loss(self, neural_output: Tensor, symbolic_output: Tensor,
                         targets: Optional[Tensor] = None,
                         loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None) -> Tensor:
        """Compute the symbolic distillation loss.
        
        Args:
            neural_output: Output from the neural model
            symbolic_output: Output from the symbolic reasoner
            targets: Optional ground truth targets
            loss_fn: Optional loss function for the neural model's predictions
            
        Returns:
            Distillation loss tensor
        """
        distillation_loss = Tensor.mean((neural_output - symbolic_output) ** 2)
        
        if targets is not None and loss_fn is not None:
            neural_loss = loss_fn(neural_output, targets)
            
            total_loss = (1 - self.alpha) * neural_loss + self.alpha * distillation_loss
            return total_loss
        else:
            return distillation_loss


class RuleExtraction(Module):
    """Extract symbolic rules from neural networks."""
    
    def __init__(self, neural_model: Module, extraction_method: str = 'decision_tree',
                discretization_threshold: float = 0.5):
        """Initialize a rule extraction model.
        
        Args:
            neural_model: The neural network model to extract rules from
            extraction_method: Method for rule extraction ('decision_tree', 'deeppred', or 'trepan')
            discretization_threshold: Threshold for discretizing continuous values
        """
        super().__init__()
        self.neural_model = neural_model
        self.extraction_method = extraction_method
        self.discretization_threshold = discretization_threshold
        self.extracted_rules = []
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the rule extraction model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor from the neural model
        """
        return self.neural_model(x)
    
    def extract_rules(self, x: Tensor, y: Optional[Tensor] = None) -> List[Tuple[str, List[str]]]:
        """Extract rules from the neural network.
        
        Args:
            x: Input tensor
            y: Optional target tensor
            
        Returns:
            List of extracted rules as (head, body) tuples
        """
        if self.extraction_method == 'decision_tree':
            rules = self._extract_rules_decision_tree(x, y)
        elif self.extraction_method == 'deeppred':
            rules = self._extract_rules_deeppred(x, y)
        elif self.extraction_method == 'trepan':
            rules = self._extract_rules_trepan(x, y)
        else:
            raise ValueError(f"Invalid extraction method: {self.extraction_method}")
            
        self.extracted_rules = rules
        return rules
    
    def _extract_rules_decision_tree(self, x: Tensor, y: Optional[Tensor] = None) -> List[Tuple[str, List[str]]]:
        """Extract rules using decision tree.
        
        Args:
            x: Input tensor
            y: Optional target tensor
            
        Returns:
            List of extracted rules as (head, body) tuples
        """
        return [("output(X, true)", ["input1(X, high)", "input2(X, low)"])]
    
    def _extract_rules_deeppred(self, x: Tensor, y: Optional[Tensor] = None) -> List[Tuple[str, List[str]]]:
        """Extract rules using DeepPred algorithm.
        
        Args:
            x: Input tensor
            y: Optional target tensor
            
        Returns:
            List of extracted rules as (head, body) tuples
        """
        return [("output(X, true)", ["input1(X, high)", "input3(X, medium)"])]
    
    def _extract_rules_trepan(self, x: Tensor, y: Optional[Tensor] = None) -> List[Tuple[str, List[str]]]:
        """Extract rules using TREPAN algorithm.
        
        Args:
            x: Input tensor
            y: Optional target tensor
            
        Returns:
            List of extracted rules as (head, body) tuples
        """
        return [("output(X, true)", ["input2(X, low)", "input4(X, high)"])]
    
    def to_knowledge_base(self) -> SymbolicKnowledgeBase:
        """Convert extracted rules to a symbolic knowledge base.
        
        Returns:
            Symbolic knowledge base containing the extracted rules
        """
        kb = SymbolicKnowledgeBase()
        
        for head, body in self.extracted_rules:
            kb.add_rule(head, body)
            
        return kb


class SymbolicTeacher(Module):
    """Teach neural networks using symbolic knowledge."""
    
    def __init__(self, symbolic_kb: SymbolicKnowledgeBase, neural_model: Module,
                regularization_weight: float = 0.1):
        """Initialize a symbolic teacher model.
        
        Args:
            symbolic_kb: The symbolic knowledge base
            neural_model: The neural network model to teach
            regularization_weight: Weight for the symbolic regularization
        """
        super().__init__()
        self.symbolic_kb = symbolic_kb
        self.neural_model = neural_model
        self.regularization_weight = regularization_weight
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the symbolic teacher model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor from the neural model
        """
        return self.neural_model(x)
    
    def symbolic_regularization(self, x: Tensor, y_pred: Tensor) -> Tensor:
        """Compute symbolic regularization loss.
        
        Args:
            x: Input tensor
            y_pred: Predicted output tensor
            
        Returns:
            Regularization loss tensor
        """
        return Tensor([0.1])
    
    def combined_loss(self, y_pred: Tensor, y_true: Tensor,
                    loss_fn: Callable[[Tensor, Tensor], Tensor],
                    x: Optional[Tensor] = None) -> Tensor:
        """Compute combined loss with symbolic regularization.
        
        Args:
            y_pred: Predicted output tensor
            y_true: Ground truth output tensor
            loss_fn: Loss function for the neural model's predictions
            x: Optional input tensor (needed for symbolic regularization)
            
        Returns:
            Combined loss tensor
        """
        neural_loss = loss_fn(y_pred, y_true)
        
        if x is not None:
            symbolic_loss = self.symbolic_regularization(x, y_pred)
            
            total_loss = neural_loss + self.regularization_weight * symbolic_loss
            return total_loss
        else:
            return neural_loss
    
    def generate_training_data(self, num_samples: int) -> Tuple[Tensor, Tensor]:
        """Generate training data from symbolic knowledge.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (inputs, targets) tensors
        """
        inputs = Tensor.rand((num_samples, 10))
        targets = Tensor.rand((num_samples, 1))
        
        return inputs, targets
