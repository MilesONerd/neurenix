"""
Differentiable logic components for hybrid neuro-symbolic models.

This module provides implementations of differentiable logic systems that can be
integrated with neural networks to create hybrid neuro-symbolic models with
end-to-end differentiability.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
from collections import defaultdict

from neurenix.tensor import Tensor
from neurenix.nn.module import Module
from neurenix.nn.functional import sigmoid, tanh, relu, softmax


class LogicTensor(Tensor):
    """A tensor with logical operations for differentiable logic."""
    
    @staticmethod
    def fuzzy_and(x: Tensor, y: Tensor) -> Tensor:
        """Fuzzy AND operation (t-norm).
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Tensor with fuzzy AND operation applied
        """
        return x * y
    
    @staticmethod
    def fuzzy_or(x: Tensor, y: Tensor) -> Tensor:
        """Fuzzy OR operation (t-conorm).
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Tensor with fuzzy OR operation applied
        """
        return x + y - x * y
    
    @staticmethod
    def fuzzy_not(x: Tensor) -> Tensor:
        """Fuzzy NOT operation (negation).
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with fuzzy NOT operation applied
        """
        return 1.0 - x
    
    @staticmethod
    def fuzzy_implies(x: Tensor, y: Tensor) -> Tensor:
        """Fuzzy IMPLIES operation (implication).
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Tensor with fuzzy IMPLIES operation applied
        """
        return LogicTensor.fuzzy_or(LogicTensor.fuzzy_not(x), y)
    
    @staticmethod
    def probabilistic_and(x: Tensor, y: Tensor) -> Tensor:
        """Probabilistic AND operation.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Tensor with probabilistic AND operation applied
        """
        return x * y
    
    @staticmethod
    def probabilistic_or(x: Tensor, y: Tensor) -> Tensor:
        """Probabilistic OR operation.
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Tensor with probabilistic OR operation applied
        """
        return x + y - x * y
    
    @staticmethod
    def probabilistic_not(x: Tensor) -> Tensor:
        """Probabilistic NOT operation.
        
        Args:
            x: Input tensor
            
        Returns:
            Tensor with probabilistic NOT operation applied
        """
        return 1.0 - x
    
    @staticmethod
    def lukasiewicz_and(x: Tensor, y: Tensor) -> Tensor:
        """Lukasiewicz AND operation (t-norm).
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Tensor with Lukasiewicz AND operation applied
        """
        return Tensor.maximum(x + y - 1.0, Tensor.zeros_like(x))
    
    @staticmethod
    def lukasiewicz_or(x: Tensor, y: Tensor) -> Tensor:
        """Lukasiewicz OR operation (t-conorm).
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Tensor with Lukasiewicz OR operation applied
        """
        return Tensor.minimum(x + y, Tensor.ones_like(x))
    
    @staticmethod
    def godel_and(x: Tensor, y: Tensor) -> Tensor:
        """Gödel AND operation (t-norm).
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Tensor with Gödel AND operation applied
        """
        return Tensor.minimum(x, y)
    
    @staticmethod
    def godel_or(x: Tensor, y: Tensor) -> Tensor:
        """Gödel OR operation (t-conorm).
        
        Args:
            x: First tensor
            y: Second tensor
            
        Returns:
            Tensor with Gödel OR operation applied
        """
        return Tensor.maximum(x, y)


class DifferentiableLogic(Module):
    """Base class for differentiable logic systems."""
    
    def __init__(self, logic_type: str = 'fuzzy'):
        """Initialize a differentiable logic system.
        
        Args:
            logic_type: Type of logic to use ('fuzzy', 'probabilistic', 'lukasiewicz', or 'godel')
        """
        super().__init__()
        self.logic_type = logic_type
        
        if logic_type == 'fuzzy':
            self.and_op = LogicTensor.fuzzy_and
            self.or_op = LogicTensor.fuzzy_or
            self.not_op = LogicTensor.fuzzy_not
            self.implies_op = LogicTensor.fuzzy_implies
        elif logic_type == 'probabilistic':
            self.and_op = LogicTensor.probabilistic_and
            self.or_op = LogicTensor.probabilistic_or
            self.not_op = LogicTensor.probabilistic_not
            self.implies_op = lambda x, y: self.or_op(self.not_op(x), y)
        elif logic_type == 'lukasiewicz':
            self.and_op = LogicTensor.lukasiewicz_and
            self.or_op = LogicTensor.lukasiewicz_or
            self.not_op = LogicTensor.fuzzy_not
            self.implies_op = lambda x, y: self.or_op(self.not_op(x), y)
        elif logic_type == 'godel':
            self.and_op = LogicTensor.godel_and
            self.or_op = LogicTensor.godel_or
            self.not_op = LogicTensor.fuzzy_not
            self.implies_op = lambda x, y: Tensor.ones_like(x) if (x <= y).all() else y
        else:
            raise ValueError(f"Invalid logic type: {logic_type}")
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the differentiable logic system.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        raise NotImplementedError("Subclasses must implement forward method")


class FuzzyLogic(DifferentiableLogic):
    """Fuzzy logic system for differentiable reasoning."""
    
    def __init__(self, logic_type: str = 'fuzzy'):
        """Initialize a fuzzy logic system.
        
        Args:
            logic_type: Type of fuzzy logic to use ('fuzzy', 'lukasiewicz', or 'godel')
        """
        super().__init__(logic_type)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the fuzzy logic system.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return x
    
    def evaluate_rule(self, antecedent: Tensor, consequent: Tensor) -> Tensor:
        """Evaluate a fuzzy logic rule.
        
        Args:
            antecedent: Antecedent tensor
            consequent: Consequent tensor
            
        Returns:
            Rule evaluation tensor
        """
        return self.implies_op(antecedent, consequent)
    
    def evaluate_conjunction(self, tensors: List[Tensor]) -> Tensor:
        """Evaluate a conjunction of tensors.
        
        Args:
            tensors: List of tensors to conjoin
            
        Returns:
            Conjunction tensor
        """
        if not tensors:
            return Tensor([1.0])
        
        result = tensors[0]
        for tensor in tensors[1:]:
            result = self.and_op(result, tensor)
            
        return result
    
    def evaluate_disjunction(self, tensors: List[Tensor]) -> Tensor:
        """Evaluate a disjunction of tensors.
        
        Args:
            tensors: List of tensors to disjoin
            
        Returns:
            Disjunction tensor
        """
        if not tensors:
            return Tensor([0.0])
        
        result = tensors[0]
        for tensor in tensors[1:]:
            result = self.or_op(result, tensor)
            
        return result
    
    def evaluate_negation(self, tensor: Tensor) -> Tensor:
        """Evaluate the negation of a tensor.
        
        Args:
            tensor: Tensor to negate
            
        Returns:
            Negation tensor
        """
        return self.not_op(tensor)


class ProbabilisticLogic(DifferentiableLogic):
    """Probabilistic logic system for differentiable reasoning."""
    
    def __init__(self):
        """Initialize a probabilistic logic system."""
        super().__init__(logic_type='probabilistic')
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the probabilistic logic system.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return x
    
    def joint_probability(self, x: Tensor, y: Tensor, 
                         conditional_prob: Optional[Tensor] = None) -> Tensor:
        """Compute the joint probability of two tensors.
        
        Args:
            x: First tensor
            y: Second tensor
            conditional_prob: Optional conditional probability tensor
            
        Returns:
            Joint probability tensor
        """
        if conditional_prob is not None:
            return x * conditional_prob
        else:
            return x * y
    
    def conditional_probability(self, joint_prob: Tensor, 
                              marginal_prob: Tensor) -> Tensor:
        """Compute the conditional probability.
        
        Args:
            joint_prob: Joint probability tensor
            marginal_prob: Marginal probability tensor
            
        Returns:
            Conditional probability tensor
        """
        return joint_prob / (marginal_prob + 1e-8)
    
    def marginal_probability(self, joint_probs: List[Tensor]) -> Tensor:
        """Compute the marginal probability.
        
        Args:
            joint_probs: List of joint probability tensors
            
        Returns:
            Marginal probability tensor
        """
        return sum(joint_probs)
    
    def bayes_rule(self, likelihood: Tensor, prior: Tensor, 
                 evidence: Tensor) -> Tensor:
        """Apply Bayes' rule.
        
        Args:
            likelihood: Likelihood tensor
            prior: Prior tensor
            evidence: Evidence tensor
            
        Returns:
            Posterior tensor
        """
        return (likelihood * prior) / (evidence + 1e-8)
