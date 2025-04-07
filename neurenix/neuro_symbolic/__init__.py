"""
Neuro-Symbolic module for Neurenix.

This module provides implementations of hybrid neuro-symbolic models
that combine neural networks with symbolic reasoning systems.
"""

from .symbolic import SymbolicSystem, LogicProgram, RuleSet
from .neural_symbolic import NeuralSymbolicModel, DifferentiableNeuralComputer
from .differentiable_logic import DifferentiableLogic, TensorLogic, FuzzyLogic
from .reasoning import SymbolicReasoner, NeuralReasoner, HybridReasoner
from .knowledge_distillation import KnowledgeDistillation, RuleExtraction

__all__ = [
    'SymbolicSystem', 'LogicProgram', 'RuleSet',
    'NeuralSymbolicModel', 'DifferentiableNeuralComputer',
    'DifferentiableLogic', 'TensorLogic', 'FuzzyLogic',
    'SymbolicReasoner', 'NeuralReasoner', 'HybridReasoner',
    'KnowledgeDistillation', 'RuleExtraction',
]
