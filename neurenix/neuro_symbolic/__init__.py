"""
Hybrid Neuro-Symbolic Models for Neurenix.

This module provides implementations of hybrid neuro-symbolic models that combine
neural networks with symbolic reasoning systems for improved interpretability,
data efficiency, and reasoning capabilities.
"""

from .symbolic import (
    SymbolicReasoner,
    LogicProgram,
    RuleSet,
    SymbolicKnowledgeBase
)

from .neural_symbolic import (
    NeuralSymbolicModel,
    NeuralSymbolicLoss,
    NeuralSymbolicTrainer,
    NeuralSymbolicInference
)

from .differentiable_logic import (
    DifferentiableLogic,
    FuzzyLogic,
    ProbabilisticLogic,
    LogicTensor
)

from .knowledge_distillation import (
    KnowledgeDistillation,
    SymbolicDistillation,
    RuleExtraction,
    SymbolicTeacher
)

from .reasoning import (
    ConstraintSatisfaction,
    LogicalInference,
    AbductiveReasoning,
    DeductiveReasoning,
    InductiveReasoning
)

__all__ = [
    'SymbolicReasoner',
    'LogicProgram',
    'RuleSet',
    'SymbolicKnowledgeBase',
    
    'NeuralSymbolicModel',
    'NeuralSymbolicLoss',
    'NeuralSymbolicTrainer',
    'NeuralSymbolicInference',
    
    'DifferentiableLogic',
    'FuzzyLogic',
    'ProbabilisticLogic',
    'LogicTensor',
    
    'KnowledgeDistillation',
    'SymbolicDistillation',
    'RuleExtraction',
    'SymbolicTeacher',
    
    'ConstraintSatisfaction',
    'LogicalInference',
    'AbductiveReasoning',
    'DeductiveReasoning',
    'InductiveReasoning'
]
