"""
Continual learning module for Neurenix.

This module provides functionality for continual learning, allowing models
to be trained on new data without forgetting previously learned knowledge.
"""

from .ewc import EWC
from .replay import ExperienceReplay
from .regularization import L2Regularization
from .distillation import KnowledgeDistillation
from .synaptic import SynapticIntelligence

__all__ = [
    'EWC',
    'ExperienceReplay',
    'L2Regularization',
    'KnowledgeDistillation',
    'SynapticIntelligence'
]
