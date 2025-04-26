"""
Meta-learning module for the Neurenix framework.

This module provides tools and utilities for meta-learning, allowing models
to learn how to learn and adapt quickly to new tasks with minimal data.
"""

from neurenix.meta.model import MetaLearningModel
from neurenix.meta.maml import MAML
from neurenix.meta.reptile import Reptile
from neurenix.meta.prototypical import PrototypicalNetworks

__all__ = [
    'MetaLearningModel',
    'MAML',
    'Reptile',
    'PrototypicalNetworks',
]
