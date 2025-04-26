"""
AutoML module for the Neurenix framework.

This module provides tools and utilities for automated machine learning,
allowing for automatic model selection, hyperparameter optimization,
and neural architecture search.
"""

from neurenix.automl.search import (
    HyperparameterSearch,
    GridSearch,
    RandomSearch,
    BayesianOptimization,
    EvolutionarySearch
)
from neurenix.automl.nas import (
    NeuralArchitectureSearch,
    ENAS,
    DARTS,
    PNAS
)
from neurenix.automl.model_selection import (
    AutoModelSelection,
    CrossValidation,
    NestedCrossValidation
)
from neurenix.automl.pipeline import (
    AutoPipeline,
    FeatureSelection,
    DataPreprocessing
)

__all__ = [
    'HyperparameterSearch',
    'GridSearch',
    'RandomSearch',
    'BayesianOptimization',
    'EvolutionarySearch',
    'NeuralArchitectureSearch',
    'ENAS',
    'DARTS',
    'PNAS',
    'AutoModelSelection',
    'CrossValidation',
    'NestedCrossValidation',
    'AutoPipeline',
    'FeatureSelection',
    'DataPreprocessing',
]
