"""
Explainable AI module for Neurenix.

This module provides tools and techniques for explaining and interpreting
machine learning models, making AI systems more transparent and trustworthy.
"""

from neurenix.explainable.shap import ShapExplainer, KernelShap, TreeShap, DeepShap
from neurenix.explainable.lime import LimeExplainer, LimeTabular, LimeText, LimeImage
from neurenix.explainable.feature_importance import FeatureImportance, PermutationImportance
from neurenix.explainable.partial_dependence import PartialDependence
from neurenix.explainable.counterfactual import Counterfactual
from neurenix.explainable.activation import ActivationVisualization

__all__ = [
    'ShapExplainer',
    'KernelShap',
    'TreeShap',
    'DeepShap',
    'LimeExplainer',
    'LimeTabular',
    'LimeText',
    'LimeImage',
    'FeatureImportance',
    'PermutationImportance',
    'PartialDependence',
    'Counterfactual',
    'ActivationVisualization'
]
