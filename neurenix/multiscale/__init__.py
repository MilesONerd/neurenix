"""
Multi-Scale Models module for Neurenix.

This module provides tools and techniques for working with data at multiple scales
and resolutions, enabling more efficient and effective model training and inference.
"""

from neurenix.multiscale.models import MultiScaleModel, PyramidNetwork, UNet
from neurenix.multiscale.pooling import MultiScalePooling, PyramidPooling, SpatialPyramidPooling
from neurenix.multiscale.fusion import FeatureFusion, ScaleFusion, AttentionFusion
from neurenix.multiscale.transforms import MultiScaleTransform, Rescale, PyramidDownsample

__all__ = [
    'MultiScaleModel',
    'PyramidNetwork',
    'UNet',
    'MultiScalePooling',
    'PyramidPooling',
    'SpatialPyramidPooling',
    'FeatureFusion',
    'ScaleFusion',
    'AttentionFusion',
    'MultiScaleTransform',
    'Rescale',
    'PyramidDownsample'
]
