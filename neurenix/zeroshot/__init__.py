"""
Zero-shot Learning module for Neurenix.

This module provides tools and techniques for zero-shot learning, enabling models
to recognize objects or classes that were not seen during training.
"""

from neurenix.zeroshot.model import ZeroShotModel, ZeroShotTransformer
from neurenix.zeroshot.embedding import EmbeddingModel, TextEncoder, ImageEncoder, CrossModalEncoder
from neurenix.zeroshot.classifier import ZeroShotClassifier, AttributeClassifier, SemanticClassifier
from neurenix.zeroshot.utils import semantic_similarity, attribute_mapping, class_embedding

__all__ = [
    'ZeroShotModel',
    'ZeroShotTransformer',
    'EmbeddingModel',
    'TextEncoder',
    'ImageEncoder',
    'CrossModalEncoder',
    'ZeroShotClassifier',
    'AttributeClassifier',
    'SemanticClassifier',
    'semantic_similarity',
    'attribute_mapping',
    'class_embedding'
]
