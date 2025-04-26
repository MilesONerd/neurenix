"""
Zero-shot learning models for Neurenix.

This module provides model architectures for zero-shot learning, enabling
recognition of unseen classes based on semantic descriptions or attributes.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any

import neurenix
from neurenix.tensor import Tensor
from neurenix.nn import Module, Linear, Dropout, BatchNorm1d, ReLU, Sequential
from neurenix.nn.functional import cosine_similarity


class ZeroShotModel(Module):
    """
    Base class for zero-shot learning models.
    
    Zero-shot learning models can recognize objects or classes that were not seen
    during training by leveraging semantic information about classes.
    """
    
    def __init__(self, 
                 visual_dim: int, 
                 semantic_dim: int, 
                 hidden_dim: int = 512,
                 dropout: float = 0.2):
        """
        Initialize a zero-shot learning model.
        
        Args:
            visual_dim: Dimension of visual features
            semantic_dim: Dimension of semantic features (class embeddings)
            hidden_dim: Dimension of hidden layers
            dropout: Dropout probability
        """
        super().__init__()
        self.visual_dim = visual_dim
        self.semantic_dim = semantic_dim
        self.hidden_dim = hidden_dim
        
        self.visual_embedding = Sequential(
            Linear(visual_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim)
        )
        
        self.semantic_embedding = Sequential(
            Linear(semantic_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Dropout(dropout),
            Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, 
                visual_features: Tensor, 
                semantic_features: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through the zero-shot model.
        
        Args:
            visual_features: Visual features of shape (batch_size, visual_dim)
            semantic_features: Semantic features of shape (num_classes, semantic_dim)
                               If None, only visual embeddings are returned
        
        Returns:
            If semantic_features is None:
                visual_embeddings: Embedded visual features of shape (batch_size, hidden_dim)
            Else:
                compatibility: Compatibility scores between visual and semantic features
                               of shape (batch_size, num_classes)
                visual_embeddings: Embedded visual features of shape (batch_size, hidden_dim)
        """
        visual_embeddings = self.visual_embedding(visual_features)
        
        if semantic_features is None:
            return visual_embeddings
        
        semantic_embeddings = self.semantic_embedding(semantic_features)
        
        compatibility = cosine_similarity(
            visual_embeddings.unsqueeze(1),  # (batch_size, 1, hidden_dim)
            semantic_embeddings.unsqueeze(0)  # (1, num_classes, hidden_dim)
        )  # (batch_size, num_classes)
        
        return compatibility, visual_embeddings
    
    def predict(self, 
                visual_features: Tensor, 
                semantic_features: Tensor) -> Tensor:
        """
        Predict class labels for visual features.
        
        Args:
            visual_features: Visual features of shape (batch_size, visual_dim)
            semantic_features: Semantic features of shape (num_classes, semantic_dim)
        
        Returns:
            predictions: Predicted class indices of shape (batch_size,)
        """
        compatibility, _ = self.forward(visual_features, semantic_features)
        predictions = compatibility.argmax(dim=1)
        return predictions


class ZeroShotTransformer(ZeroShotModel):
    """
    Transformer-based zero-shot learning model.
    
    This model uses transformer architecture for cross-modal alignment
    between visual and semantic features.
    """
    
    def __init__(self, 
                 visual_dim: int, 
                 semantic_dim: int, 
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        """
        Initialize a transformer-based zero-shot learning model.
        
        Args:
            visual_dim: Dimension of visual features
            semantic_dim: Dimension of semantic features (class embeddings)
            hidden_dim: Dimension of hidden layers
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__(visual_dim, semantic_dim, hidden_dim, dropout)
        
        from neurenix.nn import TransformerEncoder, TransformerEncoderLayer
        
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        self.visual_transformer = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout
        )
        self.semantic_transformer = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
    
    def forward(self, 
                visual_features: Tensor, 
                semantic_features: Optional[Tensor] = None) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through the transformer-based zero-shot model.
        
        Args:
            visual_features: Visual features of shape (batch_size, visual_dim)
            semantic_features: Semantic features of shape (num_classes, semantic_dim)
                               If None, only visual embeddings are returned
        
        Returns:
            If semantic_features is None:
                visual_embeddings: Embedded visual features of shape (batch_size, hidden_dim)
            Else:
                compatibility: Compatibility scores between visual and semantic features
                               of shape (batch_size, num_classes)
                visual_embeddings: Embedded visual features of shape (batch_size, hidden_dim)
        """
        visual_embeddings = self.visual_embedding(visual_features)
        
        visual_embeddings = visual_embeddings.unsqueeze(1)  # Add sequence dimension
        visual_embeddings = self.visual_transformer(visual_embeddings)
        visual_embeddings = visual_embeddings.squeeze(1)  # Remove sequence dimension
        
        if semantic_features is None:
            return visual_embeddings
        
        semantic_embeddings = self.semantic_embedding(semantic_features)
        
        semantic_embeddings = semantic_embeddings.unsqueeze(1)  # Add sequence dimension
        semantic_embeddings = self.semantic_transformer(semantic_embeddings)
        semantic_embeddings = semantic_embeddings.squeeze(1)  # Remove sequence dimension
        
        compatibility = cosine_similarity(
            visual_embeddings.unsqueeze(1),  # (batch_size, 1, hidden_dim)
            semantic_embeddings.unsqueeze(0)  # (1, num_classes, hidden_dim)
        )  # (batch_size, num_classes)
        
        return compatibility, visual_embeddings
