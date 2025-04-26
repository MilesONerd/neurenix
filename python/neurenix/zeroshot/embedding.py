"""
Embedding models for zero-shot learning in Neurenix.

This module provides embedding models for zero-shot learning, which map
inputs (images, text, etc.) and class descriptions into a shared embedding space.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any

import neurenix
from neurenix.tensor import Tensor
from neurenix.nn import Module, Linear, Dropout, BatchNorm1d, ReLU, Sequential, Embedding


class EmbeddingModel(Module):
    """
    Base class for embedding models used in zero-shot learning.
    
    Embedding models map inputs (images, text, etc.) into a shared embedding space
    where semantic relationships can be captured and exploited for zero-shot learning.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 embedding_dim: int, 
                 hidden_dims: List[int] = None,
                 dropout: float = 0.2,
                 normalize: bool = True):
        """
        Initialize an embedding model.
        
        Args:
            input_dim: Dimension of input features
            embedding_dim: Dimension of the embedding space
            hidden_dims: Dimensions of hidden layers
            dropout: Dropout probability
            normalize: Whether to L2-normalize the embeddings
        """
        super().__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.normalize = normalize
        
        if hidden_dims is None:
            hidden_dims = [input_dim * 2, input_dim]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(Linear(prev_dim, hidden_dim))
            layers.append(BatchNorm1d(hidden_dim))
            layers.append(ReLU())
            layers.append(Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(Linear(prev_dim, embedding_dim))
        
        self.embedding_network = Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the embedding model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        
        Returns:
            embeddings: Embedded features of shape (batch_size, embedding_dim)
        """
        embeddings = self.embedding_network(x)
        
        if self.normalize:
            norm = embeddings.norm(dim=1, keepdim=True)
            embeddings = embeddings / (norm + 1e-10)
        
        return embeddings


class TextEncoder(EmbeddingModel):
    """
    Text encoder for zero-shot learning.
    
    This model encodes text descriptions (e.g., class names, attributes)
    into the shared embedding space.
    """
    
    def __init__(self, 
                 vocab_size: int, 
                 embedding_dim: int, 
                 hidden_dims: List[int] = None,
                 max_seq_length: int = 100,
                 dropout: float = 0.2,
                 normalize: bool = True):
        """
        Initialize a text encoder.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of the embedding space
            hidden_dims: Dimensions of hidden layers
            max_seq_length: Maximum sequence length
            dropout: Dropout probability
            normalize: Whether to L2-normalize the embeddings
        """
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        
        self.word_embedding = Embedding(vocab_size, embedding_dim)
        
        super().__init__(
            input_dim=embedding_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            normalize=normalize
        )
    
    def forward(self, 
                tokens: Tensor, 
                mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the text encoder.
        
        Args:
            tokens: Token indices of shape (batch_size, seq_length)
            mask: Attention mask of shape (batch_size, seq_length)
        
        Returns:
            embeddings: Text embeddings of shape (batch_size, embedding_dim)
        """
        word_embeddings = self.word_embedding(tokens)  # (batch_size, seq_length, embedding_dim)
        
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (batch_size, seq_length, 1)
            word_embeddings = word_embeddings * mask
        
        text_features = word_embeddings.mean(dim=1)  # (batch_size, embedding_dim)
        
        embeddings = super().forward(text_features)
        
        return embeddings


class ImageEncoder(EmbeddingModel):
    """
    Image encoder for zero-shot learning.
    
    This model encodes images into the shared embedding space.
    """
    
    def __init__(self, 
                 input_channels: int, 
                 embedding_dim: int, 
                 hidden_dims: List[int] = None,
                 backbone: str = "resnet50",
                 pretrained: bool = True,
                 dropout: float = 0.2,
                 normalize: bool = True):
        """
        Initialize an image encoder.
        
        Args:
            input_channels: Number of input channels
            embedding_dim: Dimension of the embedding space
            hidden_dims: Dimensions of hidden layers
            backbone: Backbone architecture for feature extraction
            pretrained: Whether to use pretrained weights
            dropout: Dropout probability
            normalize: Whether to L2-normalize the embeddings
        """
        self.input_channels = input_channels
        self.backbone_name = backbone
        
        if backbone == "resnet50":
            from neurenix.nn.models import resnet50
            self.backbone = resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone == "vit":
            from neurenix.nn.models import vit_base
            self.backbone = vit_base(pretrained=pretrained)
            backbone_dim = 768
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        super().__init__(
            input_dim=backbone_dim,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            normalize=normalize
        )
    
    def forward(self, images: Tensor) -> Tensor:
        """
        Forward pass through the image encoder.
        
        Args:
            images: Input images of shape (batch_size, channels, height, width)
        
        Returns:
            embeddings: Image embeddings of shape (batch_size, embedding_dim)
        """
        features = self.backbone(images)  # (batch_size, backbone_dim)
        
        embeddings = super().forward(features)
        
        return embeddings


class CrossModalEncoder(Module):
    """
    Cross-modal encoder for zero-shot learning.
    
    This model aligns embeddings from different modalities (e.g., images and text)
    in a shared embedding space.
    """
    
    def __init__(self, 
                 image_encoder: ImageEncoder, 
                 text_encoder: TextEncoder,
                 projection_dim: int = 512,
                 temperature: float = 0.07):
        """
        Initialize a cross-modal encoder.
        
        Args:
            image_encoder: Image encoder model
            text_encoder: Text encoder model
            projection_dim: Dimension of the projection space
            temperature: Temperature parameter for similarity scaling
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.temperature = temperature
        
        self.image_projection = Linear(
            image_encoder.embedding_dim, projection_dim
        )
        self.text_projection = Linear(
            text_encoder.embedding_dim, projection_dim
        )
    
    def encode_image(self, images: Tensor) -> Tensor:
        """
        Encode images into the shared embedding space.
        
        Args:
            images: Input images of shape (batch_size, channels, height, width)
        
        Returns:
            embeddings: Image embeddings of shape (batch_size, projection_dim)
        """
        image_features = self.image_encoder(images)
        image_embeddings = self.image_projection(image_features)
        
        norm = image_embeddings.norm(dim=1, keepdim=True)
        image_embeddings = image_embeddings / (norm + 1e-10)
        
        return image_embeddings
    
    def encode_text(self, 
                    tokens: Tensor, 
                    mask: Optional[Tensor] = None) -> Tensor:
        """
        Encode text into the shared embedding space.
        
        Args:
            tokens: Token indices of shape (batch_size, seq_length)
            mask: Attention mask of shape (batch_size, seq_length)
        
        Returns:
            embeddings: Text embeddings of shape (batch_size, projection_dim)
        """
        text_features = self.text_encoder(tokens, mask)
        text_embeddings = self.text_projection(text_features)
        
        norm = text_embeddings.norm(dim=1, keepdim=True)
        text_embeddings = text_embeddings / (norm + 1e-10)
        
        return text_embeddings
    
    def forward(self, 
                images: Tensor, 
                tokens: Tensor, 
                mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through the cross-modal encoder.
        
        Args:
            images: Input images of shape (batch_size, channels, height, width)
            tokens: Token indices of shape (batch_size, seq_length)
            mask: Attention mask of shape (batch_size, seq_length)
        
        Returns:
            image_embeddings: Image embeddings of shape (batch_size, projection_dim)
            text_embeddings: Text embeddings of shape (batch_size, projection_dim)
            logits: Similarity scores of shape (batch_size, batch_size)
        """
        image_embeddings = self.encode_image(images)
        text_embeddings = self.encode_text(tokens, mask)
        
        logits = image_embeddings @ text_embeddings.t() / self.temperature
        
        return image_embeddings, text_embeddings, logits
