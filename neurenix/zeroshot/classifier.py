"""
Zero-shot classifiers for Neurenix.

This module provides classifier models for zero-shot learning, which can
classify inputs into classes that were not seen during training.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any

import neurenix
from neurenix.tensor import Tensor
from neurenix.nn import Module, Linear, Dropout, BatchNorm1d, ReLU, Sequential
from neurenix.nn.functional import cosine_similarity
from neurenix.zeroshot.model import ZeroShotModel


class ZeroShotClassifier(Module):
    """
    Base class for zero-shot classifiers.
    
    Zero-shot classifiers can classify inputs into classes that were not seen
    during training by leveraging semantic information about classes.
    """
    
    def __init__(self, 
                 zero_shot_model: ZeroShotModel,
                 class_names: List[str] = None,
                 class_embeddings: Optional[Tensor] = None):
        """
        Initialize a zero-shot classifier.
        
        Args:
            zero_shot_model: Pre-trained zero-shot model
            class_names: List of class names
            class_embeddings: Pre-computed class embeddings of shape (num_classes, semantic_dim)
                              If None, class embeddings will be computed from class_names
        """
        super().__init__()
        self.zero_shot_model = zero_shot_model
        self.class_names = class_names
        self.class_embeddings = class_embeddings
    
    def forward(self, 
                visual_features: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the zero-shot classifier.
        
        Args:
            visual_features: Visual features of shape (batch_size, visual_dim)
        
        Returns:
            logits: Classification logits of shape (batch_size, num_classes)
            visual_embeddings: Embedded visual features of shape (batch_size, hidden_dim)
        """
        if self.class_embeddings is None:
            raise ValueError("Class embeddings must be set before inference")
        
        logits, visual_embeddings = self.zero_shot_model(
            visual_features, self.class_embeddings
        )
        
        return logits, visual_embeddings
    
    def predict(self, 
                visual_features: Tensor) -> Tuple[Tensor, List[str]]:
        """
        Predict class labels for visual features.
        
        Args:
            visual_features: Visual features of shape (batch_size, visual_dim)
        
        Returns:
            predictions: Predicted class indices of shape (batch_size,)
            class_names: List of predicted class names
        """
        logits, _ = self.forward(visual_features)
        predictions = logits.argmax(dim=1)
        
        if self.class_names is not None:
            predicted_classes = [self.class_names[idx] for idx in predictions.tolist()]
            return predictions, predicted_classes
        
        return predictions, None
    
    def set_class_embeddings(self, 
                             class_embeddings: Tensor,
                             class_names: List[str] = None):
        """
        Set class embeddings for the classifier.
        
        Args:
            class_embeddings: Class embeddings of shape (num_classes, semantic_dim)
            class_names: List of class names corresponding to the embeddings
        """
        self.class_embeddings = class_embeddings
        if class_names is not None:
            self.class_names = class_names


class AttributeClassifier(ZeroShotClassifier):
    """
    Attribute-based zero-shot classifier.
    
    This classifier uses class-attribute mappings for zero-shot classification.
    """
    
    def __init__(self, 
                 zero_shot_model: ZeroShotModel,
                 class_names: List[str] = None,
                 attribute_names: List[str] = None,
                 class_attribute_matrix: Optional[Tensor] = None):
        """
        Initialize an attribute-based zero-shot classifier.
        
        Args:
            zero_shot_model: Pre-trained zero-shot model
            class_names: List of class names
            attribute_names: List of attribute names
            class_attribute_matrix: Binary matrix of shape (num_classes, num_attributes)
                                   indicating which attributes are present in each class
        """
        super().__init__(zero_shot_model, class_names)
        self.attribute_names = attribute_names
        self.class_attribute_matrix = class_attribute_matrix
        
        if class_attribute_matrix is not None:
            self.class_embeddings = class_attribute_matrix
    
    def set_class_attributes(self, 
                             class_attribute_matrix: Tensor,
                             class_names: List[str] = None,
                             attribute_names: List[str] = None):
        """
        Set class-attribute mappings for the classifier.
        
        Args:
            class_attribute_matrix: Binary matrix of shape (num_classes, num_attributes)
                                   indicating which attributes are present in each class
            class_names: List of class names corresponding to the rows
            attribute_names: List of attribute names corresponding to the columns
        """
        self.class_attribute_matrix = class_attribute_matrix
        self.class_embeddings = class_attribute_matrix
        
        if class_names is not None:
            self.class_names = class_names
        
        if attribute_names is not None:
            self.attribute_names = attribute_names
    
    def predict_attributes(self, 
                           visual_features: Tensor) -> Tuple[Tensor, List[str]]:
        """
        Predict attributes for visual features.
        
        Args:
            visual_features: Visual features of shape (batch_size, visual_dim)
        
        Returns:
            attribute_scores: Attribute prediction scores of shape (batch_size, num_attributes)
            predicted_attributes: List of lists of predicted attribute names for each sample
        """
        visual_embeddings = self.zero_shot_model(visual_features)
        
        num_attributes = self.class_attribute_matrix.shape[1]
        attribute_embeddings = Tensor.eye(num_attributes)
        
        attribute_scores = cosine_similarity(
            visual_embeddings.unsqueeze(1),  # (batch_size, 1, hidden_dim)
            attribute_embeddings.unsqueeze(0)  # (1, num_attributes, hidden_dim)
        )  # (batch_size, num_attributes)
        
        predicted_attributes = attribute_scores > 0.5
        
        if self.attribute_names is not None:
            attribute_lists = []
            for i in range(predicted_attributes.shape[0]):
                attrs = [
                    self.attribute_names[j] 
                    for j in range(num_attributes) 
                    if predicted_attributes[i, j]
                ]
                attribute_lists.append(attrs)
            
            return attribute_scores, attribute_lists
        
        return attribute_scores, None


class SemanticClassifier(ZeroShotClassifier):
    """
    Semantic-based zero-shot classifier.
    
    This classifier uses semantic embeddings (e.g., word embeddings) for zero-shot classification.
    """
    
    def __init__(self, 
                 zero_shot_model: ZeroShotModel,
                 class_names: List[str] = None,
                 class_embeddings: Optional[Tensor] = None,
                 text_encoder = None):
        """
        Initialize a semantic-based zero-shot classifier.
        
        Args:
            zero_shot_model: Pre-trained zero-shot model
            class_names: List of class names
            class_embeddings: Pre-computed class embeddings of shape (num_classes, semantic_dim)
            text_encoder: Text encoder model for computing class embeddings from class names
        """
        super().__init__(zero_shot_model, class_names, class_embeddings)
        self.text_encoder = text_encoder
        
        if class_names is not None and text_encoder is not None and class_embeddings is None:
            self.compute_class_embeddings()
    
    def compute_class_embeddings(self):
        """
        Compute class embeddings from class names using the text encoder.
        """
        if self.class_names is None or self.text_encoder is None:
            raise ValueError("Class names and text encoder must be set")
        
        from neurenix.text import tokenize
        tokens = [tokenize(name) for name in self.class_names]
        max_length = max(len(t) for t in tokens)
        
        padded_tokens = []
        masks = []
        for t in tokens:
            padding = [0] * (max_length - len(t))
            padded_tokens.append(t + padding)
            masks.append([1] * len(t) + [0] * (max_length - len(t)))
        
        token_tensor = Tensor(padded_tokens)
        mask_tensor = Tensor(masks)
        
        self.class_embeddings = self.text_encoder(token_tensor, mask_tensor)
    
    def set_text_encoder(self, text_encoder):
        """
        Set the text encoder for computing class embeddings.
        
        Args:
            text_encoder: Text encoder model
        """
        self.text_encoder = text_encoder
        
        if self.class_names is not None and self.class_embeddings is None:
            self.compute_class_embeddings()
