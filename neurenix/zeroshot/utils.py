"""
Utility functions for zero-shot learning in Neurenix.

This module provides utility functions for zero-shot learning, such as
semantic similarity computation, attribute mapping, and class embedding.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any

import neurenix
from neurenix.tensor import Tensor
from neurenix.nn.functional import cosine_similarity


def semantic_similarity(embeddings1: Tensor, embeddings2: Tensor) -> Tensor:
    """
    Compute semantic similarity between two sets of embeddings.
    
    Args:
        embeddings1: First set of embeddings of shape (batch_size1, embedding_dim)
        embeddings2: Second set of embeddings of shape (batch_size2, embedding_dim)
    
    Returns:
        similarities: Similarity matrix of shape (batch_size1, batch_size2)
    """
    norm1 = embeddings1.norm(dim=1, keepdim=True)
    norm2 = embeddings2.norm(dim=1, keepdim=True)
    
    embeddings1_normalized = embeddings1 / (norm1 + 1e-10)
    embeddings2_normalized = embeddings2 / (norm2 + 1e-10)
    
    similarities = cosine_similarity(
        embeddings1_normalized.unsqueeze(1),  # (batch_size1, 1, embedding_dim)
        embeddings2_normalized.unsqueeze(0)   # (1, batch_size2, embedding_dim)
    )  # (batch_size1, batch_size2)
    
    return similarities


def attribute_mapping(
    class_names: List[str], 
    attribute_names: List[str], 
    class_attribute_map: Dict[str, List[str]]
) -> Tensor:
    """
    Create a class-attribute binary matrix from a mapping dictionary.
    
    Args:
        class_names: List of class names
        attribute_names: List of attribute names
        class_attribute_map: Dictionary mapping class names to lists of attribute names
    
    Returns:
        class_attribute_matrix: Binary matrix of shape (num_classes, num_attributes)
                               indicating which attributes are present in each class
    """
    num_classes = len(class_names)
    num_attributes = len(attribute_names)
    
    attribute_indices = {attr: i for i, attr in enumerate(attribute_names)}
    
    class_attribute_matrix = Tensor.zeros((num_classes, num_attributes))
    
    for i, class_name in enumerate(class_names):
        if class_name in class_attribute_map:
            for attr in class_attribute_map[class_name]:
                if attr in attribute_indices:
                    class_attribute_matrix[i, attribute_indices[attr]] = 1.0
    
    return class_attribute_matrix


def class_embedding(
    class_names: List[str], 
    text_encoder: Any, 
    tokenizer: Any = None
) -> Tensor:
    """
    Compute class embeddings from class names using a text encoder.
    
    Args:
        class_names: List of class names
        text_encoder: Text encoder model
        tokenizer: Tokenizer for preprocessing class names
                  If None, a simple whitespace tokenizer is used
    
    Returns:
        class_embeddings: Class embeddings of shape (num_classes, embedding_dim)
    """
    if tokenizer is None:
        tokens = [name.lower().split() for name in class_names]
        
        from neurenix.text import get_vocab
        vocab = get_vocab()
        token_indices = [
            [vocab.get(token, vocab['<unk>']) for token in name_tokens]
            for name_tokens in tokens
        ]
        
        max_length = max(len(indices) for indices in token_indices)
        padded_tokens = []
        masks = []
        
        for indices in token_indices:
            padding = [0] * (max_length - len(indices))
            padded_tokens.append(indices + padding)
            masks.append([1] * len(indices) + [0] * (max_length - len(indices)))
        
        token_tensor = Tensor(padded_tokens)
        mask_tensor = Tensor(masks)
    else:
        tokenized = tokenizer(
            class_names, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )
        token_tensor = tokenized['input_ids']
        mask_tensor = tokenized['attention_mask']
    
    with neurenix.no_grad():
        class_embeddings = text_encoder(token_tensor, mask_tensor)
    
    return class_embeddings


def generate_class_descriptions(
    class_names: List[str], 
    template: str = "A photo of a {}."
) -> List[str]:
    """
    Generate class descriptions from class names using a template.
    
    Args:
        class_names: List of class names
        template: Template string with {} placeholder for class name
    
    Returns:
        descriptions: List of class descriptions
    """
    descriptions = [template.format(name) for name in class_names]
    return descriptions


def compute_attribute_importance(
    model: Any, 
    visual_features: Tensor, 
    attribute_matrix: Tensor
) -> Tensor:
    """
    Compute the importance of each attribute for classification.
    
    Args:
        model: Zero-shot model
        visual_features: Visual features of shape (batch_size, visual_dim)
        attribute_matrix: Binary matrix of shape (num_classes, num_attributes)
                         indicating which attributes are present in each class
    
    Returns:
        attribute_importance: Importance scores of shape (batch_size, num_attributes)
    """
    visual_embeddings = model(visual_features)
    
    num_attributes = attribute_matrix.shape[1]
    attribute_importance = Tensor.zeros((visual_features.shape[0], num_attributes))
    
    for i in range(num_attributes):
        attribute_mask = attribute_matrix[:, i:i+1]
        
        similarity = cosine_similarity(
            visual_embeddings.unsqueeze(1),  # (batch_size, 1, hidden_dim)
            attribute_mask.unsqueeze(0)      # (1, num_classes, 1)
        )  # (batch_size, num_classes)
        
        attribute_importance[:, i] = similarity.mean(dim=1)
    
    return attribute_importance
