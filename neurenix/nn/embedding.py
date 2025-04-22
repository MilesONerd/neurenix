"""
Embedding layers for neural networks.
"""

from typing import Optional, Union, Tuple
import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor
import neurenix.binding as binding

class Embedding(Module):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    
    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        scale_grad_by_freq: bool = False,
        sparse: bool = False
    ):
        """
        Initialize an Embedding layer.
        
        Args:
            num_embeddings: Size of the dictionary of embeddings
            embedding_dim: Size of each embedding vector
            padding_idx: If specified, entries at this index do not contribute to the gradient
            max_norm: If specified, renormalizes embeddings to have norm at most max_norm
            norm_type: The p in the p-norm to compute for max_norm
            scale_grad_by_freq: If True, scales gradients by the inverse of frequency of the words
            sparse: If True, gradient w.r.t. weight matrix will be a sparse tensor
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        
        self.weight = Tensor(
            np.random.normal(0, 1.0, (num_embeddings, embedding_dim)) * 0.1
        )
        
        if self.padding_idx is not None:
            with Tensor.no_grad():
                self.weight[self.padding_idx].fill_(0)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the embedding layer.
        
        Args:
            x: Tensor containing indices into the embedding matrix
            
        Returns:
            Tensor containing the embeddings for the given indices
        """
        try:
            return binding.embedding(
                x, self.weight, self.padding_idx,
                self.max_norm, self.norm_type,
                self.scale_grad_by_freq, self.sparse
            )
        except (ImportError, AttributeError):
            return self._python_forward(x)
    
    def _python_forward(self, x: Tensor) -> Tensor:
        """
        Pure Python implementation of embedding lookup as a fallback.
        
        Args:
            x: Tensor containing indices into the embedding matrix
            
        Returns:
            Tensor containing the embeddings for the given indices
        """
        if x.dim() == 1:
            output = self.weight[x.to_numpy()]
        else:
            output_shape = x.shape + (self.embedding_dim,)
            output = self.weight[x.to_numpy().flatten()].reshape(output_shape)
        
        if self.max_norm is not None:
            norms = Tensor.norm(output, p=self.norm_type, dim=-1, keepdim=True)
            output = output * Tensor.minimum(
                self.max_norm / (norms + 1e-7),
                Tensor.ones_like(norms)
            )
        
        return output
    
    def extra_repr(self) -> str:
        """Return a string containing extra information about the module."""
        s = f"{self.num_embeddings}, {self.embedding_dim}"
        if self.padding_idx is not None:
            s += f", padding_idx={self.padding_idx}"
        if self.max_norm is not None:
            s += f", max_norm={self.max_norm}"
        if self.norm_type != 2:
            s += f", norm_type={self.norm_type}"
        if self.scale_grad_by_freq:
            s += ", scale_grad_by_freq=True"
        if self.sparse:
            s += ", sparse=True"
        return s
