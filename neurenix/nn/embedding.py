"""
Embedding layer for the Neurenix framework.
"""

import numpy as np
from typing import Optional, Tuple, Union

from neurenix.tensor import Tensor
from neurenix.nn.module import Module
from neurenix.nn.parameter import Parameter
from neurenix.core import get_config
from neurenix.device import DeviceType

class Embedding(Module):
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    
    This module is often used to store word embeddings and retrieve them using indices.
    The input to the module is a list of indices, and the output is the corresponding
    word embeddings.
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None, norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False, sparse: bool = False):
        """
        Initialize an embedding layer.
        
        Args:
            num_embeddings: Size of the dictionary of embeddings
            embedding_dim: The size of each embedding vector
            padding_idx: If specified, the entries at padding_idx do not contribute to the gradient
            max_norm: If given, each embedding vector with norm larger than max_norm is renormalized
            norm_type: The p of the p-norm to compute for the max_norm option
            scale_grad_by_freq: If given, this will scale gradients by the inverse of frequency of
                                the words in the mini-batch
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
        
        weight = np.random.normal(0, 0.1, (num_embeddings, embedding_dim))
        self.weight = Parameter(Tensor(weight))
        
        if padding_idx is not None:
            with np.nditer(self.weight.data.to_numpy(), op_flags=['readwrite']) as it:
                for x in it:
                    if x == padding_idx:
                        x[...] = 0
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the embedding layer.
        
        Args:
            x: Input tensor containing indices into the embedding matrix
            
        Returns:
            Tensor containing the embeddings for the input indices
        """
        from neurenix.core import get_config
        from neurenix.device import DeviceType
        
        tensor_cores_enabled = get_config().get("tensor_cores_enabled", False)
        
        if (tensor_cores_enabled and 
            x.device.type == DeviceType.TENSOR_CORES and
            self.weight.data.device.type == DeviceType.TENSOR_CORES):
            try:
                from neurenix.binding import tensor_cores_embedding
                return tensor_cores_embedding(x, self.weight.data, self.padding_idx,
                                             self.max_norm, self.norm_type,
                                             self.scale_grad_by_freq, self.sparse)
            except (ImportError, AttributeError):
                pass
        
        try:
            from neurenix.binding import embedding
            return embedding(x, self.weight.data, self.padding_idx,
                            self.max_norm, self.norm_type,
                            self.scale_grad_by_freq, self.sparse)
        except (ImportError, AttributeError):
            from ..core import PhynexusExtension
            
            if PhynexusExtension.is_available():
                return PhynexusExtension.embedding(x, self.weight.data, self.padding_idx,
                                                 self.max_norm, self.norm_type,
                                                 self.scale_grad_by_freq, self.sparse)
            
            indices = x.to_numpy().astype(np.int32)
            weight_np = self.weight.data.to_numpy()
            
            output_shape = list(indices.shape) + [self.embedding_dim]
            output_np = np.zeros(output_shape, dtype=np.float32)
            
            flat_indices = indices.reshape(-1)
            flat_output = output_np.reshape(-1, self.embedding_dim)
            
            for i, idx in enumerate(flat_indices):
                if idx < self.num_embeddings:
                    flat_output[i] = weight_np[idx]
            
            return Tensor(output_np, device=x.device)
    
    def __repr__(self):
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"
