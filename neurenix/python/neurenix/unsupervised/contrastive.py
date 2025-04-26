"""
Contrastive learning algorithms for unsupervised learning in the Neurenix framework.

Contrastive learning is a self-supervised learning technique that learns representations
by contrasting positive pairs against negative pairs.
"""

from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import numpy as np

from neurenix.nn.module import Module
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.tensor import Tensor
from neurenix.device import Device

class SimCLR(Module):
    """
    Simple Contrastive Learning of Representations (SimCLR) implementation.
    
    SimCLR learns representations by maximizing agreement between differently
    augmented views of the same data example.
    
    Reference:
        Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020).
        A Simple Framework for Contrastive Learning of Visual Representations.
        International Conference on Machine Learning (ICML).
    """
    
    @staticmethod
    def _get_encoder_output_dim(encoder: Module) -> int:
        """
        Determine the output dimension of an encoder by passing a dummy input.
        
        Args:
            encoder: The encoder module
            
        Returns:
            Output dimension of the encoder
        """
        if hasattr(encoder, 'layers') and len(encoder.layers) > 0:
            last_layer = encoder.layers[-1]
            if hasattr(last_layer, 'out_features'):
                return last_layer.out_features
        
        if hasattr(encoder, 'fc') and hasattr(encoder.fc, 'out_features'):
            return encoder.fc.out_features
        
        import torch
        device = next(encoder.parameters()).device
        dummy_input = torch.randn(1, 3, 224, 224, device=device)  # Assume standard image input
        with torch.no_grad():
            output = encoder(dummy_input)
        
        if len(output.shape) == 2:
            return output.shape[1]
        else:
            return output.numel() // output.shape[0]
    
    def __init__(
        self,
        encoder: Module,
        projection_dim: int = 128,
        temperature: float = 0.5,
    ):
        """
        Initialize SimCLR.
        
        Args:
            encoder: Base encoder network
            projection_dim: Dimensionality of the projection head output
            temperature: Temperature parameter for the contrastive loss
        """
        super().__init__()
        
        self.encoder = encoder
        self.temperature = temperature
        
        # Get the output dimension of the encoder
        encoder_out_dim = self._get_encoder_output_dim(encoder)
        
        # Projection head (MLP with one hidden layer)
        self.projection = Sequential([
            Linear(encoder_out_dim, encoder_out_dim),
            ReLU(),
            Linear(encoder_out_dim, projection_dim),
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the SimCLR model.
        
        Args:
            x: Input tensor
            
        Returns:
            Projected features
        """
        # Encode
        h = self.encoder(x)
        
        # Project
        z = self.projection(h)
        
        # Normalize
        z = z / z.norm(dim=1, keepdim=True)
        
        return z
    
    def contrastive_loss(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """
        Compute the contrastive loss between two sets of projected features.
        
        Args:
            z_i: Projected features for first augmented views
            z_j: Projected features for second augmented views
            
        Returns:
            Contrastive loss
        """
        batch_size = z_i.shape[0]
        
        # Concatenate projections from both augmented views
        z = Tensor.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        sim = z.matmul(z.t()) / self.temperature
        
        # Mask out self-similarity
        mask = Tensor.eye(2 * batch_size, dtype=Tensor.bool)
        sim = sim.masked_fill(mask, -float('inf'))
        
        # Create labels for positive pairs
        labels = Tensor.cat([
            Tensor.arange(batch_size, 2 * batch_size),
            Tensor.arange(batch_size),
        ], dim=0)
        
        # Compute NT-Xent loss
        loss = Tensor.cross_entropy(sim, labels)
        
        return loss

class BYOL(Module):
    """
    Bootstrap Your Own Latent (BYOL) implementation.
    
    BYOL learns representations by predicting the representations of one augmented
    view from another augmented view of the same image.
    
    Reference:
        Grill, J. B., Strub, F., Altché, F., Tallec, C., Richemond, P. H., Buchatskaya, E., ... & Valko, M. (2020).
        Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning.
        Advances in Neural Information Processing Systems (NeurIPS).
    """
    
    @staticmethod
    def _get_encoder_output_dim(encoder: Module) -> int:
        """
        Determine the output dimension of an encoder by passing a dummy input.
        
        Args:
            encoder: The encoder module
            
        Returns:
            Output dimension of the encoder
        """
        if hasattr(encoder, 'layers') and len(encoder.layers) > 0:
            last_layer = encoder.layers[-1]
            if hasattr(last_layer, 'out_features'):
                return last_layer.out_features
        
        if hasattr(encoder, 'fc') and hasattr(encoder.fc, 'out_features'):
            return encoder.fc.out_features
        
        import torch
        device = next(encoder.parameters()).device
        dummy_input = torch.randn(1, 3, 224, 224, device=device)  # Assume standard image input
        with torch.no_grad():
            output = encoder(dummy_input)
        
        if len(output.shape) == 2:
            return output.shape[1]
        else:
            return output.numel() // output.shape[0]
            
    @staticmethod
    def _clone_module(module: Module) -> Module:
        """
        Create a deep copy of a module.
        
        Args:
            module: Module to clone
            
        Returns:
            Cloned module
        """
        import copy
        import torch
        
        if hasattr(module, '__class__'):
            cloned = copy.deepcopy(module)
            
            cloned.load_state_dict(module.state_dict())
            
            return cloned
        else:
            return copy.deepcopy(module)
    
    def __init__(
        self,
        encoder: Module,
        projection_dim: int = 256,
        hidden_dim: int = 4096,
        momentum: float = 0.99,
    ):
        """
        Initialize BYOL.
        
        Args:
            encoder: Base encoder network
            projection_dim: Dimensionality of the projection head output
            hidden_dim: Dimensionality of the hidden layer in the projection head
            momentum: Momentum for updating the target network
        """
        super().__init__()
        
        self.momentum = momentum
        
        # Get the output dimension of the encoder
        encoder_out_dim = self._get_encoder_output_dim(encoder)
        
        # Online network
        self.online_encoder = encoder
        self.online_projector = Sequential([
            Linear(encoder_out_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, projection_dim),
        ])
        self.online_predictor = Sequential([
            Linear(projection_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, projection_dim),
        ])
        
        self.target_encoder = self._clone_module(encoder)
        self.target_projector = Sequential([
            Linear(encoder_out_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, projection_dim),
        ])
        
        # Initialize target network with the same parameters as the online network
        self._update_target_network(tau=1.0)
        
        # Disable gradient computation for the target network
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        for param in self.target_projector.parameters():
            param.requires_grad = False
    
    def _update_target_network(self, tau: float = None):
        """
        Update the target network parameters using exponential moving average.
        
        Args:
            tau: Update weight (if None, use self.momentum)
        """
        if tau is None:
            tau = self.momentum
        
        # Update encoder
        for online_param, target_param in zip(
            self.online_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data
        
        # Update projector
        for online_param, target_param in zip(
            self.online_projector.parameters(),
            self.target_projector.parameters()
        ):
            target_param.data = tau * target_param.data + (1 - tau) * online_param.data
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the BYOL model.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (online_projection, target_projection)
        """
        # Online network
        online_features = self.online_encoder(x)
        online_projection = self.online_projector(online_features)
        online_prediction = self.online_predictor(online_projection)
        
        # Target network
        with Tensor.no_grad():
            target_features = self.target_encoder(x)
            target_projection = self.target_projector(target_features)
        
        return online_prediction, target_projection
    
    def loss_function(self, online_prediction: Tensor, target_projection: Tensor) -> Tensor:
        """
        Compute the BYOL loss.
        
        Args:
            online_prediction: Prediction from the online network
            target_projection: Projection from the target network
            
        Returns:
            BYOL loss
        """
        # Normalize projections
        online_prediction = online_prediction / online_prediction.norm(dim=1, keepdim=True)
        target_projection = target_projection / target_projection.norm(dim=1, keepdim=True)
        
        # Compute mean squared error
        loss = 2 - 2 * (online_prediction * target_projection).sum(dim=1).mean()
        
        return loss
    
    def update_target(self):
        """
        Update the target network after each training step.
        """
        self._update_target_network()

class MoCo(Module):
    """
    Momentum Contrast (MoCo) implementation.
    
    MoCo learns representations by matching an encoded query to a dictionary
    of encoded keys using a contrastive loss.
    
    Reference:
        He, K., Fan, H., Wu, Y., Xie, S., & Girshick, R. (2020).
        Momentum Contrast for Unsupervised Visual Representation Learning.
        IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
    """
    
    @staticmethod
    def _get_encoder_output_dim(encoder: Module) -> int:
        """
        Determine the output dimension of an encoder by passing a dummy input.
        
        Args:
            encoder: The encoder module
            
        Returns:
            Output dimension of the encoder
        """
        if hasattr(encoder, 'layers') and len(encoder.layers) > 0:
            last_layer = encoder.layers[-1]
            if hasattr(last_layer, 'out_features'):
                return last_layer.out_features
        
        if hasattr(encoder, 'fc') and hasattr(encoder.fc, 'out_features'):
            return encoder.fc.out_features
        
        import torch
        device = next(encoder.parameters()).device
        dummy_input = torch.randn(1, 3, 224, 224, device=device)  # Assume standard image input
        with torch.no_grad():
            output = encoder(dummy_input)
        
        if len(output.shape) == 2:
            return output.shape[1]
        else:
            return output.numel() // output.shape[0]
            
    @staticmethod
    def _clone_module(module: Module) -> Module:
        """
        Create a deep copy of a module.
        
        Args:
            module: Module to clone
            
        Returns:
            Cloned module
        """
        import copy
        import torch
        
        if hasattr(module, '__class__'):
            cloned = copy.deepcopy(module)
            
            cloned.load_state_dict(module.state_dict())
            
            return cloned
        else:
            return copy.deepcopy(module)
    
    def __init__(
        self,
        encoder: Module,
        dim: int = 128,
        K: int = 65536,
        m: float = 0.999,
        T: float = 0.07,
    ):
        """
        Initialize MoCo.
        
        Args:
            encoder: Base encoder network
            dim: Feature dimension
            K: Queue size
            m: Momentum for updating the key encoder
            T: Temperature for the contrastive loss
        """
        super().__init__()
        
        self.K = K
        self.m = m
        self.T = T
        
        # Get the output dimension of the encoder
        encoder_out_dim = self._get_encoder_output_dim(encoder)
        
        # Query encoder
        self.encoder_q = encoder
        self.projector_q = Linear(encoder_out_dim, dim)
        
        self.encoder_k = self._clone_module(encoder)
        self.projector_k = Linear(encoder_out_dim, dim)
        
        # Initialize key encoder with the same parameters as the query encoder
        self._update_key_encoder(m=1.0)
        
        # Disable gradient computation for the key encoder
        for param in self.encoder_k.parameters():
            param.requires_grad = False
        for param in self.projector_k.parameters():
            param.requires_grad = False
        
        # Create the queue
        self.register_buffer("queue", Tensor.randn(dim, K))
        self.queue = self.queue / self.queue.norm(dim=0, keepdim=True)
        self.register_buffer("queue_ptr", Tensor.zeros(1, dtype=Tensor.long))
    
    def _update_key_encoder(self, m: float = None):
        """
        Update the key encoder parameters using exponential moving average.
        
        Args:
            m: Momentum (if None, use self.m)
        """
        if m is None:
            m = self.m
        
        # Update encoder
        for param_q, param_k in zip(
            self.encoder_q.parameters(),
            self.encoder_k.parameters()
        ):
            param_k.data = m * param_k.data + (1 - m) * param_q.data
        
        # Update projector
        for param_q, param_k in zip(
            self.projector_q.parameters(),
            self.projector_k.parameters()
        ):
            param_k.data = m * param_k.data + (1 - m) * param_q.data
    
    def _dequeue_and_enqueue(self, keys: Tensor):
        """
        Update the queue with new keys.
        
        Args:
            keys: New keys to add to the queue
        """
        batch_size = keys.shape[0]
        
        ptr = int(self.queue_ptr)
        
        # Replace the keys at ptr
        if ptr + batch_size <= self.K:
            self.queue[:, ptr:ptr + batch_size] = keys.t()
        else:
            # Handle the case where the batch wraps around the queue
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys[:remaining].t()
            self.queue[:, :batch_size - remaining] = keys[remaining:].t()
        
        # Update the pointer
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr
    
    def forward(self, im_q: Tensor, im_k: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through the MoCo model.
        
        Args:
            im_q: Query images
            im_k: Key images
            
        Returns:
            Tuple of (logits, labels, queue)
        """
        # Compute query features
        q = self.encoder_q(im_q)
        q = self.projector_q(q)
        q = q / q.norm(dim=1, keepdim=True)
        
        # Compute key features
        with Tensor.no_grad():
            # Update the key encoder
            self._update_key_encoder()
            
            # Compute key features
            k = self.encoder_k(im_k)
            k = self.projector_k(k)
            k = k / k.norm(dim=1, keepdim=True)
        
        # Compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = (q * k).sum(dim=1, keepdim=True)
        
        # negative logits: NxK
        l_neg = q.matmul(self.queue)
        
        # logits: Nx(1+K)
        logits = Tensor.cat([l_pos, l_neg], dim=1)
        
        # Apply temperature
        logits /= self.T
        
        # Labels: positives are the 0-th
        labels = Tensor.zeros(logits.shape[0], dtype=Tensor.long)
        
        # Dequeue and enqueue
        self._dequeue_and_enqueue(k)
        
        return logits, labels, self.queue
    
    def contrastive_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        Compute the contrastive loss.
        
        Args:
            logits: Logits from the forward pass
            labels: Labels from the forward pass
            
        Returns:
            Contrastive loss
        """
        return Tensor.cross_entropy(logits, labels)
