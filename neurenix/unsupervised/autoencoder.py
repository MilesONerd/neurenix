"""
Autoencoder implementations for unsupervised learning in the Neurenix framework.

Autoencoders are neural networks that learn to compress data into a lower-dimensional
representation and then reconstruct it. They are useful for dimensionality reduction,
feature learning, and generative modeling.
"""

from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import numpy as np

from neurenix.nn.module import Module
from neurenix.nn import Sequential, Linear, ReLU, Sigmoid, Tanh
from neurenix.tensor import Tensor
from neurenix.device import Device

class Autoencoder(Module):
    """
    Basic autoencoder implementation.
    
    An autoencoder consists of an encoder that compresses the input data
    into a lower-dimensional latent space, and a decoder that reconstructs
    the original input from the latent representation.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        activation: str = 'relu',
    ):
        """
        Initialize an autoencoder.
        
        Args:
            input_dim: Dimensionality of the input data
            hidden_dims: List of hidden layer dimensions for the encoder and decoder
            latent_dim: Dimensionality of the latent space
            activation: Activation function to use ('relu', 'sigmoid', or 'tanh')
        """
        super().__init__()
        
        # Create activation function
        if activation == 'relu':
            act_fn = ReLU()
        elif activation == 'sigmoid':
            act_fn = Sigmoid()
        elif activation == 'tanh':
            act_fn = Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(Linear(prev_dim, hidden_dim))
            encoder_layers.append(act_fn)
            prev_dim = hidden_dim
        
        # Add latent layer
        encoder_layers.append(Linear(prev_dim, latent_dim))
        
        self.encoder = Sequential(encoder_layers)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(Linear(prev_dim, hidden_dim))
            decoder_layers.append(act_fn)
            prev_dim = hidden_dim
        
        # Add output layer
        decoder_layers.append(Linear(prev_dim, input_dim))
        
        self.decoder = Sequential(decoder_layers)
    
    def encode(self, x: Tensor) -> Tensor:
        """
        Encode input data into the latent space.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation
        """
        return self.encoder(x)
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent representation back to the input space.
        
        Args:
            z: Latent representation
            
        Returns:
            Reconstructed input
        """
        return self.decoder(z)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor
            
        Returns:
            Reconstructed input
        """
        z = self.encode(x)
        return self.decode(z)
    
    def get_latent_representation(self, x: Tensor) -> Tensor:
        """
        Get the latent representation for input data.
        
        Args:
            x: Input tensor
            
        Returns:
            Latent representation
        """
        return self.encode(x)

class VAE(Module):
    """
    Variational Autoencoder (VAE) implementation.
    
    VAEs are a type of autoencoder that learn a probabilistic mapping between
    the input space and a latent space, allowing for generative modeling.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        activation: str = 'relu',
    ):
        """
        Initialize a VAE.
        
        Args:
            input_dim: Dimensionality of the input data
            hidden_dims: List of hidden layer dimensions for the encoder and decoder
            latent_dim: Dimensionality of the latent space
            activation: Activation function to use ('relu', 'sigmoid', or 'tanh')
        """
        super().__init__()
        
        # Create activation function
        if activation == 'relu':
            act_fn = ReLU()
        elif activation == 'sigmoid':
            act_fn = Sigmoid()
        elif activation == 'tanh':
            act_fn = Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.append(Linear(prev_dim, hidden_dim))
            encoder_layers.append(act_fn)
            prev_dim = hidden_dim
        
        # Instead of directly outputting the latent vector,
        # we output mean and log variance for the latent distribution
        self.fc_mu = Linear(prev_dim, latent_dim)
        self.fc_logvar = Linear(prev_dim, latent_dim)
        
        self.encoder = Sequential(encoder_layers)
        
        # Build decoder
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.append(Linear(prev_dim, hidden_dim))
            decoder_layers.append(act_fn)
            prev_dim = hidden_dim
        
        # Add output layer
        decoder_layers.append(Linear(prev_dim, input_dim))
        
        self.decoder = Sequential(decoder_layers)
        
        # For tracking the KL divergence and reconstruction loss
        self.kl_divergence = 0.0
        self.reconstruction_loss = 0.0
    
    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode input data into the latent space.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (mean, log_variance) for the latent distribution
        """
        hidden = self.encoder(x)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar
    
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from the latent distribution.
        
        Args:
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = Tensor.exp(0.5 * logvar)
        eps = Tensor.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: Tensor) -> Tensor:
        """
        Decode latent representation back to the input space.
        
        Args:
            z: Latent representation
            
        Returns:
            Reconstructed input
        """
        return self.decoder(z)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (reconstructed_input, mean, log_variance)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss_function(self, recon_x: Tensor, x: Tensor, mu: Tensor, logvar: Tensor, beta: float = 1.0) -> Tensor:
        """
        Compute the VAE loss function.
        
        The loss consists of a reconstruction term (how well the model reconstructs the input)
        and a KL divergence term (how close the latent distribution is to a standard normal).
        
        Args:
            recon_x: Reconstructed input
            x: Original input
            mu: Mean of the latent distribution
            logvar: Log variance of the latent distribution
            beta: Weight for the KL divergence term (beta-VAE)
            
        Returns:
            Total loss
        """
        # Reconstruction loss (mean squared error)
        recon_loss = ((recon_x - x) ** 2).sum(dim=1).mean()
        
        # KL divergence
        kl_div = -0.5 * Tensor.sum(1 + logvar - mu ** 2 - Tensor.exp(logvar), dim=1).mean()
        
        # Store the components for monitoring
        self.reconstruction_loss = recon_loss.item()
        self.kl_divergence = kl_div.item()
        
        # Total loss
        return recon_loss + beta * kl_div
    
    def sample(self, num_samples: int, device: Optional[Device] = None) -> Tensor:
        """
        Generate samples from the latent space.
        
        Args:
            num_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            Generated samples
        """
        # Sample from the latent distribution
        z = Tensor.randn(num_samples, self.fc_mu.out_features, device=device)
        
        # Decode the samples
        return self.decode(z)

class DenoisingAutoencoder(Autoencoder):
    """
    Denoising Autoencoder implementation.
    
    A denoising autoencoder is trained to reconstruct clean inputs from corrupted ones,
    making it more robust and better at learning useful features.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        latent_dim: int,
        noise_factor: float = 0.3,
        activation: str = 'relu',
    ):
        """
        Initialize a denoising autoencoder.
        
        Args:
            input_dim: Dimensionality of the input data
            hidden_dims: List of hidden layer dimensions for the encoder and decoder
            latent_dim: Dimensionality of the latent space
            noise_factor: Amount of noise to add to the input during training
            activation: Activation function to use ('relu', 'sigmoid', or 'tanh')
        """
        super().__init__(input_dim, hidden_dims, latent_dim, activation)
        self.noise_factor = noise_factor
    
    def add_noise(self, x: Tensor) -> Tensor:
        """
        Add noise to the input.
        
        Args:
            x: Input tensor
            
        Returns:
            Noisy input
        """
        noise = Tensor.randn_like(x) * self.noise_factor
        return x + noise
    
    def forward(self, x: Tensor, add_noise: bool = True) -> Tensor:
        """
        Forward pass through the denoising autoencoder.
        
        Args:
            x: Input tensor
            add_noise: Whether to add noise to the input
            
        Returns:
            Reconstructed input
        """
        # Add noise during training
        if add_noise and self.training:
            x_noisy = self.add_noise(x)
            z = self.encode(x_noisy)
        else:
            z = self.encode(x)
        
        # Decode
        return self.decode(z)
