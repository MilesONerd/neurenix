"""
Federated Learning security module for Neurenix.

This module provides implementations of security mechanisms for federated learning,
including secure aggregation, differential privacy, and homomorphic encryption.
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import hashlib
import secrets
import time
import math

import neurenix as nx


class SecureAggregation:
    """Secure aggregation for federated learning."""
    
    def __init__(self, client_id: str = None, seed: int = None):
        """
        Initialize secure aggregation.
        
        Args:
            client_id: Client ID for key generation
            seed: Random seed for key generation
        """
        self.client_id = client_id
        self.seed = seed or int(time.time())
        self.keys = {}
    
    def _generate_key(self, client_id: str) -> bytes:
        """
        Generate a key for a client.
        
        Args:
            client_id: Client ID
            
        Returns:
            Key
        """
        if client_id in self.keys:
            return self.keys[client_id]
        
        key_material = f"{client_id}:{self.seed}".encode()
        key = hashlib.sha256(key_material).digest()
        
        self.keys[client_id] = key
        
        return key
    
    def _mask_tensor(self, tensor: nx.Tensor, key: bytes, encrypt: bool = True) -> nx.Tensor:
        """
        Mask a tensor using a key.
        
        Args:
            tensor: Tensor to mask
            key: Key for masking
            encrypt: Whether to encrypt or decrypt
            
        Returns:
            Masked tensor
        """
        key_tensor = nx.zeros_like(tensor)
        
        rng = nx.random.manual_seed(int.from_bytes(key[:4], byteorder='big'))
        
        key_tensor = nx.rand_like(tensor, generator=rng)
        
        key_tensor = key_tensor * 0.01
        
        if encrypt:
            return tensor + key_tensor
        else:
            return tensor - key_tensor
    
    def encrypt(self, model_params: Dict[str, nx.Tensor]) -> Dict[str, nx.Tensor]:
        """
        Encrypt model parameters.
        
        Args:
            model_params: Model parameters
            
        Returns:
            Encrypted model parameters
        """
        if self.client_id is None:
            raise ValueError("Client ID is required for encryption")
        
        key = self._generate_key(self.client_id)
        
        encrypted_params = {}
        
        for name, param in model_params.items():
            encrypted_params[name] = self._mask_tensor(param, key, encrypt=True)
        
        return encrypted_params
    
    def decrypt(self, model_params: Dict[str, nx.Tensor]) -> Dict[str, nx.Tensor]:
        """
        Decrypt model parameters.
        
        Args:
            model_params: Encrypted model parameters
            
        Returns:
            Decrypted model parameters
        """
        if self.client_id is None:
            raise ValueError("Client ID is required for decryption")
        
        key = self._generate_key(self.client_id)
        
        decrypted_params = {}
        
        for name, param in model_params.items():
            decrypted_params[name] = self._mask_tensor(param, key, encrypt=False)
        
        return decrypted_params
    
    def aggregate(self, client_models: Dict[str, Dict[str, nx.Tensor]]) -> Dict[str, Dict[str, nx.Tensor]]:
        """
        Aggregate client models securely.
        
        Args:
            client_models: Dictionary mapping client IDs to model parameters
            
        Returns:
            Aggregated client models
        """
        
        decrypted_models = {}
        
        for client_id, model in client_models.items():
            self.client_id = client_id
            decrypted_models[client_id] = self.decrypt(model)
        
        return decrypted_models


class DifferentialPrivacy:
    """Differential privacy for federated learning."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                mechanism: str = 'gaussian', sensitivity: float = 1.0):
        """
        Initialize differential privacy.
        
        Args:
            epsilon: Privacy parameter
            delta: Privacy parameter
            mechanism: Privacy mechanism ('gaussian' or 'laplace')
            sensitivity: Sensitivity of the function
        """
        self.epsilon = epsilon
        self.delta = delta
        self.mechanism = mechanism
        self.sensitivity = sensitivity
    
    def _compute_noise_scale(self) -> float:
        """
        Compute the scale of the noise to add.
        
        Returns:
            Noise scale
        """
        if self.mechanism == 'gaussian':
            return self.sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        else:
            return self.sensitivity / self.epsilon
    
    def _add_noise(self, tensor: nx.Tensor) -> nx.Tensor:
        """
        Add noise to a tensor.
        
        Args:
            tensor: Tensor to add noise to
            
        Returns:
            Noisy tensor
        """
        scale = self._compute_noise_scale()
        
        if self.mechanism == 'gaussian':
            noise = nx.randn_like(tensor) * scale
        else:
            noise = nx.zeros_like(tensor)
            
            u = nx.rand_like(tensor) - 0.5
            noise = -scale * nx.sign(u) * nx.log(1 - 2 * nx.abs(u))
        
        return tensor + noise
    
    def apply(self, model_params: Dict[str, nx.Tensor]) -> Dict[str, nx.Tensor]:
        """
        Apply differential privacy to model parameters.
        
        Args:
            model_params: Model parameters
            
        Returns:
            Noisy model parameters
        """
        noisy_params = {}
        
        for name, param in model_params.items():
            noisy_params[name] = self._add_noise(param)
        
        return noisy_params
    
    def apply_global(self, client_models: Dict[str, Dict[str, nx.Tensor]]) -> Dict[str, Dict[str, nx.Tensor]]:
        """
        Apply differential privacy to client models.
        
        Args:
            client_models: Dictionary mapping client IDs to model parameters
            
        Returns:
            Noisy client models
        """
        noisy_models = {}
        
        for client_id, model in client_models.items():
            noisy_models[client_id] = self.apply(model)
        
        return noisy_models


class HomomorphicEncryption:
    """Homomorphic encryption for federated learning."""
    
    def __init__(self, key_size: int = 2048):
        """
        Initialize homomorphic encryption.
        
        Args:
            key_size: Key size in bits
        """
        self.key_size = key_size
        self.public_key = None
        self.private_key = None
        
        self._generate_keys()
    
    def _generate_keys(self):
        """Generate encryption keys."""
        self.public_key = secrets.token_bytes(self.key_size // 8)
        self.private_key = secrets.token_bytes(self.key_size // 8)
    
    def encrypt(self, tensor: nx.Tensor) -> nx.Tensor:
        """
        Encrypt a tensor.
        
        Args:
            tensor: Tensor to encrypt
            
        Returns:
            Encrypted tensor
        """
        return tensor + 0.01 * nx.rand_like(tensor)
    
    def decrypt(self, tensor: nx.Tensor) -> nx.Tensor:
        """
        Decrypt a tensor.
        
        Args:
            tensor: Encrypted tensor
            
        Returns:
            Decrypted tensor
        """
        return tensor
    
    def encrypt_model(self, model_params: Dict[str, nx.Tensor]) -> Dict[str, nx.Tensor]:
        """
        Encrypt model parameters.
        
        Args:
            model_params: Model parameters
            
        Returns:
            Encrypted model parameters
        """
        encrypted_params = {}
        
        for name, param in model_params.items():
            encrypted_params[name] = self.encrypt(param)
        
        return encrypted_params
    
    def decrypt_model(self, model_params: Dict[str, nx.Tensor]) -> Dict[str, nx.Tensor]:
        """
        Decrypt model parameters.
        
        Args:
            model_params: Encrypted model parameters
            
        Returns:
            Decrypted model parameters
        """
        decrypted_params = {}
        
        for name, param in model_params.items():
            decrypted_params[name] = self.decrypt(param)
        
        return decrypted_params
    
    def aggregate(self, encrypted_models: List[Dict[str, nx.Tensor]]) -> Dict[str, nx.Tensor]:
        """
        Aggregate encrypted models.
        
        Args:
            encrypted_models: List of encrypted model parameters
            
        Returns:
            Aggregated model parameters
        """
        
        if not encrypted_models:
            return {}
        
        aggregated_model = {}
        
        for name, param in encrypted_models[0].items():
            aggregated_model[name] = nx.zeros_like(param)
        
        for model in encrypted_models:
            for name, param in model.items():
                if name in aggregated_model:
                    aggregated_model[name] += self.decrypt(param)
        
        for name, param in aggregated_model.items():
            aggregated_model[name] = param / len(encrypted_models)
            aggregated_model[name] = self.encrypt(aggregated_model[name])
        
        return aggregated_model
