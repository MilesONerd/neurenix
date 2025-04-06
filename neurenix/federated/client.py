"""
Federated Learning client module for Neurenix.

This module provides implementations of federated learning clients
for distributed training across multiple devices or clients.
"""

from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from enum import Enum
import time
import copy

import neurenix as nx
from neurenix.nn import Module


class ClientState(Enum):
    """Enum representing the state of a federated client."""
    IDLE = 0
    TRAINING = 1
    EVALUATING = 2
    SENDING = 3
    RECEIVING = 4


class ClientConfig:
    """Configuration for a federated client."""
    
    def __init__(self, 
                 client_id: str,
                 batch_size: int = 32,
                 epochs: int = 1,
                 learning_rate: float = 0.01,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0,
                 max_grad_norm: Optional[float] = None,
                 proximal_mu: float = 0.0,
                 device: str = 'cpu',
                 secure_aggregation: bool = False,
                 differential_privacy: bool = False,
                 dp_epsilon: float = 1.0,
                 dp_delta: float = 1e-5,
                 dp_mechanism: str = 'gaussian',
                 compression: bool = False,
                 compression_ratio: float = 0.1):
        """
        Initialize client configuration.
        
        Args:
            client_id: Unique identifier for the client
            batch_size: Batch size for training
            epochs: Number of local epochs
            learning_rate: Learning rate for optimization
            momentum: Momentum for optimization
            weight_decay: Weight decay for optimization
            max_grad_norm: Maximum gradient norm for gradient clipping
            proximal_mu: Proximal term coefficient for FedProx
            device: Device to use for training ('cpu', 'cuda', etc.)
            secure_aggregation: Whether to use secure aggregation
            differential_privacy: Whether to use differential privacy
            dp_epsilon: Epsilon parameter for differential privacy
            dp_delta: Delta parameter for differential privacy
            dp_mechanism: Mechanism for differential privacy ('gaussian' or 'laplace')
            compression: Whether to use model compression
            compression_ratio: Compression ratio for model compression
        """
        self.client_id = client_id
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.proximal_mu = proximal_mu
        self.device = device
        self.secure_aggregation = secure_aggregation
        self.differential_privacy = differential_privacy
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_mechanism = dp_mechanism
        self.compression = compression
        self.compression_ratio = compression_ratio


class FederatedClient:
    """Base class for federated learning clients."""
    
    def __init__(self, config: ClientConfig):
        """
        Initialize a federated client.
        
        Args:
            config: Client configuration
        """
        self.config = config
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.train_data = None
        self.val_data = None
        self.state = ClientState.IDLE
        self.initial_model_params = None
        self.metrics = {}
    
    def initialize(self, model: Module, criterion: Callable, 
                  train_data: Any, val_data: Optional[Any] = None):
        """
        Initialize the client with a model, criterion, and data.
        
        Args:
            model: Model to train
            criterion: Loss function
            train_data: Training data
            val_data: Validation data
        """
        self.model = model.to(self.config.device)
        self.criterion = criterion
        self.train_data = train_data
        self.val_data = val_data
        
        self.optimizer = nx.optim.SGD(
            self.model.parameters(),
            lr=self.config.learning_rate,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
    
    def train(self, global_round: int) -> Dict[str, float]:
        """
        Train the model for one round of federated learning.
        
        Args:
            global_round: Current global round
            
        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Client not initialized")
        
        self.state = ClientState.TRAINING
        
        if self.config.proximal_mu > 0:
            self.initial_model_params = copy.deepcopy(self.model.state_dict())
        
        self.model.train()
        train_loss = 0.0
        train_acc = 0.0
        n_samples = 0
        
        for epoch in range(self.config.epochs):
            for batch_idx, (data, target) in enumerate(self.train_data):
                data, target = data.to(self.config.device), target.to(self.config.device)
                
                self.optimizer.zero_grad()
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                if self.config.proximal_mu > 0:
                    proximal_term = 0.0
                    for name, param in self.model.named_parameters():
                        proximal_term += (param - self.initial_model_params[name]).norm(2) ** 2
                    loss += 0.5 * self.config.proximal_mu * proximal_term
                
                loss.backward()
                
                if self.config.max_grad_norm is not None:
                    nx.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
                
                train_loss += loss.item() * data.size(0)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                train_acc += correct
                n_samples += data.size(0)
        
        train_loss /= n_samples
        train_acc /= n_samples
        
        val_loss, val_acc = 0.0, 0.0
        if self.val_data is not None:
            val_loss, val_acc = self.evaluate()
        
        self.metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'n_samples': n_samples
        }
        
        self.state = ClientState.IDLE
        
        return self.metrics
    
    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the model on validation data.
        
        Returns:
            Tuple of (loss, accuracy)
        """
        if self.model is None or self.val_data is None:
            return 0.0, 0.0
        
        self.state = ClientState.EVALUATING
        
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_samples = 0
        
        with nx.no_grad():
            for data, target in self.val_data:
                data, target = data.to(self.config.device), target.to(self.config.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                val_loss += loss.item() * data.size(0)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                val_acc += correct
                n_samples += data.size(0)
        
        val_loss /= n_samples
        val_acc /= n_samples
        
        self.state = ClientState.IDLE
        
        return val_loss, val_acc
    
    def get_model_update(self) -> Dict[str, nx.Tensor]:
        """
        Get the model update for federated aggregation.
        
        Returns:
            Dictionary of model parameters
        """
        if self.model is None:
            raise ValueError("Client not initialized")
        
        self.state = ClientState.SENDING
        
        model_params = self.model.state_dict()
        
        if self.config.differential_privacy:
            from neurenix.federated.security import DifferentialPrivacy
            
            dp = DifferentialPrivacy(
                epsilon=self.config.dp_epsilon,
                delta=self.config.dp_delta,
                mechanism=self.config.dp_mechanism
            )
            
            model_params = dp.apply(model_params)
        
        if self.config.compression:
            from neurenix.federated.utils import ModelCompressor
            
            compressor = ModelCompressor(ratio=self.config.compression_ratio)
            model_params = compressor.compress(model_params)
        
        if self.config.secure_aggregation:
            from neurenix.federated.security import SecureAggregation
            
            secure_agg = SecureAggregation(client_id=self.config.client_id)
            model_params = secure_agg.encrypt(model_params)
        
        self.state = ClientState.IDLE
        
        return model_params
    
    def set_model_parameters(self, model_params: Dict[str, nx.Tensor]):
        """
        Set the model parameters from the global model.
        
        Args:
            model_params: Dictionary of model parameters
        """
        if self.model is None:
            raise ValueError("Client not initialized")
        
        self.state = ClientState.RECEIVING
        
        if self.config.secure_aggregation:
            from neurenix.federated.security import SecureAggregation
            
            secure_agg = SecureAggregation(client_id=self.config.client_id)
            model_params = secure_agg.decrypt(model_params)
        
        if self.config.compression:
            from neurenix.federated.utils import ModelCompressor
            
            compressor = ModelCompressor(ratio=self.config.compression_ratio)
            model_params = compressor.decompress(model_params)
        
        self.model.load_state_dict(model_params)
        
        self.state = ClientState.IDLE
