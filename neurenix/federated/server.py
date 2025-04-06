"""
Federated Learning server module for Neurenix.

This module provides implementations of federated learning servers
for coordinating distributed training across multiple clients.
"""

from typing import Dict, List, Optional, Union, Tuple, Callable, Any
from enum import Enum
import time
import copy
import logging
import random

import neurenix as nx
from neurenix.nn import Module
from neurenix.federated.client import FederatedClient, ClientConfig


class ServerState(Enum):
    """Enum representing the state of a federated server."""
    IDLE = 0
    INITIALIZING = 1
    SELECTING_CLIENTS = 2
    DISTRIBUTING = 3
    AGGREGATING = 4
    EVALUATING = 5


class AggregationStrategy(Enum):
    """Enum representing the aggregation strategy for federated learning."""
    FED_AVG = 0
    FED_PROX = 1
    FED_NOVA = 2
    FED_OPT = 3
    FED_ADAGRAD = 4
    FED_ADAM = 5
    FED_YOGI = 6


class ServerConfig:
    """Configuration for a federated server."""
    
    def __init__(self,
                 num_rounds: int = 10,
                 clients_per_round: int = 10,
                 client_fraction: float = 0.1,
                 aggregation_strategy: AggregationStrategy = AggregationStrategy.FED_AVG,
                 min_clients: int = 2,
                 min_sample_size: int = 10,
                 min_num_samples: int = 100,
                 eval_every: int = 1,
                 accept_failures: bool = True,
                 timeout: float = 60.0,
                 secure_aggregation: bool = False,
                 differential_privacy: bool = False,
                 dp_epsilon: float = 1.0,
                 dp_delta: float = 1e-5,
                 dp_mechanism: str = 'gaussian',
                 compression: bool = False,
                 compression_ratio: float = 0.1,
                 device: str = 'cpu'):
        """
        Initialize server configuration.
        
        Args:
            num_rounds: Number of federated learning rounds
            clients_per_round: Number of clients to select per round
            client_fraction: Fraction of clients to select per round
            aggregation_strategy: Strategy for aggregating client updates
            min_clients: Minimum number of clients required for aggregation
            min_sample_size: Minimum sample size per client
            min_num_samples: Minimum number of samples across all clients
            eval_every: Evaluate global model every n rounds
            accept_failures: Whether to accept client failures
            timeout: Timeout for client operations in seconds
            secure_aggregation: Whether to use secure aggregation
            differential_privacy: Whether to use differential privacy
            dp_epsilon: Epsilon parameter for differential privacy
            dp_delta: Delta parameter for differential privacy
            dp_mechanism: Mechanism for differential privacy ('gaussian' or 'laplace')
            compression: Whether to use model compression
            compression_ratio: Compression ratio for model compression
            device: Device to use for training ('cpu', 'cuda', etc.)
        """
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.client_fraction = client_fraction
        self.aggregation_strategy = aggregation_strategy
        self.min_clients = min_clients
        self.min_sample_size = min_sample_size
        self.min_num_samples = min_num_samples
        self.eval_every = eval_every
        self.accept_failures = accept_failures
        self.timeout = timeout
        self.secure_aggregation = secure_aggregation
        self.differential_privacy = differential_privacy
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_mechanism = dp_mechanism
        self.compression = compression
        self.compression_ratio = compression_ratio
        self.device = device


class FederatedServer:
    """Base class for federated learning servers."""
    
    def __init__(self, config: ServerConfig):
        """
        Initialize a federated server.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.model = None
        self.criterion = None
        self.clients = {}
        self.state = ServerState.IDLE
        self.current_round = 0
        self.metrics = {}
        self.logger = logging.getLogger("FederatedServer")
    
    def initialize(self, model: Module, criterion: Callable):
        """
        Initialize the server with a model and criterion.
        
        Args:
            model: Global model
            criterion: Loss function
        """
        self.state = ServerState.INITIALIZING
        
        self.model = model.to(self.config.device)
        self.criterion = criterion
        
        self.state = ServerState.IDLE
    
    def add_client(self, client: FederatedClient):
        """
        Add a client to the server.
        
        Args:
            client: Federated client
        """
        self.clients[client.config.client_id] = client
    
    def select_clients(self, num_clients: Optional[int] = None) -> List[str]:
        """
        Select clients for the current round.
        
        Args:
            num_clients: Number of clients to select
            
        Returns:
            List of selected client IDs
        """
        self.state = ServerState.SELECTING_CLIENTS
        
        if num_clients is None:
            num_clients = max(
                self.config.min_clients,
                min(
                    self.config.clients_per_round,
                    int(self.config.client_fraction * len(self.clients))
                )
            )
        
        client_ids = list(self.clients.keys())
        selected_client_ids = random.sample(client_ids, min(num_clients, len(client_ids)))
        
        self.state = ServerState.IDLE
        
        return selected_client_ids
    
    def distribute_model(self, client_ids: List[str]):
        """
        Distribute the global model to selected clients.
        
        Args:
            client_ids: List of client IDs
        """
        self.state = ServerState.DISTRIBUTING
        
        model_params = self.model.state_dict()
        
        for client_id in client_ids:
            if client_id in self.clients:
                self.clients[client_id].set_model_parameters(model_params)
        
        self.state = ServerState.IDLE
    
    def train_clients(self, client_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Train selected clients.
        
        Args:
            client_ids: List of client IDs
            
        Returns:
            Dictionary mapping client IDs to metrics
        """
        client_metrics = {}
        
        for client_id in client_ids:
            if client_id in self.clients:
                try:
                    metrics = self.clients[client_id].train(self.current_round)
                    client_metrics[client_id] = metrics
                except Exception as e:
                    self.logger.error(f"Error training client {client_id}: {e}")
                    if not self.config.accept_failures:
                        raise
        
        return client_metrics
    
    def aggregate_models(self, client_ids: List[str]) -> Dict[str, nx.Tensor]:
        """
        Aggregate client models.
        
        Args:
            client_ids: List of client IDs
            
        Returns:
            Aggregated model parameters
        """
        self.state = ServerState.AGGREGATING
        
        if self.config.aggregation_strategy == AggregationStrategy.FED_AVG:
            from neurenix.federated.strategies import FedAvg
            
            strategy = FedAvg()
            
        elif self.config.aggregation_strategy == AggregationStrategy.FED_PROX:
            from neurenix.federated.strategies import FedProx
            
            strategy = FedProx()
            
        elif self.config.aggregation_strategy == AggregationStrategy.FED_NOVA:
            from neurenix.federated.strategies import FedNova
            
            strategy = FedNova()
            
        elif self.config.aggregation_strategy == AggregationStrategy.FED_OPT:
            from neurenix.federated.strategies import FedOpt
            
            strategy = FedOpt()
            
        elif self.config.aggregation_strategy == AggregationStrategy.FED_ADAGRAD:
            from neurenix.federated.strategies import FedAdagrad
            
            strategy = FedAdagrad()
            
        elif self.config.aggregation_strategy == AggregationStrategy.FED_ADAM:
            from neurenix.federated.strategies import FedAdam
            
            strategy = FedAdam()
            
        elif self.config.aggregation_strategy == AggregationStrategy.FED_YOGI:
            from neurenix.federated.strategies import FedYogi
            
            strategy = FedYogi()
            
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.config.aggregation_strategy}")
        
        client_models = {}
        client_weights = {}
        
        for client_id in client_ids:
            if client_id in self.clients:
                try:
                    client_models[client_id] = self.clients[client_id].get_model_update()
                    client_weights[client_id] = self.clients[client_id].metrics.get('n_samples', 1)
                except Exception as e:
                    self.logger.error(f"Error getting model from client {client_id}: {e}")
                    if not self.config.accept_failures:
                        raise
        
        if self.config.secure_aggregation:
            from neurenix.federated.security import SecureAggregation
            
            secure_agg = SecureAggregation()
            client_models = secure_agg.aggregate(client_models)
        
        if self.config.differential_privacy:
            from neurenix.federated.security import DifferentialPrivacy
            
            dp = DifferentialPrivacy(
                epsilon=self.config.dp_epsilon,
                delta=self.config.dp_delta,
                mechanism=self.config.dp_mechanism
            )
            
            client_models = dp.apply_global(client_models)
        
        aggregated_model = strategy.aggregate(
            global_model=self.model.state_dict(),
            client_models=client_models,
            client_weights=client_weights
        )
        
        self.state = ServerState.IDLE
        
        return aggregated_model
    
    def update_global_model(self, aggregated_model: Dict[str, nx.Tensor]):
        """
        Update the global model with aggregated parameters.
        
        Args:
            aggregated_model: Aggregated model parameters
        """
        self.model.load_state_dict(aggregated_model)
    
    def evaluate_global_model(self, test_data: Any) -> Dict[str, float]:
        """
        Evaluate the global model on test data.
        
        Args:
            test_data: Test data
            
        Returns:
            Dictionary of metrics
        """
        self.state = ServerState.EVALUATING
        
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        n_samples = 0
        
        with nx.no_grad():
            for data, target in test_data:
                data, target = data.to(self.config.device), target.to(self.config.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                test_loss += loss.item() * data.size(0)
                
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                test_acc += correct
                n_samples += data.size(0)
        
        test_loss /= n_samples
        test_acc /= n_samples
        
        metrics = {
            'test_loss': test_loss,
            'test_acc': test_acc,
            'n_samples': n_samples
        }
        
        self.metrics[f'round_{self.current_round}'] = metrics
        
        self.state = ServerState.IDLE
        
        return metrics
    
    def train(self, test_data: Optional[Any] = None) -> Dict[int, Dict[str, float]]:
        """
        Train the global model using federated learning.
        
        Args:
            test_data: Test data for evaluation
            
        Returns:
            Dictionary mapping rounds to metrics
        """
        if self.model is None:
            raise ValueError("Server not initialized")
        
        if len(self.clients) < self.config.min_clients:
            raise ValueError(f"Not enough clients: {len(self.clients)} < {self.config.min_clients}")
        
        round_metrics = {}
        
        for round_idx in range(self.config.num_rounds):
            self.current_round = round_idx
            self.logger.info(f"Round {round_idx + 1}/{self.config.num_rounds}")
            
            selected_clients = self.select_clients()
            self.logger.info(f"Selected {len(selected_clients)} clients")
            
            self.distribute_model(selected_clients)
            
            client_metrics = self.train_clients(selected_clients)
            
            aggregated_model = self.aggregate_models(selected_clients)
            
            self.update_global_model(aggregated_model)
            
            if test_data is not None and (round_idx + 1) % self.config.eval_every == 0:
                metrics = self.evaluate_global_model(test_data)
                round_metrics[round_idx] = metrics
                
                self.logger.info(
                    f"Round {round_idx + 1} metrics: "
                    f"loss={metrics['test_loss']:.4f}, "
                    f"acc={metrics['test_acc']:.4f}"
                )
        
        return round_metrics
    
    def get_model(self) -> Module:
        """
        Get the global model.
        
        Returns:
            Global model
        """
        return self.model
