"""
Federated Learning module for Neurenix.

This module provides implementations of federated learning algorithms
and utilities for distributed training across multiple devices or clients.
"""

from neurenix.federated.client import (
    FederatedClient,
    ClientConfig,
    ClientState
)

from neurenix.federated.server import (
    FederatedServer,
    ServerConfig,
    ServerState,
    AggregationStrategy
)

from neurenix.federated.strategies import (
    FedAvg,
    FedProx,
    FedNova,
    FedOpt,
    FedAdagrad,
    FedAdam,
    FedYogi
)

from neurenix.federated.security import (
    SecureAggregation,
    DifferentialPrivacy,
    HomomorphicEncryption
)

from neurenix.federated.utils import (
    ClientSelector,
    RandomClientSelector,
    PowerOfChoiceSelector,
    ModelCompressor,
    GradientCompressor
)

__all__ = [
    'FederatedClient',
    'ClientConfig',
    'ClientState',
    
    'FederatedServer',
    'ServerConfig',
    'ServerState',
    'AggregationStrategy',
    
    'FedAvg',
    'FedProx',
    'FedNova',
    'FedOpt',
    'FedAdagrad',
    'FedAdam',
    'FedYogi',
    
    'SecureAggregation',
    'DifferentialPrivacy',
    'HomomorphicEncryption',
    
    'ClientSelector',
    'RandomClientSelector',
    'PowerOfChoiceSelector',
    'ModelCompressor',
    'GradientCompressor'
]
