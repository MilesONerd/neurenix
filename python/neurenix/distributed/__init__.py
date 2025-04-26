"""
Distributed computing module for Neurenix.

This module provides functionality for distributed training and inference
across multiple GPUs and compute nodes.
"""

from .distributed import DistributedContext, init_distributed, get_rank, get_world_size
from .data_parallel import DataParallel
from .sync import SyncBatchNorm
from .rpc import RpcContext, rpc_sync, rpc_async
from .dotnet import DotNetDistributedClient, DotNetDistributedContext

__all__ = [
    'DistributedContext',
    'init_distributed',
    'get_rank',
    'get_world_size',
    'DataParallel',
    'SyncBatchNorm',
    'RpcContext',
    'rpc_sync',
    'rpc_async',
    'DotNetDistributedClient',
    'DotNetDistributedContext',
]
