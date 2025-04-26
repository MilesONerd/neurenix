"""
Graph Neural Networks (GNNs) module for Neurenix.

This module provides implementations of various graph neural network architectures
and operations for processing graph-structured data.
"""

from neurenix.gnn.layers import (
    GraphConv,
    GraphAttention,
    GraphSage,
    EdgeConv,
    GINConv,
    GatedGraphConv,
    RelationalGraphConv
)

from neurenix.gnn.models import (
    GCN,
    GAT,
    GraphSAGE,
    GIN,
    RGCN,
    GGNN
)

from neurenix.gnn.data import (
    Graph,
    BatchedGraph,
    GraphDataset,
    GraphDataLoader
)

from neurenix.gnn.utils import (
    to_edge_index,
    to_adjacency_matrix,
    add_self_loops,
    remove_self_loops,
    normalize_adjacency
)

from neurenix.gnn.pooling import (
    GlobalPooling,
    GlobalAddPooling,
    GlobalMeanPooling,
    GlobalMaxPooling,
    TopKPooling,
    SAGPooling,
    DiffPooling
)

__all__ = [
    'GraphConv',
    'GraphAttention',
    'GraphSage',
    'EdgeConv',
    'GINConv',
    'GatedGraphConv',
    'RelationalGraphConv',
    
    'GCN',
    'GAT',
    'GraphSAGE',
    'GIN',
    'RGCN',
    'GGNN',
    
    'Graph',
    'BatchedGraph',
    'GraphDataset',
    'GraphDataLoader',
    
    'to_edge_index',
    'to_adjacency_matrix',
    'add_self_loops',
    'remove_self_loops',
    'normalize_adjacency',
    
    'GlobalPooling',
    'GlobalAddPooling',
    'GlobalMeanPooling',
    'GlobalMaxPooling',
    'TopKPooling',
    'SAGPooling',
    'DiffPooling'
]
