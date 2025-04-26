"""
Graph Neural Network layers for Neurenix.

This module provides implementations of various graph neural network layers
for processing graph-structured data.
"""

import math
from typing import Optional, Callable, Union, Tuple, List

import neurenix as nx
from neurenix.nn import Module, Linear, Parameter, Sequential, ReLU, Dropout
from neurenix.nn.functional import relu, softmax


class MessagePassing(Module):
    """Base class for message passing neural networks."""
    
    def __init__(self, aggr: str = 'add'):
        """
        Initialize message passing layer.
        
        Args:
            aggr: Aggregation method ('add', 'mean', or 'max')
        """
        super().__init__()
        self.aggr = aggr
        
    def message(self, x_i, x_j, edge_attr=None, edge_index_i=None):
        """
        Construct messages for each edge.
        
        Args:
            x_i: Source node features
            x_j: Target node features
            edge_attr: Edge features
            edge_index_i: Edge indices
            
        Returns:
            Messages
        """
        raise NotImplementedError("Subclasses must implement message method")
    
    def aggregate(self, messages, edge_index, num_nodes):
        """
        Aggregate messages from neighbors.
        
        Args:
            messages: Messages
            edge_index: Edge indices
            num_nodes: Number of nodes
            
        Returns:
            Aggregated messages
        """
        if self.aggr == 'add':
            return self._aggregate_add(messages, edge_index, num_nodes)
        elif self.aggr == 'mean':
            return self._aggregate_mean(messages, edge_index, num_nodes)
        elif self.aggr == 'max':
            return self._aggregate_max(messages, edge_index, num_nodes)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggr}")
    
    def _aggregate_add(self, messages, edge_index, num_nodes):
        """Aggregate messages by summation."""
        row, col = edge_index
        
        out = nx.zeros((num_nodes, messages.shape[1]), device=messages.device)
        
        for i in range(len(row)):
            out[row[i]] = out[row[i]] + messages[i]
        
        return out
    
    def _aggregate_mean(self, messages, edge_index, num_nodes):
        """Aggregate messages by mean."""
        row, col = edge_index
        
        out = nx.zeros((num_nodes, messages.shape[1]), device=messages.device)
        count = nx.zeros((num_nodes, 1), device=messages.device)
        
        for i in range(len(row)):
            out[row[i]] = out[row[i]] + messages[i]
            count[row[i]] += 1
        
        count = nx.maximum(count, nx.ones_like(count))  # Avoid division by zero
        return out / count
    
    def _aggregate_max(self, messages, edge_index, num_nodes):
        """Aggregate messages by max pooling."""
        row, col = edge_index
        
        out = nx.full((num_nodes, messages.shape[1]), float('-inf'), device=messages.device)
        
        for i in range(len(row)):
            out[row[i]] = nx.maximum(out[row[i]], messages[i])
        
        out = nx.where(out == float('-inf'), nx.zeros_like(out), out)
        
        return out
    
    def update(self, aggr_out, x):
        """
        Update node embeddings.
        
        Args:
            aggr_out: Aggregated messages
            x: Node features
            
        Returns:
            Updated node features
        """
        return aggr_out
    
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            
        Returns:
            Updated node features
        """
        row, col = edge_index
        
        x_i, x_j = x[row], x[col]
        
        messages = self.message(x_i, x_j, edge_attr, row)
        
        aggr_out = self.aggregate(messages, edge_index, x.shape[0])
        
        return self.update(aggr_out, x)


class GraphConv(MessagePassing):
    """Graph Convolutional Layer (GCN)."""
    
    def __init__(self, in_channels: int, out_channels: int, aggr: str = 'add',
                 bias: bool = True, normalize: bool = True):
        """
        Initialize GCN layer.
        
        Args:
            in_channels: Size of each input sample
            out_channels: Size of each output sample
            aggr: Aggregation method ('add', 'mean', or 'max')
            bias: If set to False, the layer will not learn an additive bias
            normalize: If set to True, output features will be normalized by degree
        """
        super().__init__(aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self._cached_edge_index = None
        
        self.weight = Parameter(nx.empty((in_channels, out_channels)))
        
        if bias:
            self.bias = Parameter(nx.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset learnable parameters."""
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def message(self, x_i, x_j, edge_attr=None, edge_index_i=None):
        """Construct messages."""
        return x_j @ self.weight
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge features
            
        Returns:
            Updated node features
        """
        self._cached_edge_index = edge_index
        
        return super().forward(x, edge_index, edge_attr)
    
    def update(self, aggr_out, x):
        """Update node embeddings."""
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        
        if self.normalize and self._cached_edge_index is not None:
            row, col = self._cached_edge_index
            deg = nx.zeros(x.shape[0], device=x.device)
            
            for i in range(len(row)):
                deg[row[i]] += 1
            
            deg = nx.maximum(deg, nx.ones_like(deg))  # Avoid division by zero
            deg_inv_sqrt = deg.pow(-0.5)
            
            aggr_out = aggr_out * deg_inv_sqrt.view(-1, 1)
        
        return aggr_out


class GraphAttention(MessagePassing):
    """Graph Attention Layer (GAT)."""
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 aggr: str = 'add', bias: bool = True):
        """
        Initialize GAT layer.
        
        Args:
            in_channels: Size of each input sample
            out_channels: Size of each output sample
            heads: Number of attention heads
            negative_slope: LeakyReLU angle of the negative slope
            dropout: Dropout probability
            aggr: Aggregation method ('add', 'mean', or 'max')
            bias: If set to False, the layer will not learn an additive bias
        """
        super().__init__(aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        self.weight = Parameter(nx.empty((in_channels, heads * out_channels)))
        self.att = Parameter(nx.empty((1, heads, 2 * out_channels)))
        
        if bias:
            self.bias = Parameter(nx.empty(heads * out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset learnable parameters."""
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Updated node features
        """
        x = x @ self.weight
        
        x = x.view(-1, self.heads, self.out_channels)
        
        row, col = edge_index
        
        x_i, x_j = x[row], x[col]
        
        alpha = nx.cat([x_i, x_j], dim=-1)
        alpha = (alpha * self.att).sum(dim=-1)
        alpha = nx.leaky_relu(alpha, self.negative_slope)
        
        alpha = softmax(alpha, row, x.shape[0])
        
        if self.training and self.dropout > 0:
            alpha = nx.dropout(alpha, p=self.dropout)
        
        out = alpha.view(-1, self.heads, 1) * x_j
        
        out = self._aggregate_add(out, edge_index, x.shape[0])
        
        out = out.view(-1, self.heads * self.out_channels)
        
        if self.bias is not None:
            out = out + self.bias
        
        return out


class GraphSage(MessagePassing):
    """GraphSAGE Layer."""
    
    def __init__(self, in_channels: int, out_channels: int, aggr: str = 'mean',
                 bias: bool = True, normalize: bool = False):
        """
        Initialize GraphSAGE layer.
        
        Args:
            in_channels: Size of each input sample
            out_channels: Size of each output sample
            aggr: Aggregation method ('add', 'mean', or 'max')
            bias: If set to False, the layer will not learn an additive bias
            normalize: If set to True, output features will be L2-normalized
        """
        super().__init__(aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        
        self.lin_self = Linear(in_channels, out_channels, bias=bias)
        self.lin_neigh = Linear(in_channels, out_channels, bias=False)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset learnable parameters."""
        self.lin_self.reset_parameters()
        self.lin_neigh.reset_parameters()
    
    def message(self, x_i, x_j, edge_attr=None, edge_index_i=None):
        """Construct messages."""
        return x_j
    
    def update(self, aggr_out, x):
        """Update node embeddings."""
        out = self.lin_self(x) + self.lin_neigh(aggr_out)
        
        if self.normalize:
            out = nx.normalize(out, p=2, dim=-1)
        
        return out


class EdgeConv(MessagePassing):
    """Edge Convolutional Layer."""
    
    def __init__(self, nn: Module, aggr: str = 'max'):
        """
        Initialize EdgeConv layer.
        
        Args:
            nn: Neural network to be applied to edge features
            aggr: Aggregation method ('add', 'mean', or 'max')
        """
        super().__init__(aggr)
        self.nn = nn
    
    def message(self, x_i, x_j, edge_attr=None, edge_index_i=None):
        """Construct messages."""
        edge_features = nx.cat([x_i, x_j - x_i], dim=-1)
        
        return self.nn(edge_features)


class GINConv(MessagePassing):
    """Graph Isomorphism Network (GIN) Layer."""
    
    def __init__(self, nn: Module, eps: float = 0.0, train_eps: bool = False,
                 aggr: str = 'add'):
        """
        Initialize GIN layer.
        
        Args:
            nn: Neural network to be applied to node features
            eps: Initial epsilon value
            train_eps: If True, epsilon will be a learnable parameter
            aggr: Aggregation method ('add', 'mean', or 'max')
        """
        super().__init__(aggr)
        
        self.nn = nn
        
        if train_eps:
            self.eps = Parameter(nx.tensor([eps]))
        else:
            self.register_buffer('eps', nx.tensor([eps]))
    
    def message(self, x_i, x_j, edge_attr=None, edge_index_i=None):
        """Construct messages."""
        return x_j
    
    def update(self, aggr_out, x):
        """Update node embeddings."""
        out = (1 + self.eps) * x + aggr_out
        
        return self.nn(out)


class GatedGraphConv(MessagePassing):
    """Gated Graph Convolutional Layer."""
    
    def __init__(self, out_channels: int, num_layers: int, aggr: str = 'add',
                 bias: bool = True):
        """
        Initialize Gated GCN layer.
        
        Args:
            out_channels: Size of each output sample
            num_layers: Number of message passing layers
            aggr: Aggregation method ('add', 'mean', or 'max')
            bias: If set to False, the layer will not learn an additive bias
        """
        super().__init__(aggr)
        
        self.out_channels = out_channels
        self.num_layers = num_layers
        
        self.weight = Parameter(nx.empty((num_layers, out_channels, out_channels)))
        
        if bias:
            self.bias = Parameter(nx.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.rnn = nx.nn.GRUCell(out_channels, out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset learnable parameters."""
        stdv = 1. / math.sqrt(self.out_channels)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
        
        self.rnn.reset_parameters()
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Updated node features
        """
        if x.shape[1] != self.out_channels:
            x = Linear(x.shape[1], self.out_channels)(x)
        
        h = x
        
        for i in range(self.num_layers):
            weight = self.weight[i]
            
            m = x @ weight
            
            m = self.propagate(edge_index, x=m)
            
            if self.bias is not None:
                m = m + self.bias
            
            h = self.rnn(m, h)
        
        return h
    
    def message(self, x_j):
        """Construct messages."""
        return x_j


class RelationalGraphConv(MessagePassing):
    """Relational Graph Convolutional Layer."""
    
    def __init__(self, in_channels: int, out_channels: int, num_relations: int,
                 aggr: str = 'mean', bias: bool = True, self_loop: bool = True):
        """
        Initialize RGCN layer.
        
        Args:
            in_channels: Size of each input sample
            out_channels: Size of each output sample
            num_relations: Number of relations
            aggr: Aggregation method ('add', 'mean', or 'max')
            bias: If set to False, the layer will not learn an additive bias
            self_loop: If set to False, the layer will not add self loops
        """
        super().__init__(aggr)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.self_loop = self_loop
        
        self.weight = Parameter(nx.empty((num_relations, in_channels, out_channels)))
        
        if self_loop:
            self.self_weight = Parameter(nx.empty((in_channels, out_channels)))
        else:
            self.register_parameter('self_weight', None)
        
        if bias:
            self.bias = Parameter(nx.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset learnable parameters."""
        stdv = 1. / math.sqrt(self.weight.shape[2])
        self.weight.data.uniform_(-stdv, stdv)
        if self.self_loop:
            self.self_weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
    
    def forward(self, x, edge_index, edge_type):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_type: Edge types
            
        Returns:
            Updated node features
        """
        if self.self_loop:
            out = x @ self.self_weight
        else:
            out = nx.zeros((x.shape[0], self.out_channels), device=x.device)
        
        for i in range(self.num_relations):
            mask = edge_type == i
            if not mask.any():
                continue
            
            rel_edge_index = edge_index[:, mask]
            
            weight = self.weight[i]
            
            rel_x = x @ weight
            
            rel_out = self.propagate(rel_edge_index, x=rel_x)
            
            out = out + rel_out
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def message(self, x_j):
        """Construct messages."""
        return x_j
