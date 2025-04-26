"""
Graph Neural Network models for Neurenix.

This module provides implementations of various graph neural network architectures
for processing graph-structured data.
"""

from typing import List, Optional, Union, Tuple

import neurenix as nx
from neurenix.nn import Module, Linear, Sequential, ReLU, Dropout
from neurenix.gnn.layers import (
    GraphConv, 
    GraphAttention, 
    GraphSage, 
    GINConv, 
    GatedGraphConv,
    RelationalGraphConv
)
from neurenix.gnn.pooling import (
    GlobalAddPooling,
    GlobalMeanPooling,
    GlobalMaxPooling
)


class GCN(Module):
    """Graph Convolutional Network."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.0, 
                 activation: str = 'relu', normalize: bool = True):
        """
        Initialize GCN model.
        
        Args:
            in_channels: Size of each input sample
            hidden_channels: Size of hidden layers
            out_channels: Size of each output sample
            num_layers: Number of GCN layers
            dropout: Dropout probability
            activation: Activation function ('relu', 'elu', etc.)
            normalize: If set to True, output features will be normalized by degree
        """
        super().__init__()
        
        self.convs = nx.nn.ModuleList()
        
        self.convs.append(GraphConv(in_channels, hidden_channels, normalize=normalize))
        
        for _ in range(num_layers - 2):
            self.convs.append(GraphConv(hidden_channels, hidden_channels, normalize=normalize))
        
        if num_layers > 1:
            self.convs.append(GraphConv(hidden_channels, out_channels, normalize=normalize))
        
        self.dropout = dropout
        self.activation = activation
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Node embeddings
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            
            if self.activation == 'relu':
                x = nx.relu(x)
            elif self.activation == 'elu':
                x = nx.elu(x)
            
            x = nx.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        return x


class GAT(Module):
    """Graph Attention Network."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, heads: int = 8, output_heads: int = 1,
                 dropout: float = 0.6, activation: str = 'elu'):
        """
        Initialize GAT model.
        
        Args:
            in_channels: Size of each input sample
            hidden_channels: Size of hidden layers
            out_channels: Size of each output sample
            num_layers: Number of GAT layers
            heads: Number of attention heads for hidden layers
            output_heads: Number of attention heads for output layer
            dropout: Dropout probability
            activation: Activation function ('relu', 'elu', etc.)
        """
        super().__init__()
        
        self.convs = nx.nn.ModuleList()
        
        self.convs.append(GraphAttention(in_channels, hidden_channels, heads=heads, dropout=dropout))
        
        for _ in range(num_layers - 2):
            self.convs.append(GraphAttention(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
        
        if num_layers > 1:
            self.convs.append(GraphAttention(hidden_channels * heads, out_channels, heads=output_heads, dropout=dropout))
        
        self.dropout = dropout
        self.activation = activation
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Node embeddings
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            
            if self.activation == 'relu':
                x = nx.relu(x)
            elif self.activation == 'elu':
                x = nx.elu(x)
            
            x = nx.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        return x


class GraphSAGE(Module):
    """GraphSAGE model."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.0, 
                 activation: str = 'relu', normalize: bool = False):
        """
        Initialize GraphSAGE model.
        
        Args:
            in_channels: Size of each input sample
            hidden_channels: Size of hidden layers
            out_channels: Size of each output sample
            num_layers: Number of GraphSAGE layers
            dropout: Dropout probability
            activation: Activation function ('relu', 'elu', etc.)
            normalize: If set to True, output features will be L2-normalized
        """
        super().__init__()
        
        self.convs = nx.nn.ModuleList()
        
        self.convs.append(GraphSage(in_channels, hidden_channels, normalize=normalize))
        
        for _ in range(num_layers - 2):
            self.convs.append(GraphSage(hidden_channels, hidden_channels, normalize=normalize))
        
        if num_layers > 1:
            self.convs.append(GraphSage(hidden_channels, out_channels, normalize=normalize))
        
        self.dropout = dropout
        self.activation = activation
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Node embeddings
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            
            if self.activation == 'relu':
                x = nx.relu(x)
            elif self.activation == 'elu':
                x = nx.elu(x)
            
            x = nx.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        return x


class GIN(Module):
    """Graph Isomorphism Network."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.0, 
                 activation: str = 'relu', train_eps: bool = False):
        """
        Initialize GIN model.
        
        Args:
            in_channels: Size of each input sample
            hidden_channels: Size of hidden layers
            out_channels: Size of each output sample
            num_layers: Number of GIN layers
            dropout: Dropout probability
            activation: Activation function ('relu', 'elu', etc.)
            train_eps: If True, epsilon will be a learnable parameter
        """
        super().__init__()
        
        self.convs = nx.nn.ModuleList()
        
        mlp = Sequential(
            Linear(in_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(mlp, train_eps=train_eps))
        
        for _ in range(num_layers - 2):
            mlp = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
        
        if num_layers > 1:
            mlp = Sequential(
                Linear(hidden_channels, hidden_channels),
                ReLU(),
                Linear(hidden_channels, out_channels)
            )
            self.convs.append(GINConv(mlp, train_eps=train_eps))
        
        self.dropout = dropout
        self.activation = activation
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Node embeddings
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            
            if self.activation == 'relu':
                x = nx.relu(x)
            elif self.activation == 'elu':
                x = nx.elu(x)
            
            x = nx.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        
        return x


class RGCN(Module):
    """Relational Graph Convolutional Network."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_relations: int, num_layers: int = 2, dropout: float = 0.0, 
                 activation: str = 'relu', self_loop: bool = True):
        """
        Initialize RGCN model.
        
        Args:
            in_channels: Size of each input sample
            hidden_channels: Size of hidden layers
            out_channels: Size of each output sample
            num_relations: Number of relations
            num_layers: Number of RGCN layers
            dropout: Dropout probability
            activation: Activation function ('relu', 'elu', etc.)
            self_loop: If set to False, the layer will not add self loops
        """
        super().__init__()
        
        self.convs = nx.nn.ModuleList()
        
        self.convs.append(RelationalGraphConv(in_channels, hidden_channels, num_relations, self_loop=self_loop))
        
        for _ in range(num_layers - 2):
            self.convs.append(RelationalGraphConv(hidden_channels, hidden_channels, num_relations, self_loop=self_loop))
        
        if num_layers > 1:
            self.convs.append(RelationalGraphConv(hidden_channels, out_channels, num_relations, self_loop=self_loop))
        
        self.dropout = dropout
        self.activation = activation
    
    def forward(self, x, edge_index, edge_type):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_type: Edge types
            
        Returns:
            Node embeddings
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_type)
            
            if self.activation == 'relu':
                x = nx.relu(x)
            elif self.activation == 'elu':
                x = nx.elu(x)
            
            x = nx.dropout(x, p=self.dropout, training=self.training)
        
        x = self.convs[-1](x, edge_index, edge_type)
        
        return x


class GGNN(Module):
    """Gated Graph Neural Network."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 5, dropout: float = 0.0):
        """
        Initialize GGNN model.
        
        Args:
            in_channels: Size of each input sample
            hidden_channels: Size of hidden layers
            out_channels: Size of each output sample
            num_layers: Number of propagation steps
            dropout: Dropout probability
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        
        if in_channels != hidden_channels:
            self.lin_in = Linear(in_channels, hidden_channels)
        else:
            self.lin_in = None
        
        self.ggnn = GatedGraphConv(hidden_channels, num_layers=num_layers)
        
        self.lin_out = Linear(hidden_channels, out_channels)
        
        self.dropout = dropout
    
    def forward(self, x, edge_index):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            
        Returns:
            Node embeddings
        """
        if self.lin_in is not None:
            x = self.lin_in(x)
        
        x = self.ggnn(x, edge_index)
        
        x = nx.dropout(x, p=self.dropout, training=self.training)
        
        x = self.lin_out(x)
        
        return x
