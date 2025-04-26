"""
Graph pooling operations for Neurenix.

This module provides implementations of various graph pooling operations
for aggregating node features into graph-level representations.
"""

import math
from typing import Optional, Callable, Union, Tuple, List

import neurenix as nx
from neurenix.nn import Module, Linear, Parameter, Sequential, ReLU, Dropout
from neurenix.nn.functional import relu, softmax


class GlobalPooling(Module):
    """Base class for global pooling operations."""
    
    def forward(self, x, batch=None, size=None):
        """
        Forward pass.
        
        Args:
            x: Node features
            batch: Batch vector, which assigns each node to a specific example
            size: Size of the batch
            
        Returns:
            Graph-level representations
        """
        if batch is None:
            return x.mean(dim=0, keepdim=True)
        
        if size is None:
            size = int(batch.max()) + 1
        
        return self._pool(x, batch, size)
    
    def _pool(self, x, batch, size):
        """
        Pool node features to graph representations.
        
        Args:
            x: Node features
            batch: Batch vector
            size: Size of the batch
            
        Returns:
            Graph-level representations
        """
        raise NotImplementedError("Subclasses must implement _pool method")


class GlobalAddPooling(GlobalPooling):
    """Global add pooling."""
    
    def _pool(self, x, batch, size):
        """
        Pool node features to graph representations by summation.
        
        Args:
            x: Node features
            batch: Batch vector
            size: Size of the batch
            
        Returns:
            Graph-level representations
        """
        out = nx.zeros((size, x.shape[1]), device=x.device)
        
        for i in range(x.shape[0]):
            out[batch[i]] = out[batch[i]] + x[i]
        
        return out


class GlobalMeanPooling(GlobalPooling):
    """Global mean pooling."""
    
    def _pool(self, x, batch, size):
        """
        Pool node features to graph representations by averaging.
        
        Args:
            x: Node features
            batch: Batch vector
            size: Size of the batch
            
        Returns:
            Graph-level representations
        """
        out = nx.zeros((size, x.shape[1]), device=x.device)
        count = nx.zeros((size, 1), device=x.device)
        
        for i in range(x.shape[0]):
            out[batch[i]] = out[batch[i]] + x[i]
            count[batch[i]] += 1
        
        count = nx.maximum(count, nx.ones_like(count))  # Avoid division by zero
        return out / count


class GlobalMaxPooling(GlobalPooling):
    """Global max pooling."""
    
    def _pool(self, x, batch, size):
        """
        Pool node features to graph representations by max pooling.
        
        Args:
            x: Node features
            batch: Batch vector
            size: Size of the batch
            
        Returns:
            Graph-level representations
        """
        out = nx.full((size, x.shape[1]), float('-inf'), device=x.device)
        
        for i in range(x.shape[0]):
            out[batch[i]] = nx.maximum(out[batch[i]], x[i])
        
        out = nx.where(out == float('-inf'), nx.zeros_like(out), out)
        
        return out


class TopKPooling(Module):
    """Top-k pooling layer."""
    
    def __init__(self, in_channels: int, ratio: float = 0.5, min_score: float = None,
                 multiplier: float = 1.0, nonlinearity: str = 'tanh'):
        """
        Initialize top-k pooling layer.
        
        Args:
            in_channels: Size of each input sample
            ratio: Pooling ratio
            min_score: Minimum score for nodes to be selected
            multiplier: Multiplier for output features
            nonlinearity: Nonlinearity to apply to scores
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        
        self.weight = Parameter(nx.empty(in_channels))
        
        if nonlinearity == 'tanh':
            self.nonlinearity = nx.tanh
        elif nonlinearity == 'sigmoid':
            self.nonlinearity = nx.sigmoid
        else:
            self.nonlinearity = lambda x: x
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset learnable parameters."""
        stdv = 1. / math.sqrt(self.weight.shape[0])
        self.weight.data.uniform_(-stdv, stdv)
    
    def forward(self, x, edge_index, batch=None, attn=None):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch vector
            attn: Optional attention scores
            
        Returns:
            Pooled node features, pooled edge indices, pooled batch vector, attention scores
        """
        if batch is None:
            batch = nx.zeros(x.shape[0], dtype=nx.int64, device=x.device)
        
        attn = x @ self.weight if attn is None else attn
        attn = self.nonlinearity(attn)
        
        if self.min_score is None:
            score = attn
        else:
            score = nx.where(attn > self.min_score, attn, nx.zeros_like(attn))
        
        perm = self._topk(score, self.ratio, batch)
        
        x = x[perm] * self.multiplier
        
        row, col = edge_index
        mask = (row.unsqueeze(0) == perm.unsqueeze(1)).any(dim=0)
        mask = mask & (col.unsqueeze(0) == perm.unsqueeze(1)).any(dim=0)
        
        edge_index = edge_index[:, mask]
        
        node_idx = nx.full((x.shape[0],), -1, dtype=nx.int64, device=x.device)
        node_idx[perm] = nx.arange(perm.shape[0], device=x.device)
        
        edge_index = node_idx[edge_index]
        
        return x, edge_index, batch[perm], attn[perm]
    
    def _topk(self, x, ratio, batch):
        """
        Select top-k elements.
        
        Args:
            x: Scores
            ratio: Pooling ratio
            batch: Batch vector
            
        Returns:
            Indices of selected elements
        """
        if ratio < 1:
            num_nodes = nx.zeros(int(batch.max()) + 1, dtype=nx.int64, device=x.device)
            
            for i in range(batch.shape[0]):
                num_nodes[batch[i]] += 1
            
            batch_size = int(num_nodes.sum())
            
            k = (ratio * num_nodes.to(nx.float32)).ceil().to(nx.int64)
            
            perm = nx.zeros(batch_size, dtype=nx.int64, device=x.device)
            
            for i in range(int(batch.max()) + 1):
                mask = batch == i
                scores = x[mask]
                
                if scores.shape[0] > 0:
                    _, indices = nx.topk(scores, min(int(k[i]), scores.shape[0]))
                    perm[mask] = indices
            
            return perm
        else:
            return nx.arange(x.shape[0], device=x.device)


class SAGPooling(TopKPooling):
    """Self-Attention Graph Pooling."""
    
    def __init__(self, in_channels: int, ratio: float = 0.5, min_score: float = None,
                 multiplier: float = 1.0, nonlinearity: str = 'tanh', GNN=None):
        """
        Initialize SAGPooling layer.
        
        Args:
            in_channels: Size of each input sample
            ratio: Pooling ratio
            min_score: Minimum score for nodes to be selected
            multiplier: Multiplier for output features
            nonlinearity: Nonlinearity to apply to scores
            GNN: Graph neural network to compute attention scores
        """
        super().__init__(in_channels, ratio, min_score, multiplier, nonlinearity)
        
        if GNN is None:
            from neurenix.gnn.layers import GraphConv
            self.gnn = GraphConv(in_channels, 1)
        else:
            self.gnn = GNN(in_channels, 1)
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch vector
            
        Returns:
            Pooled node features, pooled edge indices, pooled batch vector, attention scores
        """
        attn = self.gnn(x, edge_index).view(-1)
        
        return super().forward(x, edge_index, batch, attn)


class DiffPooling(Module):
    """Differentiable Pooling."""
    
    def __init__(self, in_channels: int, out_channels: int, num_clusters: int, GNN=None):
        """
        Initialize DiffPooling layer.
        
        Args:
            in_channels: Size of each input sample
            out_channels: Size of each output sample
            num_clusters: Number of clusters
            GNN: Graph neural network to compute embeddings and cluster assignments
        """
        super().__init__()
        
        if GNN is None:
            from neurenix.gnn.layers import GraphConv
            self.gnn_embed = GraphConv(in_channels, out_channels)
            self.gnn_pool = GraphConv(in_channels, num_clusters)
        else:
            self.gnn_embed = GNN(in_channels, out_channels)
            self.gnn_pool = GNN(in_channels, num_clusters)
        
        self.num_clusters = num_clusters
    
    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            batch: Batch vector
            
        Returns:
            Pooled node features, pooled edge indices, pooled batch vector
        """
        if batch is None:
            batch = nx.zeros(x.shape[0], dtype=nx.int64, device=x.device)
        
        x_embed = self.gnn_embed(x, edge_index)
        s = self.gnn_pool(x, edge_index)
        s = nx.softmax(s, dim=1)
        
        row, col = edge_index
        adj = nx.zeros((x.shape[0], x.shape[0]), device=x.device)
        
        for i in range(len(row)):
            adj[row[i], col[i]] = 1
        
        x_pool = s.t() @ x_embed
        adj_pool = s.t() @ adj @ s
        
        edge_index_pool = adj_pool.nonzero().t()
        
        batch_size = int(batch.max()) + 1
        batch_pool = nx.repeat(nx.arange(batch_size, device=x.device), self.num_clusters)
        
        return x_pool, edge_index_pool, batch_pool
