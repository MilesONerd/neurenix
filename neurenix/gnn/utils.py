"""
Graph utility functions for Neurenix.

This module provides utility functions for working with graph-structured data.
"""

from typing import Optional, Tuple, List

import neurenix as nx


def to_edge_index(adj_matrix):
    """
    Convert an adjacency matrix to edge indices.
    
    Args:
        adj_matrix: Adjacency matrix
        
    Returns:
        Edge indices in COO format
    """
    if isinstance(adj_matrix, list):
        adj_matrix = nx.tensor(adj_matrix)
    
    if adj_matrix.dim() != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square")
    
    row, col = adj_matrix.nonzero()
    
    edge_index = nx.stack([row, col], dim=0)
    
    return edge_index


def to_adjacency_matrix(edge_index, num_nodes=None, edge_weight=None):
    """
    Convert edge indices to an adjacency matrix.
    
    Args:
        edge_index: Edge indices in COO format
        num_nodes: Number of nodes
        edge_weight: Edge weights
        
    Returns:
        Adjacency matrix
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    
    row, col = edge_index
    
    adj_matrix = nx.zeros((num_nodes, num_nodes), device=edge_index.device)
    
    if edge_weight is None:
        for i in range(len(row)):
            adj_matrix[row[i], col[i]] = 1
    else:
        for i in range(len(row)):
            adj_matrix[row[i], col[i]] = edge_weight[i]
    
    return adj_matrix


def add_self_loops(edge_index, edge_weight=None, num_nodes=None):
    """
    Add self-loops to edge indices.
    
    Args:
        edge_index: Edge indices in COO format
        edge_weight: Edge weights
        num_nodes: Number of nodes
        
    Returns:
        Edge indices with self-loops and corresponding edge weights
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    
    row, col = edge_index
    
    mask = row != col
    
    loop_index = nx.arange(0, num_nodes, dtype=edge_index.dtype, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    
    edge_index = nx.cat([edge_index[:, mask], loop_index], dim=1)
    
    if edge_weight is not None:
        loop_weight = nx.ones(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
        edge_weight = nx.cat([edge_weight[mask], loop_weight], dim=0)
    
    return edge_index, edge_weight


def remove_self_loops(edge_index, edge_weight=None):
    """
    Remove self-loops from edge indices.
    
    Args:
        edge_index: Edge indices in COO format
        edge_weight: Edge weights
        
    Returns:
        Edge indices without self-loops and corresponding edge weights
    """
    row, col = edge_index
    
    mask = row != col
    
    edge_index = edge_index[:, mask]
    
    if edge_weight is not None:
        edge_weight = edge_weight[mask]
    
    return edge_index, edge_weight


def normalize_adjacency(edge_index, edge_weight=None, num_nodes=None, add_self_loops=True):
    """
    Normalize adjacency matrix (symmetric normalization).
    
    Args:
        edge_index: Edge indices in COO format
        edge_weight: Edge weights
        num_nodes: Number of nodes
        add_self_loops: Whether to add self-loops
        
    Returns:
        Normalized edge indices and corresponding edge weights
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    
    if add_self_loops:
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes)
    
    row, col = edge_index
    
    deg = nx.zeros(num_nodes, device=edge_index.device)
    
    for i in range(len(row)):
        deg[row[i]] += 1
    
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt = nx.where(deg_inv_sqrt == float('inf'), 0, deg_inv_sqrt)
    
    if edge_weight is None:
        edge_weight = nx.ones(edge_index.shape[1], device=edge_index.device)
    
    for i in range(len(row)):
        edge_weight[i] = edge_weight[i] * deg_inv_sqrt[row[i]] * deg_inv_sqrt[col[i]]
    
    return edge_index, edge_weight


def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False, num_nodes=None):
    """
    Compute the k-hop subgraph of a node.
    
    Args:
        node_idx: Index of the source node
        num_hops: Number of hops
        edge_index: Edge indices in COO format
        relabel_nodes: Whether to relabel nodes
        num_nodes: Number of nodes
        
    Returns:
        Subgraph edge indices, subset of nodes, and mapping
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    
    row, col = edge_index
    
    node_mask = nx.zeros(num_nodes, dtype=nx.bool, device=edge_index.device)
    edge_mask = nx.zeros(row.shape[0], dtype=nx.bool, device=edge_index.device)
    
    if isinstance(node_idx, int):
        node_idx = [node_idx]
    
    node_idx = nx.tensor(node_idx, dtype=nx.int64, device=edge_index.device)
    
    subsets = [node_idx]
    
    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        
        edge_mask = node_mask[row]
        subsets.append(col[edge_mask])
    
    subset = nx.cat(subsets).unique()
    
    node_mask.fill_(False)
    node_mask[subset] = True
    
    edge_mask = node_mask[row] & node_mask[col]
    
    edge_index = edge_index[:, edge_mask]
    
    if relabel_nodes:
        node_idx = nx.zeros(num_nodes, dtype=nx.int64, device=edge_index.device)
        node_idx[subset] = nx.arange(subset.shape[0], device=edge_index.device)
        edge_index = node_idx[edge_index]
    
    return edge_index, subset, edge_mask


def subgraph(subset, edge_index, edge_attr=None, relabel_nodes=False, num_nodes=None):
    """
    Extract a subgraph given a subset of nodes.
    
    Args:
        subset: Subset of nodes
        edge_index: Edge indices in COO format
        edge_attr: Edge attributes
        relabel_nodes: Whether to relabel nodes
        num_nodes: Number of nodes
        
    Returns:
        Subgraph edge indices, edge attributes, and mapping
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    
    row, col = edge_index
    
    node_mask = nx.zeros(num_nodes, dtype=nx.bool, device=edge_index.device)
    node_mask[subset] = True
    
    edge_mask = node_mask[row] & node_mask[col]
    
    edge_index = edge_index[:, edge_mask]
    
    if edge_attr is not None:
        edge_attr = edge_attr[edge_mask]
    
    if relabel_nodes:
        node_idx = nx.zeros(num_nodes, dtype=nx.int64, device=edge_index.device)
        node_idx[subset] = nx.arange(subset.shape[0], device=edge_index.device)
        edge_index = node_idx[edge_index]
    
    return edge_index, edge_attr, edge_mask


def to_undirected(edge_index, edge_attr=None, num_nodes=None):
    """
    Convert a directed graph to an undirected graph.
    
    Args:
        edge_index: Edge indices in COO format
        edge_attr: Edge attributes
        num_nodes: Number of nodes
        
    Returns:
        Undirected edge indices and edge attributes
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    
    row, col = edge_index
    
    edge_index_rev = nx.stack([col, row], dim=0)
    
    edge_index = nx.cat([edge_index, edge_index_rev], dim=1)
    
    if edge_attr is not None:
        edge_attr = nx.cat([edge_attr, edge_attr], dim=0)
    
    edge_index, edge_attr = coalesce(edge_index, edge_attr, num_nodes)
    
    return edge_index, edge_attr


def coalesce(edge_index, edge_attr=None, num_nodes=None):
    """
    Remove duplicate edges from a graph.
    
    Args:
        edge_index: Edge indices in COO format
        edge_attr: Edge attributes
        num_nodes: Number of nodes
        
    Returns:
        Coalesced edge indices and edge attributes
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    
    row, col = edge_index
    
    idx = row * num_nodes + col
    
    perm = nx.argsort(idx)
    
    edge_index = edge_index[:, perm]
    
    if edge_attr is not None:
        edge_attr = edge_attr[perm]
    
    mask = nx.cat([nx.tensor([True], device=edge_index.device), idx[perm][1:] != idx[perm][:-1]])
    
    edge_index = edge_index[:, mask]
    
    if edge_attr is not None:
        edge_attr = edge_attr[mask]
    
    return edge_index, edge_attr


def is_undirected(edge_index, edge_attr=None, num_nodes=None):
    """
    Check if a graph is undirected.
    
    Args:
        edge_index: Edge indices in COO format
        edge_attr: Edge attributes
        num_nodes: Number of nodes
        
    Returns:
        Whether the graph is undirected
    """
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    
    edge_index_undirected, edge_attr_undirected = to_undirected(edge_index, edge_attr, num_nodes)
    
    return edge_index.shape[1] == edge_index_undirected.shape[1]
