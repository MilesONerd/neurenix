"""
Graph data structures and utilities for Neurenix.

This module provides data structures and utilities for working with
graph-structured data in Neurenix.
"""

from typing import List, Dict, Optional, Union, Tuple, Any

import neurenix as nx


class Graph:
    """A graph data structure for storing node and edge features."""
    
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None, **kwargs):
        """
        Initialize a graph.
        
        Args:
            x: Node feature matrix
            edge_index: Graph connectivity in COO format
            edge_attr: Edge feature matrix
            y: Graph or node targets
            pos: Node position matrix
            **kwargs: Additional attributes
        """
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property
    def num_nodes(self):
        """Return the number of nodes in the graph."""
        if self.x is not None:
            return self.x.shape[0]
        elif self.pos is not None:
            return self.pos.shape[0]
        elif self.edge_index is not None:
            return int(self.edge_index.max()) + 1
        else:
            return 0
    
    @property
    def num_edges(self):
        """Return the number of edges in the graph."""
        if self.edge_index is not None:
            return self.edge_index.shape[1]
        else:
            return 0
    
    @property
    def num_features(self):
        """Return the number of node features."""
        if self.x is not None:
            return self.x.shape[1]
        else:
            return 0
    
    @property
    def num_edge_features(self):
        """Return the number of edge features."""
        if self.edge_attr is not None:
            return self.edge_attr.shape[1]
        else:
            return 0
    
    def to(self, device):
        """
        Move the graph to the specified device.
        
        Args:
            device: Device to move the graph to
            
        Returns:
            Graph on the specified device
        """
        for key, value in self.__dict__.items():
            if hasattr(value, 'to'):
                setattr(self, key, value.to(device))
        
        return self
    
    def __repr__(self):
        return f"{self.__class__.__name__}(num_nodes={self.num_nodes}, num_edges={self.num_edges})"


class BatchedGraph(Graph):
    """A batched graph data structure for storing multiple graphs."""
    
    def __init__(self, x=None, edge_index=None, edge_attr=None, y=None, pos=None,
                 batch=None, num_graphs=None, **kwargs):
        """
        Initialize a batched graph.
        
        Args:
            x: Node feature matrix
            edge_index: Graph connectivity in COO format
            edge_attr: Edge feature matrix
            y: Graph or node targets
            pos: Node position matrix
            batch: Batch vector, which assigns each node to a specific example
            num_graphs: Number of graphs in the batch
            **kwargs: Additional attributes
        """
        super().__init__(x, edge_index, edge_attr, y, pos, **kwargs)
        
        self.batch = batch
        self.num_graphs = num_graphs
    
    @classmethod
    def from_graph_list(cls, graphs: List[Graph]):
        """
        Create a batched graph from a list of graphs.
        
        Args:
            graphs: List of graphs
            
        Returns:
            Batched graph
        """
        if len(graphs) == 0:
            return cls()
        
        keys = set()
        for graph in graphs:
            keys.update(graph.__dict__.keys())
        
        for key in list(keys):
            if all(getattr(graph, key, None) is None for graph in graphs):
                keys.remove(key)
        
        batch_dict = {}
        
        for key in keys:
            values = [getattr(graph, key, None) for graph in graphs]
            
            if key == 'edge_index':
                node_offset = 0
                edge_indices = []
                
                for i, edge_index in enumerate(values):
                    if edge_index is None:
                        continue
                    
                    edge_indices.append(edge_index + node_offset)
                    
                    if 'x' in keys:
                        if graphs[i].x is not None:
                            node_offset += graphs[i].x.shape[0]
                    elif 'pos' in keys:
                        if graphs[i].pos is not None:
                            node_offset += graphs[i].pos.shape[0]
                    else:
                        node_offset += edge_index.max() + 1
                
                if edge_indices:
                    batch_dict[key] = nx.cat(edge_indices, dim=1)
            elif key == 'batch':
                continue  # We'll create this later
            else:
                valid_values = [v for v in values if v is not None]
                
                if not valid_values:
                    continue
                
                if isinstance(valid_values[0], nx.Tensor):
                    batch_dict[key] = nx.cat(valid_values, dim=0)
                else:
                    batch_dict[key] = valid_values
        
        if 'x' in batch_dict:
            num_nodes_per_graph = [graph.x.shape[0] if graph.x is not None else 0 for graph in graphs]
        elif 'pos' in batch_dict:
            num_nodes_per_graph = [graph.pos.shape[0] if graph.pos is not None else 0 for graph in graphs]
        elif 'edge_index' in batch_dict:
            num_nodes_per_graph = []
            for graph in graphs:
                if graph.edge_index is not None:
                    num_nodes_per_graph.append(int(graph.edge_index.max()) + 1)
                else:
                    num_nodes_per_graph.append(0)
        else:
            num_nodes_per_graph = [0] * len(graphs)
        
        batch = []
        for i, num_nodes in enumerate(num_nodes_per_graph):
            batch.extend([i] * num_nodes)
        
        if batch:
            batch_dict['batch'] = nx.tensor(batch, dtype=nx.int64)
        
        batch_dict['num_graphs'] = len(graphs)
        
        return cls(**batch_dict)
    
    def to_graph_list(self) -> List[Graph]:
        """
        Convert a batched graph to a list of graphs.
        
        Returns:
            List of graphs
        """
        if self.batch is None:
            return [Graph(self.x, self.edge_index, self.edge_attr, self.y, self.pos)]
        
        num_graphs = self.num_graphs or int(self.batch.max()) + 1
        
        graphs = []
        for i in range(num_graphs):
            node_mask = self.batch == i
            
            x = self.x[node_mask] if self.x is not None else None
            pos = self.pos[node_mask] if self.pos is not None else None
            
            if self.y is not None and len(self.y) == num_graphs:
                y = self.y[i]
            elif self.y is not None:
                y = self.y[node_mask]
            else:
                y = None
            
            if self.edge_index is not None:
                edge_mask = node_mask[self.edge_index[0]]
                edge_index = self.edge_index[:, edge_mask]
                
                node_indices = nx.zeros(node_mask.shape[0], dtype=nx.int64, device=node_mask.device)
                node_indices[node_mask] = nx.arange(node_mask.sum(), device=node_mask.device)
                edge_index = node_indices[edge_index]
                
                edge_attr = self.edge_attr[edge_mask] if self.edge_attr is not None else None
            else:
                edge_index = None
                edge_attr = None
            
            graph = Graph(x, edge_index, edge_attr, y, pos)
            
            for key, value in self.__dict__.items():
                if key not in ['x', 'edge_index', 'edge_attr', 'y', 'pos', 'batch', 'num_graphs']:
                    setattr(graph, key, value)
            
            graphs.append(graph)
        
        return graphs


class GraphDataset:
    """A dataset of graphs."""
    
    def __init__(self, graphs: List[Graph] = None, transform=None, pre_transform=None):
        """
        Initialize a graph dataset.
        
        Args:
            graphs: List of graphs
            transform: A function/transform that takes in a graph and returns a transformed version
            pre_transform: A function/transform that takes in a graph and returns a transformed version
        """
        self.graphs = graphs or []
        self.transform = transform
        self.pre_transform = pre_transform
        
        if self.pre_transform is not None:
            self.graphs = [self.pre_transform(graph) for graph in self.graphs]
    
    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)
    
    def __getitem__(self, idx):
        """
        Get a graph from the dataset.
        
        Args:
            idx: Index of the graph
            
        Returns:
            Graph at the specified index
        """
        if isinstance(idx, int):
            graph = self.graphs[idx]
            
            if self.transform is not None:
                graph = self.transform(graph)
            
            return graph
        else:
            return [self[i] for i in idx]
    
    def __repr__(self):
        return f"{self.__class__.__name__}(num_graphs={len(self)})"


class GraphDataLoader:
    """A data loader for loading graphs in batches."""
    
    def __init__(self, dataset: GraphDataset, batch_size: int = 1, shuffle: bool = False,
                 drop_last: bool = False):
        """
        Initialize a graph data loader.
        
        Args:
            dataset: Graph dataset
            batch_size: How many samples per batch to load
            shuffle: Whether to shuffle the data
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        self.indices = list(range(len(dataset)))
    
    def __iter__(self):
        """Return an iterator over the dataset."""
        if self.shuffle:
            import random
            random.shuffle(self.indices)
        
        batch_indices = []
        
        for idx in self.indices:
            batch_indices.append(idx)
            
            if len(batch_indices) == self.batch_size:
                yield self._collate_batch(batch_indices)
                batch_indices = []
        
        if batch_indices and not self.drop_last:
            yield self._collate_batch(batch_indices)
    
    def __len__(self):
        """Return the number of batches."""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def _collate_batch(self, indices):
        """
        Collate a batch of graphs.
        
        Args:
            indices: Indices of graphs to include in the batch
            
        Returns:
            Batched graph
        """
        graphs = [self.dataset[idx] for idx in indices]
        
        return BatchedGraph.from_graph_list(graphs)
