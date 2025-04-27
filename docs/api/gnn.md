# Graph Neural Networks API Documentation

## Overview

The Graph Neural Networks (GNNs) module provides implementations of various graph neural network architectures and operations for processing graph-structured data. Graph Neural Networks extend traditional neural networks to operate on graph-structured data, enabling the modeling of complex relationships and interactions between entities.

Unlike standard neural networks that process data in Euclidean space (e.g., grids for images or sequences for text), GNNs can handle irregular data structures represented as graphs with nodes and edges. This makes them particularly valuable for applications involving relational data, such as social networks, molecular structures, knowledge graphs, recommendation systems, and physical simulations.

## Key Concepts

### Graph Representation

A graph consists of nodes (vertices) and edges (connections between nodes). In Neurenix, graphs can be represented in various formats:

- **Edge Index**: A pair of node index arrays representing the source and target nodes of each edge
- **Adjacency Matrix**: A matrix where each element (i,j) indicates whether there is an edge from node i to node j
- **Edge List**: A list of pairs of nodes that are connected by edges

Graphs can also have additional features:
- **Node Features**: Attributes or features associated with each node
- **Edge Features**: Attributes or features associated with each edge
- **Graph Features**: Global attributes of the entire graph

### Message Passing

The core concept in GNNs is message passing, where nodes exchange information with their neighbors through edges. This process typically involves:

1. **Message Computation**: Computing messages to send along edges
2. **Aggregation**: Combining messages from neighbors
3. **Update**: Updating node representations based on aggregated messages

### Graph Convolution

Graph convolution generalizes the convolution operation from regular grids (as in CNNs) to irregular graph structures. Common graph convolution methods include:

- **Spectral Methods**: Based on graph signal processing theory
- **Spatial Methods**: Based on aggregating information from spatial neighbors

### Graph Pooling

Graph pooling reduces the size of graphs by combining or selecting nodes, similar to pooling in CNNs. Pooling methods include:

- **Global Pooling**: Aggregating all node features to obtain a graph-level representation
- **Hierarchical Pooling**: Progressively coarsening the graph structure

## API Reference

### Graph Layers

```python
neurenix.gnn.GraphConv(in_channels, out_channels, bias=True)
```

Implements the graph convolutional layer as described in the GCN paper.

**Parameters:**
- `in_channels`: Size of each input sample
- `out_channels`: Size of each output sample
- `bias`: If set to False, the layer will not learn an additive bias

```python
neurenix.gnn.GraphAttention(in_channels, out_channels, heads=1, concat=True, negative_slope=0.2, dropout=0.0, bias=True)
```

Implements the graph attention layer as described in the GAT paper.

**Parameters:**
- `in_channels`: Size of each input sample
- `out_channels`: Size of each output sample
- `heads`: Number of attention heads
- `concat`: Whether to concatenate or average multi-head attention outputs
- `negative_slope`: LeakyReLU angle of negative slope
- `dropout`: Dropout probability

```python
neurenix.gnn.GraphSage(in_channels, out_channels, normalize=True, bias=True, aggr='mean')
```

Implements the GraphSAGE layer as described in the GraphSAGE paper.

**Parameters:**
- `in_channels`: Size of each input sample
- `out_channels`: Size of each output sample
- `normalize`: If set to True, output features will be L2-normalized
- `aggr`: Aggregation method ('mean', 'sum', 'max')

```python
neurenix.gnn.EdgeConv(in_channels, out_channels, aggr='max', bias=True)
```

Implements the edge convolutional layer as described in the DGCNN paper.

```python
neurenix.gnn.GINConv(nn, eps=0, train_eps=False)
```

Implements the graph isomorphism network layer as described in the GIN paper.

```python
neurenix.gnn.GatedGraphConv(out_channels, num_layers, aggr='add', bias=True)
```

Implements the gated graph convolutional layer as described in the GGNN paper.

```python
neurenix.gnn.RelationalGraphConv(in_channels, out_channels, num_relations, num_bases=None, bias=True)
```

Implements the relational graph convolutional layer as described in the R-GCN paper.

### Graph Models

```python
neurenix.gnn.GCN(in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.0, activation='relu')
```

Implements the Graph Convolutional Network model.

```python
neurenix.gnn.GAT(in_channels, hidden_channels, out_channels, heads=1, num_layers=2, dropout=0.0, activation='elu')
```

Implements the Graph Attention Network model.

```python
neurenix.gnn.GraphSAGE(in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.0, activation='relu', aggr='mean')
```

Implements the GraphSAGE model.

```python
neurenix.gnn.GIN(in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.0, activation='relu', eps=0, train_eps=False)
```

Implements the Graph Isomorphism Network model.

```python
neurenix.gnn.RGCN(in_channels, hidden_channels, out_channels, num_relations, num_bases=None, num_layers=2, dropout=0.0, activation='relu')
```

Implements the Relational Graph Convolutional Network model.

```python
neurenix.gnn.GGNN(in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.0, activation='relu')
```

Implements the Gated Graph Neural Network model.

### Graph Data Structures

```python
neurenix.gnn.Graph(x=None, edge_index=None, edge_attr=None, y=None, pos=None)
```

Represents a single graph.

**Parameters:**
- `x`: Node feature matrix with shape [num_nodes, num_node_features]
- `edge_index`: Graph connectivity in COO format with shape [2, num_edges]
- `edge_attr`: Edge feature matrix with shape [num_edges, num_edge_features]
- `y`: Graph-level or node-level targets with arbitrary shape
- `pos`: Node position matrix with shape [num_nodes, num_dimensions]

```python
neurenix.gnn.BatchedGraph(x=None, edge_index=None, edge_attr=None, y=None, pos=None, batch=None)
```

Represents a batch of graphs.

```python
neurenix.gnn.GraphDataset()
```

Base class for graph datasets.

```python
neurenix.gnn.GraphDataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
```

Data loader for graph datasets.

### Graph Utilities

```python
neurenix.gnn.to_edge_index(adj_matrix)
```

Converts an adjacency matrix to edge index format.

```python
neurenix.gnn.to_adjacency_matrix(edge_index, num_nodes=None)
```

Converts an edge index to adjacency matrix format.

```python
neurenix.gnn.add_self_loops(edge_index, num_nodes=None)
```

Adds self-loops to the graph.

```python
neurenix.gnn.remove_self_loops(edge_index)
```

Removes self-loops from the graph.

```python
neurenix.gnn.normalize_adjacency(edge_index, num_nodes=None)
```

Normalizes the adjacency matrix as described in the GCN paper.

### Graph Pooling

```python
neurenix.gnn.GlobalPooling(aggr='add')
```

Base class for global pooling layers.

```python
neurenix.gnn.GlobalAddPooling()
```

Global add pooling layer.

```python
neurenix.gnn.GlobalMeanPooling()
```

Global mean pooling layer.

```python
neurenix.gnn.GlobalMaxPooling()
```

Global max pooling layer.

```python
neurenix.gnn.TopKPooling(in_channels, ratio=0.5)
```

Top-k pooling layer as described in the TopKPool paper.

```python
neurenix.gnn.SAGPooling(in_channels, ratio=0.5, GNN=None)
```

Self-attention graph pooling layer as described in the SAGPool paper.

```python
neurenix.gnn.DiffPooling(in_channels, out_channels, num_nodes)
```

Differentiable pooling layer as described in the DiffPool paper.

## Framework Comparison

### Neurenix vs. PyTorch Geometric (PyG)

| Feature | Neurenix | PyTorch Geometric |
|---------|----------|-------------------|
| **API Design** | Unified, consistent API | Comprehensive but sometimes complex API |
| **GNN Architectures** | Comprehensive (GCN, GAT, GraphSAGE, GIN, RGCN, GGNN) | Extensive library of GNN architectures |
| **Graph Pooling** | Multiple methods (Global, TopK, SAG, DiffPool) | Comprehensive pooling methods |
| **Scalability** | Optimized for large graphs | Good scalability with some limitations |
| **Integration with Core Framework** | Seamless integration with Neurenix | Requires PyTorch |
| **Ease of Use** | Simplified API with consistent patterns | Steeper learning curve |

### Neurenix vs. Deep Graph Library (DGL)

| Feature | Neurenix | DGL |
|---------|----------|-----|
| **API Design** | Unified, consistent API | Flexible but sometimes verbose API |
| **GNN Architectures** | Comprehensive (GCN, GAT, GraphSAGE, GIN, RGCN, GGNN) | Extensive library of GNN architectures |
| **Graph Pooling** | Multiple methods (Global, TopK, SAG, DiffPool) | Limited pooling methods |
| **Scalability** | Optimized for large graphs | Excellent scalability |
| **Integration with Core Framework** | Seamless integration with Neurenix | Supports multiple backends |
| **Ease of Use** | Simplified API with consistent patterns | More complex API |

### Neurenix vs. TensorFlow GNN

| Feature | Neurenix | TensorFlow GNN |
|---------|----------|----------------|
| **API Design** | Unified, consistent API | TensorFlow-style API with Keras integration |
| **GNN Architectures** | Comprehensive (GCN, GAT, GraphSAGE, GIN, RGCN, GGNN) | Limited selection of GNN architectures |
| **Graph Pooling** | Multiple methods (Global, TopK, SAG, DiffPool) | Basic pooling methods |
| **Scalability** | Optimized for large graphs | Good scalability with TensorFlow's distributed training |
| **Integration with Core Framework** | Seamless integration with Neurenix | Integrated with TensorFlow ecosystem |

## Best Practices

### Graph Construction

When constructing graphs, consider the following:

1. **Appropriate Representation**: Choose the right graph representation for your problem
2. **Feature Engineering**: Design meaningful node and edge features
3. **Normalization**: Normalize features to improve training stability
4. **Self-Loops**: Add self-loops when appropriate to preserve node information

### Model Architecture

When designing GNN architectures, consider the following:

1. **Layer Selection**: Choose appropriate GNN layers based on your problem
2. **Depth**: Be cautious with deep GNNs due to over-smoothing
3. **Skip Connections**: Use skip connections to mitigate over-smoothing
4. **Pooling**: Select appropriate pooling methods for graph-level tasks

### Training Strategies

When training GNNs, consider the following:

1. **Batch Size**: Use appropriate batch sizes for your graph sizes
2. **Learning Rate**: Start with a small learning rate and adjust as needed
3. **Regularization**: Apply dropout and weight decay to prevent overfitting
4. **Early Stopping**: Monitor validation performance and stop when it plateaus

## Tutorials

### Node Classification with GCN

```python
import neurenix as nx
import numpy as np

# Create a citation network dataset
class CitationDataset(nx.gnn.GraphDataset):
    def __init__(self, num_nodes=2708, num_features=1433, num_classes=7):
        super().__init__()
        
        # Create a random citation network
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_classes = num_classes
        
        # Random node features (paper content)
        self.x = np.random.randn(num_nodes, num_features)
        
        # Random edges (citations)
        num_edges = num_nodes * 5  # Average degree of 5
        self.edge_index = np.zeros((2, num_edges), dtype=np.int64)
        for i in range(num_edges):
            self.edge_index[0, i] = np.random.randint(0, num_nodes)
            self.edge_index[1, i] = np.random.randint(0, num_nodes)
        
        # Random node labels (paper categories)
        self.y = np.random.randint(0, num_classes, size=num_nodes)
        
        # Create train/val/test masks
        indices = np.random.permutation(num_nodes)
        train_size = int(0.6 * num_nodes)
        val_size = int(0.2 * num_nodes)
        
        self.train_mask = np.zeros(num_nodes, dtype=bool)
        self.val_mask = np.zeros(num_nodes, dtype=bool)
        self.test_mask = np.zeros(num_nodes, dtype=bool)
        
        self.train_mask[indices[:train_size]] = True
        self.val_mask[indices[train_size:train_size+val_size]] = True
        self.test_mask[indices[train_size+val_size:]] = True

# Create GCN model and train for node classification
model = nx.gnn.GCN(
    in_channels=1433,
    hidden_channels=64,
    out_channels=7,
    num_layers=2,
    dropout=0.5
)

# Training loop would follow with forward/backward passes and evaluation
```

### Graph Classification with GIN

```python
import neurenix as nx
import numpy as np

# Create a molecular graph dataset
class MoleculeDataset(nx.gnn.GraphDataset):
    def __init__(self, num_graphs=1000, min_nodes=10, max_nodes=20, num_features=10, num_classes=2):
        super().__init__()
        self.graphs = []
        
        for i in range(num_graphs):
            # Create a random molecular graph
            num_nodes = np.random.randint(min_nodes, max_nodes + 1)
            
            # Random node features (atom properties)
            x = np.random.randn(num_nodes, num_features)
            
            # Random edges (bonds)
            num_edges = num_nodes * 2  # Average degree of 2
            edge_index = np.zeros((2, num_edges), dtype=np.int64)
            for j in range(num_edges):
                edge_index[0, j] = np.random.randint(0, num_nodes)
                edge_index[1, j] = np.random.randint(0, num_nodes)
            
            # Random graph label (molecule property)
            y = np.random.randint(0, num_classes)
            
            # Create graph
            graph = nx.gnn.Graph(
                x=nx.Tensor(x, dtype=nx.float32),
                edge_index=nx.Tensor(edge_index, dtype=nx.int64),
                y=nx.Tensor([y], dtype=nx.int64)
            )
            self.graphs.append(graph)

# Create GIN model for graph classification
model = nx.gnn.GIN(
    in_channels=10,
    hidden_channels=64,
    out_channels=2,
    num_layers=3
)

# Training loop would follow with batched graphs and global pooling
```

This documentation provides a comprehensive overview of the Graph Neural Networks module in Neurenix, including key concepts, API reference, framework comparisons, best practices, and tutorials for common GNN tasks.
