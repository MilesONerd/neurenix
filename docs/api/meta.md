# Meta-Learning API Documentation

## Overview

The Meta-Learning API provides functionality for few-shot learning and rapid adaptation to new tasks. This module implements various meta-learning algorithms including Model-Agnostic Meta-Learning (MAML), Reptile, and Prototypical Networks.

Meta-learning, or "learning to learn," enables models to quickly adapt to new tasks with minimal data, making it particularly useful for applications where labeled data is scarce or expensive to obtain.

## Key Concepts

### Few-Shot Learning

Few-shot learning refers to the ability of a model to learn new concepts from only a few examples. The meta-learning module supports N-way K-shot tasks, where N is the number of classes and K is the number of examples per class.

### Meta-Learning Algorithms

Neurenix implements several meta-learning algorithms:

- **MAML (Model-Agnostic Meta-Learning)**: Optimizes model parameters to be easily fine-tuned for new tasks
- **Reptile**: A simplified version of MAML that uses standard stochastic gradient descent without second-order derivatives
- **Prototypical Networks**: Learns a metric space where classification is performed by computing distances to prototype representations of each class

### Task Generation

The module provides utilities for generating synthetic tasks for meta-learning, including regression and classification tasks.

## API Reference

### MAML (Model-Agnostic Meta-Learning)

```python
neurenix.meta.MAML(
    model: neurenix.nn.Module,
    inner_lr: float = 0.01,
    meta_lr: float = 0.001,
    first_order: bool = False,
    inner_steps: int = 5
)
```

Creates a MAML meta-learner.

**Parameters:**
- `model`: The model architecture to meta-train
- `inner_lr`: Learning rate for the inner adaptation loop
- `meta_lr`: Learning rate for the meta-update
- `first_order`: If True, use first-order approximation (ignore second-order derivatives)
- `inner_steps`: Number of gradient updates for the inner adaptation loop

**Methods:**
- `meta_learn(tasks, meta_optimizer, epochs, tasks_per_batch)`: Performs meta-training on the provided tasks
- `adapt_to_task(support_x, support_y, steps=None)`: Adapts the model to a new task using the support set
- `parameters()`: Returns the model parameters for optimization

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import Adam
from neurenix.meta import MAML

# Create a model for meta-learning
model = Sequential([
    Linear(1, 40),
    ReLU(),
    Linear(40, 40),
    ReLU(),
    Linear(40, 1),
])

# Create MAML meta-learner
meta_model = MAML(
    model=model,
    inner_lr=0.01,
    meta_lr=0.001,
    first_order=False,
    inner_steps=5,
)

# Create optimizer
meta_optimizer = Adam(meta_model.parameters(), lr=meta_model.meta_lr)

# Meta-train the model
history = meta_model.meta_learn(
    tasks=train_tasks,
    meta_optimizer=meta_optimizer,
    epochs=10,
    tasks_per_batch=4,
)

# Adapt to a new task
adapted_model = meta_model.adapt_to_task(support_x, support_y)
predictions = adapted_model(query_x)
```

### Reptile

```python
neurenix.meta.Reptile(
    model: neurenix.nn.Module,
    inner_lr: float = 0.02,
    meta_lr: float = 0.001,
    inner_steps: int = 8
)
```

Creates a Reptile meta-learner.

**Parameters:**
- `model`: The model architecture to meta-train
- `inner_lr`: Learning rate for the inner adaptation loop
- `meta_lr`: Learning rate for the meta-update
- `inner_steps`: Number of gradient updates for the inner adaptation loop

**Methods:**
- `meta_learn(tasks, meta_optimizer, epochs, tasks_per_batch)`: Performs meta-training on the provided tasks
- `adapt_to_task(support_x, support_y, steps=None)`: Adapts the model to a new task using the support set
- `parameters()`: Returns the model parameters for optimization

### Prototypical Networks

```python
neurenix.meta.PrototypicalNetworks(
    embedding_model: neurenix.nn.Module,
    distance_metric: str = "euclidean",
    meta_lr: float = 0.001
)
```

Creates a Prototypical Networks meta-learner.

**Parameters:**
- `embedding_model`: The model that maps inputs to embedding space
- `distance_metric`: Distance metric to use ("euclidean" or "cosine")
- `meta_lr`: Learning rate for the meta-update

**Methods:**
- `meta_learn(tasks, meta_optimizer, epochs, tasks_per_batch)`: Performs meta-training on the provided tasks
- `adapt_to_task(support_x, support_y)`: Computes prototypes from the support set
- `parameters()`: Returns the embedding model parameters for optimization

### Task Generation

```python
neurenix.meta.generate_regression_tasks(
    num_tasks: int,
    num_samples_per_task: int,
    input_dim: int = 1
) -> List[Tuple[Tensor, Tensor, Tensor, Tensor]]
```

Generates synthetic regression tasks for meta-learning.

**Parameters:**
- `num_tasks`: Number of tasks to generate
- `num_samples_per_task`: Number of samples per task
- `input_dim`: Dimensionality of the input space

**Returns:**
- List of (support_x, support_y, query_x, query_y) tuples

```python
neurenix.meta.generate_classification_tasks(
    num_tasks: int,
    num_classes: int = 5,
    num_samples_per_class: int = 10,
    input_dim: int = 20
) -> List[Tuple[Tensor, Tensor, Tensor, Tensor]]
```

Generates synthetic classification tasks for meta-learning.

**Parameters:**
- `num_tasks`: Number of tasks to generate
- `num_classes`: Number of classes per task
- `num_samples_per_class`: Number of samples per class
- `input_dim`: Dimensionality of the input space

**Returns:**
- List of (support_x, support_y, query_x, query_y) tuples

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Meta-Learning Support** | Native implementation of MAML, Reptile, Prototypical Networks | Limited native support, requires custom implementation |
| **API Simplicity** | Unified API for different meta-learning algorithms | No unified API, requires custom implementation |
| **Task Generation** | Built-in utilities for task generation | Manual implementation required |
| **Hardware Acceleration** | Automatic device selection and optimization | Manual device placement |

Neurenix provides a more comprehensive meta-learning framework compared to TensorFlow, with built-in implementations of popular algorithms and utilities for task generation. TensorFlow requires custom implementations of meta-learning algorithms, which can be complex and error-prone.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Meta-Learning Support** | Native implementation of MAML, Reptile, Prototypical Networks | Third-party libraries like learn2learn, higher |
| **API Simplicity** | Unified API for different meta-learning algorithms | Varies by library |
| **Task Generation** | Built-in utilities for task generation | Manual implementation usually required |
| **Hardware Acceleration** | Support for CPU, CUDA, ROCm, WebGPU | Support for CPU, CUDA |

While PyTorch has strong support for meta-learning through third-party libraries, Neurenix's native implementation provides a more unified and consistent API. The integration with Neurenix's hardware abstraction layer also enables better performance across a wider range of devices.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Meta-Learning Support** | Comprehensive implementation | No support |
| **Deep Learning Integration** | Seamless | Limited |
| **Few-Shot Learning** | Native support | No support |
| **Hardware Acceleration** | Multiple backends | CPU only |

Scikit-Learn does not provide meta-learning capabilities, focusing instead on traditional machine learning algorithms. Neurenix fills this gap with its comprehensive meta-learning module, which is fully integrated with its deep learning framework.

## Best Practices

### Task Design

For effective meta-learning, design tasks that share underlying structure but differ in specific parameters:

```python
# Generate related regression tasks with different amplitude and phase
tasks = neurenix.meta.generate_regression_tasks(
    num_tasks=100,
    num_samples_per_task=20
)

# Split into train and test tasks
train_tasks = tasks[:80]
test_tasks = tasks[80:]
```

### Meta-Batch Size

Choose an appropriate batch size for meta-training:

```python
# For complex tasks, use smaller batch sizes
history = meta_model.meta_learn(
    tasks=train_tasks,
    meta_optimizer=meta_optimizer,
    epochs=10,
    tasks_per_batch=4  # Start with 4-8 tasks per batch
)
```

### Model Architecture

Select a model architecture appropriate for the meta-learning algorithm:

```python
# For MAML and Reptile, use standard architectures
model = Sequential([
    Linear(input_dim, 64),
    ReLU(),
    Linear(64, 64),
    ReLU(),
    Linear(64, output_dim),
])

# For Prototypical Networks, focus on good embeddings
embedding_model = Sequential([
    Linear(input_dim, 64),
    ReLU(),
    Linear(64, 64),
    ReLU(),
    Linear(64, embedding_dim),
])
```

## Tutorials

### Few-Shot Classification with Prototypical Networks

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import Adam
from neurenix.meta import PrototypicalNetworks, generate_classification_tasks

# Initialize Neurenix
nx.init({"device": "cuda:0" if nx.get_device_count(nx.DeviceType.CUDA) > 0 else "cpu"})

# Create embedding model
embedding_model = Sequential([
    Linear(20, 64),
    ReLU(),
    Linear(64, 64),
    ReLU(),
    Linear(64, 32),  # Embedding dimension
])

# Create Prototypical Networks meta-learner
meta_model = PrototypicalNetworks(
    embedding_model=embedding_model,
    distance_metric="euclidean",
    meta_lr=0.001,
)

# Generate tasks (5-way, 10-shot)
tasks = generate_classification_tasks(
    num_tasks=100,
    num_classes=5,
    num_samples_per_class=10,
    input_dim=20,
)

# Split into train and test
train_tasks = tasks[:80]
test_tasks = tasks[80:]

# Create optimizer
meta_optimizer = Adam(meta_model.parameters(), lr=meta_model.meta_lr)

# Meta-train the model
history = meta_model.meta_learn(
    tasks=train_tasks,
    meta_optimizer=meta_optimizer,
    epochs=10,
    tasks_per_batch=4,
)

# Evaluate on test tasks
test_accuracies = []

for support_x, support_y, query_x, query_y in test_tasks:
    # Adapt to the task
    adapted_model = meta_model.adapt_to_task(support_x, support_y)
    
    # Evaluate on query set
    predictions = adapted_model(query_x)
    pred_classes = predictions.argmax(dim=1)
    true_classes = query_y.argmax(dim=1)
    accuracy = (pred_classes == true_classes).float().mean().item()
    test_accuracies.append(accuracy)

# Print results
avg_accuracy = sum(test_accuracies) / len(test_accuracies)
print(f"Average test accuracy: {avg_accuracy:.4f}")
```

### Regression with MAML

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import Adam
from neurenix.meta import MAML, generate_regression_tasks

# Initialize Neurenix
nx.init({"device": "cuda:0" if nx.get_device_count(nx.DeviceType.CUDA) > 0 else "cpu"})

# Create model
model = Sequential([
    Linear(1, 40),
    ReLU(),
    Linear(40, 40),
    ReLU(),
    Linear(40, 1),
])

# Create MAML meta-learner
meta_model = MAML(
    model=model,
    inner_lr=0.01,
    meta_lr=0.001,
    first_order=False,
    inner_steps=5,
)

# Generate sine wave tasks
tasks = generate_regression_tasks(
    num_tasks=100,
    num_samples_per_task=20,
)

# Split into train and test
train_tasks = tasks[:80]
test_tasks = tasks[80:]

# Create optimizer
meta_optimizer = Adam(meta_model.parameters(), lr=meta_model.meta_lr)

# Meta-train the model
history = meta_model.meta_learn(
    tasks=train_tasks,
    meta_optimizer=meta_optimizer,
    epochs=10,
    tasks_per_batch=4,
)

# Evaluate on test tasks
test_losses = []

for support_x, support_y, query_x, query_y in test_tasks:
    # Adapt to the task
    adapted_model = meta_model.adapt_to_task(support_x, support_y)
    
    # Evaluate on query set
    predictions = adapted_model(query_x)
    mse = ((predictions - query_y) ** 2).mean().item()
    test_losses.append(mse)

# Print results
avg_mse = sum(test_losses) / len(test_losses)
print(f"Average test MSE: {avg_mse:.6f}")
```
