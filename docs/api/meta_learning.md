# Meta-Learning Documentation

## Overview

The Meta-Learning module in Neurenix provides tools and utilities for implementing meta-learning algorithms, which enable models to learn how to learn and adapt quickly to new tasks with minimal data. Meta-learning is particularly useful for few-shot learning scenarios, where models need to generalize from very few examples.

Neurenix's meta-learning capabilities are implemented using a multi-language architecture, where the high-performance operations are implemented in the Rust/C++ Phynexus engine, while the Python API provides a user-friendly interface. This architecture enables Neurenix to deliver optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

## Key Concepts

### Meta-Learning

Meta-learning, or "learning to learn," is a paradigm where a model is trained on a variety of learning tasks, such that it can solve new learning tasks using only a small number of training examples. The goal is to train models that can quickly adapt to new tasks, rather than having to train from scratch for each new task.

### Few-Shot Learning

Few-shot learning refers to the challenge of making predictions based on only a few examples. For instance, in a 5-way 1-shot classification task, the model must classify an image into one of five classes after seeing only one example of each class. Meta-learning provides a framework for addressing few-shot learning problems.

### Support and Query Sets

In meta-learning, tasks are typically divided into:
- **Support set**: The few examples provided for adaptation to a new task.
- **Query set**: The examples used to evaluate the model's performance after adaptation.

### Inner and Outer Loops

Many meta-learning algorithms use a nested optimization process:
- **Inner loop**: Adapts the model to a specific task using the support set.
- **Outer loop**: Updates the meta-parameters to improve the model's ability to adapt to new tasks.

## API Reference

### MetaLearningModel

```python
neurenix.meta.MetaLearningModel(model, inner_lr=0.01, meta_lr=0.001, first_order=False)
```

Base class for meta-learning models in the Neurenix framework.

**Parameters:**
- `model`: The base model to meta-train.
- `inner_lr`: Learning rate for the inner loop (task-specific adaptation).
- `meta_lr`: Learning rate for the outer loop (meta-update).
- `first_order`: Whether to use first-order approximation (ignore second derivatives).

**Methods:**
- `clone_model()`: Create a clone of the base model with the same architecture and parameters.
- `adapt_to_task(support_x, support_y, steps=5)`: Adapt the model to a new task using the support set.
- `forward(x)`: Forward pass through the meta-learning model.
- `meta_learn(tasks, epochs=10, tasks_per_batch=4)`: Perform meta-learning on a set of tasks.

### MAML (Model-Agnostic Meta-Learning)

```python
neurenix.meta.MAML(model, inner_lr=0.01, meta_lr=0.001, first_order=False, inner_steps=5)
```

Model-Agnostic Meta-Learning (MAML) implementation. MAML learns an initialization for the model parameters that can be quickly adapted to new tasks with just a few gradient steps.

**Parameters:**
- `model`: The base model to meta-train.
- `inner_lr`: Learning rate for the inner loop (task-specific adaptation).
- `meta_lr`: Learning rate for the outer loop (meta-update).
- `first_order`: Whether to use first-order approximation (ignore second derivatives).
- `inner_steps`: Number of gradient steps in the inner loop.

**Methods:**
- `adapt_to_task(support_x, support_y, steps=None)`: Adapt the model to a new task using the support set.
- `meta_learn(tasks, meta_optimizer, epochs=10, tasks_per_batch=4, loss_fn=None)`: Perform meta-learning using MAML.

**Example:**
```python
import neurenix
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.meta import MAML
from neurenix.optim import Adam

# Create a base model
base_model = Sequential(
    Linear(28*28, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 5)  # 5-way classification
)

# Create a MAML model
maml = MAML(
    model=base_model,
    inner_lr=0.01,
    meta_lr=0.001,
    first_order=True,
    inner_steps=5
)

# Create a meta-optimizer
meta_optimizer = Adam(maml.parameters(), lr=0.001)

# Meta-train the model
tasks = [...]  # List of (support_x, support_y, query_x, query_y) tuples
history = maml.meta_learn(
    tasks=tasks,
    meta_optimizer=meta_optimizer,
    epochs=50,
    tasks_per_batch=4
)

# Adapt to a new task
support_x = neurenix.Tensor(...)  # Support set inputs
support_y = neurenix.Tensor(...)  # Support set labels
adapted_model = maml.adapt_to_task(support_x, support_y)

# Make predictions
query_x = neurenix.Tensor(...)  # Query set inputs
predictions = adapted_model(query_x)
```

### Reptile

```python
neurenix.meta.Reptile(model, inner_lr=0.01, meta_lr=0.001, inner_steps=5)
```

Reptile meta-learning algorithm implementation. Reptile is a first-order meta-learning algorithm that is simpler than MAML but often achieves comparable performance.

**Parameters:**
- `model`: The base model to meta-train.
- `inner_lr`: Learning rate for the inner loop (task-specific adaptation).
- `meta_lr`: Learning rate for the outer loop (meta-update).
- `inner_steps`: Number of gradient steps in the inner loop.

**Methods:**
- `adapt_to_task(support_x, support_y, steps=None)`: Adapt the model to a new task using the support set.
- `meta_learn(tasks, meta_optimizer=None, epochs=10, tasks_per_batch=4, loss_fn=None)`: Perform meta-learning using Reptile.

**Example:**
```python
import neurenix
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.meta import Reptile
from neurenix.optim import Adam

# Create a base model
base_model = Sequential(
    Linear(28*28, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 5)  # 5-way classification
)

# Create a Reptile model
reptile = Reptile(
    model=base_model,
    inner_lr=0.05,
    meta_lr=0.001,
    inner_steps=10
)

# Meta-train the model
tasks = [...]  # List of (support_x, support_y, query_x, query_y) tuples
history = reptile.meta_learn(
    tasks=tasks,
    epochs=50,
    tasks_per_batch=4
)

# Adapt to a new task
support_x = neurenix.Tensor(...)  # Support set inputs
support_y = neurenix.Tensor(...)  # Support set labels
adapted_model = reptile.adapt_to_task(support_x, support_y)

# Make predictions
query_x = neurenix.Tensor(...)  # Query set inputs
predictions = adapted_model(query_x)
```

### PrototypicalNetworks

```python
neurenix.meta.PrototypicalNetworks(embedding_model, distance_metric='euclidean', meta_lr=0.001)
```

Prototypical Networks for few-shot classification. Prototypical Networks learn an embedding space where classes can be represented by a single prototype (the mean of embedded support examples). Classification is performed by finding the nearest prototype.

**Parameters:**
- `embedding_model`: Model that maps inputs to an embedding space.
- `distance_metric`: Distance metric to use ('euclidean' or 'cosine').
- `meta_lr`: Learning rate for the meta-update.

**Methods:**
- `compute_prototypes(support_x, support_y)`: Compute class prototypes from support examples.
- `compute_distances(query_embeddings, prototypes)`: Compute distances between query embeddings and class prototypes.
- `forward(x, support_x=None, support_y=None)`: Forward pass through the Prototypical Networks model.
- `meta_learn(tasks, meta_optimizer, epochs=10, tasks_per_batch=4)`: Perform meta-learning using Prototypical Networks.

**Example:**
```python
import neurenix
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.meta import PrototypicalNetworks
from neurenix.optim import Adam

# Create an embedding model
embedding_model = Sequential(
    Linear(28*28, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 64)  # Embedding dimension
)

# Create a Prototypical Networks model
proto_net = PrototypicalNetworks(
    embedding_model=embedding_model,
    distance_metric='euclidean',
    meta_lr=0.001
)

# Create a meta-optimizer
meta_optimizer = Adam(proto_net.parameters(), lr=0.001)

# Meta-train the model
tasks = [...]  # List of (support_x, support_y, query_x, query_y) tuples
history = proto_net.meta_learn(
    tasks=tasks,
    meta_optimizer=meta_optimizer,
    epochs=50,
    tasks_per_batch=4
)

# Make predictions for a new task
support_x = neurenix.Tensor(...)  # Support set inputs
support_y = neurenix.Tensor(...)  # Support set labels (one-hot)
query_x = neurenix.Tensor(...)  # Query set inputs
logits = proto_net(query_x, support_x, support_y)
predictions = logits.argmax(dim=1)
```

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Meta-Learning API** | Unified API with base class and specific algorithm implementations | No official meta-learning API, relies on community implementations |
| **MAML Implementation** | Built-in with first-order option | Available through third-party libraries (e.g., meta-learning-tensorflow) |
| **Prototypical Networks** | Native implementation with multiple distance metrics | Available through third-party libraries |
| **Edge Device Support** | Native optimization for edge devices | TensorFlow Lite for edge devices |
| **Hardware Acceleration** | Multi-device support (CPU, CUDA, ROCm, WebGPU) | Primarily optimized for CPU and CUDA |
| **Integration with Framework** | Seamless integration with other Neurenix components | Requires additional setup and dependencies |

Neurenix's meta-learning capabilities offer a more unified and integrated approach compared to TensorFlow, which relies on third-party libraries for meta-learning algorithms. The native implementation of multiple meta-learning algorithms in Neurenix provides a consistent API and seamless integration with other framework components. Additionally, Neurenix's multi-language architecture and edge device optimization make it particularly well-suited for deploying meta-learning models in resource-constrained environments.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Meta-Learning API** | Unified API with base class and specific algorithm implementations | No official meta-learning API, relies on community implementations |
| **MAML Implementation** | Built-in with configurable inner/outer loops | Available through third-party libraries (e.g., learn2learn, higher) |
| **Reptile Implementation** | Native implementation with simplified API | Available through third-party libraries |
| **Prototypical Networks** | Built-in with multiple distance metrics | Available through third-party libraries |
| **Hardware Support** | CPU, CUDA, ROCm, WebGPU | CPU, CUDA |
| **Edge Device Optimization** | Native optimization for edge devices | PyTorch Mobile for edge devices |

Neurenix provides a more comprehensive and integrated meta-learning solution compared to PyTorch, which relies on external libraries like learn2learn or higher for meta-learning algorithms. While PyTorch's dynamic computation graph makes it flexible for implementing custom meta-learning algorithms, Neurenix's built-in implementations offer a more streamlined experience with less boilerplate code. Neurenix also extends hardware support to include ROCm and WebGPU, making it more versatile across different hardware platforms.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Meta-Learning Support** | Comprehensive meta-learning algorithms | No meta-learning support |
| **Few-Shot Learning** | Native support for few-shot learning | Limited support through transfer learning |
| **Neural Network Support** | Full neural network architecture | Limited neural network support through MLPClassifier/MLPRegressor |
| **GPU Acceleration** | Native support for multiple GPU types | No native GPU support |
| **Edge Device Support** | Native optimization for edge devices | No specific edge device support |
| **Algorithm Diversity** | Multiple meta-learning algorithms (MAML, Reptile, Prototypical Networks) | No meta-learning algorithms |

Neurenix provides much more comprehensive meta-learning capabilities compared to Scikit-Learn, which does not have built-in support for meta-learning or few-shot learning. While Scikit-Learn excels at traditional machine learning algorithms, it lacks the neural network infrastructure and GPU acceleration necessary for modern meta-learning approaches. Neurenix's implementation of multiple meta-learning algorithms, combined with its hardware acceleration and edge device optimization, makes it a superior choice for few-shot learning tasks.

## Best Practices

### Choosing the Right Meta-Learning Algorithm

Different meta-learning algorithms have different strengths and weaknesses:

1. **MAML**: Best when task adaptation requires significant changes to the model. More computationally expensive due to second-order derivatives, but can be approximated with first-order version.
2. **Reptile**: Simpler and more computationally efficient than MAML. Works well when tasks are closely related.
3. **Prototypical Networks**: Best for few-shot classification tasks where class prototypes are meaningful. More efficient during inference than gradient-based methods.

```python
import neurenix
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.meta import MAML, Reptile, PrototypicalNetworks

# Create a base model
base_model = Sequential(
    Linear(28*28, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 5)  # 5-way classification
)

# For tasks requiring significant adaptation
maml = MAML(
    model=base_model,
    inner_lr=0.01,
    meta_lr=0.001,
    first_order=False  # Use second-order derivatives for better performance
)

# For computationally constrained environments
reptile = Reptile(
    model=base_model,
    inner_lr=0.05,
    meta_lr=0.001,
    inner_steps=10
)

# For few-shot classification tasks
embedding_model = Sequential(
    Linear(28*28, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 64)  # Embedding dimension
)
proto_net = PrototypicalNetworks(
    embedding_model=embedding_model,
    distance_metric='euclidean'
)
```

### Task Construction and Sampling

The way tasks are constructed and sampled can significantly impact meta-learning performance:

1. **Task Diversity**: Include a diverse set of tasks during meta-training to improve generalization.
2. **Task Similarity**: Ensure that meta-training tasks are similar to the target tasks you expect at meta-test time.
3. **Support/Query Split**: Typically, use 1-5 examples per class for the support set and the rest for the query set.
4. **Balanced Classes**: Ensure that classes are balanced in both support and query sets.

### Hyperparameter Tuning

Meta-learning algorithms have several important hyperparameters that need careful tuning:

1. **Inner Learning Rate**: Controls how quickly the model adapts to new tasks. Too high can cause overfitting to the support set, too low can result in insufficient adaptation.
2. **Meta Learning Rate**: Controls how quickly the meta-parameters are updated. Typically smaller than the inner learning rate.
3. **Inner Steps**: Number of gradient steps for task adaptation. More steps can lead to better adaptation but may cause overfitting.
4. **First-Order Approximation**: Using first-order approximation (ignoring second derivatives) can significantly reduce computational cost with minimal performance impact.

### Optimizing for Edge Devices

When deploying meta-learning models to edge devices, consider these optimizations:

1. **Model Size**: Use smaller base models to reduce memory footprint.
2. **First-Order Approximation**: Always use first-order approximation to reduce computational requirements.
3. **Fewer Inner Steps**: Reduce the number of inner steps during adaptation.
4. **Quantization**: Quantize model weights to reduce memory usage.
5. **Pre-computed Features**: For Prototypical Networks, pre-compute embeddings when possible.

## Tutorials

### Few-Shot Image Classification with MAML

```python
import neurenix
import numpy as np
from neurenix.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU
from neurenix.meta import MAML
from neurenix.optim import Adam

# Step 1: Create a convolutional neural network for image classification
def create_cnn():
    return Sequential(
        Conv2d(3, 32, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(2),
        Conv2d(32, 32, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(2),
        Conv2d(32, 32, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(32 * 4 * 4, 256),
        ReLU(),
        Linear(256, 5)  # 5-way classification
    )

# Step 2: Create a MAML model
base_model = create_cnn()
maml = MAML(
    model=base_model,
    inner_lr=0.01,
    meta_lr=0.001,
    first_order=True,
    inner_steps=5
)

# Step 3: Create a meta-optimizer
meta_optimizer = Adam(maml.parameters(), lr=0.001)

# Step 4: Create meta-learning tasks
# (Simplified for brevity)
train_tasks = [...]  # List of (support_x, support_y, query_x, query_y) tuples
test_tasks = [...]   # List of (support_x, support_y, query_x, query_y) tuples

# Step 5: Meta-train the model
history = maml.meta_learn(
    tasks=train_tasks,
    meta_optimizer=meta_optimizer,
    epochs=50,
    tasks_per_batch=4
)

# Step 6: Evaluate on test tasks
total_accuracy = 0.0
for support_x, support_y, query_x, query_y in test_tasks:
    # Adapt to the task
    adapted_model = maml.adapt_to_task(support_x, support_y)
    
    # Make predictions
    with neurenix.no_grad():
        logits = adapted_model(query_x)
        predictions = logits.argmax(dim=1)
        
        # Convert one-hot labels to indices
        query_y_indices = query_y.argmax(dim=1)
        
        # Compute accuracy
        accuracy = (predictions == query_y_indices).float().mean().item()
        total_accuracy += accuracy

# Compute average accuracy
average_accuracy = total_accuracy / len(test_tasks)
print(f"Average 5-way 1-shot accuracy: {average_accuracy:.4f}")
```

### Few-Shot Classification with Prototypical Networks

```python
import neurenix
import numpy as np
from neurenix.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU
from neurenix.meta import PrototypicalNetworks
from neurenix.optim import Adam

# Step 1: Create an embedding network for images
def create_embedding_network():
    return Sequential(
        Conv2d(1, 64, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(2),
        Conv2d(64, 64, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(2),
        Conv2d(64, 64, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(64 * 3 * 3, 256),
        ReLU(),
        Linear(256, 64)  # Embedding dimension
    )

# Step 2: Create a Prototypical Networks model
embedding_model = create_embedding_network()
proto_net = PrototypicalNetworks(
    embedding_model=embedding_model,
    distance_metric='euclidean',
    meta_lr=0.001
)

# Step 3: Create a meta-optimizer
meta_optimizer = Adam(proto_net.parameters(), lr=0.001)

# Step 4: Meta-train the model
# (Simplified for brevity)
train_tasks = [...]  # List of (support_x, support_y, query_x, query_y) tuples
test_tasks = [...]   # List of (support_x, support_y, query_x, query_y) tuples

history = proto_net.meta_learn(
    tasks=train_tasks,
    meta_optimizer=meta_optimizer,
    epochs=50,
    tasks_per_batch=4
)

# Step 5: Evaluate on test tasks
total_accuracy = 0.0
for support_x, support_y, query_x, query_y in test_tasks:
    # Make predictions
    with neurenix.no_grad():
        logits = proto_net(query_x, support_x, support_y)
        predictions = logits.argmax(dim=1)
        
        # Convert one-hot labels to indices
        query_y_indices = query_y.argmax(dim=1)
        
        # Compute accuracy
        accuracy = (predictions == query_y_indices).float().mean().item()
        total_accuracy += accuracy

# Compute average accuracy
average_accuracy = total_accuracy / len(test_tasks)
print(f"Average accuracy: {average_accuracy:.4f}")
```

## Conclusion

The Meta-Learning module of Neurenix provides a comprehensive set of tools for implementing meta-learning algorithms, enabling models to learn how to learn and adapt quickly to new tasks with minimal data. Its multi-language architecture with a high-performance Rust/C++ core enables optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

Compared to other frameworks like TensorFlow, PyTorch, and Scikit-Learn, Neurenix's Meta-Learning module offers advantages in terms of API design, hardware support, and edge device optimization. The unified MetaLearningModel base class and implementations of multiple meta-learning algorithms (MAML, Reptile, Prototypical Networks) provide a consistent and integrated experience, making Neurenix particularly well-suited for few-shot learning tasks and AI agent development.
