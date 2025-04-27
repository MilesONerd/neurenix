# Continual Learning API Documentation

## Overview

The Continual Learning module provides functionality for training neural networks on new data without forgetting previously learned knowledge. This is a critical capability for real-world AI systems that need to adapt to changing environments and new tasks over time, without requiring complete retraining on all data.

Neurenix's Continual Learning module implements several state-of-the-art techniques to mitigate catastrophic forgetting, the phenomenon where neural networks tend to forget previously learned information when trained on new data. These techniques include regularization-based methods, replay-based methods, and parameter isolation methods.

## Key Concepts

### Catastrophic Forgetting

Catastrophic forgetting occurs when a neural network, after being trained on a new task, loses its ability to perform well on previously learned tasks. This happens because the network updates its parameters to optimize performance on the new task, potentially overwriting important parameters for previous tasks.

### Regularization-Based Methods

Regularization-based methods for continual learning add constraints to the learning process to prevent drastic changes to parameters that are important for previously learned tasks. Examples include Elastic Weight Consolidation (EWC) and Synaptic Intelligence.

### Replay-Based Methods

Replay-based methods maintain a memory of previous tasks by storing a subset of data from those tasks or by generating synthetic examples. During training on new tasks, the model is also trained on this stored or generated data to maintain performance on previous tasks.

### Knowledge Distillation

Knowledge distillation in continual learning involves using the outputs of the model on previous tasks to guide the learning of new tasks. This helps preserve the knowledge acquired from previous tasks while learning new ones.

## API Reference

### Elastic Weight Consolidation (EWC)

```python
neurenix.continual.EWC(
    model: neurenix.nn.Module,
    importance: float = 1000.0,
    fisher_sample_size: int = 1000,
    device: Optional[neurenix.Device] = None
)
```

Implements the Elastic Weight Consolidation algorithm for continual learning.

**Parameters:**
- `model`: The neural network model
- `importance`: Importance factor for previous tasks (higher values give more importance to previous tasks)
- `fisher_sample_size`: Number of samples to use for computing the Fisher information matrix
- `device`: Device to use for computation

**Methods:**
- `register_task(task_id, dataloader)`: Register a new task and compute Fisher information matrix
- `update_fisher(task_id, dataloader)`: Update the Fisher information matrix for a task
- `get_ewc_loss()`: Get the EWC regularization loss
- `train_step(optimizer, loss_fn, inputs, targets, task_id)`: Perform a training step with EWC regularization

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import Adam
from neurenix.continual import EWC
from neurenix.data import DataLoader

# Create a model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# Create an EWC wrapper
ewc = EWC(model, importance=1000.0)

# Create an optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Create a loss function
loss_fn = nx.nn.CrossEntropyLoss()

# Train on task 1
task1_dataloader = DataLoader(task1_dataset, batch_size=32, shuffle=True)
ewc.register_task(task_id=1, dataloader=task1_dataloader)

for epoch in range(10):
    for inputs, targets in task1_dataloader:
        ewc.train_step(optimizer, loss_fn, inputs, targets, task_id=1)

# Train on task 2 with EWC regularization
task2_dataloader = DataLoader(task2_dataset, batch_size=32, shuffle=True)
ewc.register_task(task_id=2, dataloader=task2_dataloader)

for epoch in range(10):
    for inputs, targets in task2_dataloader:
        ewc.train_step(optimizer, loss_fn, inputs, targets, task_id=2)
```

### Experience Replay

```python
neurenix.continual.ExperienceReplay(
    model: neurenix.nn.Module,
    memory_size: int = 1000,
    replay_ratio: float = 0.5,
    device: Optional[neurenix.Device] = None
)
```

Implements the Experience Replay method for continual learning.

**Parameters:**
- `model`: The neural network model
- `memory_size`: Maximum number of samples to store in memory
- `replay_ratio`: Ratio of replay samples to new samples during training
- `device`: Device to use for computation

**Methods:**
- `add_to_memory(inputs, targets, task_id)`: Add samples to the replay memory
- `get_replay_batch(batch_size)`: Get a batch of samples from the replay memory
- `train_step(optimizer, loss_fn, inputs, targets, task_id)`: Perform a training step with experience replay

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import Adam
from neurenix.continual import ExperienceReplay
from neurenix.data import DataLoader

# Create a model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# Create an Experience Replay wrapper
replay = ExperienceReplay(model, memory_size=1000, replay_ratio=0.5)

# Create an optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Create a loss function
loss_fn = nx.nn.CrossEntropyLoss()

# Train on task 1
task1_dataloader = DataLoader(task1_dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for inputs, targets in task1_dataloader:
        replay.add_to_memory(inputs, targets, task_id=1)
        replay.train_step(optimizer, loss_fn, inputs, targets, task_id=1)

# Train on task 2 with Experience Replay
task2_dataloader = DataLoader(task2_dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for inputs, targets in task2_dataloader:
        replay.add_to_memory(inputs, targets, task_id=2)
        replay.train_step(optimizer, loss_fn, inputs, targets, task_id=2)
```

### L2 Regularization

```python
neurenix.continual.L2Regularization(
    model: neurenix.nn.Module,
    importance: float = 1.0,
    device: Optional[neurenix.Device] = None
)
```

Implements L2 regularization for continual learning, penalizing changes to parameters from their values after training on previous tasks.

**Parameters:**
- `model`: The neural network model
- `importance`: Importance factor for previous tasks
- `device`: Device to use for computation

**Methods:**
- `register_task(task_id)`: Register a new task and store current parameter values
- `get_regularization_loss()`: Get the L2 regularization loss
- `train_step(optimizer, loss_fn, inputs, targets, task_id)`: Perform a training step with L2 regularization

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import Adam
from neurenix.continual import L2Regularization
from neurenix.data import DataLoader

# Create a model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# Create an L2 Regularization wrapper
l2_reg = L2Regularization(model, importance=1.0)

# Create an optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Create a loss function
loss_fn = nx.nn.CrossEntropyLoss()

# Train on task 1
task1_dataloader = DataLoader(task1_dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for inputs, targets in task1_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

# Register task 1 parameters
l2_reg.register_task(task_id=1)

# Train on task 2 with L2 regularization
task2_dataloader = DataLoader(task2_dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for inputs, targets in task2_dataloader:
        l2_reg.train_step(optimizer, loss_fn, inputs, targets, task_id=2)
```

### Knowledge Distillation

```python
neurenix.continual.KnowledgeDistillation(
    model: neurenix.nn.Module,
    temperature: float = 2.0,
    alpha: float = 0.5,
    device: Optional[neurenix.Device] = None
)
```

Implements Knowledge Distillation for continual learning, using the model's previous outputs to guide learning on new tasks.

**Parameters:**
- `model`: The neural network model
- `temperature`: Temperature parameter for softening probability distributions
- `alpha`: Weight for the distillation loss (higher values give more importance to matching previous outputs)
- `device`: Device to use for computation

**Methods:**
- `register_task(task_id, dataloader)`: Register a new task and store model outputs
- `get_distillation_loss(inputs, outputs)`: Get the distillation loss
- `train_step(optimizer, loss_fn, inputs, targets, task_id)`: Perform a training step with knowledge distillation

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import Adam
from neurenix.continual import KnowledgeDistillation
from neurenix.data import DataLoader

# Create a model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# Create a Knowledge Distillation wrapper
distillation = KnowledgeDistillation(model, temperature=2.0, alpha=0.5)

# Create an optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Create a loss function
loss_fn = nx.nn.CrossEntropyLoss()

# Train on task 1
task1_dataloader = DataLoader(task1_dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for inputs, targets in task1_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

# Register task 1 outputs
distillation.register_task(task_id=1, dataloader=task1_dataloader)

# Train on task 2 with Knowledge Distillation
task2_dataloader = DataLoader(task2_dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for inputs, targets in task2_dataloader:
        distillation.train_step(optimizer, loss_fn, inputs, targets, task_id=2)
```

### Synaptic Intelligence

```python
neurenix.continual.SynapticIntelligence(
    model: neurenix.nn.Module,
    importance: float = 1.0,
    xi: float = 0.1,
    device: Optional[neurenix.Device] = None
)
```

Implements the Synaptic Intelligence algorithm for continual learning, which estimates parameter importance based on their contribution to reducing the loss.

**Parameters:**
- `model`: The neural network model
- `importance`: Importance factor for previous tasks
- `xi`: Damping parameter to avoid division by zero
- `device`: Device to use for computation

**Methods:**
- `register_task(task_id)`: Register a new task and initialize parameter importance
- `update_importance(loss)`: Update parameter importance based on loss reduction
- `get_regularization_loss()`: Get the regularization loss
- `train_step(optimizer, loss_fn, inputs, targets, task_id)`: Perform a training step with synaptic intelligence regularization

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import Adam
from neurenix.continual import SynapticIntelligence
from neurenix.data import DataLoader

# Create a model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 10)
)

# Create a Synaptic Intelligence wrapper
si = SynapticIntelligence(model, importance=1.0, xi=0.1)

# Create an optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Create a loss function
loss_fn = nx.nn.CrossEntropyLoss()

# Train on task 1
task1_dataloader = DataLoader(task1_dataset, batch_size=32, shuffle=True)
si.register_task(task_id=1)

for epoch in range(10):
    for inputs, targets in task1_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        si.update_importance(loss)
        optimizer.step()

# Train on task 2 with Synaptic Intelligence
task2_dataloader = DataLoader(task2_dataset, batch_size=32, shuffle=True)
si.register_task(task_id=2)

for epoch in range(10):
    for inputs, targets in task2_dataloader:
        si.train_step(optimizer, loss_fn, inputs, targets, task_id=2)
```

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Continual Learning Support** | Comprehensive set of algorithms (EWC, Experience Replay, L2, Knowledge Distillation, Synaptic Intelligence) | Limited native support, requires custom implementation |
| **API Consistency** | Unified API for different continual learning methods | No unified API for continual learning |
| **Integration with Core Framework** | Seamless integration | Requires additional libraries or custom code |
| **Edge Device Optimization** | Native optimization for edge devices | Limited support for continual learning on edge devices |
| **Implementation Complexity** | Simple, high-level API | Complex, requires deep understanding of TensorFlow internals |

Neurenix provides a more comprehensive and integrated continual learning solution compared to TensorFlow. While TensorFlow requires custom implementations or third-party libraries for most continual learning algorithms, Neurenix offers a unified API with native support for multiple state-of-the-art methods. Additionally, Neurenix's optimization for edge devices makes it more suitable for continual learning in resource-constrained environments.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Continual Learning Support** | Comprehensive set of algorithms with unified API | Limited native support, requires libraries like Avalanche or custom implementation |
| **API Consistency** | Unified API for different continual learning methods | Different APIs depending on the library used |
| **Integration with Core Framework** | Seamless integration | Requires additional libraries |
| **Edge Device Optimization** | Native optimization for edge devices | Limited support for continual learning on edge devices |
| **Implementation Complexity** | Simple, high-level API | Varies depending on the library used |

PyTorch has good support for continual learning through third-party libraries like Avalanche, but lacks native integration in the core framework. Neurenix's unified API and native optimization for edge devices provide advantages for deploying continual learning systems in production environments, especially on resource-constrained hardware.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Continual Learning Support** | Comprehensive set of algorithms | No native support for continual learning |
| **Neural Network Integration** | Seamless integration with neural networks | Limited neural network support |
| **Deep Learning Capabilities** | Full deep learning support | Limited to shallow models |
| **Edge Device Optimization** | Native optimization for edge devices | No specific edge device support |
| **Implementation Complexity** | Simple, high-level API | Not applicable (no continual learning support) |

Scikit-Learn does not provide native support for continual learning, focusing instead on batch learning for traditional machine learning algorithms. Neurenix fills this gap with its comprehensive continual learning module, which is fully integrated with its deep learning framework and optimized for edge devices.

## Best Practices

### Choosing the Right Method

Different continual learning methods are suitable for different scenarios:

1. **Regularization-Based Methods** (EWC, Synaptic Intelligence): Best for scenarios where task boundaries are clear and computational resources are limited.
2. **Replay-Based Methods** (Experience Replay): Best for scenarios where memory constraints are not severe and data privacy is not a concern.
3. **Knowledge Distillation**: Best for scenarios where model size is a concern and previous task performance can be slightly compromised.

```python
import neurenix as nx
from neurenix.continual import EWC, ExperienceReplay, KnowledgeDistillation

# For resource-constrained environments
if memory_limited:
    continual_method = EWC(model, importance=1000.0)
# For scenarios with available memory
elif memory_available:
    continual_method = ExperienceReplay(model, memory_size=1000)
# For scenarios where model size is a concern
else:
    continual_method = KnowledgeDistillation(model, temperature=2.0)
```

### Hyperparameter Tuning

Proper hyperparameter tuning is crucial for effective continual learning:

1. **Importance Factor**: For regularization-based methods, the importance factor controls the trade-off between learning new tasks and retaining old ones.
2. **Memory Size**: For replay-based methods, the memory size determines how many samples from previous tasks are retained.
3. **Temperature**: For knowledge distillation, the temperature controls the softness of the probability distributions.

```python
import neurenix as nx
from neurenix.continual import EWC
from neurenix.automl import GridSearch

# Define a function to create an EWC model with different hyperparameters
def create_ewc_model(importance=1000.0, fisher_sample_size=1000):
    model = nx.nn.Sequential(
        nx.nn.Linear(784, 256),
        nx.nn.ReLU(),
        nx.nn.Linear(256, 10)
    )
    return EWC(model, importance=importance, fisher_sample_size=fisher_sample_size)

# Define hyperparameter space
param_space = {
    'importance': [100.0, 1000.0, 10000.0],
    'fisher_sample_size': [100, 500, 1000]
}

# Use grid search to find the best hyperparameters
grid_search = GridSearch(
    model_fn=create_ewc_model,
    param_space=param_space,
    scoring='accuracy',
    cv=3
)

grid_search.fit(X, y)
best_params = grid_search.best_params()
print(f"Best hyperparameters: {best_params}")
```

### Task Boundary Detection

In many real-world scenarios, task boundaries are not clearly defined. Implementing task boundary detection can improve continual learning performance:

```python
import neurenix as nx
from neurenix.continual import EWC
import numpy as np

# Create a model and EWC wrapper
model = nx.nn.Sequential(
    nx.nn.Linear(784, 256),
    nx.nn.ReLU(),
    nx.nn.Linear(256, 10)
)
ewc = EWC(model, importance=1000.0)

# Function to detect task boundaries based on performance drop
def detect_task_boundary(model, data_stream, window_size=100, threshold=0.1):
    recent_accuracies = []
    current_task_id = 1
    
    for batch_idx, (inputs, targets) in enumerate(data_stream):
        # Evaluate model on current batch
        model.eval()
        outputs = model(inputs)
        predictions = outputs.argmax(dim=1)
        accuracy = (predictions == targets).float().mean().item()
        
        # Update recent accuracies
        recent_accuracies.append(accuracy)
        if len(recent_accuracies) > window_size:
            recent_accuracies.pop(0)
        
        # Check for task boundary
        if len(recent_accuracies) == window_size:
            avg_accuracy = np.mean(recent_accuracies[:window_size//2])
            recent_accuracy = np.mean(recent_accuracies[window_size//2:])
            
            if avg_accuracy - recent_accuracy > threshold:
                # Task boundary detected
                current_task_id += 1
                print(f"Task boundary detected at batch {batch_idx}, switching to task {current_task_id}")
                
                # Register new task for EWC
                ewc.register_task(current_task_id, dataloader=None)  # We'll update Fisher later
        
        # Train model
        model.train()
        # ... training code ...
```

## Tutorials

### Continual Learning with Elastic Weight Consolidation (EWC)

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import Adam
from neurenix.continual import EWC
from neurenix.data import DataLoader
import matplotlib.pyplot as plt

# Initialize Neurenix
nx.init()

# Create a model for digit classification
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
)

# Create an EWC wrapper
ewc = EWC(model, importance=1000.0, fisher_sample_size=200)

# Create an optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Create a loss function
loss_fn = nx.nn.CrossEntropyLoss()

# Load MNIST dataset (task 1: digits 0-4)
task1_dataset = nx.data.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=lambda x: x.reshape(-1) / 255.0,
    target_transform=None,
    filter_labels=lambda y: y < 5
)

# Load MNIST dataset (task 2: digits 5-9)
task2_dataset = nx.data.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=lambda x: x.reshape(-1) / 255.0,
    target_transform=lambda y: y - 5,  # Remap labels to 0-4
    filter_labels=lambda y: y >= 5
)

# Create data loaders
task1_loader = DataLoader(task1_dataset, batch_size=32, shuffle=True)
task2_loader = DataLoader(task2_dataset, batch_size=32, shuffle=True)

# Create test datasets
task1_test = nx.data.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=lambda x: x.reshape(-1) / 255.0,
    target_transform=None,
    filter_labels=lambda y: y < 5
)

task2_test = nx.data.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=lambda x: x.reshape(-1) / 255.0,
    target_transform=lambda y: y - 5,
    filter_labels=lambda y: y >= 5
)

# Create test loaders
task1_test_loader = DataLoader(task1_test, batch_size=100, shuffle=False)
task2_test_loader = DataLoader(task2_test, batch_size=100, shuffle=False)

# Function to evaluate model on a dataset
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        outputs = model(inputs)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
    
    return correct / total

# Train on task 1
print("Training on task 1 (digits 0-4)...")
task1_accuracies = []

for epoch in range(5):
    model.train()
    for inputs, targets in task1_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Evaluate on both tasks
    task1_acc = evaluate(model, task1_test_loader)
    task2_acc = evaluate(model, task2_test_loader)
    task1_accuracies.append((task1_acc, task2_acc))
    
    print(f"Epoch {epoch+1}, Task 1 accuracy: {task1_acc:.4f}, Task 2 accuracy: {task2_acc:.4f}")

# Register task 1 for EWC
print("Computing Fisher information matrix for task 1...")
ewc.register_task(task_id=1, dataloader=task1_loader)

# Train on task 2 with EWC
print("Training on task 2 (digits 5-9) with EWC...")
ewc_accuracies = []

for epoch in range(5):
    model.train()
    for inputs, targets in task2_loader:
        ewc.train_step(optimizer, loss_fn, inputs, targets, task_id=2)
    
    # Evaluate on both tasks
    task1_acc = evaluate(model, task1_test_loader)
    task2_acc = evaluate(model, task2_test_loader)
    ewc_accuracies.append((task1_acc, task2_acc))
    
    print(f"Epoch {epoch+1}, Task 1 accuracy: {task1_acc:.4f}, Task 2 accuracy: {task2_acc:.4f}")

# Train on task 2 without EWC (for comparison)
print("Training on task 2 (digits 5-9) without EWC (for comparison)...")

# Reset the model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
)

optimizer = Adam(model.parameters(), lr=0.001)

# Train on task 1
for epoch in range(5):
    model.train()
    for inputs, targets in task1_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

# Train on task 2 without EWC
no_ewc_accuracies = []

for epoch in range(5):
    model.train()
    for inputs, targets in task2_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Evaluate on both tasks
    task1_acc = evaluate(model, task1_test_loader)
    task2_acc = evaluate(model, task2_test_loader)
    no_ewc_accuracies.append((task1_acc, task2_acc))
    
    print(f"Epoch {epoch+1}, Task 1 accuracy: {task1_acc:.4f}, Task 2 accuracy: {task2_acc:.4f}")

# Plot results
plt.figure(figsize=(12, 5))

# Plot task 1 accuracy
plt.subplot(1, 2, 1)
plt.plot([acc[0] for acc in ewc_accuracies], 'b-', label='With EWC')
plt.plot([acc[0] for acc in no_ewc_accuracies], 'r-', label='Without EWC')
plt.xlabel('Epoch')
plt.ylabel('Task 1 Accuracy')
plt.title('Task 1 (Digits 0-4) Accuracy During Task 2 Training')
plt.legend()

# Plot task 2 accuracy
plt.subplot(1, 2, 2)
plt.plot([acc[1] for acc in ewc_accuracies], 'b-', label='With EWC')
plt.plot([acc[1] for acc in no_ewc_accuracies], 'r-', label='Without EWC')
plt.xlabel('Epoch')
plt.ylabel('Task 2 Accuracy')
plt.title('Task 2 (Digits 5-9) Accuracy During Task 2 Training')
plt.legend()

plt.tight_layout()
plt.savefig('ewc_comparison.png')
plt.show()
```

### Continual Learning with Experience Replay

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import Adam
from neurenix.continual import ExperienceReplay
from neurenix.data import DataLoader
import matplotlib.pyplot as plt

# Initialize Neurenix
nx.init()

# Create a model for digit classification
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
)

# Create an Experience Replay wrapper
replay = ExperienceReplay(model, memory_size=1000, replay_ratio=0.5)

# Create an optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Create a loss function
loss_fn = nx.nn.CrossEntropyLoss()

# Load MNIST dataset (task 1: digits 0-4)
task1_dataset = nx.data.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=lambda x: x.reshape(-1) / 255.0,
    target_transform=None,
    filter_labels=lambda y: y < 5
)

# Load MNIST dataset (task 2: digits 5-9)
task2_dataset = nx.data.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=lambda x: x.reshape(-1) / 255.0,
    target_transform=lambda y: y - 5,  # Remap labels to 0-4
    filter_labels=lambda y: y >= 5
)

# Create data loaders
task1_loader = DataLoader(task1_dataset, batch_size=32, shuffle=True)
task2_loader = DataLoader(task2_dataset, batch_size=32, shuffle=True)

# Create test datasets
task1_test = nx.data.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=lambda x: x.reshape(-1) / 255.0,
    target_transform=None,
    filter_labels=lambda y: y < 5
)

task2_test = nx.data.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=lambda x: x.reshape(-1) / 255.0,
    target_transform=lambda y: y - 5,
    filter_labels=lambda y: y >= 5
)

# Create test loaders
task1_test_loader = DataLoader(task1_test, batch_size=100, shuffle=False)
task2_test_loader = DataLoader(task2_test, batch_size=100, shuffle=False)

# Function to evaluate model on a dataset
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    
    for inputs, targets in dataloader:
        outputs = model(inputs)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total += targets.size(0)
    
    return correct / total

# Train on task 1
print("Training on task 1 (digits 0-4)...")
task1_accuracies = []

for epoch in range(5):
    model.train()
    for inputs, targets in task1_loader:
        # Add samples to replay memory
        replay.add_to_memory(inputs, targets, task_id=1)
        
        # Regular training (no replay yet)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Evaluate on both tasks
    task1_acc = evaluate(model, task1_test_loader)
    task2_acc = evaluate(model, task2_test_loader)
    task1_accuracies.append((task1_acc, task2_acc))
    
    print(f"Epoch {epoch+1}, Task 1 accuracy: {task1_acc:.4f}, Task 2 accuracy: {task2_acc:.4f}")

# Train on task 2 with Experience Replay
print("Training on task 2 (digits 5-9) with Experience Replay...")
replay_accuracies = []

for epoch in range(5):
    model.train()
    for inputs, targets in task2_loader:
        # Train with experience replay
        replay.train_step(optimizer, loss_fn, inputs, targets, task_id=2)
        
        # Add current samples to memory
        replay.add_to_memory(inputs, targets, task_id=2)
    
    # Evaluate on both tasks
    task1_acc = evaluate(model, task1_test_loader)
    task2_acc = evaluate(model, task2_test_loader)
    replay_accuracies.append((task1_acc, task2_acc))
    
    print(f"Epoch {epoch+1}, Task 1 accuracy: {task1_acc:.4f}, Task 2 accuracy: {task2_acc:.4f}")

# Train on task 2 without Experience Replay (for comparison)
print("Training on task 2 (digits 5-9) without Experience Replay (for comparison)...")

# Reset the model
model = Sequential(
    Linear(784, 256),
    ReLU(),
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
)

optimizer = Adam(model.parameters(), lr=0.001)

# Train on task 1
for epoch in range(5):
    model.train()
    for inputs, targets in task1_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()

# Train on task 2 without Experience Replay
no_replay_accuracies = []

for epoch in range(5):
    model.train()
    for inputs, targets in task2_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Evaluate on both tasks
    task1_acc = evaluate(model, task1_test_loader)
    task2_acc = evaluate(model, task2_test_loader)
    no_replay_accuracies.append((task1_acc, task2_acc))
    
    print(f"Epoch {epoch+1}, Task 1 accuracy: {task1_acc:.4f}, Task 2 accuracy: {task2_acc:.4f}")

# Plot results
plt.figure(figsize=(12, 5))

# Plot task 1 accuracy
plt.subplot(1, 2, 1)
plt.plot([acc[0] for acc in replay_accuracies], 'g-', label='With Experience Replay')
plt.plot([acc[0] for acc in no_replay_accuracies], 'r-', label='Without Experience Replay')
plt.xlabel('Epoch')
plt.ylabel('Task 1 Accuracy')
plt.title('Task 1 (Digits 0-4) Accuracy During Task 2 Training')
plt.legend()

# Plot task 2 accuracy
plt.subplot(1, 2, 2)
plt.plot([acc[1] for acc in replay_accuracies], 'g-', label='With Experience Replay')
plt.plot([acc[1] for acc in no_replay_accuracies], 'r-', label='Without Experience Replay')
plt.xlabel('Epoch')
plt.ylabel('Task 2 Accuracy')
plt.title('Task 2 (Digits 5-9) Accuracy During Task 2 Training')
plt.legend()

plt.tight_layout()
plt.savefig('experience_replay_comparison.png')
plt.show()
```

## Conclusion

The Continual Learning module in Neurenix provides a comprehensive set of tools for training neural networks on new data without forgetting previously learned knowledge. With support for various continual learning techniques, including regularization-based methods, replay-based methods, and knowledge distillation, the module enables developers to create AI systems that can adapt to changing environments and new tasks over time.

Compared to other frameworks, Neurenix's Continual Learning module offers advantages in terms of API consistency, integration with the core framework, and optimization for edge devices. These features make Neurenix particularly well-suited for developing adaptive AI systems that can learn continuously in real-world environments.
