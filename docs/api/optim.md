# Optimization Module

## Overview

The Optimization module in Neurenix provides a comprehensive suite of optimization algorithms for training neural networks and other machine learning models. Built on Neurenix's high-performance multi-language architecture, this module delivers efficient implementations of classical and state-of-the-art optimization techniques.

The module features a unified optimizer interface that works seamlessly with Neurenix's tensor and neural network modules, enabling users to easily switch between different optimization algorithms without changing their training code. It supports various optimization strategies, including first-order methods, second-order methods, adaptive learning rate methods, and distributed optimization techniques.

Implemented with a combination of Rust and C++ for performance-critical components and Python for the user-friendly interface, the Optimization module ensures both computational efficiency and ease of use. It provides native support for various hardware accelerators, including GPUs, TPUs, and specialized AI hardware, with automatic optimization for the available hardware.

The module also includes tools for learning rate scheduling, gradient processing, and hyperparameter tuning, making it a complete solution for optimizing machine learning models across different domains and applications.

## Key Concepts

### Optimizer Interface

The Optimization module provides a unified interface for all optimizers, making it easy to switch between different algorithms:

- **Parameter Management**: Automatic tracking and updating of model parameters
- **Gradient Processing**: Support for gradient clipping, accumulation, and normalization
- **State Management**: Efficient storage and retrieval of optimizer states
- **Device Compatibility**: Seamless operation across different hardware devices
- **Serialization**: Support for saving and loading optimizer states

This unified interface ensures consistency and interoperability across different optimization algorithms.

### First-Order Methods

The module includes various first-order optimization methods that use gradient information to update model parameters:

- **Stochastic Gradient Descent (SGD)**: The fundamental optimization algorithm with support for momentum, weight decay, and Nesterov acceleration
- **AdaGrad**: Adaptive gradient algorithm that adapts learning rates based on historical gradient information
- **RMSProp**: Root Mean Square Propagation algorithm that addresses AdaGrad's radically diminishing learning rates
- **Adam**: Adaptive Moment Estimation algorithm that combines the benefits of AdaGrad and RMSProp
- **AdamW**: Adam with decoupled weight decay regularization for better generalization
- **NAdam**: Adam with Nesterov momentum for improved convergence
- **RAdam**: Rectified Adam with a variance rectification term for more stable training
- **AdaBelief**: Adam with belief in observed gradients for more stable and faster convergence
- **LAMB**: Layer-wise Adaptive Moments optimizer for large batch training
- **LARS**: Layer-wise Adaptive Rate Scaling for large batch training

These methods provide a range of options for different training scenarios and model architectures.

### Second-Order Methods

The module includes second-order optimization methods that use Hessian information for more efficient parameter updates:

- **L-BFGS**: Limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm for approximating second-order information
- **Newton-CG**: Newton's method with conjugate gradient for solving the Newton step
- **Trust Region**: Trust region methods for robust optimization
- **Gauss-Newton**: Approximation of the Hessian using the Jacobian
- **Natural Gradient**: Fisher information matrix-based optimization for preserving the model's probabilistic interpretation

These methods can provide faster convergence for certain types of problems, especially in the convex setting.

### Adaptive Learning Rate Methods

The module provides various strategies for adapting learning rates during training:

- **Learning Rate Schedulers**: Predefined schedules for adjusting learning rates over time
  - Step scheduler: Reduces learning rate at predefined steps
  - Exponential scheduler: Exponentially decays learning rate
  - Cosine annealing: Cyclical learning rate based on cosine function
  - Reduce on plateau: Reduces learning rate when metrics plateau
  - One-cycle policy: Learning rate that follows a one-cycle policy
  - Warm restarts: Cosine annealing with warm restarts
  - Custom schedulers: User-defined learning rate schedules

- **Adaptive Methods**: Algorithms that automatically adapt learning rates based on gradient information
  - Per-parameter adaptation: Different learning rates for different parameters
  - Layer-wise adaptation: Different learning rates for different layers
  - Gradient-based adaptation: Learning rates based on gradient statistics

These methods help in achieving faster convergence and better generalization.

### Distributed Optimization

The module supports distributed optimization for training models across multiple devices and nodes:

- **Data Parallelism**: Distributing data across devices with synchronized parameter updates
- **Model Parallelism**: Distributing model components across devices
- **Hybrid Parallelism**: Combining data and model parallelism
- **Parameter Server**: Centralized parameter management with distributed workers
- **Ring AllReduce**: Efficient gradient aggregation without a central server
- **Gradient Compression**: Reducing communication overhead through gradient compression
- **Asynchronous Updates**: Non-blocking parameter updates for improved throughput
- **Federated Optimization**: Privacy-preserving distributed optimization

These techniques enable efficient training of large models on distributed hardware.

## API Reference

### Base Optimizer

```python
import neurenix
from neurenix.optim import Optimizer

# Base optimizer class (abstract)
class Optimizer:
    def __init__(self, params, defaults):
        """
        Initialize the optimizer.
        
        Args:
            params: Iterable of parameters to optimize or dicts defining parameter groups
            defaults: Dict containing default values of optimization options
        """
        pass
    
    def zero_grad(self, set_to_none=False):
        """
        Reset the gradients of all optimized parameters.
        
        Args:
            set_to_none: If True, set gradients to None instead of zeroing them
        """
        pass
    
    def step(self, closure=None):
        """
        Perform a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
        """
        pass
    
    def add_param_group(self, param_group):
        """
        Add a parameter group to the optimizer's param_groups.
        
        Args:
            param_group: Dict containing parameters and optimization options
        """
        pass
    
    def load_state_dict(self, state_dict):
        """
        Load optimizer state.
        
        Args:
            state_dict: Optimizer state dict
        """
        pass
    
    def state_dict(self):
        """
        Return the optimizer's state dict.
        """
        pass
```

### First-Order Optimizers

```python
from neurenix.optim import SGD, Adam, AdamW, RMSProp, AdaGrad

# Stochastic Gradient Descent
optimizer = SGD(
    params=model.parameters(),
    lr=0.01,                # Learning rate
    momentum=0.9,           # Momentum factor
    weight_decay=1e-5,      # Weight decay (L2 penalty)
    dampening=0,            # Dampening for momentum
    nesterov=True           # Enable Nesterov momentum
)

# Adam optimizer
optimizer = Adam(
    params=model.parameters(),
    lr=0.001,               # Learning rate
    betas=(0.9, 0.999),     # Coefficients for computing running averages of gradient and its square
    eps=1e-8,               # Term added to the denominator to improve numerical stability
    weight_decay=0,         # Weight decay (L2 penalty)
    amsgrad=False           # Whether to use the AMSGrad variant
)

# AdamW optimizer
optimizer = AdamW(
    params=model.parameters(),
    lr=0.001,               # Learning rate
    betas=(0.9, 0.999),     # Coefficients for computing running averages of gradient and its square
    eps=1e-8,               # Term added to the denominator to improve numerical stability
    weight_decay=0.01,      # Weight decay (decoupled from learning rate)
    amsgrad=False           # Whether to use the AMSGrad variant
)
```

### Learning Rate Schedulers

```python
from neurenix.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, 
    ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts
)

# Step LR scheduler
scheduler = StepLR(
    optimizer=optimizer,
    step_size=30,           # Period of learning rate decay
    gamma=0.1               # Multiplicative factor of learning rate decay
)

# Cosine Annealing LR scheduler
scheduler = CosineAnnealingLR(
    optimizer=optimizer,
    T_max=100,              # Maximum number of iterations
    eta_min=0               # Minimum learning rate
)

# Reduce LR On Plateau scheduler
scheduler = ReduceLROnPlateau(
    optimizer=optimizer,
    mode='min',             # In 'min' mode, lr will be reduced when the quantity monitored has stopped decreasing
    factor=0.1,             # Factor by which the learning rate will be reduced
    patience=10,            # Number of epochs with no improvement after which learning rate will be reduced
    threshold=1e-4,         # Threshold for measuring the new optimum
    min_lr=0                # Lower bound on the learning rate
)
```

## Framework Comparison

### Neurenix Optimization vs. TensorFlow Optimization

| Feature | Neurenix Optimization | TensorFlow Optimization |
|---------|------------------------|--------------------------|
| Performance | Multi-language implementation with Rust/C++ backends | C++ backend with Python interface |
| Hardware Support | Comprehensive support for various hardware | Primarily focused on TPUs and GPUs |
| Optimizer Variety | Extensive collection of classical and modern optimizers | Good selection of common optimizers |
| Learning Rate Scheduling | Comprehensive scheduling options with adaptive techniques | Basic scheduling options |
| Gradient Processing | Advanced gradient processing with clipping, accumulation, and noise | Basic gradient processing |
| Distributed Optimization | Multiple strategies with efficient communication | Limited to specific distributed strategies |
| Hyperparameter Tuning | Integrated hyperparameter tuning tools | Requires external libraries |
| Second-Order Methods | Comprehensive support for second-order methods | Limited support for second-order methods |
| Edge Device Support | Native support for edge devices | Limited through TensorFlow Lite |

Neurenix's Optimization module provides better performance through its multi-language implementation and offers more comprehensive hardware support, especially for edge devices. It also provides a wider variety of optimization algorithms, more advanced gradient processing techniques, and better support for distributed optimization.

### Neurenix Optimization vs. PyTorch Optimization

| Feature | Neurenix Optimization | PyTorch Optimization |
|---------|------------------------|----------------------|
| Performance | Multi-language implementation with Rust/C++ backends | C++ backend with Python interface |
| Hardware Support | Comprehensive support for various hardware | Primarily focused on CUDA devices |
| Optimizer Variety | Extensive collection of classical and modern optimizers | Good selection of common optimizers |
| Learning Rate Scheduling | Comprehensive scheduling options with adaptive techniques | Good scheduling options |
| Gradient Processing | Advanced gradient processing with clipping, accumulation, and noise | Basic gradient processing |
| Distributed Optimization | Multiple strategies with efficient communication | Good support through PyTorch Distributed |
| Hyperparameter Tuning | Integrated hyperparameter tuning tools | Requires external libraries |
| Second-Order Methods | Comprehensive support for second-order methods | Limited support for second-order methods |
| Edge Device Support | Native support for edge devices | Limited through separate tools |

While PyTorch has a good optimization module, Neurenix's Optimization module offers better performance through its multi-language implementation and provides more comprehensive hardware support, especially for edge devices. It also offers more advanced gradient processing techniques and better support for second-order methods.

### Neurenix Optimization vs. Scikit-Learn Optimization

| Feature | Neurenix Optimization | Scikit-Learn Optimization |
|---------|------------------------|----------------------------|
| Deep Learning Support | Full support for deep learning optimization | Limited to classical machine learning |
| Hardware Acceleration | Native support for various hardware accelerators | Limited hardware acceleration |
| Optimizer Variety | Extensive collection of classical and modern optimizers | Focus on classical optimization algorithms |
| Learning Rate Scheduling | Comprehensive scheduling options with adaptive techniques | Limited scheduling options |
| Gradient Processing | Advanced gradient processing techniques | Basic gradient processing |
| Distributed Optimization | Multiple strategies for distributed optimization | Limited distributed capabilities |
| Hyperparameter Tuning | Integrated hyperparameter tuning tools | Good hyperparameter tuning tools |
| Second-Order Methods | Comprehensive support for second-order methods | Good support for second-order methods |
| Edge Device Support | Native support for edge devices | Limited edge support |

Scikit-Learn's optimization is primarily focused on classical machine learning algorithms, while Neurenix's Optimization module is designed for both deep learning and classical machine learning. Neurenix provides better hardware acceleration, more comprehensive support for various optimization algorithms, and better integration with the deep learning ecosystem.

## Best Practices

### Choosing the Right Optimizer

1. **Start with Adam**: For most deep learning tasks, Adam is a good default choice due to its adaptive learning rates and momentum.

```python
optimizer = neurenix.optim.Adam(
    params=model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0
)
```

2. **Use SGD with Momentum for CNNs**: For convolutional neural networks, SGD with momentum often provides better generalization.

```python
optimizer = neurenix.optim.SGD(
    params=model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)
```

3. **Consider AdamW for Transformers**: For transformer models, AdamW with decoupled weight decay often works better.

```python
optimizer = neurenix.optim.AdamW(
    params=model.parameters(),
    lr=5e-5,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01
)
```

### Learning Rate Scheduling

1. **Use Learning Rate Warm-up**: Start with a small learning rate and gradually increase it to the target value.

```python
def warmup_scheduler(optimizer, warmup_steps, target_lr):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    
    return neurenix.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

2. **Implement Learning Rate Decay**: Reduce the learning rate over time to fine-tune the model.

```python
scheduler = neurenix.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.1
)
```

### Gradient Processing

1. **Clip Gradients to Prevent Exploding Gradients**: Use gradient clipping to stabilize training.

```python
def train_step(model, optimizer, inputs, targets):
    # Forward pass
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Clip gradients
    neurenix.optim.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # Update parameters
    optimizer.step()
    
    return loss.item()
```

2. **Use Gradient Accumulation for Larger Batch Sizes**: Accumulate gradients over multiple batches to simulate larger batch sizes.

```python
def train_with_accumulation(model, optimizer, data_loader, accumulation_steps=4):
    model.train()
    optimizer.zero_grad()
    
    for i, (inputs, targets) in enumerate(data_loader):
        # Forward pass
        outputs = model(inputs)
        loss = loss_function(outputs, targets) / accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights after accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

## Tutorials

### Basic Optimization with SGD and Adam

```python
import neurenix
from neurenix.nn import Module, Linear, ReLU
from neurenix.optim import SGD, Adam
from neurenix.data import DataLoader

# Define a simple neural network
class SimpleNN(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create a model
model = SimpleNN(input_size=784, hidden_size=128, output_size=10)

# Create a loss function
loss_fn = neurenix.nn.CrossEntropyLoss()

# Create an optimizer (SGD)
optimizer_sgd = SGD(
    params=model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=1e-5,
    nesterov=True
)

# Create an optimizer (Adam)
optimizer_adam = Adam(
    params=model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-5
)

# Choose which optimizer to use
optimizer = optimizer_adam

# Training loop
def train(model, optimizer, data_loader, loss_fn, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in data_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Print epoch statistics
        accuracy = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(data_loader):.4f}, Accuracy: {accuracy:.2f}%")

# Create a data loader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Train the model
train(model, optimizer, train_loader, loss_fn, epochs=10)
```

### Learning Rate Scheduling

```python
import neurenix
from neurenix.optim import Adam
from neurenix.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

# Create a model and optimizer
model = create_model()
optimizer = Adam(model.parameters(), lr=0.001)

# Create a step scheduler
step_scheduler = StepLR(
    optimizer=optimizer,
    step_size=30,
    gamma=0.1
)

# Create a cosine annealing scheduler
cosine_scheduler = CosineAnnealingLR(
    optimizer=optimizer,
    T_max=100,
    eta_min=1e-6
)

# Choose which scheduler to use
scheduler = cosine_scheduler

# Training loop with learning rate scheduling
def train_with_scheduler(model, optimizer, scheduler, data_loader, val_loader, loss_fn, epochs=100):
    model.train()
    
    for epoch in range(epochs):
        # Training
        for inputs, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with neurenix.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = 100.0 * correct / total
        
        # Update learning rate
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # Print statistics
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%, LR: {current_lr:.6f}")
        
        model.train()
```
