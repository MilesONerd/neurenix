# Optimization Documentation

## Overview

The Optimization module provides algorithms for training machine learning models in the Neurenix framework. These optimizers update model parameters based on gradients to minimize the loss function, enabling the model to learn from data.

Neurenix's optimization algorithms are implemented using a multi-language architecture, where the high-performance operations are implemented in the Rust/C++ Phynexus engine, while the Python API provides a user-friendly interface. This architecture enables Neurenix to deliver optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

## Key Concepts

### Gradient-Based Optimization

Neurenix's optimizers use gradient-based methods to update model parameters. The gradient of the loss function with respect to each parameter indicates the direction of steepest ascent, so optimizers move parameters in the opposite direction to minimize the loss.

### Parameter Groups

Optimizers in Neurenix support parameter groups, allowing different parts of a model to use different hyperparameters (e.g., learning rates). This is particularly useful for fine-tuning pre-trained models or implementing learning rate schedules.

### Edge Device Optimization

Neurenix's optimizers are designed with edge devices in mind, with efficient implementations that minimize memory usage and computational requirements while maintaining high performance. This makes Neurenix particularly well-suited for AI agent applications on resource-constrained hardware.

### Multi-Device Support

Optimizers in Neurenix can work with parameters stored on different devices (CPU, CUDA, ROCm, WebGPU), enabling efficient training across a wide range of hardware configurations.

## API Reference

### Base Optimizer

```python
neurenix.optim.Optimizer(params, defaults)
```

Base class for all optimizers.

**Parameters:**
- `params`: An iterable of tensors to optimize.
- `defaults`: Default hyperparameters for the optimizer.

**Methods:**
- `zero_grad()`: Reset the gradients of all optimized tensors.
- `step()`: Update the parameters based on the current gradients.
- `add_param_group(param_group)`: Add a parameter group to the optimizer.

**Example:**
```python
import neurenix
from neurenix.optim import Optimizer

# Create a custom optimizer
class MyOptimizer(Optimizer):
    def __init__(self, params, lr=0.01):
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    
    def step(self):
        for group in self._parameter_groups:
            for param in group["params"]:
                if param.grad is not None:
                    # Simple gradient descent update
                    param._numpy_data -= group["lr"] * param.grad.numpy()

# Create a model
model = neurenix.nn.Sequential(
    neurenix.nn.Linear(10, 5),
    neurenix.nn.ReLU(),
    neurenix.nn.Linear(5, 1)
)

# Create an optimizer
optimizer = MyOptimizer(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    # Forward pass
    output = model(input_data)
    loss = loss_function(output, target_data)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Stochastic Gradient Descent (SGD)

```python
neurenix.optim.SGD(params, lr=0.01, momentum=0, dampening=0, weight_decay=0, nesterov=False)
```

Implements stochastic gradient descent (optionally with momentum).

**Parameters:**
- `params`: Iterable of parameters to optimize or dicts defining parameter groups.
- `lr`: Learning rate.
- `momentum`: Momentum factor.
- `dampening`: Dampening for momentum.
- `weight_decay`: Weight decay (L2 penalty).
- `nesterov`: Enables Nesterov momentum.

**Example:**
```python
import neurenix
from neurenix.nn import Linear, Sequential, ReLU
from neurenix.optim import SGD

# Create a model
model = Sequential(
    Linear(10, 5),
    ReLU(),
    Linear(5, 1)
)

# Create an SGD optimizer
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

# Training loop
for epoch in range(10):
    # Forward pass
    output = model(input_data)
    loss = loss_function(output, target_data)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### Adam

```python
neurenix.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
```

Implements the Adam algorithm.

**Parameters:**
- `params`: Iterable of parameters to optimize or dicts defining parameter groups.
- `lr`: Learning rate.
- `betas`: Coefficients used for computing running averages of gradient and its square.
- `eps`: Term added to the denominator to improve numerical stability.
- `weight_decay`: Weight decay (L2 penalty).

**Example:**
```python
import neurenix
from neurenix.nn import Linear, Sequential, ReLU
from neurenix.optim import Adam

# Create a model
model = Sequential(
    Linear(10, 5),
    ReLU(),
    Linear(5, 1)
)

# Create an Adam optimizer
optimizer = Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

# Training loop
for epoch in range(10):
    # Forward pass
    output = model(input_data)
    loss = loss_function(output, target_data)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Optimizer API** | Object-oriented, similar to PyTorch | Functional and object-oriented APIs |
| **Parameter Groups** | Supported through add_param_group | Limited support through variable collections |
| **Edge Optimization** | Native optimization for edge devices | TensorFlow Lite for edge devices |
| **Custom Optimizers** | Easy to create through Optimizer subclassing | Requires custom optimizer implementation through tf.keras.optimizers.Optimizer |
| **Multi-Device Support** | CPU, CUDA, ROCm, WebGPU | CPU, CUDA, TPU |
| **Multi-Language Architecture** | Rust/C++ core with Python interface | C++ core with Python interface |

Neurenix's optimization algorithms offer a more intuitive, PyTorch-like API compared to TensorFlow's optimizer API, making it easier for researchers and developers to create custom optimizers. The native optimization for edge devices in Neurenix provides better performance on resource-constrained hardware compared to TensorFlow Lite, which is an add-on component rather than being integrated into the core architecture.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Optimizer API** | Similar to PyTorch | Object-oriented, intuitive |
| **Parameter Groups** | Supported through add_param_group | Supported through parameter groups |
| **Edge Optimization** | Native optimization for edge devices | PyTorch Mobile for edge devices |
| **Hardware Support** | CPU, CUDA, ROCm, WebGPU | CPU, CUDA |
| **Multi-Language Architecture** | Rust/C++ core with Python interface | C++ core with Python interface |
| **WebAssembly Support** | Native WebGPU support | Limited support via PyTorch.js |

Neurenix's optimization algorithms are very similar to PyTorch's in terms of API design and functionality, making it easy for PyTorch users to transition to Neurenix. However, Neurenix extends hardware support to include ROCm and WebGPU, making it more versatile across different hardware platforms. The native edge device optimization in Neurenix also provides advantages over PyTorch Mobile, particularly for AI agent applications.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Optimizer Types** | Gradient-based optimizers for neural networks | Various optimizers for different algorithms |
| **GPU Acceleration** | Native support for multiple GPU types | No native GPU support |
| **Edge Device Support** | Native optimization for edge devices | No specific edge device support |
| **API Design** | Object-oriented, focused on neural networks | Function-based, focused on specific algorithms |
| **Parameter Groups** | Supported through add_param_group | Not applicable (different optimization paradigm) |
| **Multi-Language Architecture** | Rust/C++ core with Python interface | Pure Python with C/C++ extensions |

Neurenix provides optimizers specifically designed for neural networks, while Scikit-Learn offers a variety of optimization algorithms for different machine learning models. While Scikit-Learn's optimizers are integrated with specific algorithms, Neurenix's optimizers are more general-purpose and can be used with any differentiable model. Additionally, Neurenix's optimizers support GPU acceleration and edge device optimization, making them more suitable for deep learning and AI agent applications.

## Best Practices

### Choosing an Optimizer

When choosing an optimizer for your model, consider these factors:

1. **Task Complexity**: For simple tasks, SGD with momentum is often sufficient. For more complex tasks, Adam is generally a good default choice.
2. **Convergence Speed**: Adam typically converges faster than SGD, but SGD may reach better final solutions in some cases.
3. **Memory Requirements**: Adam requires more memory than SGD due to its moment estimates.
4. **Hardware Constraints**: On edge devices with limited memory, SGD may be preferable to Adam.

```python
import neurenix
from neurenix.nn import Linear, Sequential, ReLU
from neurenix.optim import SGD, Adam

# Create a model
model = Sequential(
    Linear(10, 5),
    ReLU(),
    Linear(5, 1)
)

# For simple tasks or memory-constrained devices
optimizer_sgd = SGD(model.parameters(), lr=0.01, momentum=0.9)

# For complex tasks or faster convergence
optimizer_adam = Adam(model.parameters(), lr=0.001)
```

### Setting Learning Rates

Choosing an appropriate learning rate is crucial for effective training:

1. **Start with a Reasonable Default**: 0.01 for SGD, 0.001 for Adam.
2. **Learning Rate Schedules**: Decrease the learning rate over time to fine-tune the model.
3. **Learning Rate Warmup**: Gradually increase the learning rate at the beginning of training.
4. **Different Learning Rates for Different Layers**: Use parameter groups to set different learning rates for different parts of the model.

```python
import neurenix
from neurenix.nn import Linear, Sequential, ReLU
from neurenix.optim import SGD

# Create a model
model = Sequential(
    Linear(10, 5),
    ReLU(),
    Linear(5, 1)
)

# Create an optimizer with different learning rates for different layers
optimizer = SGD([
    {"params": model[0].parameters(), "lr": 0.01},  # First layer
    {"params": model[2].parameters(), "lr": 0.001}  # Last layer
], lr=0.005)  # Default learning rate for other parameters
```

### Using Weight Decay

Weight decay (L2 regularization) can help prevent overfitting:

1. **Choose an Appropriate Value**: Typical values range from 1e-6 to 1e-4.
2. **Different Weight Decay for Different Layers**: Use parameter groups to set different weight decay values for different parts of the model.
3. **No Weight Decay for Bias Terms**: It's common practice to apply weight decay only to weights, not biases.

```python
import neurenix
from neurenix.nn import Linear, Sequential, ReLU
from neurenix.optim import Adam

# Create a model
model = Sequential(
    Linear(10, 5),
    ReLU(),
    Linear(5, 1)
)

# Create an optimizer with weight decay
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
```

### Optimizing for Edge Devices

When deploying models to edge devices, consider these optimizations:

1. **Choose Memory-Efficient Optimizers**: SGD requires less memory than Adam.
2. **Reduce Precision**: Use lower precision data types when possible.
3. **Minimize Parameter Count**: Use smaller models or techniques like pruning and quantization.
4. **Batch Size Adjustment**: Use smaller batch sizes to reduce memory requirements.

```python
import neurenix
from neurenix import Device, DeviceType
from neurenix.nn import Linear, Sequential, ReLU
from neurenix.optim import SGD

# Create a small model for edge devices
model = Sequential(
    Linear(10, 5),
    ReLU(),
    Linear(5, 1)
)

# Use the most efficient available device
devices = neurenix.get_available_devices()
edge_device = None

# Prioritize WebGPU for browser-based edge devices
for device in devices:
    if device.type == DeviceType.WEBGPU:
        edge_device = device
        break

# Fall back to CPU if no accelerator is available
if edge_device is None:
    edge_device = Device(DeviceType.CPU)

# Move model to the selected device
for param in model.parameters():
    param.to(edge_device, inplace=True)

# Create a memory-efficient optimizer
optimizer = SGD(model.parameters(), lr=0.01)
```

## Tutorials

### Basic Optimization with SGD

```python
import neurenix
from neurenix.nn import Linear, Sequential, ReLU, MSELoss
from neurenix.optim import SGD

# Create a simple model
model = Sequential(
    Linear(10, 5),
    ReLU(),
    Linear(5, 1)
)

# Generate some dummy data
input_data = neurenix.Tensor.randn((100, 10))
target_data = neurenix.Tensor.randn((100, 1))

# Create loss function and optimizer
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
model.train()
for epoch in range(100):
    # Forward pass
    output = model(input_data)
    loss = criterion(output, target_data)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

# Evaluation
model.eval()
with neurenix.no_grad():
    test_input = neurenix.Tensor.randn((20, 10))
    predictions = model(test_input)
    print(f"Predictions shape: {predictions.shape}")
```

### Using Adam with Parameter Groups

```python
import neurenix
from neurenix.nn import Module, Linear, ReLU
from neurenix.optim import Adam

# Create a custom model
class TwoPartModel(Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = Linear(10, 5)
        self.classifier = Linear(5, 1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features = ReLU()(features)
        return self.classifier(features)

# Create the model
model = TwoPartModel()

# Generate some dummy data
input_data = neurenix.Tensor.randn((100, 10))
target_data = neurenix.Tensor.randn((100, 1))

# Create an optimizer with different parameter groups
optimizer = Adam([
    {"params": model.feature_extractor.parameters(), "lr": 0.0001},  # Lower learning rate for feature extractor
    {"params": model.classifier.parameters(), "lr": 0.001}           # Higher learning rate for classifier
])

# Training loop
model.train()
for epoch in range(100):
    # Forward pass
    output = model(input_data)
    loss = neurenix.nn.MSELoss()(output, target_data)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

### Implementing a Learning Rate Schedule

```python
import neurenix
from neurenix.nn import Linear, Sequential, ReLU, MSELoss
from neurenix.optim import SGD

# Create a simple model
model = Sequential(
    Linear(10, 5),
    ReLU(),
    Linear(5, 1)
)

# Generate some dummy data
input_data = neurenix.Tensor.randn((100, 10))
target_data = neurenix.Tensor.randn((100, 1))

# Create loss function and optimizer
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.1)

# Training loop with learning rate schedule
model.train()
for epoch in range(100):
    # Adjust learning rate
    if epoch == 30:
        for param_group in optimizer._parameter_groups:
            param_group["lr"] = 0.01
    elif epoch == 60:
        for param_group in optimizer._parameter_groups:
            param_group["lr"] = 0.001
    
    # Forward pass
    output = model(input_data)
    loss = criterion(output, target_data)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        current_lr = optimizer._parameter_groups[0]["lr"]
        print(f"Epoch {epoch+1}, Loss: {loss.item()}, LR: {current_lr}")
```

## Conclusion

The Optimization module of Neurenix provides a comprehensive set of algorithms for training machine learning models. Its multi-language architecture with a high-performance Rust/C++ core enables optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

Compared to other frameworks like TensorFlow, PyTorch, and Scikit-Learn, Neurenix's Optimization module offers advantages in terms of API design, hardware support, and edge device optimization. These features make Neurenix particularly well-suited for AI agent development, edge computing, and browser-based applications.
