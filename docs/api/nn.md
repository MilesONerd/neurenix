# Neural Networks Documentation

## Overview

The Neural Networks module provides the building blocks for creating and training neural networks in the Neurenix framework. It includes a comprehensive set of components such as layers, activation functions, loss functions, and containers that can be combined to create complex neural network architectures.

Neurenix's neural network components are designed with a multi-language architecture, where the high-performance operations are implemented in the Rust/C++ Phynexus engine, while the Python interface provides a user-friendly API for rapid development. This architecture enables Neurenix to deliver optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

## Key Concepts

### Module System

The foundation of Neurenix's neural network components is the `Module` class, which is similar to PyTorch's `nn.Module`. It provides a way to organize parameters and submodules in a hierarchical structure, making it easy to create complex neural network architectures.

### Layer Types

Neurenix provides a variety of layer types for different neural network architectures:

- **Linear Layers**: Fully connected layers for traditional neural networks
- **Convolutional Layers**: 1D, 2D, and 3D convolutional layers for processing structured data
- **Recurrent Layers**: RNN, LSTM, and GRU layers for sequence processing
- **Activation Functions**: Various activation functions like ReLU, Sigmoid, Tanh, etc.

### Loss Functions

The framework includes a comprehensive set of loss functions for training neural networks, such as Mean Squared Error (MSE), Cross Entropy, and Binary Cross Entropy (BCE).

### Edge Device Optimization

Neurenix's neural network components are optimized for edge devices, with efficient implementations that minimize memory usage and computational requirements while maintaining high performance.

## API Reference

### Base Module

```python
neurenix.nn.Module
```

Base class for all neural network modules. This is similar to PyTorch's `nn.Module`, providing a way to organize parameters and submodules in a hierarchical structure.

**Methods:**
- `forward(*args, **kwargs)`: Forward pass of the module. This method should be overridden by all subclasses.
- `parameters()`: Get all parameters of the module and its submodules.
- `train(mode=True)`: Set the module in training mode.
- `eval()`: Set the module in evaluation mode.
- `is_training()`: Check if the module is in training mode.

**Example:**
```python
import neurenix
from neurenix.nn import Module

class MyModule(Module):
    def __init__(self):
        super().__init__()
        self.linear = neurenix.nn.Linear(10, 5)
        self.activation = neurenix.nn.ReLU()
    
    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x
```

### Linear Layers

```python
neurenix.nn.Linear(in_features, out_features, bias=True, dtype=None, device=None)
```

Applies a linear transformation to the incoming data: y = xW^T + b

**Parameters:**
- `in_features`: Size of each input sample
- `out_features`: Size of each output sample
- `bias`: If True, adds a learnable bias to the output
- `dtype`: Data type of the parameters
- `device`: Device to store the parameters on

**Example:**
```python
import neurenix
from neurenix.nn import Linear

# Create a linear layer
linear = Linear(10, 5)

# Forward pass
input_tensor = neurenix.Tensor.randn((32, 10))  # Batch of 32 samples, 10 features each
output = linear(input_tensor)  # Shape: (32, 5)
```

### Convolutional Layers

```python
neurenix.nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

Applies a 1D convolution over an input signal composed of several input channels.

```python
neurenix.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

Applies a 2D convolution over an input image composed of several input channels.

```python
neurenix.nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
```

Applies a 3D convolution over an input volume composed of several input channels.

**Parameters:**
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels
- `kernel_size`: Size of the convolving kernel
- `stride`: Stride of the convolution
- `padding`: Zero-padding added to both sides of the input
- `dilation`: Spacing between kernel elements
- `groups`: Number of blocked connections from input to output channels
- `bias`: If True, adds a learnable bias to the output

**Example:**
```python
import neurenix
from neurenix.nn import Conv2d

# Create a 2D convolutional layer
conv = Conv2d(3, 16, kernel_size=3, stride=1, padding=1)

# Forward pass
input_tensor = neurenix.Tensor.randn((32, 3, 28, 28))  # Batch of 32 images, 3 channels, 28x28 pixels
output = conv(input_tensor)  # Shape: (32, 16, 28, 28)
```

### Recurrent Layers

```python
neurenix.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False)
```

Applies a multi-layer Elman RNN with tanh or ReLU non-linearity to an input sequence.

```python
neurenix.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False)
```

Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

```python
neurenix.nn.GRU(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False)
```

Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

**Parameters:**
- `input_size`: The number of expected features in the input
- `hidden_size`: The number of features in the hidden state
- `num_layers`: Number of recurrent layers
- `nonlinearity`: The non-linearity to use (for RNN only). Can be 'tanh' or 'relu'
- `bias`: If False, then the layer does not use bias weights
- `batch_first`: If True, then the input and output tensors are provided as (batch, seq, feature)
- `dropout`: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
- `bidirectional`: If True, becomes a bidirectional RNN

**Example:**
```python
import neurenix
from neurenix.nn import LSTM

# Create an LSTM layer
lstm = LSTM(10, 20, num_layers=2, batch_first=True, bidirectional=True)

# Forward pass
input_tensor = neurenix.Tensor.randn((32, 15, 10))  # Batch of 32 sequences, 15 time steps, 10 features each
output, (h_n, c_n) = lstm(input_tensor)
# output shape: (32, 15, 40)  # 40 = 20 * 2 (bidirectional)
# h_n shape: (4, 32, 20)  # 4 = 2 (layers) * 2 (bidirectional)
# c_n shape: (4, 32, 20)
```

### Activation Functions

```python
neurenix.nn.ReLU(inplace=False)
```

Applies the rectified linear unit function element-wise: ReLU(x) = max(0, x)

```python
neurenix.nn.Sigmoid()
```

Applies the sigmoid function element-wise: Sigmoid(x) = 1 / (1 + exp(-x))

```python
neurenix.nn.Tanh()
```

Applies the hyperbolic tangent function element-wise: Tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

```python
neurenix.nn.LeakyReLU(negative_slope=0.01, inplace=False)
```

Applies the leaky rectified linear unit function element-wise: LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)

```python
neurenix.nn.ELU(alpha=1.0, inplace=False)
```

Applies the exponential linear unit function element-wise: ELU(x) = max(0, x) + min(0, alpha * (exp(x) - 1))

```python
neurenix.nn.SELU(inplace=False)
```

Applies the scaled exponential linear unit function element-wise.

```python
neurenix.nn.Softmax(dim=-1)
```

Applies the softmax function element-wise: Softmax(x_i) = exp(x_i) / sum_j(exp(x_j))

```python
neurenix.nn.GELU(approximate=False)
```

Applies the Gaussian Error Linear Unit (GELU) function element-wise: GELU(x) = x * Φ(x)

**Example:**
```python
import neurenix
from neurenix.nn import Linear, ReLU, Sigmoid

# Create a simple neural network with activation functions
model = neurenix.nn.Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 5),
    Sigmoid()
)

# Forward pass
input_tensor = neurenix.Tensor.randn((32, 10))
output = model(input_tensor)  # Shape: (32, 5)
```

### Loss Functions

```python
neurenix.nn.MSELoss(reduction='mean')
```

Mean Squared Error (MSE) loss. Measures the average squared difference between the predicted values and the target values.

```python
neurenix.nn.L1Loss(reduction='mean')
```

Mean Absolute Error (MAE) loss. Measures the average absolute difference between the predicted values and the target values.

```python
neurenix.nn.CrossEntropyLoss(weight=None, ignore_index=-100, reduction='mean')
```

Cross Entropy Loss. Combines LogSoftmax and NLLLoss in one single class.

```python
neurenix.nn.BCELoss(weight=None, reduction='mean')
```

Binary Cross Entropy Loss.

```python
neurenix.nn.BCEWithLogitsLoss(weight=None, pos_weight=None, reduction='mean')
```

Binary Cross Entropy with Logits Loss. Combines a Sigmoid layer and the BCELoss in one single class.

**Parameters:**
- `reduction`: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
- `weight`: A manual rescaling weight given to each class or batch element
- `ignore_index`: Specifies a target value that is ignored and does not contribute to the input gradient
- `pos_weight`: A weight of positive examples

**Example:**
```python
import neurenix
from neurenix.nn import Linear, CrossEntropyLoss

# Create a model and loss function
model = Linear(10, 5)
criterion = CrossEntropyLoss()

# Forward pass and loss calculation
input_tensor = neurenix.Tensor.randn((32, 10))
target = neurenix.Tensor.randint(0, 5, (32,))
output = model(input_tensor)
loss = criterion(output, target)
```

### Sequential Container

```python
neurenix.nn.Sequential(*args, **kwargs)
```

A sequential container for neural network modules. Modules will be added to it in the order they are passed in the constructor.

**Example:**
```python
import neurenix
from neurenix.nn import Linear, ReLU, Sequential

# Using Sequential with a list of modules
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 5)
)

# Using Sequential with named modules
model = Sequential({
    'fc1': Linear(10, 20),
    'relu': ReLU(),
    'fc2': Linear(20, 5)
})

# Forward pass
input_tensor = neurenix.Tensor.randn((32, 10))
output = model(input_tensor)  # Shape: (32, 5)
```

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Module System** | Hierarchical module system similar to PyTorch | Keras layers and models |
| **API Design** | Object-oriented, intuitive | Functional and object-oriented APIs |
| **Edge Optimization** | Native optimization for edge devices | TensorFlow Lite for edge devices |
| **Activation Functions** | Comprehensive set including modern functions like GELU | Comprehensive set through tf.keras.activations |
| **Recurrent Layers** | RNN, LSTM, GRU with bidirectional support | RNN, LSTM, GRU through tf.keras.layers |
| **Custom Layers** | Easy to create through Module subclassing | Requires custom layer implementation through tf.keras.layers.Layer |

Neurenix's neural network components offer a more intuitive, PyTorch-like API compared to TensorFlow's Keras API, making it easier for researchers and developers to create custom architectures. The native optimization for edge devices in Neurenix provides better performance on resource-constrained hardware compared to TensorFlow Lite, which is an add-on component rather than being integrated into the core architecture.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Module System** | Similar to PyTorch's nn.Module | nn.Module |
| **Edge Optimization** | Native optimization for edge devices | PyTorch Mobile for edge devices |
| **Hardware Support** | CPU, CUDA, ROCm, WebGPU | CPU, CUDA |
| **Activation Functions** | Comprehensive set including modern functions | Comprehensive set through torch.nn |
| **Recurrent Layers** | RNN, LSTM, GRU with bidirectional support | RNN, LSTM, GRU through torch.nn |
| **Multi-Language Architecture** | Rust/C++ core with Python interface | C++ core with Python interface |

Neurenix's neural network components are very similar to PyTorch's in terms of API design and functionality, making it easy for PyTorch users to transition to Neurenix. However, Neurenix extends hardware support to include ROCm and WebGPU, making it more versatile across different hardware platforms. The native edge device optimization in Neurenix also provides advantages over PyTorch Mobile, particularly for AI agent applications.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Neural Network Support** | Comprehensive deep learning capabilities | Limited neural network support through MLPClassifier/MLPRegressor |
| **GPU Acceleration** | Native support for multiple GPU types | No native GPU support |
| **Activation Functions** | Comprehensive set including modern functions | Limited set (identity, logistic, tanh, relu) |
| **Recurrent Layers** | RNN, LSTM, GRU with bidirectional support | No recurrent layer support |
| **Edge Device Support** | Native optimization for edge devices | No specific edge device support |
| **API Design** | Object-oriented, hierarchical | Estimator API pattern |

Neurenix provides much more comprehensive neural network capabilities compared to Scikit-Learn, which primarily focuses on traditional machine learning algorithms. While Scikit-Learn offers only basic neural network support through its MLPClassifier and MLPRegressor, Neurenix provides a full suite of deep learning components, including convolutional and recurrent layers, along with GPU acceleration and edge device optimization.

## Best Practices

### Creating Neural Networks

When creating neural networks in Neurenix, follow these best practices:

1. **Use the Module System**: Inherit from `Module` to create custom layers and models.
2. **Initialize Weights Properly**: Use appropriate initialization methods for weights to ensure stable training.
3. **Use Sequential for Simple Networks**: For linear pipelines of layers, use the `Sequential` container for cleaner code.
4. **Set Training Mode**: Use `model.train()` during training and `model.eval()` during evaluation.

```python
import neurenix
from neurenix.nn import Module, Linear, ReLU, Sequential

# Using Module inheritance for custom architecture
class MyModel(Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = Sequential(
            Linear(10, 20),
            ReLU(),
            Linear(20, 20),
            ReLU()
        )
        self.classifier = Linear(20, 5)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

# Create and use the model
model = MyModel()
model.train()  # Set to training mode
```

### Optimizing for Edge Devices

When deploying neural networks to edge devices, consider these optimizations:

1. **Reduce Model Size**: Use smaller architectures or techniques like pruning and quantization.
2. **Minimize Memory Usage**: Avoid creating unnecessary intermediate tensors.
3. **Use Appropriate Hardware Abstraction**: Leverage Neurenix's device abstraction to run on the most efficient available hardware.

```python
import neurenix
from neurenix import DeviceType, Device
from neurenix.nn import Sequential, Linear, ReLU

# Create a small model for edge devices
model = Sequential(
    Linear(10, 8),
    ReLU(),
    Linear(8, 5)
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
```

### Training Neural Networks

For effective training of neural networks:

1. **Choose Appropriate Loss Functions**: Select loss functions that match your task.
2. **Monitor Training Progress**: Track loss and metrics during training.
3. **Use Regularization**: Apply techniques like dropout to prevent overfitting.
4. **Adjust Learning Rate**: Start with a reasonable learning rate and adjust as needed.

```python
import neurenix
from neurenix.nn import Linear, ReLU, Sequential, MSELoss, Dropout
from neurenix.optim import Adam

# Create a model with dropout for regularization
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Dropout(0.2),
    Linear(20, 20),
    ReLU(),
    Dropout(0.2),
    Linear(20, 5)
)

# Set up loss function and optimizer
criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(10):
    # Forward pass
    output = model(input_data)
    loss = criterion(output, target_data)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

## Tutorials

### Creating a Simple Neural Network

```python
import neurenix
from neurenix.nn import Module, Linear, ReLU, Sequential, MSELoss
from neurenix.optim import SGD

# Create a simple neural network
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 5)
)

# Generate some dummy data
input_data = neurenix.Tensor.randn((100, 10))
target_data = neurenix.Tensor.randn((100, 5))

# Set up loss function and optimizer
criterion = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)

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

### Creating a Convolutional Neural Network

```python
import neurenix
from neurenix.nn import Module, Conv2d, Linear, ReLU, MaxPool2d, Sequential, CrossEntropyLoss
from neurenix.optim import Adam

# Create a CNN for image classification
class CNN(Module):
    def __init__(self):
        super().__init__()
        self.features = Sequential(
            Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2),
            Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = Sequential(
            Linear(32 * 7 * 7, 128),
            ReLU(),
            Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.classifier(x)
        return x

# Create the model
model = CNN()

# Generate some dummy data (28x28 RGB images)
input_data = neurenix.Tensor.randn((32, 3, 28, 28))
target_data = neurenix.Tensor.randint(0, 10, (32,))

# Set up loss function and optimizer
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(10):
    # Forward pass
    output = model(input_data)
    loss = criterion(output, target_data)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

### Creating a Recurrent Neural Network

```python
import neurenix
from neurenix.nn import Module, LSTM, Linear, Sequential, MSELoss
from neurenix.optim import Adam

# Create an RNN for sequence prediction
class RNN(Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        output, (h_n, c_n) = self.lstm(x)
        # Use the output of the last time step
        last_output = output[:, -1, :]
        return self.fc(last_output)

# Create the model
input_size = 10
hidden_size = 20
output_size = 5
model = RNN(input_size, hidden_size, output_size)

# Generate some dummy data (sequences of length 15)
batch_size = 32
seq_length = 15
input_data = neurenix.Tensor.randn((batch_size, seq_length, input_size))
target_data = neurenix.Tensor.randn((batch_size, output_size))

# Set up loss function and optimizer
criterion = MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
model.train()
for epoch in range(10):
    # Forward pass
    output = model(input_data)
    loss = criterion(output, target_data)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
```

## Conclusion

The Neural Networks module of Neurenix provides a comprehensive set of components for building and training neural networks. Its intuitive, PyTorch-like API makes it easy for researchers and developers to create custom architectures, while its multi-language architecture with a high-performance Rust/C++ core enables optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

Compared to other frameworks like TensorFlow, PyTorch, and Scikit-Learn, Neurenix's Neural Networks module offers advantages in terms of API design, edge device optimization, and hardware support. These features make Neurenix particularly well-suited for AI agent development, edge computing, and advanced learning paradigms.
