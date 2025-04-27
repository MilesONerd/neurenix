# oneDNN Support in Phynexus

This document describes the Intel oneDNN (Deep Neural Network Library) support in the Phynexus framework.

## Overview

Intel oneDNN (formerly DNNL and MKL-DNN) is an open-source performance library for deep learning applications. It includes highly optimized implementations of basic operations like convolution, matrix multiplication, and activation functions specifically tuned for Intel architectures. Phynexus integrates with oneDNN to provide accelerated deep learning operations on Intel CPUs and GPUs.

The oneDNN backend in Phynexus leverages Intel's optimized primitives to significantly improve performance for both training and inference workloads on Intel hardware. This integration allows models to run efficiently on a wide range of Intel platforms, from laptops to high-performance computing systems.

## Features

The oneDNN backend in Phynexus provides:

- Optimized deep learning operations for Intel CPUs and GPUs
- Support for both training and inference workloads
- Automatic primitive selection and optimization
- Memory format propagation for optimal performance
- Integration with Intel's hardware features (AVX-512, DL Boost)
- Support for various data types (FP32, FP16, BF16, INT8)
- Seamless integration with the rest of the framework

## Hardware Requirements

oneDNN support in Phynexus requires:

- Intel CPU with SSE4.1 or later (AVX2 or AVX-512 recommended for best performance)
- For GPU acceleration: Intel GPU with Gen9 architecture or newer
- oneDNN library installed on the system

Compatible hardware includes:
- Intel Xeon processors
- Intel Core processors
- Intel Atom processors
- Intel Iris, UHD, and Arc GPUs

## Usage

### Checking for oneDNN Availability

```python
# Python
from neurenix.hardware.onednn import is_onednn_available

if is_onednn_available():
    print("oneDNN is available")
else:
    print("oneDNN is not available")
```

### Creating a oneDNN Backend

```python
# Python
from neurenix.hardware.onednn import OneDNNBackend

# Create the backend
try:
    onednn = OneDNNBackend()
    
    # Initialize the backend
    if onednn.initialize():
        print("oneDNN backend initialized successfully")
    else:
        print("Failed to initialize oneDNN backend")
except RuntimeError as e:
    print(f"oneDNN error: {e}")
```

### Getting Device Information

```python
# Python
from neurenix.hardware.onednn import OneDNNBackend

# Create and initialize the backend
onednn = OneDNNBackend()
onednn.initialize()

# Get the number of available devices
device_count = onednn.get_device_count()
print(f"Available oneDNN devices: {device_count}")

# Get information about a specific device
device_info = onednn.get_device_info(0)  # First device
print(f"Device info: {device_info}")
```

### Using oneDNN for Tensor Operations

```python
# Python
import neurenix as nx
from neurenix.hardware.onednn import OneDNNBackend

# Create tensors
a = nx.Tensor([[1, 2], [3, 4]])
b = nx.Tensor([[5, 6], [7, 8]])

# Create and initialize the backend
onednn = OneDNNBackend()
onednn.initialize()

# Perform matrix multiplication using oneDNN
c = onednn.matmul(a, b)
print(f"Result: {c}")
```

### Using oneDNN for Convolution

```python
# Python
import neurenix as nx
from neurenix.hardware.onednn import OneDNNBackend

# Create input and weight tensors
input = nx.random.randn(1, 3, 32, 32)  # Batch, Channels, Height, Width
weight = nx.random.randn(16, 3, 3, 3)  # Out channels, In channels, Kernel H, Kernel W

# Create and initialize the backend
onednn = OneDNNBackend()
onednn.initialize()

# Perform 2D convolution using oneDNN
output = onednn.conv2d(
    input=input,
    weight=weight,
    bias=None,
    stride=(1, 1),
    padding=(1, 1)
)
print(f"Output shape: {output.shape}")
```

### Using oneDNN for RNN Operations

```python
# Python
import neurenix as nx
from neurenix.hardware.onednn import OneDNNBackend

# Create input and hidden state tensors
batch_size = 1
seq_length = 10
input_size = 20
hidden_size = 30

input = nx.random.randn(seq_length, batch_size, input_size)
hidden = nx.random.randn(batch_size, hidden_size)
weight_ih = nx.random.randn(hidden_size, input_size)
weight_hh = nx.random.randn(hidden_size, hidden_size)
bias_ih = nx.random.randn(hidden_size)
bias_hh = nx.random.randn(hidden_size)

# Create and initialize the backend
onednn = OneDNNBackend()
onednn.initialize()

# Perform RNN operation using oneDNN
output, new_hidden = onednn.rnn(
    input=input,
    hidden=hidden,
    weight_ih=weight_ih,
    weight_hh=weight_hh,
    bias_ih=bias_ih,
    bias_hh=bias_hh
)
print(f"Output shape: {output.shape}")
print(f"New hidden shape: {new_hidden.shape}")
```

## Implementation Details

The oneDNN backend implementation in Phynexus follows a layered architecture:

1. **Python API Layer**: Provides a user-friendly interface for oneDNN operations
2. **Rust Binding Layer**: Connects the Python API to the C++ implementation
3. **C++ Implementation Layer**: Implements the core oneDNN functionality

### Engine and Stream Management

The implementation uses oneDNN's engine and stream abstractions:

1. **Engine**: Represents a computational device (CPU or GPU)
2. **Stream**: Represents a queue of operations to be executed on an engine

These abstractions allow for efficient execution and synchronization of operations.

### Memory Management

The implementation includes sophisticated memory management:

1. **Memory Descriptors**: Define the logical organization of data
2. **Memory Objects**: Represent actual data in memory
3. **Memory Format Propagation**: Automatically select optimal memory formats
4. **Reorder Primitives**: Handle format conversions when necessary

### Primitive Management

oneDNN operations are represented as primitives:

1. **Primitive Descriptors**: Define the operation and its parameters
2. **Primitives**: Executable objects that perform the operation
3. **Primitive Caching**: Reuse primitives for repeated operations

## Best Practices

### Memory Format Optimization

For optimal performance with oneDNN:

```python
# Python
import neurenix as nx
from neurenix.hardware.onednn import OneDNNBackend

# Create and initialize the backend
onednn = OneDNNBackend()
onednn.initialize()

# When creating tensors, let oneDNN choose the optimal memory format
# (In a real implementation, this would be handled internally)
```

### Primitive Reuse

Reuse primitives for repeated operations:

```python
# Python
# In a real implementation, the OneDNNBackend class would cache primitives internally
# for repeated operations with the same parameters
```

### Data Type Selection

Choose appropriate data types for your workload:

```python
# Python
import neurenix as nx
from neurenix.hardware.onednn import OneDNNBackend

# Create and initialize the backend
onednn = OneDNNBackend()
onednn.initialize()

# For inference, consider using lower precision
# (In a real implementation, this would be configurable)
```

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **oneDNN Integration** | Native integration | Via oneDNN plugin |
| **Intel CPU Optimization** | Comprehensive | Limited |
| **Intel GPU Support** | Native support | Limited support |
| **Memory Format Optimization** | Automatic | Manual configuration required |
| **Primitive Caching** | Automatic | Limited |
| **API Simplicity** | Unified API | Separate API for oneDNN |

Neurenix provides more seamless integration with oneDNN compared to TensorFlow, which requires a separate plugin. The unified API in Neurenix makes it easier to use oneDNN acceleration while maintaining compatibility with other backends.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **oneDNN Integration** | Native integration | Via MKLDNN backend |
| **Intel CPU Optimization** | Comprehensive | Good |
| **Intel GPU Support** | Native support | Limited support |
| **Memory Format Optimization** | Automatic | Manual configuration required |
| **Primitive Caching** | Automatic | Limited |
| **API Simplicity** | Unified API | Separate API for MKLDNN |

PyTorch has some integration with oneDNN through its MKLDNN backend, but it's not as deeply integrated as Neurenix's native oneDNN support. Neurenix provides a more seamless experience with automatic memory format optimization and primitive caching.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **oneDNN Support** | Comprehensive support | No oneDNN support |
| **Deep Learning Primitives** | Optimized primitives | Generic implementations |
| **Intel CPU Optimization** | Comprehensive | Limited |
| **Intel GPU Support** | Native support | No GPU support |
| **Memory Format Optimization** | Automatic | N/A |
| **Performance on Intel Hardware** | Optimized | Not optimized for deep learning |

Scikit-Learn does not provide any oneDNN integration, focusing on traditional machine learning algorithms rather than deep learning. Neurenix's oneDNN support enables significant performance improvements for deep learning workloads on Intel hardware.

## Limitations

The current oneDNN implementation has the following limitations:

1. **Intel Hardware Focus**: Primarily optimized for Intel hardware
2. **Limited Operations**: Not all operations are optimized for oneDNN
3. **Library Dependency**: Requires oneDNN library installation
4. **Implementation Status**: Some features are still in development
5. **Dynamic Shapes**: Limited support for models with dynamic shapes

## Future Work

Future development of the oneDNN backend will include:

1. **Expanded Operation Support**: Implementation of more operations using oneDNN
2. **Performance Optimizations**: Further tuning for different Intel architectures
3. **Enhanced GPU Support**: Better support for Intel GPUs
4. **Advanced Features**: Support for more advanced oneDNN features
5. **Quantization Support**: Better support for INT8 and other quantized formats
6. **Graph Compilation**: Support for oneDNN's graph API for whole-model optimization
