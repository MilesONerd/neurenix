# oneAPI Support in Phynexus

This document describes the Intel oneAPI support in the Phynexus framework.

## Overview

Intel oneAPI is a cross-architecture programming model that enables developers to build applications that seamlessly target CPUs, GPUs, FPGAs, and other accelerators. Phynexus includes oneAPI support to provide optimized hardware acceleration on Intel platforms, enabling high-performance computation across Intel's diverse hardware portfolio.

The oneAPI backend in Phynexus leverages the SYCL programming model and Intel's performance libraries to provide efficient execution on Intel hardware, from client devices to high-performance computing systems. This integration allows Phynexus to take full advantage of Intel's hardware capabilities while maintaining a consistent programming interface.

## Features

The oneAPI backend in Phynexus provides:

- Hardware acceleration on Intel CPUs, GPUs, and FPGAs
- Cross-architecture support through a single programming model
- Integration with Intel's optimized libraries (oneMKL, oneDNN)
- Automatic device discovery and selection
- Support for common deep learning operations
- Seamless integration with the rest of the framework
- Performance optimizations specific to Intel hardware

## Hardware Requirements

oneAPI support in Phynexus requires:

- Intel oneAPI Base Toolkit installed on the system
- Compatible Intel hardware with appropriate drivers

Compatible hardware includes:
- Intel CPUs (Core, Xeon)
- Intel GPUs (Iris Xe, Arc)
- Intel FPGAs (Arria, Stratix)
- Future Intel accelerators

## Usage

### Checking for oneAPI Availability

```python
# Python
from neurenix.hardware.oneapi import is_oneapi_available

if is_oneapi_available():
    print("oneAPI is available")
else:
    print("oneAPI is not available")
```

### Creating a oneAPI Backend

```python
# Python
from neurenix.hardware.oneapi import OneAPIBackend

# Create the backend
try:
    oneapi = OneAPIBackend()
    
    # Initialize the backend
    if oneapi.initialize():
        print("oneAPI backend initialized successfully")
    else:
        print("Failed to initialize oneAPI backend")
except RuntimeError as e:
    print(f"oneAPI error: {e}")
```

### Getting Device Information

```python
# Python
from neurenix.hardware.oneapi import OneAPIBackend

# Create and initialize the backend
oneapi = OneAPIBackend()
oneapi.initialize()

# Get the number of available devices
device_count = oneapi.get_device_count()
print(f"Available oneAPI devices: {device_count}")

# Get information about a specific device
device_info = oneapi.get_device_info(0)  # First device
print(f"Device info: {device_info}")
```

### Using oneAPI for Tensor Operations

```python
# Python
import neurenix as nx
from neurenix.hardware.oneapi import OneAPIBackend

# Create tensors
a = nx.Tensor([[1, 2], [3, 4]])
b = nx.Tensor([[5, 6], [7, 8]])

# Create and initialize the backend
oneapi = OneAPIBackend()
oneapi.initialize()

# Perform matrix multiplication using oneAPI
c = oneapi.matmul(a, b)
print(f"Result: {c}")
```

### Using oneAPI for Convolution

```python
# Python
import neurenix as nx
from neurenix.hardware.oneapi import OneAPIBackend

# Create input and weight tensors
input = nx.random.randn(1, 3, 32, 32)  # Batch, Channels, Height, Width
weight = nx.random.randn(16, 3, 3, 3)  # Out channels, In channels, Kernel H, Kernel W

# Create and initialize the backend
oneapi = OneAPIBackend()
oneapi.initialize()

# Perform 2D convolution using oneAPI
output = oneapi.conv2d(
    input=input,
    weight=weight,
    bias=None,
    stride=(1, 1),
    padding=(1, 1)
)
print(f"Output shape: {output.shape}")
```

## Implementation Details

The oneAPI backend implementation in Phynexus follows a layered architecture:

1. **Python API Layer**: Provides a user-friendly interface for oneAPI operations
2. **Rust Binding Layer**: Connects the Python API to the C++ implementation
3. **C++ Implementation Layer**: Implements the core oneAPI functionality using SYCL

### SYCL Integration

The implementation uses the SYCL programming model to:

1. Discover and select Intel devices
2. Create queues for command submission
3. Manage memory across different devices
4. Execute kernels on the selected devices
5. Coordinate data transfer between host and devices

### Intel Library Integration

The oneAPI backend integrates with Intel's optimized libraries:

1. **oneMKL**: For optimized linear algebra operations
2. **oneDNN**: For optimized deep neural network primitives
3. **oneDAL**: For data analytics algorithms
4. **oneTBB**: For parallel programming patterns

### Automatic Fallback

The oneAPI backend includes automatic fallback to CPU implementations when:
- oneAPI is not available on the system
- The operation is not supported by oneAPI
- An error occurs during oneAPI execution

This ensures that code written to use oneAPI will still work on systems without oneAPI support, albeit with reduced performance.

## Best Practices

### Device Selection

When multiple oneAPI-compatible devices are available:

```python
# Python
from neurenix.hardware.oneapi import OneAPIBackend

# Create the backend
oneapi = OneAPIBackend()
oneapi.initialize()

# Get the number of available devices
device_count = oneapi.get_device_count()

# Get information about all devices
for i in range(device_count):
    device_info = oneapi.get_device_info(i)
    print(f"Device {i}: {device_info}")

# Select the best device based on your requirements
# (In a real implementation, you would choose based on device capabilities)
```

### Memory Management

For optimal performance with oneAPI:

1. **Unified Shared Memory (USM)**: Use USM for simplified memory management
2. **Minimize Host-Device Transfers**: Keep data on the device as much as possible
3. **Reuse Buffers**: Avoid creating new buffers for intermediate results
4. **Batch Operations**: Group operations together when possible

### Performance Optimization

For best performance with oneAPI:

1. **Use Intel Libraries**: Leverage oneMKL and oneDNN for optimized implementations
2. **Appropriate Data Types**: Use FP16 or BF16 when precision requirements allow
3. **Vectorization**: Ensure operations can be vectorized
4. **Work-Group Size**: Choose appropriate work-group sizes for your device

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **oneAPI Support** | Native integration | Limited support via plugins |
| **Intel Hardware Support** | CPUs, GPUs, FPGAs | Primarily CPUs |
| **SYCL Integration** | Comprehensive | Limited |
| **Intel Library Integration** | oneMKL, oneDNN | Limited integration |
| **Fallback Mechanism** | Automatic CPU fallback | Manual fallback required |
| **API Simplicity** | Unified API | Separate API for different backends |

Neurenix provides more comprehensive oneAPI support compared to TensorFlow, which has limited support for Intel hardware acceleration. The unified API in Neurenix makes it easier to use oneAPI acceleration while maintaining compatibility with other backends.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **oneAPI Support** | Native integration | Limited support via Intel extensions |
| **Intel Hardware Support** | CPUs, GPUs, FPGAs | Primarily CPUs |
| **SYCL Integration** | Comprehensive | Limited |
| **Intel Library Integration** | oneMKL, oneDNN | Limited integration |
| **Fallback Mechanism** | Automatic CPU fallback | Manual fallback required |
| **API Simplicity** | Unified API | Separate API for different backends |

PyTorch has some support for Intel hardware through Intel extensions, but it's not as deeply integrated as Neurenix's native oneAPI support. Neurenix provides a more seamless experience with automatic fallback to CPU when needed.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **oneAPI Support** | Comprehensive support | Limited support via Intel extensions |
| **Hardware Acceleration** | Multiple acceleration options | Primarily CPU |
| **Intel GPU Support** | Native support | No GPU support |
| **Intel FPGA Support** | Native support | No FPGA support |
| **Performance on Intel Hardware** | Optimized for Intel hardware | Limited optimization |
| **API Consistency** | Consistent API across devices | Different API for accelerated code |

Scikit-Learn has some Intel optimizations through Intel extensions, but these are primarily focused on CPU performance. Neurenix's oneAPI support enables acceleration across Intel's diverse hardware portfolio, including GPUs and FPGAs.

## Limitations

The current oneAPI implementation has the following limitations:

1. **Intel Hardware Focus**: Primarily optimized for Intel hardware
2. **Limited Operations**: Not all operations are optimized for oneAPI
3. **Toolkit Requirement**: Requires Intel oneAPI Base Toolkit installation
4. **Driver Compatibility**: Requires up-to-date Intel drivers
5. **Implementation Status**: Some features are still in development

## Future Work

Future development of the oneAPI backend will include:

1. **Expanded Operation Support**: Implementation of more operations using oneAPI
2. **Performance Optimizations**: Further tuning for different Intel architectures
3. **Enhanced Integration**: Deeper integration with Intel's performance libraries
4. **Support for New Hardware**: Integration with future Intel accelerators
5. **Advanced Features**: Support for advanced oneAPI features like Level Zero
6. **Distributed Computing**: Support for multi-device and multi-node computation
