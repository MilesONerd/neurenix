# TensorRT Support in Phynexus

This document describes the NVIDIA TensorRT support in the Phynexus framework.

## Overview

NVIDIA TensorRT is a high-performance deep learning inference optimizer and runtime that delivers low latency and high throughput for deep learning inference applications. Phynexus integrates with TensorRT to provide accelerated inference on NVIDIA GPUs, enabling significant performance improvements for deployed models.

The TensorRT backend in Phynexus leverages NVIDIA's optimized runtime to apply optimizations such as layer fusion, precision calibration, kernel auto-tuning, and dynamic tensor memory management. These optimizations allow models to run efficiently on NVIDIA GPUs, making it ideal for production deployment of deep learning models.

## Features

The TensorRT backend in Phynexus provides:

- High-performance inference on NVIDIA GPUs
- Model optimization through layer fusion and other techniques
- Support for multiple precision modes (FP32, FP16, INT8)
- Dynamic tensor memory allocation
- Automatic kernel selection and tuning
- Integration with CUDA for efficient GPU execution
- Seamless integration with the rest of the framework

## Hardware Requirements

TensorRT support in Phynexus requires:

- NVIDIA GPU with compute capability 3.0 or higher (Pascal, Volta, Turing, Ampere, or newer architectures recommended)
- CUDA Toolkit 10.0 or later
- cuDNN 7.5 or later
- TensorRT 7.0 or later

Compatible hardware includes:
- NVIDIA GeForce RTX series
- NVIDIA Tesla series
- NVIDIA Quadro series
- NVIDIA A100, A10, A30, A40
- NVIDIA T4, V100, P100

## Usage

### Checking for TensorRT Availability

```python
# Python
from neurenix.hardware.tensorrt import is_tensorrt_available

if is_tensorrt_available():
    print("TensorRT is available")
else:
    print("TensorRT is not available")
```

### Creating a TensorRT Backend

```python
# Python
from neurenix.hardware.tensorrt import TensorRTBackend

# Create the backend
try:
    tensorrt = TensorRTBackend()
    
    # Initialize the backend
    if tensorrt.initialize():
        print("TensorRT backend initialized successfully")
    else:
        print("Failed to initialize TensorRT backend")
except RuntimeError as e:
    print(f"TensorRT error: {e}")
```

### Getting Device Information

```python
# Python
from neurenix.hardware.tensorrt import TensorRTBackend

# Create and initialize the backend
tensorrt = TensorRTBackend()
tensorrt.initialize()

# Get the number of available devices
device_count = tensorrt.get_device_count()
print(f"Available TensorRT devices: {device_count}")

# Get information about a specific device
device_info = tensorrt.get_device_info(0)  # First device
print(f"Device info: {device_info}")
```

### Optimizing a Model with TensorRT

```python
# Python
import neurenix as nx
from neurenix.hardware.tensorrt import TensorRTBackend
from neurenix.nn import Sequential, Conv2d, ReLU, Linear

# Create a model
model = Sequential(
    Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    ReLU(),
    Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
    ReLU(),
    Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
    ReLU(),
    Linear(64 * 8 * 8, 10)
)

# Create and initialize the backend
tensorrt = TensorRTBackend()
tensorrt.initialize()

# Define input shapes
input_shapes = {"input": (1, 3, 32, 32)}

# Optimize the model with TensorRT
optimized_model = tensorrt.optimize_model(
    model=model,
    input_shapes=input_shapes,
    precision="fp16",
    workspace_size=1 << 30  # 1 GB
)

print("Model optimized with TensorRT")
```

### Running Inference with TensorRT

```python
# Python
import neurenix as nx
from neurenix.hardware.tensorrt import TensorRTBackend

# Create input tensor
input_tensor = nx.random.randn(1, 3, 32, 32)

# Create and initialize the backend
tensorrt = TensorRTBackend()
tensorrt.initialize()

# Run inference using TensorRT
outputs = tensorrt.inference(
    model=optimized_model,
    inputs={"input": input_tensor}
)

print(f"Output shape: {outputs['output'].shape}")
```

## Implementation Details

The TensorRT backend implementation in Phynexus follows a layered architecture:

1. **Python API Layer**: Provides a user-friendly interface for TensorRT operations
2. **Rust Binding Layer**: Connects the Python API to the C++ implementation
3. **C++ Implementation Layer**: Implements the core TensorRT functionality

### TensorRT Integration

The implementation uses the TensorRT API to:

1. Create a TensorRT builder, network, and config
2. Parse and optimize models for inference
3. Build optimized TensorRT engines
4. Execute inference using TensorRT runtime
5. Manage memory and resources efficiently

### Builder and Engine Management

The TensorRT backend manages the following components:

1. **Builder**: Creates optimized TensorRT engines from network definitions
2. **Network**: Represents the model's computational graph
3. **Config**: Specifies optimization parameters like precision and workspace size
4. **Engine**: Optimized runtime for executing the model
5. **Context**: Execution context for running inference

### Precision Modes

The implementation supports multiple precision modes:

1. **FP32**: Full precision for maximum accuracy
2. **FP16**: Half precision for improved performance with minimal accuracy loss
3. **INT8**: 8-bit integer quantization for maximum performance (requires calibration)

## Best Practices

### Precision Selection

Choose the appropriate precision for your workload:

```python
# Python
from neurenix.hardware.tensorrt import TensorRTBackend

# Create and initialize the backend
tensorrt = TensorRTBackend()
tensorrt.initialize()

# For maximum performance with acceptable accuracy loss
optimized_model = tensorrt.optimize_model(
    model=model,
    input_shapes=input_shapes,
    precision="fp16"  # Use FP16 for good balance of performance and accuracy
)

# For maximum performance where accuracy is less critical
optimized_model = tensorrt.optimize_model(
    model=model,
    input_shapes=input_shapes,
    precision="int8"  # Use INT8 for maximum performance
)

# For maximum accuracy
optimized_model = tensorrt.optimize_model(
    model=model,
    input_shapes=input_shapes,
    precision="fp32"  # Use FP32 for maximum accuracy
)
```

### Workspace Size

Allocate sufficient workspace for TensorRT:

```python
# Python
from neurenix.hardware.tensorrt import TensorRTBackend

# Create and initialize the backend
tensorrt = TensorRTBackend()
tensorrt.initialize()

# Allocate a large workspace for better optimization
optimized_model = tensorrt.optimize_model(
    model=model,
    input_shapes=input_shapes,
    workspace_size=2 << 30  # 2 GB
)
```

### Fixed Input Shapes

For best performance, use fixed input shapes:

```python
# Python
from neurenix.hardware.tensorrt import TensorRTBackend

# Create and initialize the backend
tensorrt = TensorRTBackend()
tensorrt.initialize()

# Specify exact input shapes for optimal performance
input_shapes = {"input": (1, 3, 224, 224)}  # Fixed batch size and dimensions

optimized_model = tensorrt.optimize_model(
    model=model,
    input_shapes=input_shapes
)
```

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **TensorRT Integration** | Native integration | Via TF-TRT plugin |
| **Optimization Process** | Seamless | Requires manual configuration |
| **Precision Control** | Simple API | Complex configuration |
| **Model Compatibility** | High compatibility | Limited compatibility |
| **API Simplicity** | Unified API | Separate API for TensorRT |
| **Integration with Framework** | Fully integrated | Plugin-based integration |

Neurenix provides more seamless integration with TensorRT compared to TensorFlow, which requires a separate TF-TRT plugin. The unified API in Neurenix makes it easier to optimize models with TensorRT while maintaining compatibility with other backends.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **TensorRT Integration** | Native integration | Via torch-tensorrt |
| **Optimization Process** | Seamless | Requires manual configuration |
| **Precision Control** | Simple API | Complex configuration |
| **Model Compatibility** | High compatibility | Limited compatibility |
| **API Simplicity** | Unified API | Separate API for TensorRT |
| **Integration with Framework** | Fully integrated | Plugin-based integration |

PyTorch requires the torch-tensorrt package for TensorRT integration, which introduces a separate API and workflow. Neurenix's native TensorRT support provides a more integrated experience with simpler model optimization and inference.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **TensorRT Support** | Comprehensive support | No TensorRT support |
| **GPU Acceleration** | Native GPU support | Limited GPU support |
| **Model Optimization** | Automatic optimization | No deep learning optimization |
| **Inference Performance** | Optimized for inference | Not optimized for deep learning |
| **Precision Control** | Multiple precision options | No precision control |
| **Deep Learning Support** | Comprehensive | Limited |

Scikit-Learn does not provide any TensorRT integration, focusing on traditional machine learning algorithms rather than deep learning. Neurenix's TensorRT support enables significant performance improvements for deep learning inference on NVIDIA GPUs.

## Limitations

The current TensorRT implementation has the following limitations:

1. **NVIDIA GPUs Only**: Only works with NVIDIA GPUs
2. **Model Compatibility**: Not all model architectures are supported
3. **Dynamic Shapes**: Limited support for models with dynamic input shapes
4. **Custom Operations**: May require custom implementation for unsupported operations
5. **Library Dependencies**: Requires TensorRT, CUDA, and cuDNN installations
6. **Implementation Status**: Some features are still in development

## Future Work

Future development of the TensorRT backend will include:

1. **Expanded Model Support**: Support for more model architectures
2. **Dynamic Shape Support**: Better handling of dynamic input shapes
3. **Custom Operation Support**: Easier integration of custom operations
4. **INT8 Calibration**: Improved tools for INT8 calibration
5. **TensorRT 8.x Features**: Support for newer TensorRT features
6. **Automatic Fallback**: Graceful fallback for unsupported operations
