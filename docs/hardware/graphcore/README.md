# GraphCore IPU Support in Phynexus

This document describes the GraphCore Intelligent Processing Unit (IPU) support in the Phynexus framework.

## Overview

GraphCore IPUs are specialized processors designed specifically for machine learning workloads, offering high performance and efficiency for both training and inference. Phynexus includes native support for GraphCore IPUs, enabling accelerated execution of AI models on these dedicated AI processors.

The IPU architecture is fundamentally different from GPUs and CPUs, with a massively parallel, memory-centric design optimized for the fine-grained parallelism found in modern AI workloads. Phynexus leverages these unique capabilities through a dedicated backend that optimizes models for IPU execution.

## Features

The GraphCore IPU backend in Phynexus provides:

- Support for multiple IPUs with automatic scaling
- Mixed-precision training and inference (float16/float32)
- Model compilation and optimization for IPU execution
- Memory management optimized for IPU architecture
- Integration with the Poplar SDK
- Automatic device selection and configuration
- Support for IPU-specific optimizations

## Hardware Requirements

Phynexus supports various GraphCore IPU systems:

- GraphCore C2 IPU Processor Cards (PCIe)
- GraphCore Bow IPU Processor Cards
- GraphCore IPU-POD systems
- GraphCore IPU-M2000 systems

## Usage

### Basic IPU Manager

```python
# Python
from neurenix.hardware.graphcore import GraphCoreManager

# Create an IPU manager
ipu_manager = GraphCoreManager(
    num_ipus=2,                  # Number of IPUs to use
    precision="float16",         # Precision to use
    memory_proportion=0.6,       # Proportion of IPU memory to use for model
    enable_half_partials=True,   # Use half-precision for partial results
    compile_only=False           # Whether to compile without executing
)

# Initialize the IPU environment
ipu_manager.initialize()

# Get information about available IPUs
ipu_info = ipu_manager.get_ipu_info()
print(f"IPU information: {ipu_info}")

# Compile a model for IPU execution
compiled_model = ipu_manager.compile_model(model, example_inputs)

# Execute the model on IPU
outputs = ipu_manager.execute_model(compiled_model, inputs)

# Clean up
ipu_manager.finalize()
```

### Using the Global Manager

```python
# Python
from neurenix.hardware.graphcore import get_graphcore_manager

# Get the global GraphCore IPU manager
ipu_manager = get_graphcore_manager()

# Get the number of available IPUs
ipu_count = ipu_manager.get_ipu_count()
print(f"Available IPUs: {ipu_count}")

# Optimize a model for IPU execution
optimized_model = ipu_manager.optimize_model(model, example_inputs)

# Execute the optimized model
outputs = ipu_manager.execute_model(optimized_model, inputs)
```

### Using Context Manager

```python
# Python
from neurenix.hardware.graphcore import GraphCoreManager

# Use the IPU manager as a context manager
with GraphCoreManager(num_ipus=4, precision="float16") as ipu:
    # The IPU environment is automatically initialized
    
    # Compile and execute a model
    compiled_model = ipu.compile_model(model, example_inputs)
    outputs = ipu.execute_model(compiled_model, inputs)
    
    # The IPU environment is automatically finalized when exiting the context
```

## Implementation Details

The GraphCore IPU backend implementation in Phynexus follows a layered architecture:

1. **Python API Layer**: Provides a user-friendly interface for IPU operations
2. **Rust Binding Layer**: Connects the Python API to the C++ implementation
3. **C++ Implementation Layer**: Implements the core IPU functionality using the Poplar SDK

### Poplar Integration

The implementation uses the GraphCore Poplar SDK to:

1. Create and manage IPU devices
2. Compile models for IPU execution
3. Optimize memory usage and data flow
4. Execute models on IPUs
5. Manage multi-IPU configurations

### Memory Management

The IPU architecture uses a unique approach to memory, with distributed In-Processor Memory (IPM) rather than a traditional memory hierarchy. The Phynexus implementation optimizes memory usage through:

1. Efficient tensor placement
2. Memory proportion control
3. Recomputation of activations when beneficial
4. Optimized data transfer between host and IPU

### Precision Control

The implementation supports multiple precision modes:

1. **float32**: Full precision for maximum accuracy
2. **float16**: Half precision for improved performance
3. **Mixed precision**: Using half-precision for weights and activations but higher precision for sensitive operations

## Best Practices

### Model Optimization

For optimal performance on GraphCore IPUs:

```python
# Python
from neurenix.hardware.graphcore import GraphCoreManager

# Create an IPU manager with optimization settings
ipu_manager = GraphCoreManager(
    num_ipus=2,
    precision="float16",         # Use half precision for better performance
    memory_proportion=0.6,       # Balance between available memory and recomputation
    enable_half_partials=True    # Use half-precision for partial results
)

# Optimize the model for IPU execution
optimized_model = ipu_manager.optimize_model(model, example_inputs)
```

### Multi-IPU Configuration

When working with multiple IPUs:

```python
# Python
from neurenix.hardware.graphcore import GraphCoreManager

# Create a multi-IPU manager
ipu_manager = GraphCoreManager(
    num_ipus=4,                  # Use 4 IPUs
    precision="float16",
    memory_proportion=0.6
)

# The model will automatically be distributed across the IPUs
compiled_model = ipu_manager.compile_model(model, example_inputs)
```

### Batch Size Selection

IPUs perform best with specific batch sizes:

```python
# Python
# For IPU-optimized batch sizes, use powers of 2 that fit in IPU memory
batch_sizes = [16, 32, 64, 128]  # Example batch sizes to try

# Find the optimal batch size for your model and IPU configuration
for batch_size in batch_sizes:
    try:
        # Create example inputs with this batch size
        example_inputs = create_example_inputs(batch_size)
        
        # Try to compile the model with this batch size
        compiled_model = ipu_manager.compile_model(model, example_inputs)
        
        print(f"Successfully compiled with batch size {batch_size}")
        break
    except Exception as e:
        print(f"Failed with batch size {batch_size}: {e}")
```

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **IPU Support** | Native integration | Requires TensorFlow-Poplar plugin |
| **Multi-IPU Support** | Automatic scaling | Manual configuration required |
| **Precision Control** | Flexible precision options | Limited precision control |
| **Memory Management** | Automatic memory optimization | Manual memory configuration |
| **Model Optimization** | Built-in model optimization | Requires manual optimization |
| **API Simplicity** | Unified API for IPU operations | Complex integration with TF API |

Neurenix provides more seamless integration with GraphCore IPUs compared to TensorFlow, which requires a separate plugin and more manual configuration. The unified API in Neurenix makes it easier to optimize and deploy models on IPUs.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **IPU Support** | Native integration | Requires PopTorch plugin |
| **Multi-IPU Support** | Automatic scaling | Manual configuration required |
| **Precision Control** | Flexible precision options | Limited precision control |
| **Memory Management** | Automatic memory optimization | Manual memory configuration |
| **Model Optimization** | Built-in model optimization | Requires manual optimization |
| **API Simplicity** | Unified API for IPU operations | Separate API for IPU operations |

PyTorch requires the PopTorch plugin for IPU support, which introduces a separate API and workflow. Neurenix's native IPU support provides a more integrated experience with automatic optimization and scaling.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **IPU Support** | Comprehensive IPU support | No IPU support |
| **Hardware Acceleration** | Multiple acceleration options | CPU only |
| **Model Compilation** | Built-in model compilation for IPUs | No hardware compilation |
| **Precision Control** | Multiple precision options | Limited precision control |
| **Memory Management** | Optimized for IPU architecture | No hardware-specific optimization |
| **Distributed Training** | Support for multi-IPU training | Limited distributed training support |

Scikit-Learn does not provide any IPU or hardware acceleration support, focusing solely on CPU execution. Neurenix's IPU support enables significant performance improvements for suitable workloads.

## Limitations

The current GraphCore IPU implementation has the following limitations:

1. **Model Size**: Very large models may not fit on a single IPU
2. **Dynamic Shapes**: Models with dynamic shapes may have limited support
3. **Custom Operations**: Some custom operations may not be optimized for IPUs
4. **Memory Constraints**: IPUs have limited memory compared to some GPUs
5. **Software Compatibility**: Requires specific versions of the Poplar SDK

## Future Work

Future development of the GraphCore IPU backend will include:

1. **Improved Model Compatibility**: Support for more model architectures
2. **Enhanced Multi-IPU Scaling**: Better automatic distribution of models across IPUs
3. **Advanced Memory Optimization**: More sophisticated memory management techniques
4. **IPU-Specific Operators**: Custom operators optimized for IPU architecture
5. **Gradient Accumulation**: Better support for training with accumulated gradients
6. **Pipelining Improvements**: Enhanced model pipelining across multiple IPUs
