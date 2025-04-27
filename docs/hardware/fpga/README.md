# FPGA Support in Phynexus

This document describes the FPGA (Field-Programmable Gate Array) support in the Phynexus framework.

## Overview

FPGAs are reconfigurable hardware devices that can be programmed to implement custom digital circuits, making them ideal for accelerating specific computational tasks. Phynexus includes support for FPGA acceleration through multiple frameworks, enabling high-performance, energy-efficient execution of AI workloads.

The FPGA support in Phynexus is designed to be flexible and extensible, supporting various FPGA vendors and programming models while providing a unified API for developers.

## Features

The FPGA backend in Phynexus provides:

- Support for multiple FPGA frameworks:
  - OpenCL for cross-vendor FPGA programming
  - Xilinx Vitis for Xilinx FPGAs
  - Intel OpenVINO for Intel FPGAs
- Model compilation and optimization for FPGA execution
- Custom kernel creation and execution
- Bitstream loading and management
- Automatic device selection and configuration
- Precision control (float32, float16, int8)
- Performance optimization options

## Hardware Requirements

Phynexus supports various FPGA devices:

### Xilinx FPGAs
- Alveo U200, U250, U280
- Versal AI Core series
- Kintex and Virtex UltraScale+

### Intel FPGAs
- Arria 10 GX
- Stratix 10 GX
- Agilex series

### Other Vendors
- Any FPGA with OpenCL support

## Usage

### Basic FPGA Manager

```python
# Python
from neurenix.hardware.fpga import FPGAManager

# Create an FPGA manager
fpga_manager = FPGAManager(
    framework="opencl",  # Options: "opencl", "vitis", "openvino"
    precision="float32", # Options: "float32", "float16", "int8"
    optimize_for="throughput"  # Options: "throughput", "latency", "power"
)

# Initialize the FPGA environment
fpga_manager.initialize()

# Get information about available FPGAs
fpga_info = fpga_manager.get_fpga_info()
print(f"FPGA information: {fpga_info}")

# Compile a model for FPGA execution
compiled_model = fpga_manager.compile_model(model, example_inputs)

# Execute the model on FPGA
outputs = fpga_manager.execute_model(compiled_model, inputs)

# Clean up
fpga_manager.finalize()
```

### Using OpenCL for FPGA

```python
# Python
from neurenix.hardware.fpga import OpenCLManager

# Create an OpenCL FPGA manager
opencl_manager = OpenCLManager(
    precision="float32",
    optimize_for="throughput"
)

# Get available platforms
platforms = opencl_manager.get_platforms()
print(f"Available OpenCL platforms: {platforms}")

# Get devices for a specific platform
devices = opencl_manager.get_devices(platform_id=0)
print(f"Available OpenCL devices: {devices}")

# Create and execute a kernel
kernel = opencl_manager.create_kernel(
    kernel_name="vector_add",
    kernel_source="""
    __kernel void vector_add(__global const float* a, __global const float* b, __global float* c) {
        int gid = get_global_id(0);
        c[gid] = a[gid] + b[gid];
    }
    """
)

# Execute the kernel
opencl_manager.execute_kernel(
    kernel=kernel,
    global_size=(1024,),
    local_size=(64,),
    args=[input_a, input_b, output]
)
```

### Using Xilinx Vitis for FPGA

```python
# Python
from neurenix.hardware.fpga import VitisManager

# Create a Vitis FPGA manager
vitis_manager = VitisManager(
    target_device="u250",
    precision="float32",
    optimize_for="throughput"
)

# Load an XCLBIN file
vitis_manager.load_xclbin("/path/to/accelerator.xclbin")

# Get available kernels
kernels = vitis_manager.get_kernels()
print(f"Available kernels: {kernels}")

# Execute a kernel
vitis_manager.execute_kernel(
    kernel_name="matrix_multiply",
    args=[input_a, input_b, output]
)
```

### Using Intel OpenVINO for FPGA

```python
# Python
from neurenix.hardware.fpga import OpenVINOManager

# Create an OpenVINO FPGA manager
openvino_manager = OpenVINOManager(
    precision="float32",
    optimize_for="throughput",
    cache_dir="/tmp/openvino_cache"
)

# Load a model
model = openvino_manager.load_model(
    model_path="/path/to/model.xml",
    weights_path="/path/to/model.bin"
)

# Get input and output information
input_info = openvino_manager.get_input_info(model)
output_info = openvino_manager.get_output_info(model)

print(f"Input info: {input_info}")
print(f"Output info: {output_info}")

# Run inference
outputs = openvino_manager.infer(model, {"input": input_tensor})
```

### Using Context Manager

```python
# Python
from neurenix.hardware.fpga import FPGAManager

# Use the FPGA manager as a context manager
with FPGAManager(framework="opencl") as fpga:
    # The FPGA environment is automatically initialized
    
    # Compile and execute a model
    compiled_model = fpga.compile_model(model, example_inputs)
    outputs = fpga.execute_model(compiled_model, inputs)
    
    # The FPGA environment is automatically finalized when exiting the context
```

## Implementation Details

The FPGA backend implementation in Phynexus follows a layered architecture:

1. **Python API Layer**: Provides a user-friendly interface for FPGA operations
2. **Rust Binding Layer**: Connects the Python API to the C++ implementation
3. **C++ Implementation Layer**: Implements the core FPGA functionality

### OpenCL Implementation

The OpenCL implementation uses the OpenCL API to interact with FPGA devices from various vendors. It provides:

- Platform and device discovery
- Context and command queue management
- Kernel compilation and execution
- Memory management for buffers and images

### Xilinx Vitis Implementation

The Xilinx Vitis implementation uses the Vitis AI and Vitis HLS tools to:

- Compile models for Xilinx FPGAs
- Load and manage XCLBIN files
- Execute kernels on the FPGA
- Optimize data transfers between host and device

### Intel OpenVINO Implementation

The Intel OpenVINO implementation uses the OpenVINO toolkit to:

- Compile models for Intel FPGAs
- Optimize models for FPGA execution
- Run inference on the FPGA
- Manage model caching for faster loading

## Best Practices

### Model Optimization

For optimal performance on FPGAs:

```python
# Python
from neurenix.hardware.fpga import FPGAManager

# Create an FPGA manager with optimization settings
fpga_manager = FPGAManager(
    framework="opencl",
    precision="float16",  # Use lower precision for better performance
    optimize_for="throughput",
    memory_allocation="static"  # Use static memory allocation for predictable performance
)

# Compile the model with optimization
compiled_model = fpga_manager.compile_model(model, example_inputs)
```

### Kernel Design

When designing custom kernels for FPGAs:

1. **Maximize Parallelism**: Design kernels to exploit the parallel nature of FPGAs
2. **Minimize Data Transfers**: Reduce host-device data transfers
3. **Use Fixed-Point Arithmetic**: When possible, use fixed-point instead of floating-point
4. **Pipeline Operations**: Design kernels with pipelined operations for higher throughput

### Memory Management

Efficient memory management is crucial for FPGA performance:

```python
# Python
from neurenix.hardware.fpga import OpenCLManager

# Create an OpenCL FPGA manager
opencl_manager = OpenCLManager()

# Create a kernel with efficient memory access patterns
kernel = opencl_manager.create_kernel(
    kernel_name="efficient_kernel",
    kernel_source="""
    __kernel void efficient_kernel(__global const float* input, __global float* output) {
        // Use local memory for frequently accessed data
        __local float local_data[256];
        
        // Load data into local memory
        int lid = get_local_id(0);
        int gid = get_global_id(0);
        local_data[lid] = input[gid];
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Process data from local memory
        output[gid] = local_data[lid] * 2.0f;
    }
    """
)
```

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **FPGA Support** | Native support for multiple FPGA frameworks | Limited support through third-party extensions |
| **Xilinx Integration** | Native Vitis support | Requires TensorFlow-Vitis AI bridge |
| **Intel Integration** | Native OpenVINO support | Requires OpenVINO-TensorFlow bridge |
| **OpenCL Support** | Comprehensive OpenCL support | Limited OpenCL support |
| **Custom Kernel API** | Unified API for custom kernels | Complex integration for custom kernels |
| **Multi-vendor Support** | Support for multiple FPGA vendors | Primarily focused on specific vendors |

Neurenix provides more comprehensive and integrated FPGA support compared to TensorFlow, which relies on third-party extensions for FPGA acceleration. The unified API in Neurenix makes it easier to work with different FPGA vendors and frameworks.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **FPGA Support** | Native support for multiple FPGA frameworks | No official FPGA support |
| **Xilinx Integration** | Native Vitis support | Requires custom integration |
| **Intel Integration** | Native OpenVINO support | Requires custom integration |
| **OpenCL Support** | Comprehensive OpenCL support | No native OpenCL support |
| **Custom Kernel API** | Unified API for custom kernels | No standard API for FPGA kernels |
| **Model Compilation** | Built-in model compilation for FPGAs | Requires external tools |

PyTorch lacks official FPGA support, requiring custom integrations or third-party tools for FPGA acceleration. Neurenix's native FPGA support provides a more seamless experience with built-in model compilation and optimization.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **FPGA Support** | Comprehensive FPGA support | No FPGA support |
| **Hardware Acceleration** | Multiple acceleration options | CPU only |
| **Custom Kernels** | Support for custom FPGA kernels | No hardware kernel support |
| **Model Compilation** | Built-in model compilation for FPGAs | No hardware compilation |
| **Multi-vendor Support** | Support for multiple FPGA vendors | No hardware vendor support |
| **Precision Control** | Multiple precision options | Limited precision control |

Scikit-Learn does not provide any FPGA or hardware acceleration support, focusing solely on CPU execution. Neurenix's FPGA support enables significant performance improvements for suitable workloads.

## Limitations

The current FPGA implementation has the following limitations:

1. **Model Compatibility**: Not all model architectures are suitable for FPGA acceleration
2. **Compilation Time**: FPGA compilation can be time-consuming, especially for complex models
3. **Memory Constraints**: FPGAs typically have limited on-board memory
4. **Vendor-Specific Features**: Some advanced features are vendor-specific
5. **Dynamic Shapes**: Models with dynamic shapes may have limited support

## Future Work

Future development of the FPGA backend will include:

1. **Improved Model Compatibility**: Support for more model architectures
2. **Faster Compilation**: Reduced compilation times through caching and optimization
3. **Dynamic Runtime Reconfiguration**: Support for reconfiguring FPGAs at runtime
4. **Enhanced Multi-FPGA Support**: Better support for distributed computation across multiple FPGAs
5. **Automated Kernel Generation**: Tools for automatically generating optimized FPGA kernels
6. **Quantization-Aware Training**: Better support for training models specifically for FPGA deployment
