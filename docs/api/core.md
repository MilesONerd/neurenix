# Core API Documentation

## Overview

The Core API provides the fundamental functionality for the Neurenix framework, including initialization, configuration management, device handling, and version information. It serves as the foundation upon which all other components of the framework are built.

Neurenix's Core API is designed with a multi-language architecture, where the high-performance Phynexus engine is implemented in Rust/C++, while the Python interface provides a user-friendly API for rapid development. This architecture enables Neurenix to deliver optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

## Key Concepts

### Phynexus Engine

The Phynexus engine is the high-performance core of Neurenix, implemented in Rust and C++. It provides optimized tensor operations and hardware acceleration across different platforms. The Python API communicates with the Phynexus engine through bindings, allowing users to leverage the performance of native code while enjoying the simplicity of Python.

### Device Abstraction

Neurenix provides a hardware abstraction layer that allows code to run seamlessly across different devices, including:

- CPU
- CUDA (NVIDIA GPUs)
- ROCm (AMD GPUs)
- WebGPU (for WebAssembly/browser execution)

This abstraction enables developers to write code once and run it on any supported hardware without modification.

### Configuration System

The framework uses a global configuration system that allows users to customize various aspects of its behavior, such as the default device, logging level, and debug mode.

## API Reference

### Initialization

```python
neurenix.init(config: Optional[Dict[str, Any]] = None) -> None
```

Initializes the Neurenix framework with the given configuration. This function should be called before using any other functionality in the framework.

**Parameters:**
- `config`: Optional dictionary with configuration options for the framework.

**Example:**
```python
import neurenix

# Initialize with default configuration
neurenix.init()

# Initialize with custom configuration
neurenix.init({
    "device": "cuda:0",
    "debug": True,
    "log_level": "debug"
})
```

### Version Information

```python
neurenix.version() -> str
```

Returns the version of the Neurenix framework.

**Returns:**
- The version string.

**Example:**
```python
import neurenix

# Get the framework version
version = neurenix.version()
print(f"Neurenix version: {version}")
```

### Configuration Management

```python
neurenix.get_config() -> Dict[str, Any]
```

Returns the current configuration of the Neurenix framework.

**Returns:**
- The configuration dictionary.

```python
neurenix.set_config(key: str, value: Any) -> None
```

Sets a configuration option for the Neurenix framework.

**Parameters:**
- `key`: The configuration key.
- `value`: The configuration value.

**Example:**
```python
import neurenix

# Get the current configuration
config = neurenix.get_config()
print(f"Current configuration: {config}")

# Update a configuration option
neurenix.set_config("device", "cuda:0")
```

### Device Management

```python
neurenix.Device(device_type: DeviceType, index: int = 0)
```

Creates a new device object representing a computational device (CPU, GPU, etc.).

**Parameters:**
- `device_type`: The type of the device (CPU, CUDA, ROCm, WebGPU).
- `index`: The index of the device (for multiple devices of the same type).

```python
neurenix.DeviceType
```

Enum representing the types of devices supported by Neurenix:
- `CPU`: Central Processing Unit
- `CUDA`: NVIDIA GPU using CUDA
- `ROCM`: AMD GPU using ROCm
- `WEBGPU`: WebGPU for WebAssembly context (client-side execution)

```python
neurenix.get_device_count(device_type: DeviceType) -> int
```

Returns the number of devices of the given type.

**Parameters:**
- `device_type`: The type of device to count.

**Returns:**
- The number of devices of the given type.

```python
neurenix.get_available_devices() -> List[Device]
```

Returns a list of all available devices.

**Returns:**
- A list of available devices.

**Example:**
```python
import neurenix
from neurenix import DeviceType, Device

# Create a CPU device
cpu_device = Device(DeviceType.CPU)

# Create a CUDA device (GPU)
cuda_device = Device(DeviceType.CUDA, 0)  # First CUDA device

# Get the number of CUDA devices
cuda_count = neurenix.get_device_count(DeviceType.CUDA)
print(f"Number of CUDA devices: {cuda_count}")

# Get all available devices
devices = neurenix.get_available_devices()
print(f"Available devices: {devices}")
```

### Utility Functions

```python
neurenix.seed_everything(seed: int) -> None
```

Sets the random seed for all random number generators to ensure reproducibility.

**Parameters:**
- `seed`: Random seed.

```python
neurenix.to_numpy(tensor: Union[Tensor, np.ndarray]) -> np.ndarray
```

Converts a Neurenix tensor to a NumPy array.

**Parameters:**
- `tensor`: Tensor or NumPy array.

**Returns:**
- NumPy array.

```python
neurenix.to_tensor(data: Union[Tensor, np.ndarray, List, Tuple], device: Optional[Union[str, Device]] = None) -> Tensor
```

Converts data to a Neurenix tensor.

**Parameters:**
- `data`: Data to convert.
- `device`: Device to store the tensor on.

**Returns:**
- Neurenix tensor.

**Example:**
```python
import neurenix
import numpy as np

# Set random seed for reproducibility
neurenix.seed_everything(42)

# Convert NumPy array to Neurenix tensor
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
tensor = neurenix.to_tensor(numpy_array, device="cuda:0")

# Convert Neurenix tensor back to NumPy array
numpy_array_again = neurenix.to_numpy(tensor)
```

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Core Architecture** | Multi-language (Rust/C++/Python/Go) | C++ core with Python API |
| **Device Abstraction** | CPU, CUDA, ROCm, WebGPU | CPU, GPU, TPU |
| **Edge Device Support** | Native optimization | TensorFlow Lite |
| **WebAssembly Support** | Native | TensorFlow.js |
| **API Simplicity** | High | Medium |
| **Initialization** | Simple `init()` function | Complex environment setup |

Neurenix's Core API provides a more streamlined initialization process compared to TensorFlow, with a simpler configuration system. The multi-language architecture of Neurenix, with its Rust/C++ core (Phynexus engine), offers performance advantages for edge devices. Additionally, Neurenix's native WebGPU support provides more seamless integration with WebAssembly environments compared to TensorFlow.js.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Core Architecture** | Multi-language (Rust/C++/Python/Go) | C++ core with Python API |
| **Device Abstraction** | CPU, CUDA, ROCm, WebGPU | CPU, CUDA |
| **Edge Device Support** | Native optimization | PyTorch Mobile |
| **WebAssembly Support** | Native | Limited |
| **API Simplicity** | High | High |
| **Configuration System** | Global configuration | Context-based configuration |

Neurenix and PyTorch both offer intuitive APIs, but Neurenix's multi-language architecture provides advantages for specialized use cases. While PyTorch has excellent CUDA support, Neurenix extends device support to include ROCm and WebGPU, making it more versatile across different hardware platforms. Neurenix's native edge device optimization also provides advantages over PyTorch Mobile, which is an add-on component rather than being integrated into the core architecture.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Core Architecture** | Multi-language (Rust/C++/Python/Go) | Python with Cython extensions |
| **Device Abstraction** | CPU, CUDA, ROCm, WebGPU | CPU only |
| **Edge Device Support** | Native optimization | Limited |
| **WebAssembly Support** | Native | None |
| **API Simplicity** | High | High |
| **Deep Learning Support** | Comprehensive | Limited |

Neurenix provides a much more comprehensive device abstraction layer compared to Scikit-Learn, which primarily focuses on CPU execution. While Scikit-Learn offers an excellent API for traditional machine learning, Neurenix extends this with deep learning capabilities and hardware acceleration. Neurenix's multi-language architecture also provides performance advantages over Scikit-Learn's Python/Cython implementation, particularly for computationally intensive tasks.

## Best Practices

### Initialization

Always initialize the framework before using any other functionality:

```python
import neurenix

# Initialize the framework
neurenix.init()

# Now you can use other functionality
tensor = neurenix.Tensor([1, 2, 3])
```

### Device Selection

Choose the appropriate device based on your hardware and requirements:

```python
import neurenix
from neurenix import DeviceType, Device

# Check available devices
devices = neurenix.get_available_devices()
print(f"Available devices: {devices}")

# Use CUDA if available, otherwise fall back to CPU
cuda_count = neurenix.get_device_count(DeviceType.CUDA)
if cuda_count > 0:
    device = Device(DeviceType.CUDA, 0)
else:
    device = Device(DeviceType.CPU)

# Set as default device
neurenix.set_config("device", device.name)
```

### Logging Configuration

Configure logging based on your needs:

```python
import neurenix

# For development, use debug logging
neurenix.init({"log_level": "debug"})

# For production, use warning or error logging
neurenix.init({"log_level": "warning"})
```

### Reproducibility

For reproducible results, set a random seed:

```python
import neurenix

# Set random seed for reproducibility
neurenix.seed_everything(42)
```

## Tutorials

### Basic Framework Initialization

```python
import neurenix

# Initialize the framework with default settings
neurenix.init()

# Check the version
version = neurenix.version()
print(f"Neurenix version: {version}")

# Get the current configuration
config = neurenix.get_config()
print(f"Current configuration: {config}")
```

### Custom Configuration

```python
import neurenix

# Initialize with custom configuration
neurenix.init({
    "device": "cuda:0",  # Use the first CUDA device
    "debug": True,       # Enable debug mode
    "log_level": "debug" # Set logging level to debug
})

# Update a specific configuration option
neurenix.set_config("debug", False)

# Get the updated configuration
config = neurenix.get_config()
print(f"Updated configuration: {config}")
```

### Working with Different Devices

```python
import neurenix
from neurenix import DeviceType, Device

# Create a CPU device
cpu_device = Device(DeviceType.CPU)
print(f"CPU device: {cpu_device}")

# Check if CUDA is available
cuda_count = neurenix.get_device_count(DeviceType.CUDA)
if cuda_count > 0:
    # Create a CUDA device
    cuda_device = Device(DeviceType.CUDA, 0)
    print(f"CUDA device: {cuda_device}")
    
    # Create a tensor on the CUDA device
    tensor = neurenix.Tensor([1, 2, 3], device=cuda_device)
else:
    print("CUDA is not available")
    
    # Create a tensor on the CPU
    tensor = neurenix.Tensor([1, 2, 3], device=cpu_device)

print(f"Tensor device: {tensor.device}")
```

## Conclusion

The Core API of Neurenix provides the foundation for the entire framework, offering initialization, configuration, and device management functionality. Its multi-language architecture, with a high-performance Rust/C++ core and a user-friendly Python interface, enables optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

Compared to other frameworks like TensorFlow, PyTorch, and Scikit-Learn, Neurenix's Core API offers advantages in terms of device abstraction, edge device optimization, and WebAssembly support. These features make Neurenix particularly well-suited for AI agent development, edge computing, and advanced learning paradigms.
