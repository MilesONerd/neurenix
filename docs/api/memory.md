# Memory Management API Documentation

## Overview

The Memory Management module provides advanced memory management capabilities for efficient AI workloads across heterogeneous hardware. It implements Unified Memory (UM) and Heterogeneous Memory Management (HMM) to optimize memory usage, reduce data transfer overhead, and improve performance for large-scale AI models.

Memory management is a critical component in modern AI frameworks, especially when working with large models and datasets across different hardware accelerators. This module enables seamless memory sharing between CPU and accelerators (GPUs, TPUs, etc.), automatic memory migration, and intelligent prefetching to minimize data transfer bottlenecks.

## Key Concepts

### Unified Memory (UM)

Unified Memory provides a single memory space that is accessible from both CPU and accelerators. It simplifies memory management by:

- Automatically migrating data between devices as needed
- Providing a coherent view of memory across all devices
- Enabling zero-copy access to memory from different devices
- Supporting prefetching and memory usage hints

### Heterogeneous Memory Management (HMM)

Heterogeneous Memory Management extends unified memory concepts to diverse hardware architectures:

- Manages memory across heterogeneous devices with different memory architectures
- Optimizes memory placement based on access patterns
- Provides fine-grained control over memory migration policies
- Supports devices that may not have native unified memory capabilities

### Memory Policies

The module supports various memory management policies:

- **Prefetch Policies**: Control when and how data is prefetched to devices
- **Migration Policies**: Determine when and how data is migrated between devices
- **Advise Policies**: Provide hints about memory usage patterns to optimize placement

## API Reference

### Unified Memory Manager

```python
neurenix.memory.UnifiedMemoryManager(
    mode: str = "auto",
    prefetch_policy: str = "adaptive",
    migration_policy: str = "adaptive",
    advise_policy: str = "preferred_location",
    device: Optional[str] = None
)
```

Creates a manager for Unified Memory operations.

**Parameters:**
- `mode`: Memory management mode ('auto', 'manual', 'managed')
- `prefetch_policy`: Prefetch policy ('none', 'adaptive', 'aggressive')
- `migration_policy`: Migration policy ('none', 'adaptive', 'aggressive')
- `advise_policy`: Memory advise policy ('preferred_location', 'read_mostly', 'accessed_by')
- `device`: Device to use for memory operations (default: current device)

**Methods:**
- `initialize()`: Initialize the Unified Memory environment
- `finalize()`: Finalize the Unified Memory environment
- `allocate(size, dtype)`: Allocate unified memory
- `free(handle)`: Free unified memory
- `prefetch(handle, device)`: Prefetch unified memory to a device
- `advise(handle, advice, device)`: Set memory usage advice for unified memory
- `is_managed(handle)`: Check if memory is managed by unified memory
- `get_info(handle)`: Get information about unified memory

**Example:**
```python
import neurenix as nx
from neurenix.memory import UnifiedMemoryManager

# Create a Unified Memory manager
um_manager = UnifiedMemoryManager(
    mode="auto",
    prefetch_policy="adaptive",
    migration_policy="adaptive",
    advise_policy="preferred_location"
)

# Initialize the manager
um_manager.initialize()

# Allocate unified memory
memory_handle = um_manager.allocate(size=1024*1024, dtype="float32")

# Set memory advice
um_manager.advise(memory_handle, advice="preferred_location", device="cuda:0")

# Prefetch memory to device
um_manager.prefetch(memory_handle, device="cuda:0")

# Free memory when done
um_manager.free(memory_handle)

# Finalize the manager
um_manager.finalize()
```

### Heterogeneous Memory Manager

```python
neurenix.memory.HeterogeneousMemoryManager(
    mode: str = "auto",
    migration_policy: str = "adaptive",
    device: Optional[str] = None
)
```

Creates a manager for Heterogeneous Memory Management operations.

**Parameters:**
- `mode`: Memory management mode ('auto', 'manual', 'managed')
- `migration_policy`: Migration policy ('none', 'adaptive', 'aggressive')
- `device`: Device to use for memory operations (default: current device)

**Methods:**
- `initialize()`: Initialize the HMM environment
- `finalize()`: Finalize the HMM environment
- `allocate(size, dtype)`: Allocate HMM memory
- `free(handle)`: Free HMM memory
- `migrate(handle, device)`: Migrate HMM memory to a device
- `get_info(handle)`: Get information about HMM memory

### Global Managers

```python
neurenix.memory.get_um_manager() -> UnifiedMemoryManager
```

Gets the global Unified Memory manager instance.

```python
neurenix.memory.get_hmm_manager() -> HeterogeneousMemoryManager
```

Gets the global HMM manager instance.

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Unified Memory** | Comprehensive API with fine-grained control | Limited to CUDA unified memory |
| **Heterogeneous Memory** | Native HMM support | Limited support |
| **Memory Policies** | Multiple configurable policies | Limited configuration options |
| **Device Support** | Multiple accelerator types | Primarily GPU-focused |
| **API Consistency** | Unified API across devices | Different APIs for different devices |

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Unified Memory** | Native API with fine-grained control | Limited to CUDA unified memory |
| **Heterogeneous Memory** | Native HMM support | No native support |
| **Memory Policies** | Multiple configurable policies | Limited configuration options |
| **Device Support** | Multiple accelerator types | Primarily GPU-focused |
| **API Consistency** | Unified API across devices | Different APIs for different devices |

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Unified Memory** | Comprehensive support | No support |
| **Heterogeneous Memory** | Native HMM support | No support |
| **Memory Policies** | Multiple configurable policies | No policies |
| **Device Support** | Multiple accelerator types | CPU only |
| **API Consistency** | Unified API across devices | CPU-focused API |

## Best Practices

### Memory Allocation

For optimal performance, consider the following when allocating memory:

```python
from neurenix.memory import get_um_manager

um_manager = get_um_manager()

# Allocate memory in larger chunks when possible
large_memory = um_manager.allocate(size=1024*1024*100, dtype="float32")

# Avoid frequent small allocations
# Reuse memory when possible
memory_pool = [um_manager.allocate(size=1024*1024, dtype="float32") for _ in range(10)]
```

### Memory Prefetching

Use prefetching to reduce data transfer latency:

```python
from neurenix.memory import get_um_manager

um_manager = get_um_manager()
memory = um_manager.allocate(size=1024*1024*100, dtype="float32")

# Prefetch data before it's needed
um_manager.prefetch(memory, device="cuda:0")
```

### Memory Advice

Provide memory usage hints for better performance:

```python
from neurenix.memory import get_um_manager

um_manager = get_um_manager()
memory = um_manager.allocate(size=1024*1024*100, dtype="float32")

# For data that will be read frequently by GPU
um_manager.advise(memory, advice="read_mostly", device="cuda:0")
```

## Tutorials

### Efficient Memory Management for Large Models

```python
import neurenix as nx
from neurenix.memory import get_um_manager
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.device import Device, DeviceType

# Initialize Neurenix
nx.init()

# Get the unified memory manager
um_manager = get_um_manager()

# Create a large model
model = Sequential(
    Linear(1024, 4096),
    ReLU(),
    Linear(4096, 4096),
    ReLU(),
    Linear(4096, 1024)
)

# Move model to GPU
device = Device(DeviceType.CUDA, 0)
model.to(device)

# Allocate unified memory for input data
input_size = 1024 * 1024  # 1M elements
input_memory = um_manager.allocate(size=input_size * 4, dtype="float32")

# Advise the memory system about usage patterns
um_manager.advise(input_memory, advice="read_mostly", device="cuda:0")

# Create a tensor using the allocated memory
input_tensor = nx.Tensor.from_memory(input_memory, shape=(1024, 1024), dtype="float32")

# Prefetch data to GPU before computation
um_manager.prefetch(input_memory, device="cuda:0")

# Run the model
output = model(input_tensor)

# Free memory when done
um_manager.free(input_memory)
```
