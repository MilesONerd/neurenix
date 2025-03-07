# Distributed Computing Documentation

## Overview

The Distributed Computing module in Neurenix provides tools and utilities for distributed training and inference across multiple GPUs and compute nodes. This module enables users to scale their machine learning workloads to leverage the computational power of multiple devices, from edge devices to multi-GPU clusters.

Neurenix's distributed computing capabilities are implemented using a multi-language architecture, where the high-performance operations are implemented in the Rust/C++ Phynexus engine, while the Python API provides a user-friendly interface. The distributed systems integration is powered by Go, which provides robust networking and concurrency features. This architecture enables Neurenix to deliver optimal performance across a wide range of devices and deployment scenarios.

## Key Concepts

### Distributed Context

The distributed context manages the communication between processes in a distributed training setup. It handles initialization, synchronization, and cleanup of the distributed environment. The context includes information about the world size (total number of processes), rank (process identifier), and device mapping.

### Data Parallelism

Data parallelism is a distributed training strategy where the model is replicated across multiple devices, and each device processes a different subset of the training data. The gradients from each device are then synchronized to update the model parameters. This approach is particularly effective for large datasets and models that fit on a single device.

### Remote Procedure Call (RPC)

Remote Procedure Call (RPC) enables communication between processes in a distributed environment. It allows one process to execute a function on another process, which is useful for parameter server architectures and distributed inference.

### Synchronization Primitives

Synchronization primitives ensure consistent behavior across processes in a distributed environment. These include barriers for synchronizing all processes and synchronized batch normalization for consistent normalization statistics across devices.

### External Integrations

Neurenix provides integrations with external distributed computing frameworks, such as Dask for scalable data processing and PyCUDA for low-level GPU operations. These integrations enable users to leverage the strengths of these frameworks within the Neurenix ecosystem.

## API Reference

### Distributed Context

```python
neurenix.distributed.DistributedContext(
    backend="nccl",
    init_method=None,
    world_size=-1,
    rank=-1,
    device_id=None,
    timeout=1800.0
)
```

Context manager for distributed training.

**Parameters:**
- `backend`: Communication backend ('nccl', 'gloo', or 'mpi')
- `init_method`: URL specifying how to initialize the process group (optional)
- `world_size`: Number of processes in the group (default: auto-detect)
- `rank`: Rank of the current process (default: auto-detect)
- `device_id`: Device ID for the current process (default: auto-detect)
- `timeout`: Timeout for operations (default: 1800.0 seconds)

**Methods:**
- `initialize()`: Initialize the distributed context
- `shutdown()`: Shut down the distributed context
- `is_initialized`: Check if the distributed context is initialized
- `is_main_process`: Check if this is the main process (rank 0)

### Distributed Initialization

```python
neurenix.distributed.init_distributed(
    backend="nccl",
    init_method=None,
    world_size=-1,
    rank=-1,
    device_id=None
)
```

Initialize distributed training.

**Parameters:**
- `backend`: Communication backend ('nccl', 'gloo', or 'mpi')
- `init_method`: URL specifying how to initialize the process group (optional)
- `world_size`: Number of processes in the group (default: auto-detect)
- `rank`: Rank of the current process (default: auto-detect)
- `device_id`: Device ID for the current process (default: auto-detect)

**Returns:**
- Distributed context

### Distributed Utilities

```python
neurenix.distributed.get_rank()
neurenix.distributed.get_world_size()
neurenix.distributed.is_main_process()
neurenix.distributed.barrier()
```

Utility functions for distributed training.

- `get_rank()`: Get the rank of the current process
- `get_world_size()`: Get the world size (number of processes)
- `is_main_process()`: Check if this is the main process (rank 0)
- `barrier()`: Synchronize all processes

### Data Parallel

```python
neurenix.distributed.DataParallel(module, device_ids=None)
```

Data parallel wrapper for modules.

**Parameters:**
- `module`: Module to parallelize
- `device_ids`: List of device IDs to use (default: all available devices)

**Methods:**
- `forward(*args, **kwargs)`: Forward pass with data parallelism
- `parameters()`: Get module parameters
- `to(device)`: Move module to device
- `train(mode=True)`: Set module to training mode
- `eval()`: Set module to evaluation mode

### RPC Context

```python
neurenix.distributed.RpcContext(
    backend="tensorpipe",
    init_method=None,
    world_size=-1,
    rank=-1,
    timeout=1800.0
)
```

RPC context for distributed training.

**Parameters:**
- `backend`: RPC backend ('tensorpipe' or 'gloo')
- `init_method`: URL specifying how to initialize the RPC (optional)
- `world_size`: Number of processes in the group (default: auto-detect)
- `rank`: Rank of the current process (default: auto-detect)
- `timeout`: Timeout for operations (default: 1800.0 seconds)

**Methods:**
- `initialize()`: Initialize the RPC context
- `shutdown()`: Shut down the RPC context
- `register_function(name, func)`: Register a function for RPC

### RPC Functions

```python
neurenix.distributed.init_rpc(
    backend="tensorpipe",
    init_method=None,
    world_size=-1,
    rank=-1
)
neurenix.distributed.rpc_sync(dst_rank, function_name, *args, **kwargs)
neurenix.distributed.rpc_async(dst_rank, function_name, *args, **kwargs)
```

Functions for RPC communication.

- `init_rpc()`: Initialize RPC
- `rpc_sync()`: Synchronous RPC call
- `rpc_async()`: Asynchronous RPC call

### Synchronized Batch Normalization

```python
neurenix.distributed.SyncBatchNorm(
    num_features,
    eps=1e-5,
    momentum=0.1,
    affine=True,
    track_running_stats=True
)
```

Synchronized Batch Normalization.

**Parameters:**
- `num_features`: Number of features
- `eps`: Small constant for numerical stability (default: 1e-5)
- `momentum`: Momentum for running statistics (default: 0.1)
- `affine`: Whether to use learnable affine parameters (default: True)
- `track_running_stats`: Whether to track running statistics (default: True)

**Methods:**
- `forward(x)`: Forward pass

### Dask Integration

```python
neurenix.distributed.DaskCluster(
    n_workers=4,
    threads_per_worker=2,
    memory_limit="4GB",
    scheduler_address=None
)
```

Dask cluster for distributed computing.

**Parameters:**
- `n_workers`: Number of workers (default: 4)
- `threads_per_worker`: Number of threads per worker (default: 2)
- `memory_limit`: Memory limit per worker (default: "4GB")
- `scheduler_address`: Address of existing scheduler to connect to (optional)

**Methods:**
- `start()`: Start the Dask cluster
- `stop()`: Stop the Dask cluster
- `is_running`: Check if the Dask cluster is running
- `map(func, *iterables, **kwargs)`: Map a function to iterables in parallel
- `submit(func, *args, **kwargs)`: Submit a function for execution
- `gather(futures)`: Gather results from futures
- `scatter(data, broadcast=False)`: Scatter data to workers
- `replicate(future)`: Replicate data to all workers
- `get_worker_info()`: Get worker information

### Dask Tensor Conversion

```python
neurenix.distributed.tensor_to_dask(tensor, chunks="auto")
neurenix.distributed.dask_to_tensor(dask_array)
```

Functions for converting between Neurenix tensors and Dask arrays.

- `tensor_to_dask()`: Convert a Neurenix tensor to a Dask array
- `dask_to_tensor()`: Convert a Dask array to a Neurenix tensor

### PyCUDA Integration

```python
neurenix.distributed.CudaContext(
    device_id=0,
    enable_profiling=False
)
```

CUDA context for GPU computing.

**Parameters:**
- `device_id`: GPU device ID (default: 0)
- `enable_profiling`: Whether to enable profiling (default: False)

**Methods:**
- `start()`: Start the CUDA context
- `stop()`: Stop the CUDA context
- `is_running`: Check if the CUDA context is running
- `get_kernel(name, source, function_name)`: Get a CUDA kernel
- `allocate(shape, dtype)`: Allocate memory on the GPU
- `to_gpu(array)`: Copy array to GPU
- `from_gpu(array)`: Copy array from GPU
- `synchronize()`: Synchronize the CUDA context

### PyCUDA Tensor Conversion

```python
neurenix.distributed.tensor_to_gpu(tensor)
neurenix.distributed.gpu_to_tensor(gpu_array)
```

Functions for converting between Neurenix tensors and GPU arrays.

- `tensor_to_gpu()`: Convert a Neurenix tensor to a GPU array
- `gpu_to_tensor()`: Convert a GPU array to a Neurenix tensor

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Multi-language Architecture** | Rust/C++ core, Python API, Go for distributed systems | C++ core with Python API |
| **Distributed Training** | Native support for multi-GPU and multi-node training | TensorFlow Distributed Strategy |
| **Communication Backends** | NCCL, Gloo, MPI | gRPC, NCCL |
| **RPC Framework** | Built-in RPC framework | TensorFlow Serving for inference |
| **Edge Device Support** | Native optimization for edge devices | TensorFlow Lite for edge devices |
| **Hardware Support** | CPU, CUDA, ROCm, WebGPU | CPU, CUDA, TPU |
| **External Integrations** | Dask, PyCUDA | Horovod, Ray |

Neurenix's distributed computing capabilities offer a more comprehensive solution compared to TensorFlow, with native support for a wider range of hardware platforms and communication backends. The multi-language architecture with a high-performance Rust/C++ core and Go-powered distributed systems enables optimal performance across different deployment scenarios. Additionally, Neurenix's integrations with Dask and PyCUDA provide more flexibility for distributed data processing and low-level GPU operations.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Multi-language Architecture** | Rust/C++ core, Python API, Go for distributed systems | C++ core with Python API |
| **Distributed Training** | Native support for multi-GPU and multi-node training | PyTorch Distributed Data Parallel |
| **Communication Backends** | NCCL, Gloo, MPI | NCCL, Gloo, MPI |
| **RPC Framework** | Built-in RPC framework | PyTorch RPC |
| **Hardware Support** | CPU, CUDA, ROCm, WebGPU | CPU, CUDA |
| **Synchronization Primitives** | Synchronized Batch Normalization, Barriers | Synchronized Batch Normalization, Barriers |
| **External Integrations** | Dask, PyCUDA | Ray, CUDA |

Neurenix and PyTorch both offer comprehensive distributed computing capabilities, but Neurenix's multi-language architecture with a Go-powered distributed systems layer provides better performance and scalability for large-scale deployments. Neurenix also extends hardware support to include ROCm and WebGPU, making it more versatile across different hardware platforms. The integration with Dask provides more advanced distributed data processing capabilities compared to PyTorch's integration with Ray.

### Neurenix vs. Horovod

| Feature | Neurenix | Horovod |
|---------|----------|---------|
| **Framework Integration** | Native part of Neurenix | Add-on for TensorFlow, PyTorch, and MXNet |
| **Communication Backends** | NCCL, Gloo, MPI | MPI, NCCL |
| **Ease of Use** | Integrated API with automatic device detection | Requires explicit initialization and synchronization |
| **Hardware Support** | CPU, CUDA, ROCm, WebGPU | CPU, CUDA |
| **Scalability** | Designed for edge devices to clusters | Primarily designed for clusters |
| **External Integrations** | Dask, PyCUDA | Limited to MPI-based systems |

Neurenix provides a more integrated and user-friendly distributed computing solution compared to Horovod, with automatic device detection and a consistent API across different deployment scenarios. While Horovod is primarily designed for large-scale clusters, Neurenix's distributed computing capabilities scale from edge devices to multi-node clusters, making it more versatile for different use cases. Additionally, Neurenix's multi-language architecture with a Go-powered distributed systems layer provides better performance and scalability for large-scale deployments.

## Best Practices

### Choosing the Right Backend

When selecting a communication backend for distributed training, consider the following factors:

1. **NCCL**: Best for GPU-to-GPU communication on NVIDIA hardware
2. **Gloo**: Good for CPU-to-CPU communication and heterogeneous hardware
3. **MPI**: Best for high-performance computing clusters with specialized interconnects

### Optimizing Data Parallelism

To get the best performance from data parallel training, consider these optimizations:

1. **Batch Size**: Adjust the batch size to maximize GPU utilization
2. **Gradient Accumulation**: Use gradient accumulation for larger effective batch sizes
3. **Mixed Precision Training**: Use mixed precision to reduce memory usage and increase throughput
4. **Overlap Communication and Computation**: Overlap gradient communication with backward pass computation

### Scaling to Multiple Nodes

When scaling to multiple nodes, consider these best practices:

1. **Network Bandwidth**: Ensure sufficient network bandwidth between nodes
2. **Synchronization Frequency**: Reduce synchronization frequency for better scalability
3. **Parameter Server vs. All-Reduce**: Choose the appropriate architecture based on model size and network topology
4. **Fault Tolerance**: Implement checkpointing for fault tolerance

### Edge Device Considerations

When deploying distributed training to edge devices, consider these optimizations:

1. **Model Partitioning**: Partition the model across devices based on computational capabilities
2. **Communication Efficiency**: Minimize communication between devices
3. **Asynchronous Updates**: Use asynchronous updates to handle varying device speeds
4. **Resource Awareness**: Adapt to available resources on each device

## Tutorials

### Multi-GPU Training with DataParallel

```python
import neurenix
from neurenix.nn import Module, Linear
from neurenix.distributed import DataParallel
from neurenix.optim import SGD

# Create a simple model
class SimpleModel(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 256)
        self.fc2 = Linear(256, 10)
    
    def forward(self, x):
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x

# Create model
model = SimpleModel()

# Wrap model with DataParallel
model = DataParallel(model)

# Create optimizer
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Forward pass
        output = model(data)
        loss = loss_function(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")
```

### Distributed Training with DistributedContext

```python
import neurenix
from neurenix.distributed import init_distributed, get_rank, get_world_size, is_main_process

# Initialize distributed training
dist_ctx = init_distributed(
    backend="nccl",
    init_method="tcp://localhost:23456",
    world_size=4,
    rank=0,  # Set this to the current process rank
)

# Create model and move to device
model = create_model()
device = neurenix.get_device(f"cuda:{get_rank()}")
model.to(device)

# Create optimizer
optimizer = create_optimizer(model)

# Training loop
for epoch in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data = data.to(device)
        target = target.to(device)
        
        # Forward pass
        output = model(data)
        loss = loss_function(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress on main process
        if is_main_process() and batch_idx % 100 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item()}")

# Clean up
dist_ctx.shutdown()
```

### Using Dask for Distributed Data Processing

```python
import neurenix
from neurenix.distributed import DaskCluster, tensor_to_dask, dask_to_tensor

# Create Dask cluster
with DaskCluster(n_workers=4, threads_per_worker=2) as cluster:
    # Load data
    data = neurenix.tensor.load("large_dataset.npy")
    
    # Convert to Dask array
    dask_data = tensor_to_dask(data, chunks=(1000, -1))
    
    # Define processing function
    def process_chunk(chunk):
        # Convert to Neurenix tensor
        tensor = neurenix.tensor.Tensor(chunk)
        
        # Process tensor
        processed = tensor - tensor.mean(dim=0)
        processed = processed / tensor.std(dim=0)
        
        return processed.numpy()
    
    # Process data in parallel
    processed_dask = dask_data.map_blocks(process_chunk)
    
    # Compute result
    processed_data = dask_to_tensor(processed_dask)
    
    # Save result
    processed_data.save("processed_dataset.npy")
```

### Using PyCUDA for Custom CUDA Kernels

```python
import neurenix
from neurenix.distributed import CudaContext, tensor_to_gpu, gpu_to_tensor
import numpy as np

# Define CUDA kernel
kernel_source = """
__global__ void custom_activation(float *input, float *output, int n, float alpha)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
    {
        output[idx] = input[idx] > 0 ? input[idx] : alpha * input[idx];
    }
}
"""

# Create CUDA context
with CudaContext(device_id=0) as ctx:
    # Get kernel
    kernel = ctx.get_kernel(
        name="custom_activation",
        source=kernel_source,
        function_name="custom_activation",
    )
    
    # Create input tensor
    input_tensor = neurenix.tensor.randn(1000000)
    
    # Convert to GPU array
    input_gpu = tensor_to_gpu(input_tensor)
    
    # Allocate output array
    output_gpu = ctx.allocate(input_gpu.shape, input_gpu.dtype)
    
    # Launch kernel
    block_size = 256
    grid_size = (input_gpu.size + block_size - 1) // block_size
    
    kernel(
        input_gpu,
        output_gpu,
        np.int32(input_gpu.size),
        np.float32(0.01),
        block=(block_size, 1, 1),
        grid=(grid_size, 1),
    )
    
    # Convert back to Neurenix tensor
    output_tensor = gpu_to_tensor(output_gpu)
    
    # Print result
    print(f"Input: {input_tensor[:10]}")
    print(f"Output: {output_tensor[:10]}")
```

## Conclusion

The Distributed Computing module of Neurenix provides a comprehensive set of tools for distributed training and inference across multiple GPUs and compute nodes. Its multi-language architecture with a high-performance Rust/C++ core and Go-powered distributed systems enables optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

Compared to other frameworks like TensorFlow, PyTorch, and Horovod, Neurenix's distributed computing capabilities offer advantages in terms of hardware support, ease of use, and scalability. The native support for a wider range of hardware platforms and communication backends, combined with integrations with Dask and PyCUDA, provides more flexibility for distributed data processing and low-level GPU operations.

Whether you're training a model on multiple GPUs in a single machine, scaling to a multi-node cluster, or deploying to edge devices, Neurenix's distributed computing capabilities provide the tools you need to maximize performance and scalability.
