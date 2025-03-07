# Tensor Operations Documentation

## Overview

Tensor Operations form the foundation of the Neurenix framework, providing the essential mathematical operations needed for building and training machine learning models. These operations range from basic arithmetic to complex linear algebra and are optimized for performance across various hardware platforms.

Neurenix's tensor operations are implemented using a multi-language architecture, where the high-performance operations are implemented in the Rust/C++ Phynexus engine, while the Python API provides a user-friendly interface. This architecture enables Neurenix to deliver optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

## Key Concepts

### Tensor Data Structure

The fundamental data structure in Neurenix is the `Tensor`, which represents a multi-dimensional array with support for various hardware devices. Tensors can be created from Python lists, tuples, NumPy arrays, or other tensors, and can be moved between different devices (CPU, CUDA, ROCm, WebGPU) as needed.

### Device Abstraction

Neurenix provides a device abstraction layer that allows tensors to be stored and operated on various hardware devices, including CPUs, GPUs (via CUDA or ROCm), and WebGPU for browser-based applications. This abstraction enables seamless code execution across different hardware platforms without requiring changes to the user code.

### Automatic Differentiation

Tensors in Neurenix can track gradients for automatic differentiation, which is essential for training neural networks. When a tensor is created with `requires_grad=True`, the framework automatically tracks operations performed on the tensor and builds a computational graph for backward propagation.

### Edge Device Optimization

Neurenix's tensor operations are optimized for edge devices, with efficient implementations that minimize memory usage and computational requirements while maintaining high performance. This makes Neurenix particularly well-suited for AI agent applications on resource-constrained hardware.

## API Reference

### Tensor Creation

```python
neurenix.Tensor(data=None, shape=None, dtype=None, device=None, requires_grad=False)
```

Create a new tensor.

**Parameters:**
- `data`: The data to initialize the tensor with. Can be a NumPy array, a list, a tuple, another Tensor, or None (for uninitialized tensor).
- `shape`: The shape of the tensor. If None, inferred from data.
- `dtype`: The data type of the tensor. If None, inferred from data.
- `device`: The device to store the tensor on. If None, uses the default device.
- `requires_grad`: Whether to track gradients for this tensor.

**Example:**
```python
import neurenix

# Create a tensor from a Python list
x = neurenix.Tensor([1, 2, 3, 4])

# Create a tensor with a specific shape and data type
y = neurenix.Tensor(shape=(2, 3), dtype=neurenix.DType.FLOAT32)

# Create a tensor on a specific device
z = neurenix.Tensor([1, 2, 3, 4], device=neurenix.Device(neurenix.DeviceType.CUDA))

# Create a tensor that requires gradients for automatic differentiation
w = neurenix.Tensor([1, 2, 3, 4], requires_grad=True)
```

### Tensor Properties

```python
tensor.shape
```

Get the shape of the tensor.

```python
tensor.ndim
```

Get the number of dimensions of the tensor.

```python
tensor.size
```

Get the total number of elements in the tensor.

```python
tensor.dtype
```

Get the data type of the tensor.

```python
tensor.device
```

Get the device where the tensor is stored.

```python
tensor.requires_grad
```

Check if the tensor requires gradients.

```python
tensor.grad
```

Get the gradient of the tensor.

### Tensor Conversion

```python
tensor.numpy()
```

Convert the tensor to a NumPy array. This operation will copy the tensor data from the device to the CPU if necessary.

```python
tensor.to(device)
```

Move the tensor to the specified device.

**Parameters:**
- `device`: The target device.

**Returns:**
- A new tensor on the target device.

**Example:**
```python
import neurenix
from neurenix import Device, DeviceType

# Create a tensor on the CPU
x = neurenix.Tensor([1, 2, 3, 4])

# Move the tensor to a CUDA device
cuda_device = Device(DeviceType.CUDA)
x_cuda = x.to(cuda_device)
```

### Element-wise Operations

```python
neurenix.add(a, b)
```

Add two tensors element-wise.

```python
neurenix.sub(a, b)
```

Subtract two tensors element-wise.

```python
neurenix.mul(a, b)
```

Multiply two tensors element-wise.

```python
neurenix.div(a, b)
```

Divide two tensors element-wise.

```python
neurenix.pow(a, b)
```

Compute the element-wise power.

```python
neurenix.exp(x)
```

Compute the element-wise exponential.

```python
neurenix.log(x)
```

Compute the element-wise natural logarithm.

**Example:**
```python
import neurenix

# Create tensors
a = neurenix.Tensor([1, 2, 3, 4])
b = neurenix.Tensor([5, 6, 7, 8])

# Element-wise operations
c = neurenix.add(a, b)  # [6, 8, 10, 12]
d = neurenix.sub(b, a)  # [4, 4, 4, 4]
e = neurenix.mul(a, b)  # [5, 12, 21, 32]
f = neurenix.div(b, a)  # [5, 3, 2.33, 2]
g = neurenix.pow(a, b)  # [1, 64, 2187, 65536]
h = neurenix.exp(a)     # [2.72, 7.39, 20.09, 54.60]
i = neurenix.log(b)     # [1.61, 1.79, 1.95, 2.08]
```

### Matrix Operations

```python
neurenix.matmul(a, b)
```

Perform matrix multiplication between two tensors.

**Parameters:**
- `a`: First tensor.
- `b`: Second tensor.

**Returns:**
- A new tensor containing the result of the matrix multiplication.

**Example:**
```python
import neurenix

# Create 2D tensors
a = neurenix.Tensor([[1, 2], [3, 4]])
b = neurenix.Tensor([[5, 6], [7, 8]])

# Matrix multiplication
c = neurenix.matmul(a, b)  # [[19, 22], [43, 50]]
```

### Reduction Operations

```python
neurenix.sum(x, dim=None, keepdim=False)
```

Sum of tensor elements along the specified dimension.

```python
neurenix.mean(x, dim=None, keepdim=False)
```

Mean of tensor elements along the specified dimension.

```python
neurenix.max(x, dim=None, keepdim=False)
```

Maximum value of tensor elements along the specified dimension.

```python
neurenix.min(x, dim=None, keepdim=False)
```

Minimum value of tensor elements along the specified dimension.

**Parameters:**
- `x`: Input tensor.
- `dim`: Dimension along which to reduce. If None, reduce over all dimensions.
- `keepdim`: Whether to keep the reduced dimension as a singleton dimension.

**Returns:**
- A new tensor containing the result of the reduction operation.

**Example:**
```python
import neurenix

# Create a 2D tensor
x = neurenix.Tensor([[1, 2, 3], [4, 5, 6]])

# Reduction operations
s = neurenix.sum(x)          # 21
s_dim0 = neurenix.sum(x, 0)  # [5, 7, 9]
s_dim1 = neurenix.sum(x, 1)  # [6, 15]

m = neurenix.mean(x)          # 3.5
m_dim0 = neurenix.mean(x, 0)  # [2.5, 3.5, 4.5]
m_dim1 = neurenix.mean(x, 1)  # [2, 5]
```

### BLAS Operations

```python
neurenix.dot(a, b)
```

Dot product of two tensors.

```python
neurenix.norm(x, p=2, dim=None, keepdim=False)
```

Compute the p-norm of a tensor along the specified dimension.

**Parameters:**
- `a`, `b`: Input tensors.
- `x`: Input tensor.
- `p`: The order of the norm.
- `dim`: Dimension along which to compute the norm. If None, compute over all dimensions.
- `keepdim`: Whether to keep the reduced dimension as a singleton dimension.

**Returns:**
- A new tensor containing the result of the operation.

**Example:**
```python
import neurenix

# Create tensors
a = neurenix.Tensor([1, 2, 3])
b = neurenix.Tensor([4, 5, 6])

# Dot product
c = neurenix.dot(a, b)  # 32 (1*4 + 2*5 + 3*6)

# Norm
n = neurenix.norm(a)  # 3.74 (sqrt(1^2 + 2^2 + 3^2))
```

### Convolution Operations

```python
neurenix.conv1d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
```

Apply a 1D convolution over an input signal composed of several input channels.

```python
neurenix.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
```

Apply a 2D convolution over an input image composed of several input channels.

```python
neurenix.conv3d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
```

Apply a 3D convolution over an input volume composed of several input channels.

**Parameters:**
- `input`: Input tensor of shape (batch_size, in_channels, *).
- `weight`: Filters of shape (out_channels, in_channels/groups, *).
- `bias`: Optional bias tensor of shape (out_channels).
- `stride`: Stride of the convolution.
- `padding`: Zero-padding added to both sides of the input.
- `dilation`: Spacing between kernel elements.
- `groups`: Number of blocked connections from input to output channels.

**Returns:**
- A new tensor containing the result of the convolution operation.

**Example:**
```python
import neurenix

# Create input tensor (batch_size=1, channels=3, width=5)
input = neurenix.Tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]])

# Create weight tensor (out_channels=2, in_channels=3, kernel_size=3)
weight = neurenix.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])

# Apply 1D convolution
output = neurenix.conv1d(input, weight, stride=1, padding=1)
```

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Tensor Creation** | Simple, intuitive API | Multiple APIs (tf.constant, tf.Variable, etc.) |
| **Device Support** | CPU, CUDA, ROCm, WebGPU | CPU, CUDA, TPU |
| **Edge Optimization** | Native optimization for edge devices | TensorFlow Lite for edge devices |
| **API Design** | Object-oriented, intuitive | Functional and object-oriented APIs |
| **WebAssembly Support** | Native WebGPU support | TensorFlow.js for web applications |
| **Multi-Language Architecture** | Rust/C++ core with Python interface | C++ core with Python interface |

Neurenix's tensor operations offer a more intuitive API compared to TensorFlow's multiple APIs for tensor creation and manipulation. The native optimization for edge devices in Neurenix provides better performance on resource-constrained hardware compared to TensorFlow Lite, which is an add-on component rather than being integrated into the core architecture. Additionally, Neurenix's native WebGPU support provides better performance for browser-based applications compared to TensorFlow.js.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Tensor Creation** | Similar to PyTorch | Simple, intuitive API |
| **Device Support** | CPU, CUDA, ROCm, WebGPU | CPU, CUDA |
| **Edge Optimization** | Native optimization for edge devices | PyTorch Mobile for edge devices |
| **API Design** | Object-oriented, intuitive | Object-oriented, intuitive |
| **WebAssembly Support** | Native WebGPU support | Limited support via PyTorch.js |
| **Multi-Language Architecture** | Rust/C++ core with Python interface | C++ core with Python interface |

Neurenix's tensor operations are very similar to PyTorch's in terms of API design and functionality, making it easy for PyTorch users to transition to Neurenix. However, Neurenix extends hardware support to include ROCm and WebGPU, making it more versatile across different hardware platforms. The native edge device optimization in Neurenix also provides advantages over PyTorch Mobile, particularly for AI agent applications.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Tensor Support** | Comprehensive tensor operations | Limited tensor support via NumPy |
| **GPU Acceleration** | Native support for multiple GPU types | No native GPU support |
| **Edge Device Support** | Native optimization for edge devices | No specific edge device support |
| **API Design** | Object-oriented, focused on tensors | Function-based, focused on algorithms |
| **WebAssembly Support** | Native WebGPU support | No WebAssembly support |
| **Multi-Language Architecture** | Rust/C++ core with Python interface | Pure Python with C/C++ extensions |

Neurenix provides much more comprehensive tensor operations compared to Scikit-Learn, which primarily relies on NumPy for tensor operations. While Scikit-Learn focuses on machine learning algorithms rather than tensor operations, Neurenix provides a full suite of tensor operations with GPU acceleration and edge device optimization, making it more suitable for deep learning and AI agent applications.

## Best Practices

### Efficient Tensor Operations

When working with tensor operations in Neurenix, follow these best practices:

1. **Use the Appropriate Device**: Choose the most efficient device for your operations based on the available hardware.
2. **Minimize Data Transfers**: Avoid unnecessary data transfers between devices, as these can be expensive operations.
3. **Use In-place Operations**: When possible, use in-place operations to reduce memory usage.
4. **Batch Operations**: Batch multiple operations together to reduce overhead.

```python
import neurenix
from neurenix import Device, DeviceType

# Choose the most efficient available device
devices = neurenix.get_available_devices()
compute_device = None

# Prioritize CUDA for high-performance computing
for device in devices:
    if device.type == DeviceType.CUDA:
        compute_device = device
        break

# Fall back to CPU if no GPU is available
if compute_device is None:
    compute_device = Device(DeviceType.CPU)

# Create tensors on the selected device
a = neurenix.Tensor([1, 2, 3, 4], device=compute_device)
b = neurenix.Tensor([5, 6, 7, 8], device=compute_device)

# Perform operations on the device
c = neurenix.add(a, b)
```

### Optimizing for Edge Devices

When deploying tensor operations to edge devices, consider these optimizations:

1. **Reduce Precision**: Use lower precision data types (e.g., FLOAT32 instead of FLOAT64) when possible.
2. **Minimize Memory Usage**: Avoid creating unnecessary intermediate tensors.
3. **Use Appropriate Hardware Abstraction**: Leverage Neurenix's device abstraction to run on the most efficient available hardware.

```python
import neurenix
from neurenix import Device, DeviceType, DType

# Create tensors with reduced precision
a = neurenix.Tensor([1, 2, 3, 4], dtype=DType.FLOAT32)
b = neurenix.Tensor([5, 6, 7, 8], dtype=DType.FLOAT32)

# Use the most efficient available device for edge computing
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

# Move tensors to the selected device
a = a.to(edge_device)
b = b.to(edge_device)

# Perform operations on the device
c = neurenix.add(a, b)
```

### Working with Automatic Differentiation

For effective use of automatic differentiation:

1. **Track Gradients Selectively**: Only set `requires_grad=True` for tensors that need gradients.
2. **Use No-Grad Context**: Use the `no_grad` context manager for inference to reduce memory usage.
3. **Clear Gradients**: Clear gradients before each optimization step to prevent accumulation.

```python
import neurenix
from neurenix.optim import SGD

# Create tensors with gradient tracking
x = neurenix.Tensor([1, 2, 3, 4], requires_grad=True)
y = neurenix.Tensor([5, 6, 7, 8])

# Perform operations that track gradients
z = neurenix.mul(x, y)
loss = neurenix.sum(z)

# Compute gradients
loss.backward()

# Create an optimizer
optimizer = SGD([x], lr=0.01)

# Optimization step
optimizer.zero_grad()  # Clear gradients
loss.backward()        # Compute gradients
optimizer.step()       # Update parameters

# Inference without tracking gradients
with neurenix.no_grad():
    pred = neurenix.mul(x, y)
```

## Tutorials

### Basic Tensor Operations

```python
import neurenix

# Create tensors
a = neurenix.Tensor([1, 2, 3, 4])
b = neurenix.Tensor([5, 6, 7, 8])

# Element-wise operations
c = neurenix.add(a, b)  # [6, 8, 10, 12]
d = neurenix.sub(b, a)  # [4, 4, 4, 4]
e = neurenix.mul(a, b)  # [5, 12, 21, 32]
f = neurenix.div(b, a)  # [5, 3, 2.33, 2]

# Reduction operations
sum_a = neurenix.sum(a)    # 10
mean_a = neurenix.mean(a)  # 2.5
max_a = neurenix.max(a)    # 4
min_a = neurenix.min(a)    # 1

# Matrix operations
m1 = neurenix.Tensor([[1, 2], [3, 4]])
m2 = neurenix.Tensor([[5, 6], [7, 8]])
m3 = neurenix.matmul(m1, m2)  # [[19, 22], [43, 50]]

# Print results
print(f"a: {a.numpy()}")
print(f"b: {b.numpy()}")
print(f"a + b: {c.numpy()}")
print(f"b - a: {d.numpy()}")
print(f"a * b: {e.numpy()}")
print(f"b / a: {f.numpy()}")
print(f"sum(a): {sum_a.numpy()}")
print(f"mean(a): {mean_a.numpy()}")
print(f"max(a): {max_a.numpy()}")
print(f"min(a): {min_a.numpy()}")
print(f"m1: {m1.numpy()}")
print(f"m2: {m2.numpy()}")
print(f"m1 @ m2: {m3.numpy()}")
```

### Working with Different Devices

```python
import neurenix
from neurenix import Device, DeviceType

# Get available devices
devices = neurenix.get_available_devices()
print(f"Available devices: {devices}")

# Create a tensor on the CPU
cpu_device = Device(DeviceType.CPU)
x_cpu = neurenix.Tensor([1, 2, 3, 4], device=cpu_device)
print(f"x_cpu: {x_cpu}")

# Move the tensor to a CUDA device if available
cuda_device = None
for device in devices:
    if device.type == DeviceType.CUDA:
        cuda_device = device
        break

if cuda_device is not None:
    x_cuda = x_cpu.to(cuda_device)
    print(f"x_cuda: {x_cuda}")
    
    # Perform operations on the CUDA device
    y_cuda = neurenix.Tensor([5, 6, 7, 8], device=cuda_device)
    z_cuda = neurenix.add(x_cuda, y_cuda)
    print(f"z_cuda: {z_cuda}")
    
    # Move the result back to the CPU
    z_cpu = z_cuda.to(cpu_device)
    print(f"z_cpu: {z_cpu}")
    print(f"z_cpu data: {z_cpu.numpy()}")
else:
    print("CUDA device not available")
```

### Automatic Differentiation

```python
import neurenix
from neurenix.optim import SGD

# Create tensors with gradient tracking
x = neurenix.Tensor([1, 2, 3, 4], requires_grad=True)
y = neurenix.Tensor([2, 2, 2, 2])

# Define a simple computation graph
z = neurenix.mul(x, y)  # z = x * y
w = neurenix.sum(z)     # w = sum(z)

# Compute gradients
w.backward()

# Print gradients
print(f"x: {x.numpy()}")
print(f"y: {y.numpy()}")
print(f"z: {z.numpy()}")
print(f"w: {w.numpy()}")
print(f"dw/dx: {x.grad.numpy()}")  # Should be [2, 2, 2, 2]

# Create an optimizer
optimizer = SGD([x], lr=0.1)

# Optimization loop
for i in range(5):
    # Forward pass
    z = neurenix.mul(x, y)
    w = neurenix.sum(z)
    
    # Backward pass
    optimizer.zero_grad()
    w.backward()
    
    # Update parameters
    optimizer.step()
    
    print(f"Iteration {i+1}, x: {x.numpy()}, w: {w.numpy()}")
```

## Conclusion

The Tensor Operations module of Neurenix provides a comprehensive set of operations for building and training machine learning models. Its multi-language architecture with a high-performance Rust/C++ core enables optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

Compared to other frameworks like TensorFlow, PyTorch, and Scikit-Learn, Neurenix's Tensor Operations module offers advantages in terms of API design, hardware support, and edge device optimization. These features make Neurenix particularly well-suited for AI agent development, edge computing, and browser-based applications.
