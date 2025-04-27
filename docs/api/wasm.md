# WebAssembly Support API Documentation

## Overview

The WebAssembly (Wasm) module provides functionality for exporting and running Neurenix models in web browsers using WebAssembly. This enables client-side AI execution, reducing server load and latency while enhancing privacy by keeping data on the client device.

WebAssembly is a binary instruction format designed as a portable compilation target for high-performance applications on the web. Neurenix leverages WebAssembly to enable AI models to run directly in web browsers without server roundtrips.

## Key Concepts

### Client-Side Execution

Client-side execution allows AI models to run directly in the user's browser, providing several benefits:
- Reduced latency by eliminating network delays
- Enhanced privacy by keeping data on the client device
- Offline capabilities for running models without an internet connection
- Reduced server costs by offloading computation to client devices

### WebGPU Integration

WebGPU is a modern graphics and compute API for the web that provides access to GPU capabilities. Neurenix's WebAssembly support integrates with WebGPU to accelerate tensor operations when available in the browser.

### Seamless API

Neurenix provides a seamless API that works across server and client environments, allowing developers to write code once and run it anywhere. The framework automatically detects the execution environment and selects the appropriate device.

## API Reference

### Exporting Models

```python
neurenix.wasm.export_to_wasm(
    model: neurenix.nn.Module,
    output_dir: str,
    model_name: str,
    optimize: bool = True,
    quantize: bool = False,
    simd: bool = True,
    threads: bool = True
) -> str
```

Exports a Neurenix model for WebAssembly execution.

**Parameters:**
- `model`: The model to export
- `output_dir`: Directory to save the exported model
- `model_name`: Name of the exported model
- `optimize`: Whether to optimize the model for WebAssembly
- `quantize`: Whether to quantize the model
- `simd`: Whether to enable SIMD instructions
- `threads`: Whether to enable threading

**Returns:**
- Path to the exported model

**Example:**
```python
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.wasm import export_to_wasm

# Create a model
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 5)
)

# Export the model for WebAssembly execution
output_dir = "webassembly_model"
export_path = export_to_wasm(model, output_dir, "my_model")
```

### Browser Execution

```python
neurenix.wasm.run_in_browser(
    model: neurenix.nn.Module,
    inputs: Union[Tensor, Dict[str, Tensor]],
    device: Optional[Device] = None
) -> Union[Tensor, Dict[str, Tensor]]
```

Marks a model for browser execution. This is a no-op in Python but signals intent for WebAssembly compilation.

**Parameters:**
- `model`: The model to run in the browser
- `inputs`: Input tensors
- `device`: Device to run the model on (usually WebGPU)

**Returns:**
- Output tensors (simulated in Python)

**Example:**
```python
from neurenix.device import Device, DeviceType
from neurenix.wasm import run_in_browser

# Mark the model for browser execution
result = run_in_browser(model, inputs, device=Device(DeviceType.WEBGPU))
```

### Custom WebGPU Shaders

```python
neurenix.wasm.register_custom_shader(
    name: str,
    shader_code: str
) -> None
```

Registers a custom WebGPU compute shader for use in WebAssembly context.

**Parameters:**
- `name`: Name of the shader
- `shader_code`: WGSL shader code

**Example:**
```python
from neurenix.wasm import register_custom_shader

# Register a custom WebGPU compute shader
register_custom_shader(
    "matrix_multiply",
    """
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> c: array<f32>;
    
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        // Custom matrix multiplication implementation
        // ...
    }
    """
)
```

### WebAssembly Configuration

```python
neurenix.wasm.get_wasm_config() -> Dict[str, Any]
```

Returns the current WebAssembly configuration.

**Returns:**
- Dictionary of configuration options

```python
neurenix.wasm.set_wasm_config(key: str, value: Any) -> None
```

Sets a WebAssembly configuration option.

**Parameters:**
- `key`: Configuration key
- `value`: Configuration value

**Example:**
```python
from neurenix.wasm import get_wasm_config, set_wasm_config

# Get current configuration
config = get_wasm_config()
print(f"Current WebAssembly configuration: {config}")

# Update configuration
set_wasm_config("memory_limit", 1024 * 1024 * 50)  # 50 MB
```

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **WebAssembly Support** | Native | TensorFlow.js |
| **WebGPU Integration** | Native | Limited |
| **API Consistency** | Same API for server and browser | Different APIs |
| **Model Export** | Simple export function | Complex conversion process |
| **Custom Shaders** | Supported | Limited |

Neurenix provides native WebAssembly support with seamless integration between server and client environments. TensorFlow requires using TensorFlow.js, which has a different API from the main TensorFlow library. Neurenix's approach allows developers to write code once and run it anywhere, while TensorFlow requires adapting code for different environments.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **WebAssembly Support** | Native | Limited (via third-party tools) |
| **WebGPU Integration** | Native | Not available |
| **API Consistency** | Same API for server and browser | Different APIs |
| **Model Export** | Simple export function | Complex conversion process |
| **Custom Shaders** | Supported | Not available |

PyTorch has limited WebAssembly support through third-party tools like ONNX.js or PyTorch.js, which are not as well-integrated as Neurenix's native support. Neurenix provides a more seamless experience with the same API for both server and browser environments.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **WebAssembly Support** | Native | None |
| **WebGPU Integration** | Native | None |
| **Browser Execution** | Supported | Not supported |
| **Model Export** | Simple export function | Not available |
| **Custom Shaders** | Supported | Not available |

Scikit-Learn does not provide WebAssembly support or browser execution capabilities. Neurenix fills this gap with its comprehensive WebAssembly module, allowing models to run directly in web browsers.

## Best Practices

### Model Size Optimization

For optimal performance in WebAssembly context, optimize model size:

```python
from neurenix.wasm import export_to_wasm

# Export with quantization for smaller model size
export_to_wasm(
    model,
    output_dir="webassembly_model",
    model_name="my_model",
    optimize=True,
    quantize=True  # Enable quantization
)
```

### Browser Compatibility

Check browser compatibility before using WebGPU:

```python
from neurenix.device import Device, DeviceType
from neurenix.wasm import run_in_browser

# Try WebGPU first, fall back to CPU if not available
try:
    result = run_in_browser(model, inputs, device=Device(DeviceType.WEBGPU))
except RuntimeError:
    result = run_in_browser(model, inputs, device=Device(DeviceType.CPU))
```

### Memory Management

Be mindful of memory usage in browser context:

```python
from neurenix.wasm import set_wasm_config

# Set memory limit to avoid browser crashes
set_wasm_config("memory_limit", 1024 * 1024 * 50)  # 50 MB

# Enable garbage collection
set_wasm_config("enable_gc", True)
```

## Tutorials

### Exporting and Running a Model in the Browser

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.wasm import export_to_wasm
import numpy as np

# Initialize Neurenix
nx.init()

# Create a simple model
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 5)
)

# Generate random input
x = nx.Tensor(np.random.randn(1, 10))

# Run the model normally
y_normal = model(x)
print(f"Normal output: {y_normal}")

# Export the model for WebAssembly execution
output_dir = "webassembly_model"
export_path = export_to_wasm(model, output_dir, "simple_model")
print(f"Model exported to: {export_path}")

# Generate HTML file for browser testing
html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Neurenix WebAssembly Demo</title>
</head>
<body>
    <h1>Neurenix WebAssembly Demo</h1>
    <div id="output"></div>
    
    <script type="module">
        import {{ loadModel }} from './{output_dir}/simple_model.js';
        
        async function runModel() {{
            const model = await loadModel('{output_dir}');
            const outputDiv = document.getElementById('output');
            
            outputDiv.innerHTML = `<p>Model loaded: ${{model.name}}</p>`;
            
            // Create input tensor (same as in Python example)
            const inputData = new Float32Array(10);
            for (let i = 0; i < 10; i++) {{
                inputData[i] = Math.random() * 2 - 1; // Random values between -1 and 1
            }}
            
            // Create input tensor
            const inputTensor = model.createTensor([1, 10], inputData);
            
            // Run the model
            console.log("Running model inference...");
            const startTime = performance.now();
            const outputTensor = await model.forward(inputTensor);
            const endTime = performance.now();
            
            // Get output data
            const outputData = await outputTensor.getData();
            
            // Display results
            outputDiv.innerHTML += `<p>Inference time: ${{(endTime - startTime).toFixed(2)}} ms</p>`;
            outputDiv.innerHTML += `<p>Input: [${{Array.from(inputData).map(x => x.toFixed(4)).join(', ')}}]</p>`;
            outputDiv.innerHTML += `<p>Output: [${{Array.from(outputData).map(x => x.toFixed(4)).join(', ')}}]</p>`;
        }}
        
        runModel().catch(error => {{
            document.getElementById('output').innerHTML = `<p>Error: ${{error.message}}</p>`;
        }});
    </script>
</body>
</html>
"""

with open("webassembly_demo.html", "w") as f:
    f.write(html)

print("HTML demo file created: webassembly_demo.html")
```

### Using WebGPU for Acceleration

```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.device import Device, DeviceType
from neurenix.wasm import export_to_wasm, register_custom_shader

# Initialize Neurenix
nx.init()

# Create a model
model = Sequential(
    Linear(1024, 1024),
    ReLU(),
    Linear(1024, 1024),
    ReLU(),
    Linear(1024, 10)
)

# Register a custom WebGPU shader for matrix multiplication
register_custom_shader(
    "optimized_matmul",
    """
    @group(0) @binding(0) var<storage, read> a: array<f32>;
    @group(0) @binding(1) var<storage, read> b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> c: array<f32>;
    
    @compute @workgroup_size(8, 8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let row = global_id.x;
        let col = global_id.y;
        let width = 1024;
        
        var sum = 0.0;
        for (var i = 0u; i < width; i = i + 1u) {
            sum = sum + a[row * width + i] * b[i * width + col];
        }
        
        c[row * width + col] = sum;
    }
    """
)

# Export the model with WebGPU optimization
export_to_wasm(
    model,
    output_dir="webgpu_model",
    model_name="webgpu_model",
    optimize=True,
    simd=True,
    threads=True
)

print("Model exported with WebGPU optimization")
```

## Conclusion

WebAssembly support in Neurenix provides a powerful way to run AI models directly in web browsers, offering reduced latency, enhanced privacy, and offline capabilities. By providing a seamless API that works across server and client environments, Neurenix makes it easy to develop AI applications that can run anywhere.
