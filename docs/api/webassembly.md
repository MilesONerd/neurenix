# WebAssembly Support in Neurenix

Neurenix provides native WebAssembly support, allowing AI models to run directly in web browsers. This feature enables client-side AI execution, reducing server load and latency while enhancing privacy by keeping data on the client device.

## Overview

WebAssembly (Wasm) is a binary instruction format designed as a portable compilation target for high-performance applications on the web. Neurenix leverages WebAssembly to enable:

- **Client-side AI execution**: Run models directly in the browser without server roundtrips
- **Reduced latency**: Eliminate network delays for inference
- **Enhanced privacy**: Keep sensitive data on the client device
- **Offline capabilities**: Run AI models without an internet connection
- **Reduced server costs**: Offload computation to client devices

The WebAssembly support in Neurenix is **optional** - developers can choose whether to run their models on the server or in the browser based on their specific requirements.

## Architecture

Neurenix's WebAssembly support is built on a flexible architecture:

1. **Hardware Abstraction Layer**: The Phynexus engine automatically detects the execution environment (server with CUDA/ROCm or client with WebAssembly/WebGPU)
2. **Conditional Compilation**: WebAssembly-specific code is only included when targeting WebAssembly
3. **WebGPU Integration**: Leverages the browser's WebGPU capabilities for hardware acceleration when available
4. **Seamless API**: The same API works across server and client environments

## Usage

### Exporting Models for WebAssembly

To export a Neurenix model for WebAssembly execution:

```python
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.wasm import export_to_wasm

# Create a model
model = Sequential(
    Linear(10, 20),
    ReLU(),
    Linear(20, 5)
)

# Train the model...

# Export the model for WebAssembly execution
output_dir = "webassembly_model"
export_path = export_to_wasm(model, output_dir, "my_model")
```

This will generate:
- A JSON file describing the model architecture
- Binary files containing the model parameters
- A JavaScript loader for easy integration in web applications

### Running Models in the Browser

To mark a model for browser execution:

```python
from neurenix.device import Device, DeviceType
from neurenix.wasm import run_in_browser

# Mark the model for browser execution
# This is a no-op in Python, but signals intent for WebAssembly compilation
result = run_in_browser(model, inputs, device=Device(DeviceType.WEBGPU))
```

### Browser Integration

To use the exported model in a web application:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Neurenix WebAssembly Demo</title>
</head>
<body>
    <h1>Neurenix WebAssembly Demo</h1>
    <div id="output"></div>
    
    <script type="module">
        import { loadModel } from './webassembly_model/my_model.js';
        
        async function runModel() {
            const model = await loadModel('webassembly_model');
            
            // Create input tensor
            const input = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]);
            
            // Run the model
            const output = await model.forward({ input });
            
            // Display the result
            document.getElementById('output').innerHTML = 
                `<p>Model output: ${JSON.stringify(output)}</p>`;
        }
        
        runModel();
    </script>
</body>
</html>
```

## Device Selection

Neurenix automatically detects the execution environment and selects the appropriate device:

```python
from neurenix.device import Device, DeviceType

# Create a WebGPU device
# In a browser context, this will use WebGPU
# In a server context, this will fall back to CPU if WebGPU is not available
device = Device(DeviceType.WEBGPU)

# Check if the device is available
if device.is_available():
    # Use the device
    model.to(device)
```

## Limitations

WebAssembly support in Neurenix has some limitations:

1. **Browser Compatibility**: WebGPU is not supported in all browsers yet
2. **Performance**: WebAssembly execution may be slower than native execution for complex models
3. **Memory Constraints**: Browsers have memory limitations that may affect large models
4. **Feature Support**: Some advanced features may not be available in WebAssembly context

## Browser Compatibility

WebGPU support varies by browser:

| Browser | WebGPU Support |
|---------|----------------|
| Chrome  | 113+ |
| Edge    | 113+ |
| Firefox | 118+ (behind flag) |
| Safari  | 17+ |
| Opera   | 99+ |

For browsers without WebGPU support, Neurenix will fall back to CPU execution using WebAssembly.

## Examples

See the [WebAssembly integration example](../../examples/webassembly_integration.py) for a complete demonstration of exporting and running a model in WebAssembly.

## Advanced Configuration

### WebAssembly Build Configuration

For advanced users, Neurenix provides configuration options for WebAssembly compilation:

```python
from neurenix.wasm import export_to_wasm

# Export with advanced configuration
export_to_wasm(
    model,
    output_dir="webassembly_model",
    model_name="my_model",
    optimize=True,  # Enable optimization
    quantize=False,  # Disable quantization
    simd=True,      # Enable SIMD instructions
    threads=True    # Enable threading
)
```

### Custom WebGPU Shaders

For performance-critical applications, Neurenix allows defining custom WebGPU compute shaders:

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

## Conclusion

WebAssembly support in Neurenix provides a powerful way to run AI models directly in web browsers, offering reduced latency, enhanced privacy, and offline capabilities. By providing a seamless API that works across server and client environments, Neurenix makes it easy to develop AI applications that can run anywhere.
