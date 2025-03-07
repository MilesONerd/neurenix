"""
Example of WebAssembly integration in Neurenix.

This example demonstrates how to export a model for WebAssembly execution
and run it in the browser.
"""

import os
import numpy as np

import neurenix as nx
from neurenix.nn import Linear, Sequential, ReLU
from neurenix.device import Device, DeviceType
from neurenix.wasm import export_to_wasm, run_in_browser

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

# Mark the model for browser execution (no-op in Python)
y_browser = run_in_browser(model, x, device=Device(DeviceType.WEBGPU))
print(f"Browser output (simulated): {y_browser}")

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
            outputDiv.innerHTML += `<p>Layers: ${{Object.keys(model.layers).join(', ')}}</p>`;
            
            // In a real implementation, you would run the model here
            // using WebAssembly
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
