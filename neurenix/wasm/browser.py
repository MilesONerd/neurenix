"""
Browser integration for the Neurenix framework.
"""

from typing import Optional, Dict, Any, Union
import os
import json

from neurenix.tensor import Tensor
from neurenix.device import Device, DeviceType
from neurenix.nn import Module

def run_in_browser(model: Module, inputs: Dict[str, Any], device: Optional[Device] = None) -> Dict[str, Any]:
    """
    Run a model in the browser using WebAssembly.
    
    This function is a no-op when not running in a WebAssembly context.
    It's used to mark a model for browser execution.
    
    Args:
        model: The model to run.
        inputs: The inputs to the model.
        device: The device to run the model on. If None, uses WebGPU if available.
        
    Returns:
        The outputs of the model.
    """
    if device is None:
        device = Device(DeviceType.WEBGPU)
        
    # When running in Python, this just runs the model normally
    # When compiled to WebAssembly, this will use the WebGPU backend
    return model(inputs)

def export_to_wasm(model: Module, output_dir: str, model_name: Optional[str] = None) -> str:
    """
    Export a model for WebAssembly execution.
    
    This function exports a model to be used in the browser with WebAssembly.
    
    Args:
        model: The model to export.
        output_dir: The directory to export the model to.
        model_name: The name of the model. If None, uses the class name.
        
    Returns:
        The path to the exported model.
    """
    if model_name is None:
        model_name = model.__class__.__name__
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Export model architecture
    model_config = {
        "name": model_name,
        "layers": []
    }
    
    # Serialize model parameters
    for name, param in model.named_parameters():
        # Add layer to model config
        layer_config = {
            "name": name,
            "shape": param.shape,
            "dtype": str(param.dtype)
        }
        model_config["layers"].append(layer_config)
        
        # Save parameter data
        param_path = os.path.join(output_dir, f"{name}.bin")
        with open(param_path, "wb") as f:
            f.write(param.cpu().numpy().tobytes())
    
    # Save model config
    config_path = os.path.join(output_dir, f"{model_name}.json")
    with open(config_path, "w") as f:
        json.dump(model_config, f, indent=2)
        
    # Generate JavaScript loader
    js_loader = f"""
// Neurenix WebAssembly model loader
export async function loadModel(modelPath) {{
    const response = await fetch(`${{modelPath}}/${{model_name}}.json`);
    const config = await response.json();
    
    const model = {{
        name: config.name,
        layers: {{}},
        async forward(inputs) {{
            // This will be replaced by the actual WebAssembly implementation
            // when the model is loaded in the browser
            throw new Error('Model not initialized');
        }}
    }};
    
    // Load model parameters
    for (const layer of config.layers) {{
        const paramResponse = await fetch(`${{modelPath}}/${{layer.name}}.bin`);
        const paramData = await paramResponse.arrayBuffer();
        model.layers[layer.name] = new Float32Array(paramData);
    }}
    
    return model;
}}
"""
    
    js_path = os.path.join(output_dir, f"{model_name}.js")
    with open(js_path, "w") as f:
        f.write(js_loader)
        
    return output_dir
