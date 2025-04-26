"""
WASI-NN support for the Neurenix framework.

This module provides integration with the WebAssembly System Interface for Neural Networks
(WASI-NN), enabling efficient neural network inference in WebAssembly environments.
"""

from typing import Optional, List, Dict, Any, Union, Tuple
import os
import json
import numpy as np

from neurenix.tensor import Tensor
from neurenix.device import Device, DeviceType
from neurenix.nn import Module

WASI_NN_GRAPH_EXECUTION_MODE_DEFAULT = 0
WASI_NN_GRAPH_EXECUTION_MODE_STREAMING = 1

WASI_NN_TENSOR_TYPE_F16 = 0
WASI_NN_TENSOR_TYPE_F32 = 1
WASI_NN_TENSOR_TYPE_U8 = 2
WASI_NN_TENSOR_TYPE_I32 = 3
WASI_NN_TENSOR_TYPE_I8 = 4

WASI_NN_EXECUTION_TARGET_CPU = 0
WASI_NN_EXECUTION_TARGET_GPU = 1
WASI_NN_EXECUTION_TARGET_TPU = 2

class WasiNNModel:
    """
    A wrapper for models using WASI-NN for inference.
    """
    
    def __init__(self, model: Module, execution_target: int = WASI_NN_EXECUTION_TARGET_CPU):
        """
        Initialize a WASI-NN model.
        
        Args:
            model: The Neurenix model to wrap
            execution_target: The execution target (CPU, GPU, TPU)
        """
        self.model = model
        self.execution_target = execution_target
        self.graph_id = None
        self.context_id = None
        self._initialized = False
        
    def _initialize(self) -> bool:
        """
        Initialize the WASI-NN graph and context.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self._initialized:
            return True
            
        try:
            import wasi_nn
            
            from neurenix.onnx_support import export_to_onnx_bytes
            model_bytes = export_to_onnx_bytes(self.model)
            
            self.graph_id = wasi_nn.load(
                [model_bytes],
                wasi_nn.GRAPH_ENCODING_ONNX,
                self.execution_target
            )
            
            self.context_id = wasi_nn.init_execution_context(self.graph_id)
            self._initialized = True
            return True
            
        except ImportError:
            return False
        except Exception as e:
            print(f"WASI-NN initialization error: {e}")
            return False
    
    def forward(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Run inference using WASI-NN.
        
        Args:
            inputs: Dictionary of input tensors
            
        Returns:
            Dictionary of output tensors
        """
        if not self._initialize():
            return self.model(inputs)
            
        try:
            import wasi_nn
            
            for i, (name, tensor) in enumerate(inputs.items()):
                tensor_np = tensor.cpu().numpy()
                tensor_bytes = tensor_np.tobytes()
                
                if tensor_np.dtype == np.float32:
                    tensor_type = WASI_NN_TENSOR_TYPE_F32
                elif tensor_np.dtype == np.float16:
                    tensor_type = WASI_NN_TENSOR_TYPE_F16
                elif tensor_np.dtype == np.uint8:
                    tensor_type = WASI_NN_TENSOR_TYPE_U8
                elif tensor_np.dtype == np.int32:
                    tensor_type = WASI_NN_TENSOR_TYPE_I32
                elif tensor_np.dtype == np.int8:
                    tensor_type = WASI_NN_TENSOR_TYPE_I8
                else:
                    tensor_np = tensor_np.astype(np.float32)
                    tensor_bytes = tensor_np.tobytes()
                    tensor_type = WASI_NN_TENSOR_TYPE_F32
                
                wasi_nn.set_input(
                    self.context_id,
                    i,
                    tensor_type,
                    tensor_np.shape,
                    tensor_bytes
                )
            
            wasi_nn.compute(self.context_id)
            
            outputs = {}
            output_count = wasi_nn.get_output_count(self.context_id)
            
            for i in range(output_count):
                output_size = wasi_nn.get_output_size(self.context_id, i)
                output_bytes = wasi_nn.get_output(self.context_id, i, output_size)
                
                output_meta = self.model.output_info[i] if hasattr(self.model, 'output_info') else {}
                output_name = output_meta.get('name', f'output_{i}')
                output_shape = output_meta.get('shape', None)
                output_dtype = output_meta.get('dtype', np.float32)
                
                if output_shape is None:
                    if output_dtype == np.float32:
                        element_size = 4
                    elif output_dtype == np.float16:
                        element_size = 2
                    elif output_dtype in (np.uint8, np.int8):
                        element_size = 1
                    elif output_dtype == np.int32:
                        element_size = 4
                    else:
                        element_size = 4
                        output_dtype = np.float32
                    
                    total_elements = output_size // element_size
                    output_shape = (total_elements,)
                
                output_np = np.frombuffer(output_bytes, dtype=output_dtype).reshape(output_shape)
                
                outputs[output_name] = Tensor(output_np)
            
            return outputs
            
        except Exception as e:
            print(f"WASI-NN inference error: {e}")
            return self.model(inputs)

def is_wasi_nn_available() -> bool:
    """
    Check if WASI-NN is available in the current environment.
    
    Returns:
        bool: True if WASI-NN is available, False otherwise
    """
    try:
        import wasi_nn
        return True
    except ImportError:
        return False

def use_wasi_nn(model: Module, execution_target: int = WASI_NN_EXECUTION_TARGET_CPU) -> Union[WasiNNModel, Module]:
    """
    Wrap a model to use WASI-NN for inference if available.
    
    Args:
        model: The model to wrap
        execution_target: The execution target (CPU, GPU, TPU)
        
    Returns:
        A wrapped model that uses WASI-NN for inference if available,
        otherwise the original model
    """
    if is_wasi_nn_available():
        return WasiNNModel(model, execution_target)
    else:
        return model

def export_for_wasi_nn(model: Module, output_path: str) -> str:
    """
    Export a model for use with WASI-NN.
    
    Args:
        model: The model to export
        output_path: Path to save the exported model
        
    Returns:
        Path to the exported model
    """
    try:
        from neurenix.onnx_support import export_to_onnx
        return export_to_onnx(model, output_path)
    except ImportError:
        raise ImportError("ONNX support is required for WASI-NN export. Install onnx and onnxruntime packages.")
