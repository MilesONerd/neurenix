"""
ONNX support module for the Neurenix framework.

This module provides functionality for importing and exporting models
between Neurenix and the ONNX format, enabling interoperability with
other machine learning frameworks.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple, Union, Any, Sequence

import numpy as np

from neurenix.nn import Module
from neurenix.tensor import Tensor
from neurenix.device import Device, DeviceType

logger = logging.getLogger("neurenix")

class ONNXConverter:
    """
    Converter for importing and exporting models between Neurenix and ONNX.
    """
    
    def __init__(self):
        """Initialize the ONNX converter."""
        self._supported_ops = self._get_supported_ops()
        
    def _get_supported_ops(self) -> Dict[str, Any]:
        """Get a dictionary of supported ONNX operations and their Neurenix equivalents."""
        from neurenix.nn import (
            Conv2d, Linear, BatchNorm2d, MaxPool2d, AvgPool2d,
            ReLU, Sigmoid, Tanh, LeakyReLU, Dropout
        )
        
        return {
            "Conv": Conv2d,
            
            "Gemm": Linear,
            "MatMul": Linear,
            
            "BatchNormalization": BatchNorm2d,
            
            "MaxPool": MaxPool2d,
            "AveragePool": AvgPool2d,
            
            "Relu": ReLU,
            "Sigmoid": Sigmoid,
            "Tanh": Tanh,
            "LeakyRelu": LeakyReLU,
            
            "Dropout": Dropout,
            
            "Add": lambda: None,  # Handled by Tensor.__add__
            "Mul": lambda: None,  # Handled by Tensor.__mul__
            "Sub": lambda: None,  # Handled by Tensor.__sub__
            "Div": lambda: None,  # Handled by Tensor.__truediv__
            
            "Reshape": lambda: None,  # Handled by Tensor.reshape
            "Transpose": lambda: None,  # Handled by Tensor.transpose
            "Flatten": lambda: None,  # Handled by custom flatten function
            
            "ReduceMean": lambda: None,  # Handled by Tensor.mean
            "ReduceSum": lambda: None,  # Handled by Tensor.sum
        }
    
    def export_model(self, model: Module, input_shape: Sequence[int], 
                    output_path: str, opset_version: int = 12,
                    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None) -> str:
        """
        Export a Neurenix model to ONNX format.
        
        Args:
            model: The Neurenix model to export.
            input_shape: The shape of the input tensor.
            output_path: Path to save the ONNX model.
            opset_version: ONNX opset version to use.
            dynamic_axes: Dictionary specifying dynamic axes for inputs/outputs.
                          Example: {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            
        Returns:
            Path to the exported ONNX model.
        """
        try:
            import onnx
            import torch
            import torch.onnx
        except ImportError:
            raise ImportError("ONNX export requires PyTorch and ONNX. "
                             "Please install them with: pip install torch onnx")
        
        torch_model = self._create_torch_wrapper(model)
        
        dummy_input = torch.randn(*input_shape)
        
        torch.onnx.export(
            torch_model,
            dummy_input,
            output_path,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=dynamic_axes
        )
        
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"Model exported to ONNX format at {output_path}")
        return output_path
    
    def import_model(self, onnx_path: str, device: Optional[Device] = None) -> Module:
        """
        Import a model from ONNX format to Neurenix.
        
        Args:
            onnx_path: Path to the ONNX model.
            device: Device to load the model on.
            
        Returns:
            A Neurenix model.
        """
        try:
            import onnx
        except ImportError:
            raise ImportError("ONNX import requires ONNX. "
                             "Please install it with: pip install onnx")
        
        onnx_model = onnx.load(onnx_path)
        
        onnx.checker.check_model(onnx_model)
        
        neurenix_model = self._convert_onnx_to_neurenix(onnx_model, device)
        
        logger.info(f"Model imported from ONNX format at {onnx_path}")
        return neurenix_model
    
    def _create_torch_wrapper(self, model: Module) -> Any:
        """
        Create a PyTorch wrapper for a Neurenix model.
        
        Args:
            model: The Neurenix model to wrap.
            
        Returns:
            A PyTorch module that wraps the Neurenix model.
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required for ONNX export. "
                             "Please install it with: pip install torch")
        
        class NeurenixTorchWrapper(torch.nn.Module):
            def __init__(self, neurenix_model):
                super().__init__()
                self.neurenix_model = neurenix_model
                
            def forward(self, x):
                neurenix_input = Tensor(x.detach().cpu().numpy())
                
                neurenix_output = self.neurenix_model(neurenix_input)
                
                return torch.tensor(neurenix_output.numpy())
        
        return NeurenixTorchWrapper(model)
    
    def _convert_onnx_to_neurenix(self, onnx_model, device: Optional[Device] = None) -> Module:
        """
        Convert an ONNX model to a Neurenix model.
        
        Args:
            onnx_model: The ONNX model to convert.
            device: Device to load the model on.
            
        Returns:
            A Neurenix model.
        """
        from neurenix.nn import Module, Sequential
        
        neurenix_model = Sequential()
        
        graph = onnx_model.graph
        
        tensor_map = {}
        
        for initializer in graph.initializer:
            tensor = self._extract_tensor_from_initializer(initializer)
            tensor_map[initializer.name] = tensor
        
        for node in graph.node:
            if node.op_type not in self._supported_ops:
                logger.warning(f"Skipping unsupported operation: {node.op_type}")
                continue
            
            layer = self._create_layer_from_node(node, tensor_map, device)
            
            if layer is not None:
                neurenix_model.add_module(node.name, layer)
        
        return neurenix_model
    
    def _extract_tensor_from_initializer(self, initializer) -> np.ndarray:
        """
        Extract a numpy array from an ONNX initializer.
        
        Args:
            initializer: ONNX initializer.
            
        Returns:
            Numpy array.
        """
        import onnx
        import numpy as np
        
        if initializer.data_type == onnx.TensorProto.FLOAT:
            return np.frombuffer(initializer.raw_data, dtype=np.float32).reshape(initializer.dims)
        elif initializer.data_type == onnx.TensorProto.DOUBLE:
            return np.frombuffer(initializer.raw_data, dtype=np.float64).reshape(initializer.dims)
        elif initializer.data_type == onnx.TensorProto.INT32:
            return np.frombuffer(initializer.raw_data, dtype=np.int32).reshape(initializer.dims)
        elif initializer.data_type == onnx.TensorProto.INT64:
            return np.frombuffer(initializer.raw_data, dtype=np.int64).reshape(initializer.dims)
        else:
            raise ValueError(f"Unsupported data type: {initializer.data_type}")
    
    def _create_layer_from_node(self, node, tensor_map, device):
        """
        Create a Neurenix layer from an ONNX node.
        
        Args:
            node: ONNX node.
            tensor_map: Map of value_info names to tensors.
            device: Device to load the model on.
            
        Returns:
            A Neurenix layer.
        """
        from neurenix.nn import Conv2d, Linear, BatchNorm2d, MaxPool2d, AvgPool2d
        
        op_type = node.op_type
        
        attrs = {attr.name: self._get_attribute_value(attr) for attr in node.attribute}
        
        if op_type == "Conv":
            weights = tensor_map[node.input[1]]
            bias = tensor_map[node.input[2]] if len(node.input) > 2 else None
            
            in_channels = weights.shape[1]
            out_channels = weights.shape[0]
            kernel_size = weights.shape[2:4]
            stride = attrs.get("strides", (1, 1))
            padding = attrs.get("pads", (0, 0, 0, 0))[:2]  # Only use the first two padding values
            
            layer = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias is not None
            )
            
            layer.weight.data = Tensor(weights, device=device)
            if bias is not None:
                layer.bias.data = Tensor(bias, device=device)
            
            return layer
        
        elif op_type == "Gemm" or op_type == "MatMul":
            weights = tensor_map[node.input[1]]
            bias = tensor_map[node.input[2]] if len(node.input) > 2 else None
            
            in_features = weights.shape[1]
            out_features = weights.shape[0]
            
            layer = Linear(
                in_features=in_features,
                out_features=out_features,
                bias=bias is not None
            )
            
            layer.weight.data = Tensor(weights, device=device)
            if bias is not None:
                layer.bias.data = Tensor(bias, device=device)
            
            return layer
        
        elif op_type == "BatchNormalization":
            scale = tensor_map[node.input[1]]
            bias = tensor_map[node.input[2]]
            running_mean = tensor_map[node.input[3]]
            running_var = tensor_map[node.input[4]]
            
            num_features = scale.shape[0]
            eps = attrs.get("epsilon", 1e-5)
            momentum = attrs.get("momentum", 0.1)
            
            layer = BatchNorm2d(
                num_features=num_features,
                eps=eps,
                momentum=momentum
            )
            
            layer.weight.data = Tensor(scale, device=device)
            layer.bias.data = Tensor(bias, device=device)
            layer.running_mean.data = Tensor(running_mean, device=device)
            layer.running_var.data = Tensor(running_var, device=device)
            
            return layer
        
        elif op_type == "MaxPool":
            kernel_size = attrs.get("kernel_shape", (2, 2))
            stride = attrs.get("strides", kernel_size)
            padding = attrs.get("pads", (0, 0, 0, 0))[:2]  # Only use the first two padding values
            
            layer = MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            
            return layer
        
        elif op_type == "AveragePool":
            kernel_size = attrs.get("kernel_shape", (2, 2))
            stride = attrs.get("strides", kernel_size)
            padding = attrs.get("pads", (0, 0, 0, 0))[:2]  # Only use the first two padding values
            
            layer = AvgPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding
            )
            
            return layer
        
        elif op_type in ["Relu", "Sigmoid", "Tanh", "LeakyRelu"]:
            return self._supported_ops[op_type]()
        
        elif op_type == "Dropout":
            p = attrs.get("ratio", 0.5)
            
            return self._supported_ops[op_type](p=p)
        
        return None
    
    def _get_attribute_value(self, attr):
        """
        Get the value of an ONNX attribute.
        
        Args:
            attr: ONNX attribute.
            
        Returns:
            Attribute value.
        """
        import onnx
        
        if attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode('utf-8')
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.STRINGS:
            return [s.decode('utf-8') for s in attr.strings]
        else:
            raise ValueError(f"Unsupported attribute type: {attr.type}")


def to_onnx(model: Module, input_shape: Sequence[int], output_path: str, 
           opset_version: int = 12, dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None) -> str:
    """
    Export a Neurenix model to ONNX format.
    
    Args:
        model: The Neurenix model to export.
        input_shape: The shape of the input tensor.
        output_path: Path to save the ONNX model.
        opset_version: ONNX opset version to use.
        dynamic_axes: Dictionary specifying dynamic axes for inputs/outputs.
                      Example: {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        
    Returns:
        Path to the exported ONNX model.
    """
    converter = ONNXConverter()
    return converter.export_model(model, input_shape, output_path, opset_version, dynamic_axes)


def from_onnx(onnx_path: str, device: Optional[Device] = None) -> Module:
    """
    Import a model from ONNX format to Neurenix.
    
    Args:
        onnx_path: Path to the ONNX model.
        device: Device to load the model on.
        
    Returns:
        A Neurenix model.
    """
    converter = ONNXConverter()
    return converter.import_model(onnx_path, device)
