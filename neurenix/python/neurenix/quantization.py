"""
Quantization module for the Neurenix framework.

This module provides functionality for quantizing models to lower precision
formats (INT8, FP16, FP8) and model pruning for improved performance and
reduced memory footprint.
"""

from typing import Optional, Dict, Any, Union, Tuple, List, Callable
import numpy as np
import math
import logging

from neurenix.tensor import Tensor
from neurenix.device import Device, DeviceType
from neurenix.nn import Module

logger = logging.getLogger(__name__)

class QuantizationType:
    INT8 = "int8"
    FP16 = "fp16"
    FP8 = "fp8"
    
    @staticmethod
    def all_types():
        return [QuantizationType.INT8, QuantizationType.FP16, QuantizationType.FP8]

class QuantizedTensor:
    """
    A wrapper for quantized tensors.
    """
    
    def __init__(self, tensor: Tensor, scale: float, zero_point: int, dtype: str):
        """
        Initialize a quantized tensor.
        
        Args:
            tensor: The quantized tensor data
            scale: The scale factor used for quantization
            zero_point: The zero point used for quantization
            dtype: The quantization data type
        """
        self.tensor = tensor
        self.scale = scale
        self.zero_point = zero_point
        self.dtype = dtype
    
    def dequantize(self) -> Tensor:
        """
        Dequantize the tensor back to full precision.
        
        Returns:
            Tensor: Dequantized tensor
        """
        if self.dtype == QuantizationType.INT8:
            return (self.tensor.float() - self.zero_point) * self.scale
        elif self.dtype == QuantizationType.FP16:
            return self.tensor.float()
        elif self.dtype == QuantizationType.FP8:
            return self.tensor.float()
        else:
            raise ValueError(f"Unsupported quantization type: {self.dtype}")

def quantize_tensor(tensor: Tensor, dtype: str = QuantizationType.INT8) -> QuantizedTensor:
    """
    Quantize a tensor to the specified data type.
    
    Args:
        tensor: The tensor to quantize
        dtype: The quantization data type
        
    Returns:
        QuantizedTensor: Quantized tensor
    """
    if dtype == QuantizationType.INT8:
        tensor_np = tensor.cpu().numpy()
        min_val = float(tensor_np.min())
        max_val = float(tensor_np.max())
        
        scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0
        zero_point = round(127 - max_val / scale) if scale != 0 else 0
        zero_point = max(0, min(255, zero_point))
        
        quantized_np = np.clip(np.round(tensor_np / scale + zero_point), 0, 255).astype(np.uint8)
        quantized_tensor = Tensor(quantized_np, device=tensor.device)
        
        return QuantizedTensor(quantized_tensor, scale, zero_point, dtype)
        
    elif dtype == QuantizationType.FP16:
        tensor_np = tensor.cpu().numpy()
        fp16_np = tensor_np.astype(np.float16)
        fp16_tensor = Tensor(fp16_np, device=tensor.device)
        
        return QuantizedTensor(fp16_tensor, 1.0, 0, dtype)
        
    elif dtype == QuantizationType.FP8:
        tensor_np = tensor.cpu().numpy()
        
        abs_max = np.max(np.abs(tensor_np))
        
        scale = 127.0 / abs_max if abs_max > 0 else 1.0
        
        scaled_np = np.round(tensor_np * scale).astype(np.int8)
        fp8_tensor = Tensor(scaled_np, device=tensor.device)
        
        return QuantizedTensor(fp8_tensor, 1.0/scale, 0, dtype)
        
    else:
        raise ValueError(f"Unsupported quantization type: {dtype}")

class QuantizedModule(Module):
    """
    A wrapper for quantized modules.
    """
    
    def __init__(self, module: Module, dtype: str = QuantizationType.INT8):
        """
        Initialize a quantized module.
        
        Args:
            module: The module to quantize
            dtype: The quantization data type
        """
        super().__init__()
        self.original_module = module
        self.dtype = dtype
        self.quantized_params = {}
        self.quantize_module()
    
    def quantize_module(self):
        """
        Quantize the module parameters.
        """
        for name, param in self.original_module.named_parameters():
            self.quantized_params[name] = quantize_tensor(param, self.dtype)
    
    def forward(self, *args, **kwargs):
        """
        Forward pass using the quantized parameters.
        
        This method temporarily replaces the original parameters with
        dequantized versions, performs the forward pass, and then
        restores the original parameters.
        """
        original_params = {}
        for name, param in self.original_module.named_parameters():
            original_params[name] = param.clone()
        
        for name, quantized_param in self.quantized_params.items():
            param = getattr(self.original_module, name.split('.')[0])
            for part in name.split('.')[1:-1]:
                param = getattr(param, part)
            setattr(param, name.split('.')[-1], quantized_param.dequantize())
        
        output = self.original_module(*args, **kwargs)
        
        for name, param in original_params.items():
            param_obj = getattr(self.original_module, name.split('.')[0])
            for part in name.split('.')[1:-1]:
                param_obj = getattr(param_obj, part)
            setattr(param_obj, name.split('.')[-1], param)
        
        return output

def quantize_model(model: Module, dtype: str = QuantizationType.INT8) -> QuantizedModule:
    """
    Quantize a model to the specified data type.
    
    Args:
        model: The model to quantize
        dtype: The quantization data type
        
    Returns:
        QuantizedModule: Quantized model
    """
    return QuantizedModule(model, dtype)

def quantize_model_per_layer(model: Module, dtype_map: Dict[str, str]) -> Module:
    """
    Quantize a model with different quantization types per layer.
    
    Args:
        model: The model to quantize
        dtype_map: Dictionary mapping layer names to quantization types
        
    Returns:
        Module: Quantized model
    """
    quantized_model = type(model)()
    
    for attr_name in dir(model):
        if not attr_name.startswith('_') and attr_name not in ['parameters', 'named_parameters', 'modules', 'named_modules']:
            attr = getattr(model, attr_name)
            if not callable(attr) and not isinstance(attr, Module):
                setattr(quantized_model, attr_name, attr)
    
    for name, module in model.named_modules():
        if name in dtype_map:
            quantized_module = QuantizedModule(module, dtype_map[name])
            
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = quantized_model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, quantized_module)
            else:
                setattr(quantized_model, name, quantized_module)
        else:
            if '.' in name:
                parent_name, child_name = name.rsplit('.', 1)
                parent = quantized_model
                for part in parent_name.split('.'):
                    parent = getattr(parent, part)
                setattr(parent, child_name, module)
            else:
                setattr(quantized_model, name, module)
    
    return quantized_model

def prune_model(model: Module, sparsity: float = 0.5, method: str = 'magnitude') -> Module:
    """
    Prune a model by setting small weights to zero.
    
    Args:
        model: The model to prune
        sparsity: The target sparsity (fraction of weights to prune)
        method: The pruning method ('magnitude', 'random')
        
    Returns:
        Module: Pruned model
    """
    pruned_model = type(model)()
    
    for attr_name in dir(model):
        if not attr_name.startswith('_') and attr_name not in ['parameters', 'named_parameters', 'modules', 'named_modules']:
            attr = getattr(model, attr_name)
            if not callable(attr) and not isinstance(attr, Module):
                setattr(pruned_model, attr_name, attr)
    
    for name, param in model.named_parameters():
        tensor = param.clone()
        tensor_np = tensor.cpu().numpy()
        
        if method == 'magnitude':
            abs_weights = np.abs(tensor_np.flatten())
            sorted_weights = np.sort(abs_weights)
            
            threshold_idx = int(len(sorted_weights) * sparsity)
            threshold = sorted_weights[threshold_idx] if threshold_idx < len(sorted_weights) else 0
            
            mask = np.abs(tensor_np) > threshold
            
        elif method == 'random':
            mask = np.random.rand(*tensor_np.shape) > sparsity
            
        else:
            raise ValueError(f"Unsupported pruning method: {method}")
        
        pruned_tensor_np = tensor_np * mask
        pruned_tensor = Tensor(pruned_tensor_np, device=tensor.device)
        
        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            parent = pruned_model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            setattr(parent, child_name, pruned_tensor)
        else:
            setattr(pruned_model, name, pruned_tensor)
    
    return pruned_model

def quantization_aware_training(model: Module, dtype: str = QuantizationType.INT8) -> Module:
    """
    Prepare a model for quantization-aware training.
    
    This method inserts fake quantization operations during training
    to simulate the effects of quantization.
    
    Args:
        model: The model to prepare for quantization-aware training
        dtype: The quantization data type
        
    Returns:
        Module: Model prepared for quantization-aware training
    """
    qat_model = type(model)()
    
    for attr_name in dir(model):
        if not attr_name.startswith('_') and attr_name not in ['parameters', 'named_parameters', 'modules', 'named_modules']:
            attr = getattr(model, attr_name)
            if not callable(attr) and not isinstance(attr, Module):
                setattr(qat_model, attr_name, attr)
    
    for name, param in model.named_parameters():
        quantized_param = quantize_tensor(param, dtype)
        dequantized_param = quantized_param.dequantize()
        
        if '.' in name:
            parent_name, child_name = name.rsplit('.', 1)
            parent = qat_model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            setattr(parent, child_name, dequantized_param)
        else:
            setattr(qat_model, name, dequantized_param)
    
    return qat_model

def calibrate_model(model: Module, calibration_data: List[Dict[str, Tensor]], 
                   dtype: str = QuantizationType.INT8, num_batches: int = 10) -> Dict[str, Tuple[float, int]]:
    """
    Calibrate a model for post-training quantization.
    
    This method computes optimal scale and zero point values for each
    layer based on representative data.
    
    Args:
        model: The model to calibrate
        calibration_data: List of input batches for calibration
        dtype: The quantization data type
        num_batches: Number of batches to use for calibration
        
    Returns:
        Dict: Mapping from parameter names to (scale, zero_point) tuples
    """
    activation_ranges = {}
    
    hooks = []
    
    def hook_fn(name):
        def fn(module, input, output):
            if isinstance(output, Tensor):
                output_np = output.cpu().numpy()
                min_val = float(output_np.min())
                max_val = float(output_np.max())
                
                if name in activation_ranges:
                    activation_ranges[name][0] = min(activation_ranges[name][0], min_val)
                    activation_ranges[name][1] = max(activation_ranges[name][1], max_val)
                else:
                    activation_ranges[name] = [min_val, max_val]
        return fn
    
    for name, module in model.named_modules():
        if not isinstance(module, Module):
            continue
        hook = module.register_forward_hook(hook_fn(name))
        hooks.append(hook)
    
    model.eval()
    with Tensor.no_grad():
        for i, batch in enumerate(calibration_data):
            if i >= num_batches:
                break
            model(**batch)
    
    for hook in hooks:
        hook.remove()
    
    calibration_params = {}
    for name, (min_val, max_val) in activation_ranges.items():
        if dtype == QuantizationType.INT8:
            scale = (max_val - min_val) / 255.0 if max_val > min_val else 1.0
            zero_point = round(127 - max_val / scale) if scale != 0 else 0
            zero_point = max(0, min(255, zero_point))
            
            calibration_params[name] = (scale, zero_point)
            
        elif dtype == QuantizationType.FP16:
            calibration_params[name] = (1.0, 0)
            
        elif dtype == QuantizationType.FP8:
            abs_max = max(abs(min_val), abs(max_val))
            scale = 127.0 / abs_max if abs_max > 0 else 1.0
            
            calibration_params[name] = (1.0/scale, 0)
            
        else:
            raise ValueError(f"Unsupported quantization type: {dtype}")
    
    return calibration_params
