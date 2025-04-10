"""
Functional interface for neural network operations.
"""

import numpy as np
from typing import Optional, Union, Tuple, List

from neurenix.tensor import Tensor
from neurenix.device import DeviceType

def relu(x: Tensor, inplace: bool = False) -> Tensor:
    """
    Apply ReLU activation function.
    
    Args:
        x: Input tensor
        inplace: Whether to perform the operation in-place
        
    Returns:
        Result tensor
    """
    if inplace:
        x.data = np.maximum(x.data, 0)
        return x
    else:
        result = np.maximum(x.data, 0)
        return Tensor(result, device=x.device)

def sigmoid(x: Tensor) -> Tensor:
    """
    Apply sigmoid activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Result tensor
    """
    result = 1 / (1 + np.exp(-x.data))
    return Tensor(result, device=x.device)

def tanh(x: Tensor) -> Tensor:
    """
    Apply tanh activation function.
    
    Args:
        x: Input tensor
        
    Returns:
        Result tensor
    """
    result = np.tanh(x.data)
    return Tensor(result, device=x.device)

def softmax(x: Tensor, dim: int = 1) -> Tensor:
    """
    Apply softmax activation function.
    
    Args:
        x: Input tensor
        dim: Dimension along which to apply softmax
        
    Returns:
        Result tensor
    """
    exp_x = np.exp(x.data - np.max(x.data, axis=dim, keepdims=True))
    result = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
    return Tensor(result, device=x.device)

def log_softmax(x: Tensor, dim: int = 1) -> Tensor:
    """
    Apply log softmax activation function.
    
    Args:
        x: Input tensor
        dim: Dimension along which to apply log softmax
        
    Returns:
        Result tensor
    """
    max_val = np.max(x.data, axis=dim, keepdims=True)
    exp_x = np.exp(x.data - max_val)
    sum_exp_x = np.sum(exp_x, axis=dim, keepdims=True)
    result = x.data - max_val - np.log(sum_exp_x)
    return Tensor(result, device=x.device)

def leaky_relu(x: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    """
    Apply leaky ReLU activation function.
    
    Args:
        x: Input tensor
        negative_slope: Controls the angle of the negative slope
        inplace: Whether to perform the operation in-place
        
    Returns:
        Result tensor
    """
    if inplace:
        x.data = np.where(x.data > 0, x.data, x.data * negative_slope)
        return x
    else:
        result = np.where(x.data > 0, x.data, x.data * negative_slope)
        return Tensor(result, device=x.device)

def elu(x: Tensor, alpha: float = 1.0, inplace: bool = False) -> Tensor:
    """
    Apply ELU activation function.
    
    Args:
        x: Input tensor
        alpha: Controls the value to which the function saturates for negative inputs
        inplace: Whether to perform the operation in-place
        
    Returns:
        Result tensor
    """
    if inplace:
        x.data = np.where(
            x.data > 0,
            x.data,
            alpha * (np.exp(x.data) - 1)
        )
        return x
    else:
        result = np.where(
            x.data > 0,
            x.data,
            alpha * (np.exp(x.data) - 1)
        )
        return Tensor(result, device=x.device)

def selu(x: Tensor, inplace: bool = False) -> Tensor:
    """
    Apply SELU activation function.
    
    Args:
        x: Input tensor
        inplace: Whether to perform the operation in-place
        
    Returns:
        Result tensor
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    
    if inplace:
        x.data = scale * np.where(
            x.data > 0,
            x.data,
            alpha * (np.exp(x.data) - 1)
        )
        return x
    else:
        result = scale * np.where(
            x.data > 0,
            x.data,
            alpha * (np.exp(x.data) - 1)
        )
        return Tensor(result, device=x.device)

def gelu(x: Tensor, approximate: bool = False) -> Tensor:
    """
    Apply GELU activation function.
    
    Args:
        x: Input tensor
        approximate: Whether to use an approximation of the GELU function
        
    Returns:
        Result tensor
    """
    if approximate:
        sqrt_2_over_pi = np.sqrt(2 / np.pi)
        result = 0.5 * x.data * (1 + np.tanh(sqrt_2_over_pi * (x.data + 0.044715 * np.power(x.data, 3))))
    else:
        from scipy import special
        result = 0.5 * x.data * (1 + special.erf(x.data / np.sqrt(2)))
    
    return Tensor(result, device=x.device)

def dropout(x: Tensor, p: float = 0.5, training: bool = True) -> Tensor:
    """
    Apply dropout to an input tensor.
    
    Args:
        x: Input tensor
        p: Probability of an element to be zeroed
        training: Whether in training mode or not
        
    Returns:
        Output tensor
    """
    if not training or p == 0:
        return x
    
    mask = np.random.binomial(1, 1 - p, x.shape).astype(np.float32) / (1 - p)
    return Tensor(x.data * mask, device=x.device)

def conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, 
           stride: Union[int, Tuple[int, int]] = 1, 
           padding: Union[int, Tuple[int, int]] = 0,
           dilation: Union[int, Tuple[int, int]] = 1,
           groups: int = 1) -> Tensor:
    """
    Apply 2D convolution.
    
    Args:
        input: Input tensor of shape (batch_size, in_channels, height, width)
        weight: Weight tensor of shape (out_channels, in_channels // groups, kernel_height, kernel_width)
        bias: Optional bias tensor of shape (out_channels)
        stride: Stride of the convolution
        padding: Padding added to all sides of the input
        dilation: Spacing between kernel elements
        groups: Number of blocked connections from input channels to output channels
        
    Returns:
        Output tensor
    """
    from neurenix.core import get_config
    
    tensor_cores_enabled = get_config().get("tensor_cores_enabled", False)
    
    if (tensor_cores_enabled and 
        input.device.type == DeviceType.TENSOR_CORES and 
        weight.device.type == DeviceType.TENSOR_CORES):
        try:
            from neurenix.binding import tensor_cores_conv2d
            return tensor_cores_conv2d(input, weight, bias, stride, padding, dilation, groups)
        except (ImportError, AttributeError):
            pass
    
    try:
        from neurenix.binding import conv2d as binding_conv2d
        return binding_conv2d(input, weight, bias, stride, padding, dilation, groups)
    except (ImportError, AttributeError):
        batch_size, in_channels, height, width = input.shape
        out_channels, _, kernel_height, kernel_width = weight.shape
        
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        
        output_height = ((height + 2 * padding[0] - dilation[0] * (kernel_height - 1) - 1) // stride[0]) + 1
        output_width = ((width + 2 * padding[1] - dilation[1] * (kernel_width - 1) - 1) // stride[1]) + 1
        
        output = Tensor.zeros((batch_size, out_channels, output_height, output_width), device=input.device)
        
        return output

def max_pool2d(input: Tensor, kernel_size: Union[int, Tuple[int, int]],
               stride: Optional[Union[int, Tuple[int, int]]] = None,
               padding: Union[int, Tuple[int, int]] = 0,
               dilation: Union[int, Tuple[int, int]] = 1,
               return_indices: bool = False,
               ceil_mode: bool = False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Apply 2D max pooling.
    
    Args:
        input: Input tensor of shape (batch_size, channels, height, width)
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window. Default: kernel_size
        padding: Padding added to both sides of the input. Default: 0
        dilation: Controls the spacing between kernel elements. Default: 1
        return_indices: If True, will return the indices along with the outputs
        ceil_mode: If True, will use ceil instead of floor to compute the output shape
        
    Returns:
        Output tensor, or tuple of output tensor and indices tensor if return_indices is True
    """
    from neurenix.core import get_config
    
    tensor_cores_enabled = get_config().get("tensor_cores_enabled", False)
    
    if (tensor_cores_enabled and 
        input.device.type == DeviceType.TENSOR_CORES):
        try:
            from neurenix.binding import tensor_cores_max_pool2d
            if return_indices:
                return tensor_cores_max_pool2d(input, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
            else:
                return tensor_cores_max_pool2d(input, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        except (ImportError, AttributeError):
            pass
    
    try:
        from neurenix.binding import max_pool2d as binding_max_pool2d
        return binding_max_pool2d(input, kernel_size, stride, padding, dilation, return_indices, ceil_mode)
    except (ImportError, AttributeError):
        batch_size, channels, height, width = input.shape
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if stride is None:
            stride = kernel_size
        elif isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        
        if ceil_mode:
            output_height = int(np.ceil((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1))
            output_width = int(np.ceil((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1))
        else:
            output_height = int(np.floor((height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1))
            output_width = int(np.floor((width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1))
        
        output = Tensor.zeros((batch_size, channels, output_height, output_width), device=input.device)
        
        if return_indices:
            indices = Tensor.zeros((batch_size, channels, output_height, output_width), device=input.device)
            return output, indices
        else:
            return output

def cosine_similarity(x1: Tensor, x2: Tensor, dim: int = 1, eps: float = 1e-8) -> Tensor:
    """
    Returns cosine similarity between x1 and x2, computed along dim.
    
    Args:
        x1: First input tensor
        x2: Second input tensor
        dim: Dimension along which to compute cosine similarity
        eps: Small value to avoid division by zero
        
    Returns:
        Cosine similarity tensor
    """
    try:
        from neurenix.core import PhynexusExtension
        if PhynexusExtension.is_available():
            return PhynexusExtension.cosine_similarity(x1, x2, dim, eps)
    except (ImportError, AttributeError):
        pass
    
    x1_data = x1.data
    x2_data = x2.data
    
    x1_norm = np.sqrt(np.sum(x1_data ** 2, axis=dim, keepdims=True))
    x2_norm = np.sqrt(np.sum(x2_data ** 2, axis=dim, keepdims=True))
    
    x1_normalized = x1_data / np.maximum(x1_norm, eps)
    x2_normalized = x2_data / np.maximum(x2_norm, eps)
    
    cos_sim = np.sum(x1_normalized * x2_normalized, axis=dim)
    
    return Tensor(cos_sim, device=x1.device)

def pairwise_distance(x1: Tensor, x2: Tensor, p: float = 2.0, eps: float = 1e-6) -> Tensor:
    """
    Computes the pairwise distance between vectors v1,v2 using the p-norm.
    
    Args:
        x1: First input tensor
        x2: Second input tensor
        p: The norm degree. Default: 2
        eps: Small value to avoid division by zero
        
    Returns:
        Pairwise distance tensor
    """
    try:
        from neurenix.core import PhynexusExtension
        if PhynexusExtension.is_available():
            return PhynexusExtension.pairwise_distance(x1, x2, p, eps)
    except (ImportError, AttributeError):
        pass
    
    diff = x1.data - x2.data
    
    if p == float('inf'):
        dist = np.max(np.abs(diff), axis=1)
    else:
        dist = np.sum(np.abs(diff) ** p, axis=1) ** (1 / p)
    
    return Tensor(dist, device=x1.device)

def normalize(input: Tensor, p: float = 2.0, dim: int = 1, eps: float = 1e-12) -> Tensor:
    """
    Performs L_p normalization of inputs over specified dimension.
    
    Args:
        input: Input tensor
        p: The exponent value in the norm formulation. Default: 2
        dim: The dimension to reduce. Default: 1
        eps: Small value to avoid division by zero. Default: 1e-12
        
    Returns:
        Normalized tensor
    """
    try:
        from neurenix.core import PhynexusExtension
        if PhynexusExtension.is_available():
            return PhynexusExtension.normalize(input, p, dim, eps)
    except (ImportError, AttributeError):
        pass
    
    x = input.data
    
    if p == float('inf'):
        norm = np.max(np.abs(x), axis=dim, keepdims=True)
    else:
        norm = np.sum(np.abs(x) ** p, axis=dim, keepdims=True) ** (1 / p)
    
    return Tensor(x / np.maximum(norm, eps), device=input.device)
