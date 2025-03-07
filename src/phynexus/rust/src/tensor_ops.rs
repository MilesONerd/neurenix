//! High-level tensor operations for the Phynexus engine

use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;
use crate::device::Device;
use crate::ops;

/// Perform matrix multiplication between two tensors
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Check that the devices match
    if a.device() != b.device() {
        return Err(PhynexusError::DeviceMismatch(format!(
            "Cannot multiply tensors on different devices: {:?} and {:?}",
            a.device(), b.device()
        )));
    }
    
    // Check that the shapes are compatible for matrix multiplication
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(PhynexusError::ShapeMismatch(format!(
            "Tensors must have at least 2 dimensions for matmul, got {:?} and {:?}",
            a_shape, b_shape
        )));
    }
    
    if a_shape[a_shape.len() - 1] != b_shape[b_shape.len() - 2] {
        return Err(PhynexusError::ShapeMismatch(format!(
            "Incompatible dimensions for matmul: {:?} and {:?}",
            a_shape, b_shape
        )));
    }
    
    // Determine the output shape
    let mut out_shape = Vec::new();
    
    // Handle batch dimensions
    let a_batch_dims = &a_shape[0..a_shape.len() - 2];
    let b_batch_dims = &b_shape[0..b_shape.len() - 2];
    
    // Broadcast batch dimensions
    let batch_dims = broadcast_shapes(a_batch_dims, b_batch_dims)?;
    out_shape.extend(batch_dims);
    
    // Add matrix dimensions
    out_shape.push(a_shape[a_shape.len() - 2]);
    out_shape.push(b_shape[b_shape.len() - 1]);
    
    // Create output tensor
    let mut result = Tensor::new(out_shape, a.dtype(), a.device().clone())?;
    
    // Perform the matrix multiplication using the appropriate backend
    match a.device().device_type() {
        crate::device::DeviceType::CPU => {
            ops::matmul::cpu_matmul(a, b, &mut result)?;
        },
        crate::device::DeviceType::CUDA => {
            ops::matmul::cuda_matmul(a, b, &mut result)?;
        },
        crate::device::DeviceType::ROCm => {
            ops::matmul::rocm_matmul(a, b, &mut result)?;
        },
        crate::device::DeviceType::WebGPU => {
            ops::matmul::webgpu_matmul(a, b, &mut result)?;
        },
    }
    
    Ok(result)
}

/// Add two tensors element-wise
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Check that the devices match
    if a.device() != b.device() {
        return Err(PhynexusError::DeviceMismatch(format!(
            "Cannot add tensors on different devices: {:?} and {:?}",
            a.device(), b.device()
        )));
    }
    
    // Check that the shapes are compatible for addition (broadcasting)
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    // Determine the output shape with broadcasting
    let out_shape = broadcast_shapes(a_shape, b_shape)?;
    
    // Create output tensor
    let mut result = Tensor::new(out_shape, a.dtype(), a.device().clone())?;
    
    // Perform the addition using the appropriate backend
    match a.device().device_type() {
        crate::device::DeviceType::CPU => {
            ops::elementwise::cpu_add(a, b, &mut result)?;
        },
        crate::device::DeviceType::CUDA => {
            ops::elementwise::cuda_add(a, b, &mut result)?;
        },
        crate::device::DeviceType::ROCm => {
            ops::elementwise::rocm_add(a, b, &mut result)?;
        },
        crate::device::DeviceType::WebGPU => {
            ops::elementwise::webgpu_add(a, b, &mut result)?;
        },
    }
    
    Ok(result)
}

/// Subtract two tensors element-wise
pub fn subtract(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Check that the devices match
    if a.device() != b.device() {
        return Err(PhynexusError::DeviceMismatch(format!(
            "Cannot subtract tensors on different devices: {:?} and {:?}",
            a.device(), b.device()
        )));
    }
    
    // Check that the shapes are compatible for subtraction (broadcasting)
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    // Determine the output shape with broadcasting
    let out_shape = broadcast_shapes(a_shape, b_shape)?;
    
    // Create output tensor
    let mut result = Tensor::new(out_shape, a.dtype(), a.device().clone())?;
    
    // Perform the subtraction using the appropriate backend
    match a.device().device_type() {
        crate::device::DeviceType::CPU => {
            ops::elementwise::cpu_subtract(a, b, &mut result)?;
        },
        crate::device::DeviceType::CUDA => {
            ops::elementwise::cuda_subtract(a, b, &mut result)?;
        },
        crate::device::DeviceType::ROCm => {
            ops::elementwise::rocm_subtract(a, b, &mut result)?;
        },
        crate::device::DeviceType::WebGPU => {
            ops::elementwise::webgpu_subtract(a, b, &mut result)?;
        },
    }
    
    Ok(result)
}

/// Multiply two tensors element-wise
pub fn multiply(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Check that the devices match
    if a.device() != b.device() {
        return Err(PhynexusError::DeviceMismatch(format!(
            "Cannot multiply tensors on different devices: {:?} and {:?}",
            a.device(), b.device()
        )));
    }
    
    // Check that the shapes are compatible for multiplication (broadcasting)
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    // Determine the output shape with broadcasting
    let out_shape = broadcast_shapes(a_shape, b_shape)?;
    
    // Create output tensor
    let mut result = Tensor::new(out_shape, a.dtype(), a.device().clone())?;
    
    // Perform the multiplication using the appropriate backend
    match a.device().device_type() {
        crate::device::DeviceType::CPU => {
            ops::elementwise::cpu_multiply(a, b, &mut result)?;
        },
        crate::device::DeviceType::CUDA => {
            ops::elementwise::cuda_multiply(a, b, &mut result)?;
        },
        crate::device::DeviceType::ROCm => {
            ops::elementwise::rocm_multiply(a, b, &mut result)?;
        },
        crate::device::DeviceType::WebGPU => {
            ops::elementwise::webgpu_multiply(a, b, &mut result)?;
        },
    }
    
    Ok(result)
}

/// Divide two tensors element-wise
pub fn divide(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Check that the devices match
    if a.device() != b.device() {
        return Err(PhynexusError::DeviceMismatch(format!(
            "Cannot divide tensors on different devices: {:?} and {:?}",
            a.device(), b.device()
        )));
    }
    
    // Check that the shapes are compatible for division (broadcasting)
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    // Determine the output shape with broadcasting
    let out_shape = broadcast_shapes(a_shape, b_shape)?;
    
    // Create output tensor
    let mut result = Tensor::new(out_shape, a.dtype(), a.device().clone())?;
    
    // Perform the division using the appropriate backend
    match a.device().device_type() {
        crate::device::DeviceType::CPU => {
            ops::elementwise::cpu_divide(a, b, &mut result)?;
        },
        crate::device::DeviceType::CUDA => {
            ops::elementwise::cuda_divide(a, b, &mut result)?;
        },
        crate::device::DeviceType::ROCm => {
            ops::elementwise::rocm_divide(a, b, &mut result)?;
        },
        crate::device::DeviceType::WebGPU => {
            ops::elementwise::webgpu_divide(a, b, &mut result)?;
        },
    }
    
    Ok(result)
}

/// Apply a reduction operation along specified dimensions
pub fn reduce(tensor: &Tensor, op: ops::reduction::ReductionOp, dims: &[usize], keep_dims: bool) -> Result<Tensor> {
    let shape = tensor.shape();
    
    // Validate dimensions
    for &dim in dims {
        if dim >= shape.len() {
            return Err(PhynexusError::InvalidArgument(format!(
                "Reduction dimension {} is out of bounds for tensor with {} dimensions",
                dim, shape.len()
            )));
        }
    }
    
    // Compute output shape
    let mut out_shape = Vec::new();
    for (i, &dim) in shape.iter().enumerate() {
        if !dims.contains(&i) {
            out_shape.push(dim);
        } else if keep_dims {
            out_shape.push(1);
        }
    }
    
    // Create output tensor
    let mut result = Tensor::new(out_shape, tensor.dtype(), tensor.device().clone())?;
    
    // Perform the reduction using the appropriate backend
    match tensor.device().device_type() {
        crate::device::DeviceType::CPU => {
            ops::reduction::cpu_reduce(tensor, &mut result, op, dims, keep_dims)?;
        },
        crate::device::DeviceType::CUDA => {
            ops::reduction::cuda_reduce(tensor, &mut result, op, dims, keep_dims)?;
        },
        crate::device::DeviceType::ROCm => {
            ops::reduction::rocm_reduce(tensor, &mut result, op, dims, keep_dims)?;
        },
        crate::device::DeviceType::WebGPU => {
            ops::reduction::webgpu_reduce(tensor, &mut result, op, dims, keep_dims)?;
        },
    }
    
    Ok(result)
}

/// Compute the sum of a tensor along specified dimensions
pub fn sum(tensor: &Tensor, dims: &[usize], keep_dims: bool) -> Result<Tensor> {
    reduce(tensor, ops::reduction::ReductionOp::Sum, dims, keep_dims)
}

/// Compute the mean of a tensor along specified dimensions
pub fn mean(tensor: &Tensor, dims: &[usize], keep_dims: bool) -> Result<Tensor> {
    reduce(tensor, ops::reduction::ReductionOp::Mean, dims, keep_dims)
}

/// Compute the maximum of a tensor along specified dimensions
pub fn max(tensor: &Tensor, dims: &[usize], keep_dims: bool) -> Result<Tensor> {
    reduce(tensor, ops::reduction::ReductionOp::Max, dims, keep_dims)
}

/// Compute the minimum of a tensor along specified dimensions
pub fn min(tensor: &Tensor, dims: &[usize], keep_dims: bool) -> Result<Tensor> {
    reduce(tensor, ops::reduction::ReductionOp::Min, dims, keep_dims)
}

/// Apply an activation function to a tensor
pub fn activate(tensor: &Tensor, activation: ops::activation::ActivationType) -> Result<Tensor> {
    // Create output tensor with the same shape
    let mut result = Tensor::new(tensor.shape().to_vec(), tensor.dtype(), tensor.device().clone())?;
    
    // Apply the activation function using the appropriate backend
    match tensor.device().device_type() {
        crate::device::DeviceType::CPU => {
            ops::activation::cpu_activate(tensor, &mut result, activation)?;
        },
        crate::device::DeviceType::CUDA => {
            ops::activation::cuda_activate(tensor, &mut result, activation)?;
        },
        crate::device::DeviceType::ROCm => {
            ops::activation::rocm_activate(tensor, &mut result, activation)?;
        },
        crate::device::DeviceType::WebGPU => {
            ops::activation::webgpu_activate(tensor, &mut result, activation)?;
        },
    }
    
    Ok(result)
}

/// Apply ReLU activation function to a tensor
pub fn relu(tensor: &Tensor) -> Result<Tensor> {
    activate(tensor, ops::activation::ActivationType::ReLU)
}

/// Apply sigmoid activation function to a tensor
pub fn sigmoid(tensor: &Tensor) -> Result<Tensor> {
    activate(tensor, ops::activation::ActivationType::Sigmoid)
}

/// Apply tanh activation function to a tensor
pub fn tanh(tensor: &Tensor) -> Result<Tensor> {
    activate(tensor, ops::activation::ActivationType::Tanh)
}

/// Perform convolution operation
pub fn conv(input: &Tensor, weight: &Tensor, bias: Option<&Tensor>, stride: &[usize], padding: &[usize], dilation: &[usize], groups: usize) -> Result<Tensor> {
    // Check that the devices match
    if input.device() != weight.device() {
        return Err(PhynexusError::DeviceMismatch(format!(
            "Cannot perform convolution with tensors on different devices: {:?} and {:?}",
            input.device(), weight.device()
        )));
    }
    
    if let Some(bias) = bias {
        if input.device() != bias.device() {
            return Err(PhynexusError::DeviceMismatch(format!(
                "Cannot perform convolution with tensors on different devices: {:?} and {:?}",
                input.device(), bias.device()
            )));
        }
    }
    
    // Validate input shapes and compute output shape
    let (batch_size, in_channels, input_dims) = validate_conv_input(input)?;
    let (out_channels, kernel_channels, kernel_dims) = validate_conv_weight(weight, in_channels, groups)?;
    
    if input_dims.len() != stride.len() || input_dims.len() != padding.len() || input_dims.len() != dilation.len() {
        return Err(PhynexusError::InvalidArgument(format!(
            "Convolution parameters must have the same dimensionality as the input spatial dimensions"
        )));
    }
    
    // Compute output spatial dimensions
    let mut output_dims = Vec::new();
    for i in 0..input_dims.len() {
        let output_dim = (input_dims[i] + 2 * padding[i] - dilation[i] * (kernel_dims[i] - 1) - 1) / stride[i] + 1;
        output_dims.push(output_dim);
    }
    
    // Create output shape: [batch_size, out_channels, output_dims...]
    let mut output_shape = vec![batch_size, out_channels];
    output_shape.extend(output_dims);
    
    // Create output tensor
    let mut result = Tensor::new(output_shape, input.dtype(), input.device().clone())?;
    
    // Perform the convolution using the appropriate backend
    match input.device().device_type() {
        crate::device::DeviceType::CPU => {
            ops::conv::cpu_conv(input, weight, bias, &mut result, stride, padding, dilation, groups)?;
        },
        crate::device::DeviceType::CUDA => {
            ops::conv::cuda_conv(input, weight, bias, &mut result, stride, padding, dilation, groups)?;
        },
        crate::device::DeviceType::ROCm => {
            ops::conv::rocm_conv(input, weight, bias, &mut result, stride, padding, dilation, groups)?;
        },
        crate::device::DeviceType::WebGPU => {
            ops::conv::webgpu_conv(input, weight, bias, &mut result, stride, padding, dilation, groups)?;
        },
    }
    
    Ok(result)
}

/// Transpose a tensor by swapping two dimensions
pub fn transpose(tensor: &Tensor, dim0: usize, dim1: usize) -> Result<Tensor> {
    let shape = tensor.shape();
    
    // Validate dimensions
    if dim0 >= shape.len() || dim1 >= shape.len() {
        return Err(PhynexusError::InvalidArgument(format!(
            "Transpose dimensions ({}, {}) out of bounds for tensor with {} dimensions",
            dim0, dim1, shape.len()
        )));
    }
    
    // Compute output shape
    let mut out_shape = shape.to_vec();
    out_shape.swap(dim0, dim1);
    
    // Create output tensor
    let mut result = Tensor::new(out_shape, tensor.dtype(), tensor.device().clone())?;
    
    // Perform the transpose using the appropriate backend
    match tensor.device().device_type() {
        crate::device::DeviceType::CPU => {
            ops::blas::cpu_transpose(tensor, &mut result, dim0, dim1)?;
        },
        crate::device::DeviceType::CUDA => {
            ops::blas::cuda_transpose(tensor, &mut result, dim0, dim1)?;
        },
        crate::device::DeviceType::ROCm => {
            ops::blas::rocm_transpose(tensor, &mut result, dim0, dim1)?;
        },
        crate::device::DeviceType::WebGPU => {
            ops::blas::webgpu_transpose(tensor, &mut result, dim0, dim1)?;
        },
    }
    
    Ok(result)
}

/// Reshape a tensor to a new shape
pub fn reshape(tensor: &Tensor, new_shape: &[usize]) -> Result<Tensor> {
    let old_shape = tensor.shape();
    
    // Compute the total number of elements
    let old_size: usize = old_shape.iter().product();
    let mut new_size: usize = 1;
    let mut infer_dim = None;
    
    for (i, &dim) in new_shape.iter().enumerate() {
        if dim == 0 {
            // Keep the original dimension
            if i >= old_shape.len() {
                return Err(PhynexusError::InvalidArgument(format!(
                    "Cannot use 0 for dimension {} in reshape because the original tensor has only {} dimensions",
                    i, old_shape.len()
                )));
            }
            new_size *= old_shape[i];
        } else if dim == usize::MAX {
            // Infer this dimension
            if infer_dim.is_some() {
                return Err(PhynexusError::InvalidArgument(format!(
                    "Cannot infer multiple dimensions in reshape"
                )));
            }
            infer_dim = Some(i);
        } else {
            // Use the specified dimension
            new_size *= dim;
        }
    }
    
    // Compute the inferred dimension if needed
    let mut final_shape = new_shape.to_vec();
    if let Some(dim) = infer_dim {
        if new_size == 0 {
            final_shape[dim] = old_size;
        } else {
            if old_size % new_size != 0 {
                return Err(PhynexusError::InvalidArgument(format!(
                    "Cannot reshape tensor of size {} to shape {:?} with inferred dimension",
                    old_size, new_shape
                )));
            }
            final_shape[dim] = old_size / new_size;
        }
    } else if new_size != old_size {
        return Err(PhynexusError::InvalidArgument(format!(
            "Cannot reshape tensor of size {} to shape {:?} with size {}",
            old_size, new_shape, new_size
        )));
    }
    
    // Create a new tensor with the reshaped data
    // In a real implementation, we would avoid copying the data if possible
    let mut result = Tensor::new(final_shape, tensor.dtype(), tensor.device().clone())?;
    
    // Copy the data
    // This is a simplified implementation; in reality, we would use a more efficient approach
    match tensor.device().device_type() {
        crate::device::DeviceType::CPU => {
            ops::blas::cpu_copy(tensor, &mut result)?;
        },
        crate::device::DeviceType::CUDA => {
            ops::blas::cuda_copy(tensor, &mut result)?;
        },
        crate::device::DeviceType::ROCm => {
            ops::blas::rocm_copy(tensor, &mut result)?;
        },
        crate::device::DeviceType::WebGPU => {
            ops::blas::webgpu_copy(tensor, &mut result)?;
        },
    }
    
    Ok(result)
}

/// Helper function to compute the output shape when broadcasting two shapes
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let a_len = a.len();
    let b_len = b.len();
    let max_len = a_len.max(b_len);
    
    let mut result = Vec::with_capacity(max_len);
    
    for i in 0..max_len {
        let a_dim = if i < a_len { a[a_len - 1 - i] } else { 1 };
        let b_dim = if i < b_len { b[b_len - 1 - i] } else { 1 };
        
        if a_dim == b_dim || a_dim == 1 || b_dim == 1 {
            result.push(a_dim.max(b_dim));
        } else {
            return Err(PhynexusError::ShapeMismatch(format!(
                "Cannot broadcast shapes {:?} and {:?}",
                a, b
            )));
        }
    }
    
    // Reverse the result since we computed it from the end
    result.reverse();
    
    Ok(result)
}

/// Helper function to validate convolution input tensor
fn validate_conv_input(input: &Tensor) -> Result<(usize, usize, Vec<usize>)> {
    let shape = input.shape();
    
    if shape.len() < 3 {
        return Err(PhynexusError::InvalidArgument(format!(
            "Convolution input must have at least 3 dimensions (batch, channels, spatial), got {:?}",
            shape
        )));
    }
    
    let batch_size = shape[0];
    let in_channels = shape[1];
    let input_dims = shape[2..].to_vec();
    
    Ok((batch_size, in_channels, input_dims))
}

/// Helper function to validate convolution weight tensor
fn validate_conv_weight(weight: &Tensor, in_channels: usize, groups: usize) -> Result<(usize, usize, Vec<usize>)> {
    let shape = weight.shape();
    
    if shape.len() < 3 {
        return Err(PhynexusError::InvalidArgument(format!(
            "Convolution weight must have at least 3 dimensions (out_channels, in_channels/groups, spatial), got {:?}",
            shape
        )));
    }
    
    let out_channels = shape[0];
    let kernel_channels = shape[1];
    let kernel_dims = shape[2..].to_vec();
    
    if in_channels % groups != 0 {
        return Err(PhynexusError::InvalidArgument(format!(
            "Input channels ({}) must be divisible by groups ({})",
            in_channels, groups
        )));
    }
    
    if out_channels % groups != 0 {
        return Err(PhynexusError::InvalidArgument(format!(
            "Output channels ({}) must be divisible by groups ({})",
            out_channels, groups
        )));
    }
    
    if kernel_channels != in_channels / groups {
        return Err(PhynexusError::InvalidArgument(format!(
            "Convolution weight channels ({}) must be equal to input_channels/groups ({}/{}={})",
            kernel_channels, in_channels, groups, in_channels / groups
        )));
    }
    
    Ok((out_channels, kernel_channels, kernel_dims))
}
