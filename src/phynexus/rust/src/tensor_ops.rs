//! Tensor operations for the Phynexus engine

use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;
use crate::device::DeviceType;

/// Perform matrix multiplication: C = A * B
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Check that the tensors are on the same device
    if a.device() != b.device() {
        return Err(PhynexusError::DeviceMismatch(
            "Tensors must be on the same device for matmul".to_string()
        ));
    }
    
    // Get the dimensions of the matrices
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(PhynexusError::ShapeMismatch(
            "Tensors must be 2-dimensional for matmul".to_string()
        ));
    }
    
    // Check that the inner dimensions match
    if a_shape[1] != b_shape[0] {
        return Err(PhynexusError::ShapeMismatch(format!(
            "Inner dimensions must match for matmul: {} and {}",
            a_shape[1], b_shape[0]
        )));
    }
    
    // Create the output tensor
    let out_shape = vec![a_shape[0], b_shape[1]];
    let mut result = Tensor::new(out_shape, a.dtype(), a.device().clone())?;
    
    // Dispatch to the appropriate implementation based on the device
    match a.device().device_type() {
        DeviceType::CPU => {
            crate::ops::matmul::cpu_matmul(a, b, &mut result)?;
        },
        DeviceType::CUDA => {
            crate::ops::matmul::cuda_matmul(a, b, &mut result)?;
        },
        DeviceType::ROCm => {
            crate::ops::matmul::rocm_matmul(a, b, &mut result)?;
        },
        DeviceType::WebGPU => {
            crate::ops::matmul::webgpu_matmul(a, b, &mut result)?;
        },
        DeviceType::TPU => {
            crate::ops::tpu::tpu_matmul(a, b, &mut result)?;
        },
    }
    
    Ok(result)
}

/// Perform element-wise addition: C = A + B
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Check that the tensors are on the same device
    if a.device() != b.device() {
        return Err(PhynexusError::DeviceMismatch(
            "Tensors must be on the same device for add".to_string()
        ));
    }
    
    // Check that the shapes are compatible
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    // For now, we only support tensors with the same shape
    if a_shape != b_shape {
        return Err(PhynexusError::ShapeMismatch(format!(
            "Shapes must match for add: {:?} and {:?}",
            a_shape, b_shape
        )));
    }
    
    // Create the output tensor
    let mut result = Tensor::new(a_shape.to_vec(), a.dtype(), a.device().clone())?;
    
    // Dispatch to the appropriate implementation based on the device
    match a.device().device_type() {
        DeviceType::CPU => {
            crate::ops::elementwise::cpu_add(a, b, &mut result)?;
        },
        DeviceType::CUDA => {
            crate::ops::elementwise::cuda_add(a, b, &mut result)?;
        },
        DeviceType::ROCm => {
            crate::ops::elementwise::rocm_add(a, b, &mut result)?;
        },
        DeviceType::WebGPU => {
            crate::ops::elementwise::webgpu_add(a, b, &mut result)?;
        },
        DeviceType::TPU => {
            crate::ops::tpu::tpu_add(a, b, &mut result)?;
        },
    }
    
    Ok(result)
}

/// Perform element-wise subtraction: C = A - B
pub fn subtract(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Check that the tensors are on the same device
    if a.device() != b.device() {
        return Err(PhynexusError::DeviceMismatch(
            "Tensors must be on the same device for subtract".to_string()
        ));
    }
    
    // Check that the shapes are compatible
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    // For now, we only support tensors with the same shape
    if a_shape != b_shape {
        return Err(PhynexusError::ShapeMismatch(format!(
            "Shapes must match for subtract: {:?} and {:?}",
            a_shape, b_shape
        )));
    }
    
    // Create the output tensor
    let mut result = Tensor::new(a_shape.to_vec(), a.dtype(), a.device().clone())?;
    
    // Dispatch to the appropriate implementation based on the device
    match a.device().device_type() {
        DeviceType::CPU => {
            crate::ops::elementwise::cpu_subtract(a, b, &mut result)?;
        },
        DeviceType::CUDA => {
            crate::ops::elementwise::cuda_subtract(a, b, &mut result)?;
        },
        DeviceType::ROCm => {
            crate::ops::elementwise::rocm_subtract(a, b, &mut result)?;
        },
        DeviceType::WebGPU => {
            crate::ops::elementwise::webgpu_subtract(a, b, &mut result)?;
        },
        DeviceType::TPU => {
            crate::ops::tpu::tpu_subtract(a, b, &mut result)?;
        },
    }
    
    Ok(result)
}

/// Perform element-wise multiplication: C = A * B
pub fn multiply(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Check that the tensors are on the same device
    if a.device() != b.device() {
        return Err(PhynexusError::DeviceMismatch(
            "Tensors must be on the same device for multiply".to_string()
        ));
    }
    
    // Check that the shapes are compatible
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    // For now, we only support tensors with the same shape
    if a_shape != b_shape {
        return Err(PhynexusError::ShapeMismatch(format!(
            "Shapes must match for multiply: {:?} and {:?}",
            a_shape, b_shape
        )));
    }
    
    // Create the output tensor
    let mut result = Tensor::new(a_shape.to_vec(), a.dtype(), a.device().clone())?;
    
    // Dispatch to the appropriate implementation based on the device
    match a.device().device_type() {
        DeviceType::CPU => {
            crate::ops::elementwise::cpu_multiply(a, b, &mut result)?;
        },
        DeviceType::CUDA => {
            crate::ops::elementwise::cuda_multiply(a, b, &mut result)?;
        },
        DeviceType::ROCm => {
            crate::ops::elementwise::rocm_multiply(a, b, &mut result)?;
        },
        DeviceType::WebGPU => {
            crate::ops::elementwise::webgpu_multiply(a, b, &mut result)?;
        },
        DeviceType::TPU => {
            crate::ops::tpu::tpu_multiply(a, b, &mut result)?;
        },
    }
    
    Ok(result)
}
