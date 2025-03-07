//! Matrix multiplication operations

use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;

/// Perform matrix multiplication between two tensors
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Check that the tensors are on the same device
    if a.device() != b.device() {
        return Err(PhynexusError::InvalidArgument(
            "Tensors must be on the same device for matmul".to_string()
        ));
    }
    
    // Check that the tensors have compatible shapes for matmul
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    if a_shape.len() < 2 || b_shape.len() < 2 {
        return Err(PhynexusError::ShapeMismatch(
            "Tensors must have at least 2 dimensions for matmul".to_string()
        ));
    }
    
    if a_shape[a_shape.len() - 1] != b_shape[b_shape.len() - 2] {
        return Err(PhynexusError::ShapeMismatch(format!(
            "Incompatible dimensions for matmul: {} and {}",
            a_shape[a_shape.len() - 1],
            b_shape[b_shape.len() - 2]
        )));
    }
    
    // Calculate the output shape
    let mut out_shape = Vec::new();
    
    // Handle broadcasting for batch dimensions
    let a_batch_dims = &a_shape[..a_shape.len() - 2];
    let b_batch_dims = &b_shape[..b_shape.len() - 2];
    
    // TODO: Implement proper broadcasting for batch dimensions
    
    // For now, just require the same batch dimensions
    if a_batch_dims != b_batch_dims {
        return Err(PhynexusError::ShapeMismatch(
            "Different batch dimensions not yet supported for matmul".to_string()
        ));
    }
    
    out_shape.extend_from_slice(a_batch_dims);
    out_shape.push(a_shape[a_shape.len() - 2]);
    out_shape.push(b_shape[b_shape.len() - 1]);
    
    // Create the output tensor
    let mut out = Tensor::new(out_shape, a.dtype(), a.device().clone())?;
    
    // TODO: Implement the actual matrix multiplication
    // This would depend on the device and would be dispatched accordingly
    
    Ok(out)
}

/// Perform batched matrix multiplication between two tensors
pub fn batch_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    matmul(a, b)
}
