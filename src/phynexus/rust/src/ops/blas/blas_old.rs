//! Basic Linear Algebra Subprograms (BLAS) operations

use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;
use crate::device::DeviceType;

/// Perform general matrix multiplication (GEMM): C = alpha * A * B + beta * C
pub fn gemm(
    a: &Tensor,
    b: &Tensor,
    c: &mut Tensor,
    alpha: f32,
    beta: f32,
    transpose_a: bool,
    transpose_b: bool,
) -> Result<()> {
    // Check that the tensors are on the same device
    if a.device() != b.device() || a.device() != c.device() {
        return Err(PhynexusError::InvalidArgument(
            "All tensors must be on the same device for gemm".to_string()
        ));
    }
    
    // Get the dimensions of the matrices
    let a_shape = a.shape();
    let b_shape = b.shape();
    let c_shape = c.shape();
    
    if a_shape.len() != 2 || b_shape.len() != 2 || c_shape.len() != 2 {
        return Err(PhynexusError::ShapeMismatch(
            "All tensors must be 2-dimensional for gemm".to_string()
        ));
    }
    
    // Get the dimensions of the matrices after considering transposition
    let (m, k_a) = if transpose_a {
        (a_shape[1], a_shape[0])
    } else {
        (a_shape[0], a_shape[1])
    };
    
    let (k_b, n) = if transpose_b {
        (b_shape[1], b_shape[0])
    } else {
        (b_shape[0], b_shape[1])
    };
    
    // Check that the inner dimensions match
    if k_a != k_b {
        return Err(PhynexusError::ShapeMismatch(format!(
            "Inner dimensions must match for gemm: {} and {}",
            k_a, k_b
        )));
    }
    
    // Check that the output dimensions match
    if c_shape[0] != m || c_shape[1] != n {
        return Err(PhynexusError::ShapeMismatch(format!(
            "Output dimensions must match for gemm: expected ({}, {}), got ({}, {})",
            m, n, c_shape[0], c_shape[1]
        )));
    }
    
    // Dispatch to the appropriate implementation based on the device
    match a.device().device_type() {
        DeviceType::CPU => {
            let a_data = a.data_ptr::<f32>()?;
            let b_data = b.data_ptr::<f32>()?;
            let c_data = c.data_ptr_mut::<f32>()?;
            
            let lda = if transpose_a { a_shape[0] } else { a_shape[1] };
            let ldb = if transpose_b { b_shape[0] } else { b_shape[1] };
            let ldc = c_shape[1];
            
            for i in 0..m {
                for j in 0..n {
                    let c_idx = i * ldc + j;
                    if beta == 0.0 {
                        c_data[c_idx] = 0.0;
                    } else {
                        c_data[c_idx] *= beta;
                    }
                    
                    for k in 0..k_a {
                        let a_idx = if transpose_a { k * lda + i } else { i * lda + k };
                        let b_idx = if transpose_b { j * ldb + k } else { k * ldb + j };
                        
                        c_data[c_idx] += alpha * a_data[a_idx] * b_data[b_idx];
                    }
                }
            }
            
            Ok(())
        },
        DeviceType::CUDA => {
            #[cfg(feature = "cuda")]
            {
                Err(PhynexusError::UnsupportedOperation(
                    "CUDA GEMM implementation in progress".to_string()
                ))
            }
            
            #[cfg(not(feature = "cuda"))]
            {
                Err(PhynexusError::UnsupportedOperation(
                    "CUDA support not enabled".to_string()
                ))
            }
        },
        DeviceType::ROCm => {
            // TODO: Implement ROCm GEMM
            // For now, just return an error
            Err(PhynexusError::UnsupportedOperation(
                "ROCm GEMM not yet implemented".to_string()
            ))
        },
        DeviceType::WebGPU => {
            // TODO: Implement WebGPU GEMM
            // For now, just return an error
            Err(PhynexusError::UnsupportedOperation(
                "WebGPU GEMM not yet implemented".to_string()
            ))
        },
    }
}

/// Perform vector-vector dot product: result = x^T * y
pub fn dot(x: &Tensor, y: &Tensor) -> Result<f32> {
    // Check that the tensors are on the same device
    if x.device() != y.device() {
        return Err(PhynexusError::InvalidArgument(
            "Tensors must be on the same device for dot product".to_string()
        ));
    }
    
    // Check that the tensors are 1-dimensional
    let x_shape = x.shape();
    let y_shape = y.shape();
    
    if x_shape.len() != 1 || y_shape.len() != 1 {
        return Err(PhynexusError::ShapeMismatch(
            "Tensors must be 1-dimensional for dot product".to_string()
        ));
    }
    
    // Check that the dimensions match
    if x_shape[0] != y_shape[0] {
        return Err(PhynexusError::ShapeMismatch(format!(
            "Dimensions must match for dot product: {} and {}",
            x_shape[0], y_shape[0]
        )));
    }
    
    // Dispatch to the appropriate implementation based on the device
    match x.device().device_type() {
        DeviceType::CPU => {
            let x_data = x.data_ptr::<f32>()?;
            let y_data = y.data_ptr::<f32>()?;
            let n = x_shape[0];
            
            let mut result = 0.0;
            for i in 0..n {
                result += x_data[i] * y_data[i];
            }
            
            Ok(result)
        },
        DeviceType::CUDA => {
            #[cfg(feature = "cuda")]
            {
                Err(PhynexusError::UnsupportedOperation(
                    "CUDA dot product implementation in progress".to_string()
                ))
            }
            
            #[cfg(not(feature = "cuda"))]
            {
                Err(PhynexusError::UnsupportedOperation(
                    "CUDA support not enabled".to_string()
                ))
            }
        },
        DeviceType::ROCm => {
            // TODO: Implement ROCm dot product
            // For now, just return an error
            Err(PhynexusError::UnsupportedOperation(
                "ROCm dot product not yet implemented".to_string()
            ))
        },
        DeviceType::WebGPU => {
            // TODO: Implement WebGPU dot product
            // For now, just return an error
            Err(PhynexusError::UnsupportedOperation(
                "WebGPU dot product not yet implemented".to_string()
            ))
        },
    }
}

/// Perform matrix-vector multiplication: y = A * x
pub fn gemv(a: &Tensor, x: &Tensor, y: &mut Tensor, alpha: f32, beta: f32, transpose_a: bool) -> Result<()> {
    // Check that the tensors are on the same device
    if a.device() != x.device() || a.device() != y.device() {
        return Err(PhynexusError::InvalidArgument(
            "All tensors must be on the same device for gemv".to_string()
        ));
    }
    
    // Check that the tensors have the correct dimensions
    let a_shape = a.shape();
    let x_shape = x.shape();
    let y_shape = y.shape();
    
    if a_shape.len() != 2 || x_shape.len() != 1 || y_shape.len() != 1 {
        return Err(PhynexusError::ShapeMismatch(
            "Tensors must have correct dimensions for gemv".to_string()
        ));
    }
    
    // Get the dimensions of the matrix after considering transposition
    let (m, n) = if transpose_a {
        (a_shape[1], a_shape[0])
    } else {
        (a_shape[0], a_shape[1])
    };
    
    // Check that the dimensions match
    if n != x_shape[0] {
        return Err(PhynexusError::ShapeMismatch(format!(
            "Inner dimensions must match for gemv: {} and {}",
            n, x_shape[0]
        )));
    }
    
    if m != y_shape[0] {
        return Err(PhynexusError::ShapeMismatch(format!(
            "Output dimension must match for gemv: expected {}, got {}",
            m, y_shape[0]
        )));
    }
    
    // Dispatch to the appropriate implementation based on the device
    match a.device().device_type() {
        DeviceType::CPU => {
            let a_data = a.data_ptr::<f32>()?;
            let x_data = x.data_ptr::<f32>()?;
            let y_data = y.data_ptr_mut::<f32>()?;
            
            let lda = a_shape[1];
            
            if beta == 0.0 {
                for i in 0..m {
                    y_data[i] = 0.0;
                }
            } else if beta != 1.0 {
                for i in 0..m {
                    y_data[i] *= beta;
                }
            }
            
            if !transpose_a {
                for i in 0..m {
                    for j in 0..n {
                        y_data[i] += alpha * a_data[i * lda + j] * x_data[j];
                    }
                }
            } else {
                for i in 0..m {
                    for j in 0..n {
                        y_data[i] += alpha * a_data[j * lda + i] * x_data[j];
                    }
                }
            }
            
            Ok(())
        },
        DeviceType::CUDA => {
            #[cfg(feature = "cuda")]
            {
                Err(PhynexusError::UnsupportedOperation(
                    "CUDA GEMV implementation in progress".to_string()
                ))
            }
            
            #[cfg(not(feature = "cuda"))]
            {
                Err(PhynexusError::UnsupportedOperation(
                    "CUDA support not enabled".to_string()
                ))
            }
        },
        DeviceType::ROCm => {
            // TODO: Implement ROCm GEMV
            // For now, just return an error
            Err(PhynexusError::UnsupportedOperation(
                "ROCm GEMV not yet implemented".to_string()
            ))
        },
        DeviceType::WebGPU => {
            // TODO: Implement WebGPU GEMV
            // For now, just return an error
            Err(PhynexusError::UnsupportedOperation(
                "WebGPU GEMV not yet implemented".to_string()
            ))
        },
    }
}
