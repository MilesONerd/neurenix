//! BLAS operations for the Phynexus engine

mod transpose;
mod copy;

pub use transpose::*;
pub use copy::*;

use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;
use crate::device::DeviceType;

/// Perform general matrix multiplication (GEMM)
pub fn gemm(
    a: &Tensor,
    b: &Tensor,
    c: &mut Tensor,
    alpha: f32,
    beta: f32,
    transpose_a: bool,
    transpose_b: bool,
) -> Result<()> {
    let device = a.device_type()?;
    
    let a_shape = a.shape()?;
    let b_shape = b.shape()?;
    let c_shape = c.shape()?;
    
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(PhynexusError::DimensionMismatch(
            format!("GEMM requires 2D tensors, got {}D and {}D", a_shape.len(), b_shape.len())
        ));
    }
    
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
    
    if k_a != k_b {
        return Err(PhynexusError::DimensionMismatch(
            format!("Inner dimensions must match for GEMM: {} != {}", k_a, k_b)
        ));
    }
    
    if c_shape.len() != 2 || c_shape[0] != m || c_shape[1] != n {
        return Err(PhynexusError::DimensionMismatch(
            format!("Output tensor shape {:?} does not match expected shape [{}, {}]", c_shape, m, n)
        ));
    }
    
    // Dispatch to device-specific implementation
    match device {
        DeviceType::CPU => {
            use crate::ops::matmul::cpu_matmul;
            cpu_matmul(a, b, c, alpha, beta, transpose_a, transpose_b)
        },
        DeviceType::CUDA => {
            use crate::ops::matmul::cuda_matmul;
            cuda_matmul(a, b, c, alpha, beta, transpose_a, transpose_b)
        },
        DeviceType::ROCm => {
            use crate::ops::matmul::rocm_matmul;
            rocm_matmul(a, b, c, alpha, beta, transpose_a, transpose_b)
        },
        DeviceType::WebGPU => {
            use crate::ops::matmul::webgpu_matmul;
            webgpu_matmul(a, b, c, alpha, beta, transpose_a, transpose_b)
        },
        DeviceType::TPU => {
            use crate::ops::matmul::tpu_matmul;
            tpu_matmul(a, b, c, alpha, beta, transpose_a, transpose_b)
        },
        _ => {
            Err(PhynexusError::UnsupportedOperation(
                format!("GEMM not supported on device: {:?}", device)
            ))
        }
    }
}

/// Perform vector-vector dot product
pub fn dot(x: &Tensor, y: &Tensor) -> Result<f32> {
    let device = x.device_type()?;
    
    let x_shape = x.shape()?;
    let y_shape = y.shape()?;
    
    if x_shape.len() != 1 || y_shape.len() != 1 {
        return Err(PhynexusError::DimensionMismatch(
            format!("Dot product requires 1D tensors, got {}D and {}D", x_shape.len(), y_shape.len())
        ));
    }
    
    if x_shape[0] != y_shape[0] {
        return Err(PhynexusError::DimensionMismatch(
            format!("Vector dimensions must match for dot product: {} != {}", x_shape[0], y_shape[0])
        ));
    }
    
    let mut result = Tensor::zeros(&[1], device)?;
    
    // Dispatch to device-specific implementation
    match device {
        DeviceType::CPU => {
            let x_data = x.data_cpu()?;
            let y_data = y.data_cpu()?;
            
            let mut dot_result = 0.0;
            for i in 0..x_shape[0] {
                dot_result += x_data[i] * y_data[i];
            }
            
            Ok(dot_result)
        },
        DeviceType::CUDA => {
            use crate::ops::matmul::cuda_dot;
            cuda_dot(x, y)
        },
        DeviceType::ROCm => {
            use crate::ops::matmul::rocm_dot;
            rocm_dot(x, y)
        },
        DeviceType::WebGPU => {
            use crate::ops::matmul::webgpu_dot;
            webgpu_dot(x, y)
        },
        DeviceType::TPU => {
            use crate::ops::matmul::tpu_dot;
            tpu_dot(x, y)
        },
        _ => {
            Err(PhynexusError::UnsupportedOperation(
                format!("Dot product not supported on device: {:?}", device)
            ))
        }
    }
}

/// Perform matrix-vector multiplication
pub fn gemv(
    a: &Tensor,
    x: &Tensor,
    y: &mut Tensor,
    alpha: f32,
    beta: f32,
    transpose_a: bool,
) -> Result<()> {
    let device = a.device_type()?;
    
    let a_shape = a.shape()?;
    let x_shape = x.shape()?;
    let y_shape = y.shape()?;
    
    if a_shape.len() != 2 {
        return Err(PhynexusError::DimensionMismatch(
            format!("GEMV requires a 2D matrix, got {}D", a_shape.len())
        ));
    }
    
    if x_shape.len() != 1 {
        return Err(PhynexusError::DimensionMismatch(
            format!("GEMV requires a 1D vector for x, got {}D", x_shape.len())
        ));
    }
    
    if y_shape.len() != 1 {
        return Err(PhynexusError::DimensionMismatch(
            format!("GEMV requires a 1D vector for y, got {}D", y_shape.len())
        ));
    }
    
    let (m, n) = if transpose_a {
        (a_shape[1], a_shape[0])
    } else {
        (a_shape[0], a_shape[1])
    };
    
    if x_shape[0] != n {
        return Err(PhynexusError::DimensionMismatch(
            format!("Vector x dimension {} does not match matrix dimension {}", x_shape[0], n)
        ));
    }
    
    if y_shape[0] != m {
        return Err(PhynexusError::DimensionMismatch(
            format!("Vector y dimension {} does not match matrix dimension {}", y_shape[0], m)
        ));
    }
    
    // Dispatch to device-specific implementation
    match device {
        DeviceType::CPU => {
            // CPU implementation
            let a_data = a.data_cpu()?;
            let x_data = x.data_cpu()?;
            let mut y_data = y.data_cpu_mut()?;
            
            for i in 0..m {
                y_data[i] *= beta;
            }
            
            if !transpose_a {
                for i in 0..m {
                    for j in 0..n {
                        y_data[i] += alpha * a_data[i * n + j] * x_data[j];
                    }
                }
            } else {
                for j in 0..n {
                    for i in 0..m {
                        y_data[i] += alpha * a_data[j * m + i] * x_data[j];
                    }
                }
            }
            
            Ok(())
        },
        DeviceType::CUDA => {
            use crate::ops::matmul::cuda_gemv;
            cuda_gemv(a, x, y, alpha, beta, transpose_a)
        },
        DeviceType::ROCm => {
            use crate::ops::matmul::rocm_gemv;
            rocm_gemv(a, x, y, alpha, beta, transpose_a)
        },
        DeviceType::WebGPU => {
            use crate::ops::matmul::webgpu_gemv;
            webgpu_gemv(a, x, y, alpha, beta, transpose_a)
        },
        DeviceType::TPU => {
            use crate::ops::matmul::tpu_gemv;
            tpu_gemv(a, x, y, alpha, beta, transpose_a)
        },
        _ => {
            Err(PhynexusError::UnsupportedOperation(
                format!("GEMV not supported on device: {:?}", device)
            ))
        }
    }
}

/// Perform tensor transpose
pub fn transpose(tensor: &Tensor, dim0: usize, dim1: usize) -> Result<Tensor> {
    let device = tensor.device_type()?;
    let shape = tensor.shape()?;
    
    if dim0 >= shape.len() || dim1 >= shape.len() {
        return Err(PhynexusError::InvalidArgument(
            format!("Transpose dimensions out of bounds: dim0={}, dim1={}, shape={:?}", dim0, dim1, shape)
        ));
    }
    
    let mut new_shape = shape.clone();
    new_shape[dim0] = shape[dim1];
    new_shape[dim1] = shape[dim0];
    
    let mut output = Tensor::zeros(&new_shape, device)?;
    
    // Dispatch to device-specific implementation
    match device {
        DeviceType::CPU => {
            cpu_transpose(tensor, &mut output, dim0, dim1)?;
        },
        DeviceType::CUDA => {
            cuda_transpose(tensor, &mut output, dim0, dim1)?;
        },
        DeviceType::ROCm => {
            rocm_transpose(tensor, &mut output, dim0, dim1)?;
        },
        DeviceType::WebGPU => {
            webgpu_transpose(tensor, &mut output, dim0, dim1)?;
        },
        DeviceType::TPU => {
            tpu_transpose(tensor, &mut output, dim0, dim1)?;
        },
        _ => {
            return Err(PhynexusError::UnsupportedOperation(
                format!("Transpose not supported on device: {:?}", device)
            ));
        }
    }
    
    Ok(output)
}

/// Copy tensor
pub fn copy(tensor: &Tensor) -> Result<Tensor> {
    let device = tensor.device_type()?;
    let shape = tensor.shape()?;
    
    let mut output = Tensor::zeros(&shape, device)?;
    
    // Dispatch to device-specific implementation
    match device {
        DeviceType::CPU => {
            cpu_copy(tensor, &mut output)?;
        },
        DeviceType::CUDA => {
            cuda_copy(tensor, &mut output)?;
        },
        DeviceType::ROCm => {
            rocm_copy(tensor, &mut output)?;
        },
        DeviceType::WebGPU => {
            webgpu_copy(tensor, &mut output)?;
        },
        DeviceType::TPU => {
            tpu_copy(tensor, &mut output)?;
        },
        _ => {
            return Err(PhynexusError::UnsupportedOperation(
                format!("Copy not supported on device: {:?}", device)
            ));
        }
    }
    
    Ok(output)
}
