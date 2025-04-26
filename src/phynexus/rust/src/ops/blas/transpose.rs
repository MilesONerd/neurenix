//! Transpose operations for the Phynexus engine

use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;
use ndarray::{Array, ArrayD, Axis, Ix, IxDyn};

/// Perform transpose operation on CPU
pub fn cpu_transpose(tensor: &Tensor, out: &mut Tensor, dim0: usize, dim1: usize) -> Result<()> {
    let shape = tensor.shape()?;
    let out_shape = out.shape()?;
    
    if dim0 >= shape.len() || dim1 >= shape.len() {
        return Err(PhynexusError::InvalidArgument(
            format!("Transpose dimensions out of bounds: dim0={}, dim1={}, shape={:?}", dim0, dim1, shape)
        ));
    }
    
    let mut expected_shape = shape.clone();
    expected_shape[dim0] = shape[dim1];
    expected_shape[dim1] = shape[dim0];
    
    if out_shape != expected_shape {
        return Err(PhynexusError::DimensionMismatch(
            format!("Output shape {:?} does not match expected shape {:?}", out_shape, expected_shape)
        ));
    }
    
    let tensor_data = tensor.data_cpu()?;
    let mut out_data = out.data_cpu_mut()?;
    
    let dims: Vec<Ix> = shape.iter().map(|&d| d as Ix).collect();
    let array = Array::from_shape_vec(IxDyn(&dims), tensor_data.to_vec())?;
    
    let transposed = array.swap_axes(dim0, dim1);
    
    let flat_transposed = transposed.as_standard_layout().into_raw_vec();
    out_data.copy_from_slice(&flat_transposed);
    
    Ok(())
}

/// Perform transpose operation on CUDA
pub fn cuda_transpose(tensor: &Tensor, out: &mut Tensor, dim0: usize, dim1: usize) -> Result<()> {
    let shape = tensor.shape()?;
    let out_shape = out.shape()?;
    
    if dim0 >= shape.len() || dim1 >= shape.len() {
        return Err(PhynexusError::InvalidArgument(
            format!("Transpose dimensions out of bounds: dim0={}, dim1={}, shape={:?}", dim0, dim1, shape)
        ));
    }
    
    let mut expected_shape = shape.clone();
    expected_shape[dim0] = shape[dim1];
    expected_shape[dim1] = shape[dim0];
    
    if out_shape != expected_shape {
        return Err(PhynexusError::DimensionMismatch(
            format!("Output shape {:?} does not match expected shape {:?}", out_shape, expected_shape)
        ));
    }
    
    let cuda_context = crate::hardware::cuda::get_cuda_context()?;
    let cuda_stream = crate::hardware::cuda::get_cuda_stream(cuda_context)?;
    
    let tensor_data = tensor.data_cuda()?;
    let mut out_data = out.data_cuda_mut()?;
    
    let mut in_strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        in_strides[i] = in_strides[i + 1] * shape[i + 1];
    }
    
    let mut out_strides = vec![1; out_shape.len()];
    for i in (0..out_shape.len() - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }
    
    unsafe {
        crate::hardware::cuda::execute_cuda_kernel(
            cuda_context,
            cuda_stream,
            "transpose_kernel",
            &[
                tensor_data.as_ptr() as *const std::ffi::c_void,
                out_data.as_mut_ptr() as *mut std::ffi::c_void,
                &dim0 as *const usize as *const std::ffi::c_void,
                &dim1 as *const usize as *const std::ffi::c_void,
                shape.as_ptr() as *const std::ffi::c_void,
                in_strides.as_ptr() as *const std::ffi::c_void,
                out_strides.as_ptr() as *const std::ffi::c_void,
                &(shape.len()) as *const usize as *const std::ffi::c_void,
            ],
            shape.iter().product::<usize>(),
            256,
        )?;
    }
    
    crate::hardware::cuda::synchronize_cuda_stream(cuda_stream)?;
    
    Ok(())
}

/// Perform transpose operation on ROCm
pub fn rocm_transpose(tensor: &Tensor, out: &mut Tensor, dim0: usize, dim1: usize) -> Result<()> {
    let shape = tensor.shape()?;
    let out_shape = out.shape()?;
    
    if dim0 >= shape.len() || dim1 >= shape.len() {
        return Err(PhynexusError::InvalidArgument(
            format!("Transpose dimensions out of bounds: dim0={}, dim1={}, shape={:?}", dim0, dim1, shape)
        ));
    }
    
    let mut expected_shape = shape.clone();
    expected_shape[dim0] = shape[dim1];
    expected_shape[dim1] = shape[dim0];
    
    if out_shape != expected_shape {
        return Err(PhynexusError::DimensionMismatch(
            format!("Output shape {:?} does not match expected shape {:?}", out_shape, expected_shape)
        ));
    }
    
    let rocm_context = crate::hardware::rocm::get_rocm_context()?;
    let rocm_stream = crate::hardware::rocm::get_rocm_stream(rocm_context)?;
    
    let tensor_data = tensor.data_rocm()?;
    let mut out_data = out.data_rocm_mut()?;
    
    let mut in_strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        in_strides[i] = in_strides[i + 1] * shape[i + 1];
    }
    
    let mut out_strides = vec![1; out_shape.len()];
    for i in (0..out_shape.len() - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }
    
    unsafe {
        crate::hardware::rocm::execute_rocm_kernel(
            rocm_context,
            rocm_stream,
            "transpose_kernel",
            &[
                tensor_data.as_ptr() as *const std::ffi::c_void,
                out_data.as_mut_ptr() as *mut std::ffi::c_void,
                &dim0 as *const usize as *const std::ffi::c_void,
                &dim1 as *const usize as *const std::ffi::c_void,
                shape.as_ptr() as *const std::ffi::c_void,
                in_strides.as_ptr() as *const std::ffi::c_void,
                out_strides.as_ptr() as *const std::ffi::c_void,
                &(shape.len()) as *const usize as *const std::ffi::c_void,
            ],
            shape.iter().product::<usize>(),
            256,
        )?;
    }
    
    crate::hardware::rocm::synchronize_rocm_stream(rocm_stream)?;
    
    Ok(())
}

/// Perform transpose operation on WebGPU
pub fn webgpu_transpose(tensor: &Tensor, out: &mut Tensor, dim0: usize, dim1: usize) -> Result<()> {
    let shape = tensor.shape()?;
    let out_shape = out.shape()?;
    
    if dim0 >= shape.len() || dim1 >= shape.len() {
        return Err(PhynexusError::InvalidArgument(
            format!("Transpose dimensions out of bounds: dim0={}, dim1={}, shape={:?}", dim0, dim1, shape)
        ));
    }
    
    let mut expected_shape = shape.clone();
    expected_shape[dim0] = shape[dim1];
    expected_shape[dim1] = shape[dim0];
    
    if out_shape != expected_shape {
        return Err(PhynexusError::DimensionMismatch(
            format!("Output shape {:?} does not match expected shape {:?}", out_shape, expected_shape)
        ));
    }
    
    let webgpu_device = crate::hardware::webgpu::get_webgpu_device()?;
    let webgpu_queue = crate::hardware::webgpu::get_webgpu_queue(webgpu_device)?;
    
    let tensor_data = tensor.data_webgpu()?;
    let mut out_data = out.data_webgpu_mut()?;
    
    let mut in_strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        in_strides[i] = in_strides[i + 1] * shape[i + 1];
    }
    
    let mut out_strides = vec![1; out_shape.len()];
    for i in (0..out_shape.len() - 1).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }
    
    unsafe {
        crate::hardware::webgpu::execute_webgpu_compute(
            webgpu_device,
            webgpu_queue,
            "transpose_shader",
            &[
                tensor_data.as_ptr() as *const std::ffi::c_void,
                out_data.as_mut_ptr() as *mut std::ffi::c_void,
                &dim0 as *const usize as *const std::ffi::c_void,
                &dim1 as *const usize as *const std::ffi::c_void,
                shape.as_ptr() as *const std::ffi::c_void,
                in_strides.as_ptr() as *const std::ffi::c_void,
                out_strides.as_ptr() as *const std::ffi::c_void,
                &(shape.len()) as *const usize as *const std::ffi::c_void,
            ],
            (shape.iter().product::<usize>() + 255) / 256,
            1,
            1,
        )?;
    }
    
    crate::hardware::webgpu::wait_for_webgpu_operations(webgpu_queue)?;
    
    Ok(())
}

/// Perform transpose operation on TPU
pub fn tpu_transpose(tensor: &Tensor, out: &mut Tensor, dim0: usize, dim1: usize) -> Result<()> {
    let shape = tensor.shape()?;
    let out_shape = out.shape()?;
    
    if dim0 >= shape.len() || dim1 >= shape.len() {
        return Err(PhynexusError::InvalidArgument(
            format!("Transpose dimensions out of bounds: dim0={}, dim1={}, shape={:?}", dim0, dim1, shape)
        ));
    }
    
    let mut expected_shape = shape.clone();
    expected_shape[dim0] = shape[dim1];
    expected_shape[dim1] = shape[dim0];
    
    if out_shape != expected_shape {
        return Err(PhynexusError::DimensionMismatch(
            format!("Output shape {:?} does not match expected shape {:?}", out_shape, expected_shape)
        ));
    }
    
    let tpu_device = crate::hardware::tpu::get_tpu_device()?;
    let tpu_context = crate::hardware::tpu::get_tpu_context()?;
    
    let tensor_data = tensor.data_tpu()?;
    let mut out_data = out.data_tpu_mut()?;
    
    unsafe {
        crate::hardware::tpu::execute_tpu_op(
            tpu_device,
            tpu_context,
            "tpu_transpose",
            &[tensor_data],
            &mut [out_data],
            &[
                dim0 as i32,
                dim1 as i32,
                shape.len() as i32,
                shape.iter().map(|&d| d as i32).collect::<Vec<i32>>().as_ptr() as i64,
            ],
        )?;
    }
    
    crate::hardware::tpu::wait_for_completion(tpu_context)?;
    
    Ok(())
}
