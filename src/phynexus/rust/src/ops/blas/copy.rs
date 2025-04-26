//! Copy operations for the Phynexus engine

use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;

/// Perform copy operation on CPU
pub fn cpu_copy(tensor: &Tensor, out: &mut Tensor) -> Result<()> {
    let shape = tensor.shape()?;
    let out_shape = out.shape()?;
    
    if shape != out_shape {
        return Err(PhynexusError::DimensionMismatch(
            format!("Output shape {:?} does not match input shape {:?}", out_shape, shape)
        ));
    }
    
    let tensor_data = tensor.data_cpu()?;
    let mut out_data = out.data_cpu_mut()?;
    
    out_data.copy_from_slice(&tensor_data);
    
    Ok(())
}

/// Perform copy operation on CUDA
pub fn cuda_copy(tensor: &Tensor, out: &mut Tensor) -> Result<()> {
    let shape = tensor.shape()?;
    let out_shape = out.shape()?;
    
    if shape != out_shape {
        return Err(PhynexusError::DimensionMismatch(
            format!("Output shape {:?} does not match input shape {:?}", out_shape, shape)
        ));
    }
    
    let cuda_context = crate::hardware::cuda::get_cuda_context()?;
    let cuda_stream = crate::hardware::cuda::get_cuda_stream(cuda_context)?;
    
    let tensor_data = tensor.data_cuda()?;
    let mut out_data = out.data_cuda_mut()?;
    
    let num_elements = shape.iter().product::<usize>();
    let element_size = std::mem::size_of::<f32>();
    let size_bytes = num_elements * element_size;
    
    unsafe {
        crate::hardware::cuda::cuda_memcpy(
            out_data.as_mut_ptr() as *mut std::ffi::c_void,
            tensor_data.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            cuda_stream,
        )?;
    }
    
    crate::hardware::cuda::synchronize_cuda_stream(cuda_stream)?;
    
    Ok(())
}

/// Perform copy operation on ROCm
pub fn rocm_copy(tensor: &Tensor, out: &mut Tensor) -> Result<()> {
    let shape = tensor.shape()?;
    let out_shape = out.shape()?;
    
    if shape != out_shape {
        return Err(PhynexusError::DimensionMismatch(
            format!("Output shape {:?} does not match input shape {:?}", out_shape, shape)
        ));
    }
    
    let rocm_context = crate::hardware::rocm::get_rocm_context()?;
    let rocm_stream = crate::hardware::rocm::get_rocm_stream(rocm_context)?;
    
    let tensor_data = tensor.data_rocm()?;
    let mut out_data = out.data_rocm_mut()?;
    
    let num_elements = shape.iter().product::<usize>();
    let element_size = std::mem::size_of::<f32>();
    let size_bytes = num_elements * element_size;
    
    unsafe {
        crate::hardware::rocm::rocm_memcpy(
            out_data.as_mut_ptr() as *mut std::ffi::c_void,
            tensor_data.as_ptr() as *const std::ffi::c_void,
            size_bytes,
            rocm_stream,
        )?;
    }
    
    crate::hardware::rocm::synchronize_rocm_stream(rocm_stream)?;
    
    Ok(())
}

/// Perform copy operation on WebGPU
pub fn webgpu_copy(tensor: &Tensor, out: &mut Tensor) -> Result<()> {
    let shape = tensor.shape()?;
    let out_shape = out.shape()?;
    
    if shape != out_shape {
        return Err(PhynexusError::DimensionMismatch(
            format!("Output shape {:?} does not match input shape {:?}", out_shape, shape)
        ));
    }
    
    let webgpu_device = crate::hardware::webgpu::get_webgpu_device()?;
    let webgpu_queue = crate::hardware::webgpu::get_webgpu_queue(webgpu_device)?;
    
    let tensor_data = tensor.data_webgpu()?;
    let mut out_data = out.data_webgpu_mut()?;
    
    let num_elements = shape.iter().product::<usize>();
    let element_size = std::mem::size_of::<f32>();
    let size_bytes = num_elements * element_size;
    
    unsafe {
        crate::hardware::webgpu::webgpu_buffer_copy(
            webgpu_device,
            webgpu_queue,
            tensor_data.as_ptr() as *const std::ffi::c_void,
            out_data.as_mut_ptr() as *mut std::ffi::c_void,
            size_bytes,
        )?;
    }
    
    crate::hardware::webgpu::wait_for_webgpu_operations(webgpu_queue)?;
    
    Ok(())
}

/// Perform copy operation on TPU
pub fn tpu_copy(tensor: &Tensor, out: &mut Tensor) -> Result<()> {
    let shape = tensor.shape()?;
    let out_shape = out.shape()?;
    
    if shape != out_shape {
        return Err(PhynexusError::DimensionMismatch(
            format!("Output shape {:?} does not match input shape {:?}", out_shape, shape)
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
            "tpu_copy",
            &[tensor_data],
            &mut [out_data],
            &[],
        )?;
    }
    
    crate::hardware::tpu::wait_for_completion(tpu_context)?;
    
    Ok(())
}
