//! Element-wise operations for the Phynexus engine

use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;

/// Perform element-wise addition
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let device = a.device_type()?;
    let shape = a.shape()?;
    let mut result = Tensor::zeros(shape, device)?;
    
    match device {
        crate::device::DeviceType::CPU => {
            cpu_add(a, b, &mut result)?;
        },
        crate::device::DeviceType::CUDA => {
            cuda_add(a, b, &mut result)?;
        },
        crate::device::DeviceType::ROCm => {
            rocm_add(a, b, &mut result)?;
        },
        crate::device::DeviceType::WebGPU => {
            webgpu_add(a, b, &mut result)?;
        },
        crate::device::DeviceType::TPU => {
            crate::ops::tpu_elementwise::tpu_add(a, b, &mut result)?;
        },
        _ => {
            // Fallback to CPU implementation
            cpu_add(a, b, &mut result)?;
        }
    }
    
    Ok(result)
}

/// Perform element-wise subtraction
pub fn subtract(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let device = a.device_type()?;
    let shape = a.shape()?;
    let mut result = Tensor::zeros(shape, device)?;
    
    match device {
        crate::device::DeviceType::CPU => {
            cpu_subtract(a, b, &mut result)?;
        },
        crate::device::DeviceType::CUDA => {
            cuda_subtract(a, b, &mut result)?;
        },
        crate::device::DeviceType::ROCm => {
            rocm_subtract(a, b, &mut result)?;
        },
        crate::device::DeviceType::WebGPU => {
            webgpu_subtract(a, b, &mut result)?;
        },
        crate::device::DeviceType::TPU => {
            crate::ops::tpu_elementwise::tpu_subtract(a, b, &mut result)?;
        },
        _ => {
            // Fallback to CPU implementation
            cpu_subtract(a, b, &mut result)?;
        }
    }
    
    Ok(result)
}

/// Perform element-wise multiplication
pub fn multiply(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let device = a.device_type()?;
    let shape = a.shape()?;
    let mut result = Tensor::zeros(shape, device)?;
    
    match device {
        crate::device::DeviceType::CPU => {
            cpu_multiply(a, b, &mut result)?;
        },
        crate::device::DeviceType::CUDA => {
            cuda_multiply(a, b, &mut result)?;
        },
        crate::device::DeviceType::ROCm => {
            rocm_multiply(a, b, &mut result)?;
        },
        crate::device::DeviceType::WebGPU => {
            webgpu_multiply(a, b, &mut result)?;
        },
        crate::device::DeviceType::TPU => {
            crate::ops::tpu_elementwise::tpu_multiply(a, b, &mut result)?;
        },
        _ => {
            // Fallback to CPU implementation
            cpu_multiply(a, b, &mut result)?;
        }
    }
    
    Ok(result)
}

/// Perform element-wise division
pub fn divide(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let device = a.device_type()?;
    let shape = a.shape()?;
    let mut result = Tensor::zeros(shape, device)?;
    
    match device {
        crate::device::DeviceType::CPU => {
            cpu_divide(a, b, &mut result)?;
        },
        crate::device::DeviceType::CUDA => {
            cuda_divide(a, b, &mut result)?;
        },
        crate::device::DeviceType::ROCm => {
            rocm_divide(a, b, &mut result)?;
        },
        crate::device::DeviceType::WebGPU => {
            webgpu_divide(a, b, &mut result)?;
        },
        crate::device::DeviceType::TPU => {
            crate::ops::tpu_elementwise::tpu_divide(a, b, &mut result)?;
        },
        _ => {
            // Fallback to CPU implementation
            cpu_divide(a, b, &mut result)?;
        }
    }
    
    Ok(result)
}

/// Perform element-wise power
#[allow(unused_variables)]
pub fn pow(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Placeholder implementation
    Err(PhynexusError::UnsupportedOperation(
        "Element-wise power not yet implemented".to_string()
    ))
}

/// Perform element-wise exponential
#[allow(unused_variables)]
pub fn exp(a: &Tensor) -> Result<Tensor> {
    // Placeholder implementation
    Err(PhynexusError::UnsupportedOperation(
        "Element-wise exponential not yet implemented".to_string()
    ))
}

/// Perform element-wise logarithm
#[allow(unused_variables)]
pub fn log(a: &Tensor) -> Result<Tensor> {
    // Placeholder implementation
    Err(PhynexusError::UnsupportedOperation(
        "Element-wise logarithm not yet implemented".to_string()
    ))
}

/// Perform element-wise addition on CPU
pub fn cpu_add(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let a_data = a.data()?;
    let b_data = b.data()?;
    let mut out_data = out.data_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for addition: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    // Perform element-wise addition
    for i in 0..out_data.len() {
        out_data[i] = a_data[i] + b_data[i];
    }
    
    Ok(())
}

/// Perform element-wise addition on CUDA
pub fn cuda_add(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let cuda_ctx = crate::hardware::cuda::get_cuda_context()?;
    let cuda_stream = crate::hardware::cuda::get_cuda_stream()?;
    
    let a_data = a.data_cuda()?;
    let b_data = b.data_cuda()?;
    let mut out_data = out.data_cuda_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for CUDA addition: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    // Launch CUDA kernel for element-wise addition
    unsafe {
        let block_size = 256;
        let grid_size = (out_data.len() + block_size - 1) / block_size;
        
        crate::hardware::cuda::launch_kernel(
            "cuda_elementwise_add",
            grid_size,
            block_size,
            0,
            cuda_stream,
            &[
                &a_data.as_ptr(),
                &b_data.as_ptr(),
                &out_data.as_mut_ptr(),
                &(out_data.len() as i32),
            ],
        )?;
    }
    
    crate::hardware::cuda::synchronize_stream(cuda_stream)?;
    
    Ok(())
}

/// Perform element-wise addition on ROCm
pub fn rocm_add(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let rocm_ctx = crate::hardware::rocm::get_rocm_context()?;
    let rocm_queue = crate::hardware::rocm::get_rocm_queue()?;
    
    let a_data = a.data_rocm()?;
    let b_data = b.data_rocm()?;
    let mut out_data = out.data_rocm_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for ROCm addition: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    // Launch ROCm kernel for element-wise addition
    unsafe {
        let work_group_size = 256;
        let global_work_size = ((out_data.len() + work_group_size - 1) / work_group_size) * work_group_size;
        
        // Launch ROCm kernel
        crate::hardware::rocm::launch_kernel(
            "rocm_elementwise_add",
            global_work_size,
            work_group_size,
            rocm_queue,
            &[
                &a_data.as_ptr(),
                &b_data.as_ptr(),
                &out_data.as_mut_ptr(),
                &(out_data.len() as i32),
            ],
        )?;
    }
    
    crate::hardware::rocm::wait_for_queue(rocm_queue)?;
    
    Ok(())
}

/// Perform element-wise addition on WebGPU
pub fn webgpu_add(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let webgpu_device = crate::hardware::webgpu::get_webgpu_device()?;
    let webgpu_queue = crate::hardware::webgpu::get_webgpu_queue()?;
    
    let a_data = a.data_webgpu()?;
    let b_data = b.data_webgpu()?;
    let mut out_data = out.data_webgpu_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for WebGPU addition: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    // Create shader module for element-wise addition
    let shader_module = unsafe {
        crate::hardware::webgpu::create_shader_module(
            webgpu_device,
            r#"
            @group(0) @binding(0) var<storage, read> a: array<f32>;
            @group(0) @binding(1) var<storage, read> b: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;
            
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                if (idx >= arrayLength(&output)) {
                    return;
                }
                output[idx] = a[idx] + b[idx];
            }
            "#,
        )?
    };
    
    let bind_group_layout = unsafe {
        crate::hardware::webgpu::create_bind_group_layout(
            webgpu_device,
            &[
                crate::hardware::webgpu::BindGroupLayoutEntry::buffer(0, true, false),
                crate::hardware::webgpu::BindGroupLayoutEntry::buffer(1, true, false),
                crate::hardware::webgpu::BindGroupLayoutEntry::buffer(2, false, true),
            ],
        )?
    };
    
    let pipeline = unsafe {
        crate::hardware::webgpu::create_compute_pipeline(
            webgpu_device,
            shader_module,
            bind_group_layout,
            "main",
        )?
    };
    
    let bind_group = unsafe {
        crate::hardware::webgpu::create_bind_group(
            webgpu_device,
            bind_group_layout,
            &[
                crate::hardware::webgpu::BindGroupEntry::buffer(0, a_data),
                crate::hardware::webgpu::BindGroupEntry::buffer(1, b_data),
                crate::hardware::webgpu::BindGroupEntry::buffer(2, out_data),
            ],
        )?
    };
    
    unsafe {
        crate::hardware::webgpu::dispatch_compute(
            webgpu_device,
            webgpu_queue,
            pipeline,
            bind_group,
            (out_data.len() + 255) / 256,
            1,
            1,
        )?;
    }
    
    crate::hardware::webgpu::wait_for_queue(webgpu_queue)?;
    
    Ok(())
}

/// Perform element-wise subtraction on CPU
pub fn cpu_subtract(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let a_data = a.data()?;
    let b_data = b.data()?;
    let mut out_data = out.data_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for subtraction: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    // Perform element-wise subtraction
    for i in 0..out_data.len() {
        out_data[i] = a_data[i] - b_data[i];
    }
    
    Ok(())
}

/// Perform element-wise subtraction on CUDA
pub fn cuda_subtract(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let cuda_ctx = crate::hardware::cuda::get_cuda_context()?;
    let cuda_stream = crate::hardware::cuda::get_cuda_stream()?;
    
    let a_data = a.data_cuda()?;
    let b_data = b.data_cuda()?;
    let mut out_data = out.data_cuda_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for CUDA subtraction: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    // Launch CUDA kernel for element-wise subtraction
    unsafe {
        let block_size = 256;
        let grid_size = (out_data.len() + block_size - 1) / block_size;
        
        // Launch CUDA kernel
        crate::hardware::cuda::launch_kernel(
            "cuda_elementwise_subtract",
            grid_size,
            block_size,
            0,
            cuda_stream,
            &[
                &a_data.as_ptr(),
                &b_data.as_ptr(),
                &out_data.as_mut_ptr(),
                &(out_data.len() as i32),
            ],
        )?;
    }
    
    crate::hardware::cuda::synchronize_stream(cuda_stream)?;
    
    Ok(())
}

/// Perform element-wise subtraction on ROCm
pub fn rocm_subtract(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let rocm_ctx = crate::hardware::rocm::get_rocm_context()?;
    let rocm_queue = crate::hardware::rocm::get_rocm_queue()?;
    
    let a_data = a.data_rocm()?;
    let b_data = b.data_rocm()?;
    let mut out_data = out.data_rocm_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for ROCm subtraction: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    // Launch ROCm kernel for element-wise subtraction
    unsafe {
        let work_group_size = 256;
        let global_work_size = ((out_data.len() + work_group_size - 1) / work_group_size) * work_group_size;
        
        // Launch ROCm kernel
        crate::hardware::rocm::launch_kernel(
            "rocm_elementwise_subtract",
            global_work_size,
            work_group_size,
            rocm_queue,
            &[
                &a_data.as_ptr(),
                &b_data.as_ptr(),
                &out_data.as_mut_ptr(),
                &(out_data.len() as i32),
            ],
        )?;
    }
    
    crate::hardware::rocm::wait_for_queue(rocm_queue)?;
    
    Ok(())
}

/// Perform element-wise subtraction on WebGPU
pub fn webgpu_subtract(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let webgpu_device = crate::hardware::webgpu::get_webgpu_device()?;
    let webgpu_queue = crate::hardware::webgpu::get_webgpu_queue()?;
    
    let a_data = a.data_webgpu()?;
    let b_data = b.data_webgpu()?;
    let mut out_data = out.data_webgpu_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for WebGPU subtraction: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    // Create shader module for element-wise subtraction
    let shader_module = unsafe {
        crate::hardware::webgpu::create_shader_module(
            webgpu_device,
            r#"
            @group(0) @binding(0) var<storage, read> a: array<f32>;
            @group(0) @binding(1) var<storage, read> b: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;
            
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                if (idx >= arrayLength(&output)) {
                    return;
                }
                output[idx] = a[idx] - b[idx];
            }
            "#,
        )?
    };
    
    // Create bind group layout
    let bind_group_layout = unsafe {
        crate::hardware::webgpu::create_bind_group_layout(
            webgpu_device,
            &[
                crate::hardware::webgpu::BindGroupLayoutEntry::buffer(0, true, false),
                crate::hardware::webgpu::BindGroupLayoutEntry::buffer(1, true, false),
                crate::hardware::webgpu::BindGroupLayoutEntry::buffer(2, false, true),
            ],
        )?
    };
    
    // Create compute pipeline
    let pipeline = unsafe {
        crate::hardware::webgpu::create_compute_pipeline(
            webgpu_device,
            shader_module,
            bind_group_layout,
            "main",
        )?
    };
    
    // Create bind group
    let bind_group = unsafe {
        crate::hardware::webgpu::create_bind_group(
            webgpu_device,
            bind_group_layout,
            &[
                crate::hardware::webgpu::BindGroupEntry::buffer(0, a_data),
                crate::hardware::webgpu::BindGroupEntry::buffer(1, b_data),
                crate::hardware::webgpu::BindGroupEntry::buffer(2, out_data),
            ],
        )?
    };
    
    unsafe {
        crate::hardware::webgpu::dispatch_compute(
            webgpu_device,
            webgpu_queue,
            pipeline,
            bind_group,
            (out_data.len() + 255) / 256,
            1,
            1,
        )?;
    }
    
    crate::hardware::webgpu::wait_for_queue(webgpu_queue)?;
    
    Ok(())
}

/// Perform element-wise multiplication on CPU
pub fn cpu_multiply(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let a_data = a.data()?;
    let b_data = b.data()?;
    let mut out_data = out.data_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for multiplication: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    // Perform element-wise multiplication
    for i in 0..out_data.len() {
        out_data[i] = a_data[i] * b_data[i];
    }
    
    Ok(())
}

/// Perform element-wise multiplication on CUDA
pub fn cuda_multiply(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let cuda_ctx = crate::hardware::cuda::get_cuda_context()?;
    let cuda_stream = crate::hardware::cuda::get_cuda_stream()?;
    
    let a_data = a.data_cuda()?;
    let b_data = b.data_cuda()?;
    let mut out_data = out.data_cuda_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for CUDA multiplication: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    // Launch CUDA kernel for element-wise multiplication
    unsafe {
        let block_size = 256;
        let grid_size = (out_data.len() + block_size - 1) / block_size;
        
        // Launch CUDA kernel
        crate::hardware::cuda::launch_kernel(
            "cuda_elementwise_multiply",
            grid_size,
            block_size,
            0,
            cuda_stream,
            &[
                &a_data.as_ptr(),
                &b_data.as_ptr(),
                &out_data.as_mut_ptr(),
                &(out_data.len() as i32),
            ],
        )?;
    }
    
    crate::hardware::cuda::synchronize_stream(cuda_stream)?;
    
    Ok(())
}

/// Perform element-wise multiplication on ROCm
pub fn rocm_multiply(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let rocm_ctx = crate::hardware::rocm::get_rocm_context()?;
    let rocm_queue = crate::hardware::rocm::get_rocm_queue()?;
    
    let a_data = a.data_rocm()?;
    let b_data = b.data_rocm()?;
    let mut out_data = out.data_rocm_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for ROCm multiplication: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    // Launch ROCm kernel for element-wise multiplication
    unsafe {
        let work_group_size = 256;
        let global_work_size = ((out_data.len() + work_group_size - 1) / work_group_size) * work_group_size;
        
        // Launch ROCm kernel
        crate::hardware::rocm::launch_kernel(
            "rocm_elementwise_multiply",
            global_work_size,
            work_group_size,
            rocm_queue,
            &[
                &a_data.as_ptr(),
                &b_data.as_ptr(),
                &out_data.as_mut_ptr(),
                &(out_data.len() as i32),
            ],
        )?;
    }
    
    crate::hardware::rocm::wait_for_queue(rocm_queue)?;
    
    Ok(())
}

/// Perform element-wise multiplication on WebGPU
pub fn webgpu_multiply(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let webgpu_device = crate::hardware::webgpu::get_webgpu_device()?;
    let webgpu_queue = crate::hardware::webgpu::get_webgpu_queue()?;
    
    let a_data = a.data_webgpu()?;
    let b_data = b.data_webgpu()?;
    let mut out_data = out.data_webgpu_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for WebGPU multiplication: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    // Create shader module for element-wise multiplication
    let shader_module = unsafe {
        crate::hardware::webgpu::create_shader_module(
            webgpu_device,
            r#"
            @group(0) @binding(0) var<storage, read> a: array<f32>;
            @group(0) @binding(1) var<storage, read> b: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;
            
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                if (idx >= arrayLength(&output)) {
                    return;
                }
                output[idx] = a[idx] * b[idx];
            }
            "#,
        )?
    };
    
    // Create bind group layout
    let bind_group_layout = unsafe {
        crate::hardware::webgpu::create_bind_group_layout(
            webgpu_device,
            &[
                crate::hardware::webgpu::BindGroupLayoutEntry::buffer(0, true, false),
                crate::hardware::webgpu::BindGroupLayoutEntry::buffer(1, true, false),
                crate::hardware::webgpu::BindGroupLayoutEntry::buffer(2, false, true),
            ],
        )?
    };
    
    // Create compute pipeline
    let pipeline = unsafe {
        crate::hardware::webgpu::create_compute_pipeline(
            webgpu_device,
            shader_module,
            bind_group_layout,
            "main",
        )?
    };
    
    // Create bind group
    let bind_group = unsafe {
        crate::hardware::webgpu::create_bind_group(
            webgpu_device,
            bind_group_layout,
            &[
                crate::hardware::webgpu::BindGroupEntry::buffer(0, a_data),
                crate::hardware::webgpu::BindGroupEntry::buffer(1, b_data),
                crate::hardware::webgpu::BindGroupEntry::buffer(2, out_data),
            ],
        )?
    };
    
    unsafe {
        crate::hardware::webgpu::dispatch_compute(
            webgpu_device,
            webgpu_queue,
            pipeline,
            bind_group,
            (out_data.len() + 255) / 256,
            1,
            1,
        )?;
    }
    
    crate::hardware::webgpu::wait_for_queue(webgpu_queue)?;
    
    Ok(())
}

/// Perform element-wise division on CPU
pub fn cpu_divide(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let a_data = a.data()?;
    let b_data = b.data()?;
    let mut out_data = out.data_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for division: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    // Perform element-wise division
    for i in 0..out_data.len() {
        if b_data[i] == 0.0 {
            return Err(PhynexusError::InvalidValue(
                "Division by zero encountered".to_string()
            ));
        }
        out_data[i] = a_data[i] / b_data[i];
    }
    
    Ok(())
}

/// Perform element-wise division on CUDA
pub fn cuda_divide(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let cuda_ctx = crate::hardware::cuda::get_cuda_context()?;
    let cuda_stream = crate::hardware::cuda::get_cuda_stream()?;
    
    let a_data = a.data_cuda()?;
    let b_data = b.data_cuda()?;
    let mut out_data = out.data_cuda_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for CUDA division: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    // Launch CUDA kernel for element-wise division
    unsafe {
        let block_size = 256;
        let grid_size = (out_data.len() + block_size - 1) / block_size;
        
        // Launch CUDA kernel
        crate::hardware::cuda::launch_kernel(
            "cuda_elementwise_divide",
            grid_size,
            block_size,
            0,
            cuda_stream,
            &[
                &a_data.as_ptr(),
                &b_data.as_ptr(),
                &out_data.as_mut_ptr(),
                &(out_data.len() as i32),
            ],
        )?;
    }
    
    crate::hardware::cuda::synchronize_stream(cuda_stream)?;
    
    Ok(())
}

/// Perform element-wise division on ROCm
pub fn rocm_divide(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let rocm_ctx = crate::hardware::rocm::get_rocm_context()?;
    let rocm_queue = crate::hardware::rocm::get_rocm_queue()?;
    
    let a_data = a.data_rocm()?;
    let b_data = b.data_rocm()?;
    let mut out_data = out.data_rocm_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for ROCm division: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    // Launch ROCm kernel for element-wise division
    unsafe {
        let work_group_size = 256;
        let global_work_size = ((out_data.len() + work_group_size - 1) / work_group_size) * work_group_size;
        
        // Launch ROCm kernel
        crate::hardware::rocm::launch_kernel(
            "rocm_elementwise_divide",
            global_work_size,
            work_group_size,
            rocm_queue,
            &[
                &a_data.as_ptr(),
                &b_data.as_ptr(),
                &out_data.as_mut_ptr(),
                &(out_data.len() as i32),
            ],
        )?;
    }
    
    crate::hardware::rocm::wait_for_queue(rocm_queue)?;
    
    Ok(())
}

/// Perform element-wise division on WebGPU
pub fn webgpu_divide(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let webgpu_device = crate::hardware::webgpu::get_webgpu_device()?;
    let webgpu_queue = crate::hardware::webgpu::get_webgpu_queue()?;
    
    let a_data = a.data_webgpu()?;
    let b_data = b.data_webgpu()?;
    let mut out_data = out.data_webgpu_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for WebGPU division: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    // Create shader module for element-wise division
    let shader_module = unsafe {
        crate::hardware::webgpu::create_shader_module(
            webgpu_device,
            r#"
            @group(0) @binding(0) var<storage, read> a: array<f32>;
            @group(0) @binding(1) var<storage, read> b: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;
            
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let idx = global_id.x;
                if (idx >= arrayLength(&output)) {
                    return;
                }
                if (b[idx] == 0.0) {
                    output[idx] = 0.0; // Or NaN/Inf depending on preference
                } else {
                    output[idx] = a[idx] / b[idx];
                }
            }
            "#,
        )?
    };
    
    // Create bind group layout
    let bind_group_layout = unsafe {
        crate::hardware::webgpu::create_bind_group_layout(
            webgpu_device,
            &[
                crate::hardware::webgpu::BindGroupLayoutEntry::buffer(0, true, false),
                crate::hardware::webgpu::BindGroupLayoutEntry::buffer(1, true, false),
                crate::hardware::webgpu::BindGroupLayoutEntry::buffer(2, false, true),
            ],
        )?
    };
    
    // Create compute pipeline
    let pipeline = unsafe {
        crate::hardware::webgpu::create_compute_pipeline(
            webgpu_device,
            shader_module,
            bind_group_layout,
            "main",
        )?
    };
    
    // Create bind group
    let bind_group = unsafe {
        crate::hardware::webgpu::create_bind_group(
            webgpu_device,
            bind_group_layout,
            &[
                crate::hardware::webgpu::BindGroupEntry::buffer(0, a_data),
                crate::hardware::webgpu::BindGroupEntry::buffer(1, b_data),
                crate::hardware::webgpu::BindGroupEntry::buffer(2, out_data),
            ],
        )?
    };
    
    unsafe {
        crate::hardware::webgpu::dispatch_compute(
            webgpu_device,
            webgpu_queue,
            pipeline,
            bind_group,
            (out_data.len() + 255) / 256,
            1,
            1,
        )?;
    }
    
    crate::hardware::webgpu::wait_for_queue(webgpu_queue)?;
    
    Ok(())
}
