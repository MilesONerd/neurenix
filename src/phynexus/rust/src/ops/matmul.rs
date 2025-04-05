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
    
    
    let a_batch_rank = a_batch_dims.len();
    let b_batch_rank = b_batch_dims.len();
    let max_batch_rank = a_batch_rank.max(b_batch_rank);
    
    let mut a_padded_batch = vec![1; max_batch_rank];
    let mut b_padded_batch = vec![1; max_batch_rank];
    
    for i in 0..a_batch_rank {
        a_padded_batch[max_batch_rank - a_batch_rank + i] = a_batch_dims[i];
    }
    
    for i in 0..b_batch_rank {
        b_padded_batch[max_batch_rank - b_batch_rank + i] = b_batch_dims[i];
    }
    
    let mut out_batch_dims = vec![0; max_batch_rank];
    for i in 0..max_batch_rank {
        let a_dim = a_padded_batch[i];
        let b_dim = b_padded_batch[i];
        
        if a_dim == b_dim || a_dim == 1 || b_dim == 1 {
            out_batch_dims[i] = a_dim.max(b_dim);
        } else {
            return Err(PhynexusError::ShapeMismatch(format!(
                "Cannot broadcast batch dimensions {} and {}",
                a_dim, b_dim
            )));
        }
    }
    
    out_shape.extend_from_slice(&out_batch_dims);
    out_shape.push(a_shape[a_shape.len() - 2]);
    out_shape.push(b_shape[b_shape.len() - 1]);
    
    // Create the output tensor
    let mut out = Tensor::new(out_shape, a.dtype(), a.device().clone())?;
    
    match a.device().device_type() {
        crate::device::DeviceType::CPU => {
            cpu_matmul(a, b, &mut out)?;
        },
        crate::device::DeviceType::CUDA => {
            cuda_matmul(a, b, &mut out)?;
        },
        crate::device::DeviceType::ROCm => {
            rocm_matmul(a, b, &mut out)?;
        },
        crate::device::DeviceType::WebGPU => {
            webgpu_matmul(a, b, &mut out)?;
        },
        crate::device::DeviceType::TPU => {
            tpu_matmul(a, b, &mut out)?;
        },
        _ => {
            return Err(PhynexusError::UnsupportedOperation(
                format!("Matrix multiplication not supported on device: {:?}", a.device())
            ));
        }
    }
    
    Ok(out)
}

/// Perform batched matrix multiplication between two tensors
pub fn batch_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    matmul(a, b)
}

/// Perform matrix multiplication on CPU
pub fn cpu_matmul(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let out_shape = out.shape();
    
    let a_data = a.data_as_slice()?;
    let b_data = b.data_as_slice()?;
    let out_data = out.data_as_slice_mut()?;
    
    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let n = b_shape[b_shape.len() - 1];
    
    let batch_size: usize = if a_shape.len() > 2 {
        a_shape[..a_shape.len() - 2].iter().product()
    } else {
        1
    };
    
    for batch in 0..batch_size {
        let a_offset = batch * m * k;
        let b_offset = batch * k * n;
        let out_offset = batch * m * n;
        
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a_data[a_offset + i * k + l] * b_data[b_offset + l * n + j];
                }
                out_data[out_offset + i * n + j] = sum;
            }
        }
    }
    
    Ok(())
}

/// Perform matrix multiplication on CUDA
pub fn cuda_matmul(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let out_shape = out.shape();
        
        let cuda_ctx = a.device().cuda_context()?;
        let stream = cuda_ctx.stream();
        
        let m = a_shape[a_shape.len() - 2] as i32;
        let k = a_shape[a_shape.len() - 1] as i32;
        let n = b_shape[b_shape.len() - 1] as i32;
        
        let batch_size: usize = if a_shape.len() > 2 {
            a_shape[..a_shape.len() - 2].iter().product()
        } else {
            1
        };
        
        let a_ptr = a.data_ptr()?;
        let b_ptr = b.data_ptr()?;
        let out_ptr = out.data_ptr_mut()?;
        
        let alpha = 1.0f32;
        let beta = 0.0f32;
        
        unsafe {
            let mut handle = std::ptr::null_mut();
            cublas_sys::cublasCreate_v2(&mut handle);
            
            cublas_sys::cublasSetStream_v2(handle, stream.as_ptr());
            
            for batch in 0..batch_size {
                let a_offset = batch * (m as usize) * (k as usize);
                let b_offset = batch * (k as usize) * (n as usize);
                let out_offset = batch * (m as usize) * (n as usize);
                
                let status = cublas_sys::cublasSgemm_v2(
                    handle,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                    n, m, k,
                    &alpha as *const f32,
                    (b_ptr.offset(b_offset as isize)) as *const f32, n,
                    (a_ptr.offset(a_offset as isize)) as *const f32, k,
                    &beta as *const f32,
                    (out_ptr.offset(out_offset as isize)) as *mut f32, n
                );
                
                if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    cublas_sys::cublasDestroy_v2(handle);
                    return Err(PhynexusError::DeviceError(
                        format!("cuBLAS SGEMM failed with status: {}", status)
                    ));
                }
            }
            
            cublas_sys::cublasDestroy_v2(handle);
        }
        
        Ok(())
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        Err(PhynexusError::UnsupportedOperation(
            "CUDA support is not enabled".to_string()
        ))
    }
}

/// Perform matrix multiplication on ROCm
pub fn rocm_matmul(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    #[cfg(feature = "rocm")]
    {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let out_shape = out.shape();
        
        let rocm_ctx = a.device().rocm_context()?;
        let stream = rocm_ctx.stream();
        
        let m = a_shape[a_shape.len() - 2] as i32;
        let k = a_shape[a_shape.len() - 1] as i32;
        let n = b_shape[b_shape.len() - 1] as i32;
        
        let batch_size: usize = if a_shape.len() > 2 {
            a_shape[..a_shape.len() - 2].iter().product()
        } else {
            1
        };
        
        let a_ptr = a.data_ptr()?;
        let b_ptr = b.data_ptr()?;
        let out_ptr = out.data_ptr_mut()?;
        
        let alpha = 1.0f32;
        let beta = 0.0f32;
        
        unsafe {
            let mut handle = std::ptr::null_mut();
            rocblas_sys::rocblas_create_handle(&mut handle);
            
            rocblas_sys::rocblas_set_stream(handle, stream.as_ptr());
            
            for batch in 0..batch_size {
                let a_offset = batch * (m as usize) * (k as usize);
                let b_offset = batch * (k as usize) * (n as usize);
                let out_offset = batch * (m as usize) * (n as usize);
                
                let status = rocblas_sys::rocblas_sgemm(
                    handle,
                    rocblas_sys::rocblas_operation::rocblas_operation_none,
                    rocblas_sys::rocblas_operation::rocblas_operation_none,
                    n, m, k,
                    &alpha as *const f32,
                    (b_ptr.offset(b_offset as isize)) as *const f32, n,
                    (a_ptr.offset(a_offset as isize)) as *const f32, k,
                    &beta as *const f32,
                    (out_ptr.offset(out_offset as isize)) as *mut f32, n
                );
                
                if status != rocblas_sys::rocblas_status::rocblas_status_success {
                    rocblas_sys::rocblas_destroy_handle(handle);
                    return Err(PhynexusError::DeviceError(
                        format!("rocBLAS SGEMM failed with status: {}", status)
                    ));
                }
            }
            
            rocblas_sys::rocblas_destroy_handle(handle);
        }
        
        Ok(())
    }
    
    #[cfg(not(feature = "rocm"))]
    {
        Err(PhynexusError::UnsupportedOperation(
            "ROCm support is not enabled".to_string()
        ))
    }
}

/// Perform matrix multiplication on WebGPU
pub fn webgpu_matmul(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    #[cfg(feature = "webgpu")]
    {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let out_shape = out.shape();
        
        let webgpu_ctx = a.device().webgpu_context()?;
        let device = webgpu_ctx.device();
        let queue = webgpu_ctx.queue();
        
        let m = a_shape[a_shape.len() - 2] as u32;
        let k = a_shape[a_shape.len() - 1] as u32;
        let n = b_shape[b_shape.len() - 1] as u32;
        
        let batch_size: usize = if a_shape.len() > 2 {
            a_shape[..a_shape.len() - 2].iter().product()
        } else {
            1
        };
        
        let a_buffer = a.webgpu_buffer()?;
        let b_buffer = b.webgpu_buffer()?;
        let out_buffer = out.webgpu_buffer_mut()?;
        
        let pipeline = webgpu_ctx.get_or_create_pipeline("matmul", || {
            let shader_src = r#"
                @group(0) @binding(0) var<storage, read> a: array<f32>;
                @group(0) @binding(1) var<storage, read> b: array<f32>;
                @group(0) @binding(2) var<storage, read_write> output: array<f32>;
                
                struct Params {
                    m: u32,
                    n: u32,
                    k: u32,
                    batch_offset: u32,
                };
                
                @group(0) @binding(3) var<uniform> params: Params;
                
                @compute @workgroup_size(16, 16)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let row = global_id.x;
                    let col = global_id.y;
                    
                    if (row >= params.m || col >= params.n) {
                        return;
                    }
                    
                    let batch_offset = params.batch_offset;
                    let a_offset = batch_offset * params.m * params.k;
                    let b_offset = batch_offset * params.k * params.n;
                    let out_offset = batch_offset * params.m * params.n;
                    
                    var sum: f32 = 0.0;
                    for (var i: u32 = 0; i < params.k; i = i + 1) {
                        let a_idx = a_offset + row * params.k + i;
                        let b_idx = b_offset + i * params.n + col;
                        sum = sum + a[a_idx] * b[b_idx];
                    }
                    
                    let out_idx = out_offset + row * params.n + col;
                    output[out_idx] = sum;
                }
            "#;
            
            let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("matmul_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });
            
            let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("matmul_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("matmul_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
            
            let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("matmul_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "main",
            });
            
            (compute_pipeline, bind_group_layout)
        })?;
        
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("matmul_params_buffer"),
            size: std::mem::size_of::<[u32; 4]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        for batch in 0..batch_size {
            let params = [m, n, k, batch as u32];
            queue.write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));
            
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("matmul_bind_group"),
                layout: &pipeline.1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: a_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: out_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
            
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("matmul_command_encoder"),
            });
            
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("matmul_compute_pass"),
                });
                compute_pass.set_pipeline(&pipeline.0);
                compute_pass.set_bind_group(0, &bind_group, &[]);
                
                let workgroup_count_x = (m + 15) / 16;
                let workgroup_count_y = (n + 15) / 16;
                compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
            }
            
            queue.submit(std::iter::once(encoder.finish()));
        }
        
        webgpu_ctx.device().poll(wgpu::Maintain::Wait);
        
        Ok(())
    }
    
    #[cfg(not(feature = "webgpu"))]
    {
        Err(PhynexusError::UnsupportedOperation(
            "WebGPU support is not enabled".to_string()
        ))
    }
}

/// Perform matrix multiplication on TPU
pub fn tpu_matmul(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    #[cfg(feature = "tpu")]
    {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let out_shape = out.shape();
        
        let tpu_ctx = a.device().tpu_context()?;
        
        let m = a_shape[a_shape.len() - 2];
        let k = a_shape[a_shape.len() - 1];
        let n = b_shape[b_shape.len() - 1];
        
        let batch_size: usize = if a_shape.len() > 2 {
            a_shape[..a_shape.len() - 2].iter().product()
        } else {
            1
        };
        
        let a_buffer = a.tpu_buffer()?;
        let b_buffer = b.tpu_buffer()?;
        let out_buffer = out.tpu_buffer_mut()?;
        
        unsafe {
            let params = tpu_sys::TpuMatMulParams {
                m: m as i32,
                n: n as i32,
                k: k as i32,
                batch_size: batch_size as i32,
                transpose_a: 0, // No transpose
                transpose_b: 0, // No transpose
                alpha: 1.0,
                beta: 0.0,
            };
            
            let status = tpu_sys::TpuMatMul(
                tpu_ctx.handle(),
                &params,
                a_buffer.as_ptr(),
                b_buffer.as_ptr(),
                out_buffer.as_mut_ptr()
            );
            
            if status != tpu_sys::TpuStatus::TPU_OK {
                return Err(PhynexusError::DeviceError(
                    format!("TPU matrix multiplication failed with status: {}", status)
                ));
            }
        }
        
        tpu_ctx.synchronize()?;
        
        Ok(())
    }
    
    #[cfg(not(feature = "tpu"))]
    {
        Err(PhynexusError::UnsupportedOperation(
            "TPU support is not enabled".to_string()
        ))
    }
}
