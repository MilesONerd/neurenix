//! Convolution operations for the Phynexus engine

mod conv2d;

pub use conv2d::*;

pub struct Conv2dParams {
    pub stride: Vec<usize>,
    
    pub padding: Vec<usize>,
    
    pub dilation: Vec<usize>,
    
    pub groups: usize,
}

use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;

/// Perform convolution operation on CPU
pub fn cpu_conv(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    out: &mut Tensor,
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    groups: usize
) -> Result<()> {
    let input_data = input.data()?;
    let weight_data = weight.data()?;
    let mut output_data = out.data_mut()?;
    
    let batch_size = input.shape()?[0];
    let in_channels = input.shape()?[1];
    let in_height = input.shape()?[2];
    let in_width = input.shape()?[3];
    
    let out_channels = weight.shape()?[0];
    let kernel_height = weight.shape()?[2];
    let kernel_width = weight.shape()?[3];
    
    let out_height = out.shape()?[2];
    let out_width = out.shape()?[3];
    
    let stride_h = stride[0];
    let stride_w = stride[1];
    
    let padding_h = padding[0];
    let padding_w = padding[1];
    
    let dilation_h = dilation[0];
    let dilation_w = dilation[1];
    
    for b in 0..batch_size {
        for oc in 0..out_channels {
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let mut sum = 0.0;
                    
                    for ic in 0..in_channels / groups {
                        let input_channel = ic + (oc / (out_channels / groups)) * (in_channels / groups);
                        
                        for kh in 0..kernel_height {
                            for kw in 0..kernel_width {
                                let ih = oh * stride_h + kh * dilation_h - padding_h;
                                let iw = ow * stride_w + kw * dilation_w - padding_w;
                                
                                if ih < in_height && iw < in_width && ih >= 0 && iw >= 0 {
                                    let input_idx = ((b * in_channels + input_channel) * in_height + ih) * in_width + iw;
                                    let weight_idx = ((oc * (in_channels / groups) + ic) * kernel_height + kh) * kernel_width + kw;
                                    
                                    sum += input_data[input_idx] * weight_data[weight_idx];
                                }
                            }
                        }
                    }
                    
                    let output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                    output_data[output_idx] = sum;
                }
            }
        }
    }
    
    if let Some(bias) = bias {
        let bias_data = bias.data()?;
        
        for b in 0..batch_size {
            for oc in 0..out_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        let output_idx = ((b * out_channels + oc) * out_height + oh) * out_width + ow;
                        output_data[output_idx] += bias_data[oc];
                    }
                }
            }
        }
    }
    
    Ok(())
}

/// Perform convolution operation on CUDA
pub fn cuda_conv(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    out: &mut Tensor,
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    groups: usize
) -> Result<()> {
    #[cfg(feature = "cuda")]
    {
        let cuda_device = crate::hardware::cuda::get_cuda_device()?;
        let cuda_context = crate::hardware::cuda::get_cuda_context()?;
        
        let input_data = input.data_cuda()?;
        let weight_data = weight.data_cuda()?;
        let mut output_data = out.data_cuda_mut()?;
        
        let batch_size = input.shape()?[0];
        let in_channels = input.shape()?[1];
        let in_height = input.shape()?[2];
        let in_width = input.shape()?[3];
        
        let out_channels = weight.shape()?[0];
        let kernel_height = weight.shape()?[2];
        let kernel_width = weight.shape()?[3];
        
        let out_height = out.shape()?[2];
        let out_width = out.shape()?[3];
        
        let params = [
            batch_size as i32, 
            in_channels as i32, 
            in_height as i32, 
            in_width as i32,
            out_channels as i32, 
            kernel_height as i32, 
            kernel_width as i32,
            out_height as i32, 
            out_width as i32,
            stride[0] as i32, 
            stride[1] as i32,
            padding[0] as i32, 
            padding[1] as i32,
            dilation[0] as i32, 
            dilation[1] as i32,
            groups as i32
        ];
        
        let params_buffer = crate::hardware::cuda::create_buffer(&params)?;
        
        unsafe {
            crate::hardware::cuda::execute_cuda_kernel(
                cuda_device,
                cuda_context,
                "cuda_conv2d_forward",
                &[input_data, weight_data],
                &mut [output_data],
                &params_buffer,
            )?;
        }
        
        if let Some(bias) = bias {
            let bias_data = bias.data_cuda()?;
            
            unsafe {
                crate::hardware::cuda::execute_cuda_kernel(
                    cuda_device,
                    cuda_context,
                    "cuda_add_bias",
                    &[output_data, bias_data],
                    &mut [output_data],
                    &[batch_size as i32, out_channels as i32, out_height as i32, out_width as i32],
                )?;
            }
        }
        
        crate::hardware::cuda::wait_for_completion(cuda_context)?;
        
        Ok(())
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        Err(PhynexusError::UnsupportedOperation(
            "CUDA support is not enabled".to_string()
        ))
    }
}

/// Perform convolution operation on ROCm
pub fn rocm_conv(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    out: &mut Tensor,
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    groups: usize
) -> Result<()> {
    #[cfg(feature = "rocm")]
    {
        let rocm_device = crate::hardware::rocm::get_rocm_device()?;
        let rocm_context = crate::hardware::rocm::get_rocm_context()?;
        
        let input_data = input.data_rocm()?;
        let weight_data = weight.data_rocm()?;
        let mut output_data = out.data_rocm_mut()?;
        
        let batch_size = input.shape()?[0];
        let in_channels = input.shape()?[1];
        let in_height = input.shape()?[2];
        let in_width = input.shape()?[3];
        
        let out_channels = weight.shape()?[0];
        let kernel_height = weight.shape()?[2];
        let kernel_width = weight.shape()?[3];
        
        let out_height = out.shape()?[2];
        let out_width = out.shape()?[3];
        
        let params = [
            batch_size as i32, 
            in_channels as i32, 
            in_height as i32, 
            in_width as i32,
            out_channels as i32, 
            kernel_height as i32, 
            kernel_width as i32,
            out_height as i32, 
            out_width as i32,
            stride[0] as i32, 
            stride[1] as i32,
            padding[0] as i32, 
            padding[1] as i32,
            dilation[0] as i32, 
            dilation[1] as i32,
            groups as i32
        ];
        
        let params_buffer = crate::hardware::rocm::create_buffer(&params)?;
        
        unsafe {
            crate::hardware::rocm::execute_rocm_kernel(
                rocm_device,
                rocm_context,
                "rocm_conv2d_forward",
                &[input_data, weight_data],
                &mut [output_data],
                &params_buffer,
            )?;
        }
        
        if let Some(bias) = bias {
            let bias_data = bias.data_rocm()?;
            
            unsafe {
                crate::hardware::rocm::execute_rocm_kernel(
                    rocm_device,
                    rocm_context,
                    "rocm_add_bias",
                    &[output_data, bias_data],
                    &mut [output_data],
                    &[batch_size as i32, out_channels as i32, out_height as i32, out_width as i32],
                )?;
            }
        }
        
        crate::hardware::rocm::wait_for_completion(rocm_context)?;
        
        Ok(())
    }
    
    #[cfg(not(feature = "rocm"))]
    {
        Err(PhynexusError::UnsupportedOperation(
            "ROCm support is not enabled".to_string()
        ))
    }
}

/// Perform convolution operation on WebGPU
pub fn webgpu_conv(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    out: &mut Tensor,
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    groups: usize
) -> Result<()> {
    #[cfg(feature = "webgpu")]
    {
        let webgpu_device = crate::hardware::webgpu::get_webgpu_device()?;
        let webgpu_queue = crate::hardware::webgpu::get_webgpu_queue()?;
        
        let input_data = input.data_webgpu()?;
        let weight_data = weight.data_webgpu()?;
        let mut output_data = out.data_webgpu_mut()?;
        
        let batch_size = input.shape()?[0];
        let in_channels = input.shape()?[1];
        let in_height = input.shape()?[2];
        let in_width = input.shape()?[3];
        
        let out_channels = weight.shape()?[0];
        let kernel_height = weight.shape()?[2];
        let kernel_width = weight.shape()?[3];
        
        let out_height = out.shape()?[2];
        let out_width = out.shape()?[3];
        
        let shader_code = r#"
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read> weight: array<f32>;
            @group(0) @binding(2) var<storage, read_write> output: array<f32>;
            @group(0) @binding(3) var<uniform> params: array<u32, 16>; // [batch_size, in_channels, in_height, in_width, out_channels, kernel_height, kernel_width, out_height, out_width, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups]
            
            @compute @workgroup_size(8, 8, 1)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let batch = global_id.z;
                let out_c = global_id.y;
                let out_h = global_id.x / params[8]; // out_width
                let out_w = global_id.x % params[8]; // out_width
                
                if (batch >= params[0] || out_c >= params[4] || out_h >= params[7] || out_w >= params[8]) {
                    return;
                }
                
                let stride_h = params[9];
                let stride_w = params[10];
                let padding_h = params[11];
                let padding_w = params[12];
                let dilation_h = params[13];
                let dilation_w = params[14];
                let groups = params[15];
                
                let in_channels_per_group = params[1] / groups;
                let out_channels_per_group = params[4] / groups;
                let group = out_c / out_channels_per_group;
                
                var sum: f32 = 0.0;
                
                for (var ic: u32 = 0u; ic < in_channels_per_group; ic = ic + 1u) {
                    let input_channel = ic + group * in_channels_per_group;
                    
                    for (var kh: u32 = 0u; kh < params[5]; kh = kh + 1u) {
                        for (var kw: u32 = 0u; kw < params[6]; kw = kw + 1u) {
                            let ih: i32 = i32(out_h * stride_h + kh * dilation_h) - i32(padding_h);
                            let iw: i32 = i32(out_w * stride_w + kw * dilation_w) - i32(padding_w);
                            
                            if (ih >= 0 && ih < i32(params[2]) && iw >= 0 && iw < i32(params[3])) {
                                let input_idx = ((batch * params[1] + input_channel) * params[2] + u32(ih)) * params[3] + u32(iw);
                                let weight_idx = ((out_c * in_channels_per_group + ic) * params[5] + kh) * params[6] + kw;
                                
                                sum = sum + input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                
                let output_idx = ((batch * params[4] + out_c) * params[7] + out_h) * params[8] + out_w;
                output[output_idx] = sum;
            }
        "#;
        
        let shader_module = webgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("conv2d_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });
        
        let params_data = [
            batch_size as u32, 
            in_channels as u32, 
            in_height as u32, 
            in_width as u32,
            out_channels as u32, 
            kernel_height as u32, 
            kernel_width as u32,
            out_height as u32, 
            out_width as u32,
            stride[0] as u32, 
            stride[1] as u32,
            padding[0] as u32, 
            padding[1] as u32,
            dilation[0] as u32, 
            dilation[1] as u32,
            groups as u32
        ];
        
        let params_buffer = webgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("conv2d_params_buffer"),
            contents: bytemuck::cast_slice(&params_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let bind_group_layout = webgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("conv2d_bind_group_layout"),
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
        
        let pipeline_layout = webgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("conv2d_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let compute_pipeline = webgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("conv2d_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "main",
        });
        
        let bind_group = webgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv2d_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight_data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        
        let mut encoder = webgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("conv2d_encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv2d_pass"),
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            let dispatch_x = (out_width * out_height + 7) / 8;
            let dispatch_y = (out_channels + 7) / 8;
            let dispatch_z = batch_size;
            
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
        }
        
        if let Some(bias) = bias {
            let bias_data = bias.data_webgpu()?;
            
            let bias_shader = r#"
                @group(0) @binding(0) var<storage, read_write> output: array<f32>;
                @group(0) @binding(1) var<storage, read> bias: array<f32>;
                @group(0) @binding(2) var<uniform> params: array<u32, 4>; // [batch_size, out_channels, out_height, out_width]
                
                @compute @workgroup_size(8, 8, 1)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let batch = global_id.z;
                    let out_c = global_id.y;
                    let out_h = global_id.x / params[3]; // out_width
                    let out_w = global_id.x % params[3]; // out_width
                    
                    if (batch >= params[0] || out_c >= params[1] || out_h >= params[2] || out_w >= params[3]) {
                        return;
                    }
                    
                    let output_idx = ((batch * params[1] + out_c) * params[2] + out_h) * params[3] + out_w;
                    output[output_idx] = output[output_idx] + bias[out_c];
                }
            "#;
            
            let bias_shader_module = webgpu_device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("bias_shader"),
                source: wgpu::ShaderSource::Wgsl(bias_shader.into()),
            });
            
            let bias_params = [
                batch_size as u32,
                out_channels as u32,
                out_height as u32,
                out_width as u32,
            ];
            
            let bias_params_buffer = webgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("bias_params_buffer"),
                contents: bytemuck::cast_slice(&bias_params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
            
            let bias_bind_group_layout = webgpu_device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("bias_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
            
            let bias_pipeline_layout = webgpu_device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bias_pipeline_layout"),
                bind_group_layouts: &[&bias_bind_group_layout],
                push_constant_ranges: &[],
            });
            
            let bias_compute_pipeline = webgpu_device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("bias_pipeline"),
                layout: Some(&bias_pipeline_layout),
                module: &bias_shader_module,
                entry_point: "main",
            });
            
            let bias_bind_group = webgpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("bias_bind_group"),
                layout: &bias_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: output_data.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bias_data.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bias_params_buffer.as_entire_binding(),
                    },
                ],
            });
            
            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("bias_pass"),
                });
                compute_pass.set_pipeline(&bias_compute_pipeline);
                compute_pass.set_bind_group(0, &bias_bind_group, &[]);
                
                let dispatch_x = (out_width * out_height + 7) / 8;
                let dispatch_y = (out_channels + 7) / 8;
                let dispatch_z = batch_size;
                
                compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, dispatch_z);
            }
        }
        
        webgpu_queue.submit(std::iter::once(encoder.finish()));
        
        Ok(())
    }
    
    #[cfg(not(feature = "webgpu"))]
    {
        Err(PhynexusError::UnsupportedOperation(
            "WebGPU support is not enabled".to_string()
        ))
    }
}

/// Perform convolution operation on TPU
pub fn tpu_conv(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    out: &mut Tensor,
    stride: &[usize],
    padding: &[usize],
    dilation: &[usize],
    groups: usize
) -> Result<()> {
    #[cfg(feature = "tpu")]
    {
        let tpu_device = crate::hardware::tpu::get_tpu_device()?;
        let tpu_context = crate::hardware::tpu::get_tpu_context()?;
        
        let input_data = input.data_tpu()?;
        let weight_data = weight.data_tpu()?;
        let mut output_data = out.data_tpu_mut()?;
        
        let batch_size = input.shape()?[0];
        let in_channels = input.shape()?[1];
        let in_height = input.shape()?[2];
        let in_width = input.shape()?[3];
        
        let out_channels = weight.shape()?[0];
        let kernel_height = weight.shape()?[2];
        let kernel_width = weight.shape()?[3];
        
        let out_height = out.shape()?[2];
        let out_width = out.shape()?[3];
        
        let params = [
            batch_size as i32, 
            in_channels as i32, 
            in_height as i32, 
            in_width as i32,
            out_channels as i32, 
            kernel_height as i32, 
            kernel_width as i32,
            out_height as i32, 
            out_width as i32,
            stride[0] as i32, 
            stride[1] as i32,
            padding[0] as i32, 
            padding[1] as i32,
            dilation[0] as i32, 
            dilation[1] as i32,
            groups as i32
        ];
        
        unsafe {
            crate::hardware::tpu::execute_tpu_op(
                tpu_device,
                tpu_context,
                "tpu_conv2d_forward",
                &[input_data, weight_data],
                &mut [output_data],
                &params,
            )?;
        }
        
        if let Some(bias) = bias {
            let bias_data = bias.data_tpu()?;
            
            let bias_params = [
                batch_size as i32,
                out_channels as i32,
                out_height as i32,
                out_width as i32,
            ];
            
            unsafe {
                crate::hardware::tpu::execute_tpu_op(
                    tpu_device,
                    tpu_context,
                    "tpu_add_bias",
                    &[output_data, bias_data],
                    &mut [output_data],
                    &bias_params,
                )?;
            }
        }
        
        crate::hardware::tpu::wait_for_completion(tpu_context)?;
        
        Ok(())
    }
    
    #[cfg(not(feature = "tpu"))]
    {
        Err(PhynexusError::UnsupportedOperation(
            "TPU support is not enabled".to_string()
        ))
    }
}
