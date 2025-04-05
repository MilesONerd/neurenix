//! Convolution operations

use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;
use crate::device::DeviceType;

/// Convolution parameters
pub struct ConvParams {
    /// Stride in each dimension
    pub stride: Vec<usize>,
    
    /// Padding in each dimension
    pub padding: Vec<usize>,
    
    /// Dilation in each dimension
    pub dilation: Vec<usize>,
    
    /// Groups for grouped convolution
    pub groups: usize,
}

impl Default for ConvParams {
    fn default() -> Self {
        Self {
            stride: vec![1, 1],
            padding: vec![0, 0],
            dilation: vec![1, 1],
            groups: 1,
        }
    }
}

/// Perform 2D convolution: output = conv2d(input, weight, bias)
pub fn conv2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    params: &ConvParams,
) -> Result<Tensor> {
    // Check that the tensors are on the same device
    if input.device() != weight.device() {
        return Err(PhynexusError::InvalidArgument(
            "Input and weight must be on the same device for conv2d".to_string()
        ));
    }
    
    if let Some(bias) = bias {
        if input.device() != bias.device() {
            return Err(PhynexusError::InvalidArgument(
                "Input and bias must be on the same device for conv2d".to_string()
            ));
        }
    }
    
    // Check that the tensors have the correct dimensions
    let input_shape = input.shape();
    let weight_shape = weight.shape();
    
    if input_shape.len() != 4 {
        return Err(PhynexusError::ShapeMismatch(
            "Input must be 4-dimensional (N, C, H, W) for conv2d".to_string()
        ));
    }
    
    if weight_shape.len() != 4 {
        return Err(PhynexusError::ShapeMismatch(
            "Weight must be 4-dimensional (out_channels, in_channels/groups, kH, kW) for conv2d".to_string()
        ));
    }
    
    // Extract dimensions
    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];
    
    let out_channels = weight_shape[0];
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];
    
    // Check that the number of input channels is compatible with the weight tensor
    if in_channels != weight_shape[1] * params.groups {
        return Err(PhynexusError::ShapeMismatch(format!(
            "Number of input channels must match weight tensor: {} and {}*{}",
            in_channels, weight_shape[1], params.groups
        )));
    }
    
    // Check that the number of output channels is divisible by the number of groups
    if out_channels % params.groups != 0 {
        return Err(PhynexusError::InvalidArgument(format!(
            "Number of output channels must be divisible by groups: {} and {}",
            out_channels, params.groups
        )));
    }
    
    // Check that the bias tensor has the correct shape
    if let Some(bias) = bias {
        let bias_shape = bias.shape();
        
        if bias_shape.len() != 1 || bias_shape[0] != out_channels {
            return Err(PhynexusError::ShapeMismatch(format!(
                "Bias must be 1-dimensional with shape ({}), got {:?}",
                out_channels, bias_shape
            )));
        }
    }
    
    // Calculate output dimensions
    let out_height = (in_height + 2 * params.padding[0] - params.dilation[0] * (kernel_height - 1) - 1) / params.stride[0] + 1;
    let out_width = (in_width + 2 * params.padding[1] - params.dilation[1] * (kernel_width - 1) - 1) / params.stride[1] + 1;
    
    // Create output tensor
    let out_shape = vec![batch_size, out_channels, out_height, out_width];
    let mut output = Tensor::new(out_shape, input.dtype(), input.device().clone())?;
    
    // Dispatch to the appropriate implementation based on the device
    match input.device().device_type() {
        DeviceType::CPU => {
            let input_data = input.data_ptr::<f32>()?;
            let weight_data = weight.data_ptr::<f32>()?;
            let output_data = output.data_ptr_mut::<f32>()?;
            
            // Extract dimensions
            let batch_size = input_shape[0];
            let in_channels = input_shape[1];
            let in_height = input_shape[2];
            let in_width = input_shape[3];
            
            let out_channels = weight_shape[0];
            let kernel_height = weight_shape[2];
            let kernel_width = weight_shape[3];
            
            let out_height = output.shape()[2];
            let out_width = output.shape()[3];
            
            for i in 0..output_data.len() {
                output_data[i] = 0.0;
            }
            
            for n in 0..batch_size {
                for g in 0..params.groups {
                    let in_channels_per_group = in_channels / params.groups;
                    let out_channels_per_group = out_channels / params.groups;
                    
                    for oc in 0..out_channels_per_group {
                        let out_c = g * out_channels_per_group + oc;
                        
                        for oh in 0..out_height {
                            for ow in 0..out_width {
                                let mut sum = 0.0;
                                
                                for ic in 0..in_channels_per_group {
                                    let in_c = g * in_channels_per_group + ic;
                                    
                                    for kh in 0..kernel_height {
                                        for kw in 0..kernel_width {
                                            let ih = oh * params.stride[0] + kh * params.dilation[0] - params.padding[0];
                                            let iw = ow * params.stride[1] + kw * params.dilation[1] - params.padding[1];
                                            
                                            if ih < in_height && iw < in_width && ih >= 0 && iw >= 0 {
                                                let input_idx = ((n * in_channels + in_c) * in_height + ih) * in_width + iw;
                                                let weight_idx = ((out_c * in_channels_per_group + ic) * kernel_height + kh) * kernel_width + kw;
                                                
                                                sum += input_data[input_idx] * weight_data[weight_idx];
                                            }
                                        }
                                    }
                                }
                                
                                let output_idx = ((n * out_channels + out_c) * out_height + oh) * out_width + ow;
                                output_data[output_idx] = sum;
                            }
                        }
                    }
                }
            }
            
            if let Some(bias) = bias {
                let bias_data = bias.data_ptr::<f32>()?;
                
                for n in 0..batch_size {
                    for c in 0..out_channels {
                        for h in 0..out_height {
                            for w in 0..out_width {
                                let idx = ((n * out_channels + c) * out_height + h) * out_width + w;
                                output_data[idx] += bias_data[c];
                            }
                        }
                    }
                }
            }
            
            Ok(output)
        },
        DeviceType::CUDA => {
            #[cfg(feature = "cuda")]
            {
                Err(PhynexusError::UnsupportedOperation(
                    "CUDA conv2d implementation in progress".to_string()
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
            #[cfg(feature = "rocm")]
            {
                Err(PhynexusError::UnsupportedOperation(
                    "ROCm conv2d implementation in progress".to_string()
                ))
            }
            
            #[cfg(not(feature = "rocm"))]
            {
                Err(PhynexusError::UnsupportedOperation(
                    "ROCm support not enabled".to_string()
                ))
            }
        },
        DeviceType::WebGPU => {
            #[cfg(feature = "webgpu")]
            {
                Err(PhynexusError::UnsupportedOperation(
                    "WebGPU conv2d implementation in progress".to_string()
                ))
            }
            
            #[cfg(not(feature = "webgpu"))]
            {
                Err(PhynexusError::UnsupportedOperation(
                    "WebGPU support not enabled".to_string()
                ))
            }
        },
    }
}

/// Perform 2D transposed convolution: output = conv_transpose2d(input, weight, bias)
pub fn conv_transpose2d(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    params: &ConvParams,
) -> Result<Tensor> {
    // Check that the tensors are on the same device
    if input.device() != weight.device() {
        return Err(PhynexusError::InvalidArgument(
            "Input and weight must be on the same device for conv_transpose2d".to_string()
        ));
    }
    
    if let Some(bias) = bias {
        if input.device() != bias.device() {
            return Err(PhynexusError::InvalidArgument(
                "Input and bias must be on the same device for conv_transpose2d".to_string()
            ));
        }
    }
    
    // Check that the tensors have the correct dimensions
    let input_shape = input.shape();
    let weight_shape = weight.shape();
    
    if input_shape.len() != 4 {
        return Err(PhynexusError::ShapeMismatch(
            "Input must be 4-dimensional (N, C, H, W) for conv_transpose2d".to_string()
        ));
    }
    
    if weight_shape.len() != 4 {
        return Err(PhynexusError::ShapeMismatch(
            "Weight must be 4-dimensional (in_channels, out_channels/groups, kH, kW) for conv_transpose2d".to_string()
        ));
    }
    
    // Extract dimensions
    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];
    
    let out_channels = weight_shape[1] * params.groups;
    let kernel_height = weight_shape[2];
    let kernel_width = weight_shape[3];
    
    // Check that the number of input channels is compatible with the weight tensor
    if in_channels != weight_shape[0] {
        return Err(PhynexusError::ShapeMismatch(format!(
            "Number of input channels must match weight tensor first dimension: {} and {}",
            in_channels, weight_shape[0]
        )));
    }
    
    // Check that the bias tensor has the correct shape
    if let Some(bias) = bias {
        let bias_shape = bias.shape();
        
        if bias_shape.len() != 1 || bias_shape[0] != out_channels {
            return Err(PhynexusError::ShapeMismatch(format!(
                "Bias must be 1-dimensional with shape ({}), got {:?}",
                out_channels, bias_shape
            )));
        }
    }
    
    let out_height = (in_height - 1) * params.stride[0] - 2 * params.padding[0] + params.dilation[0] * (kernel_height - 1) + 1;
    let out_width = (in_width - 1) * params.stride[1] - 2 * params.padding[1] + params.dilation[1] * (kernel_width - 1) + 1;
    
    // Create output tensor
    let out_shape = vec![batch_size, out_channels, out_height, out_width];
    let mut output = Tensor::new(out_shape, input.dtype(), input.device().clone())?;
    
    // Dispatch to the appropriate implementation based on the device
    match input.device().device_type() {
        DeviceType::CPU => {
            let input_data = input.data_ptr::<f32>()?;
            let weight_data = weight.data_ptr::<f32>()?;
            let output_data = output.data_ptr_mut::<f32>()?;
            
            for i in 0..output_data.len() {
                output_data[i] = 0.0;
            }
            
            for n in 0..batch_size {
                for g in 0..params.groups {
                    let in_channels_per_group = in_channels / params.groups;
                    let out_channels_per_group = out_channels / params.groups;
                    
                    for ic in 0..in_channels_per_group {
                        let in_c = g * in_channels_per_group + ic;
                        
                        for ih in 0..in_height {
                            for iw in 0..in_width {
                                for oc in 0..out_channels_per_group {
                                    let out_c = g * out_channels_per_group + oc;
                                    
                                    for kh in 0..kernel_height {
                                        for kw in 0..kernel_width {
                                            let oh = ih * params.stride[0] + kh * params.dilation[0] - params.padding[0];
                                            let ow = iw * params.stride[1] + kw * params.dilation[1] - params.padding[1];
                                            
                                            if oh < out_height && ow < out_width && oh >= 0 && ow >= 0 {
                                                let input_idx = ((n * in_channels + in_c) * in_height + ih) * in_width + iw;
                                                let weight_idx = ((in_c * out_channels_per_group + oc) * kernel_height + kh) * kernel_width + kw;
                                                let output_idx = ((n * out_channels + out_c) * out_height + oh) * out_width + ow;
                                                
                                                output_data[output_idx] += input_data[input_idx] * weight_data[weight_idx];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            if let Some(bias) = bias {
                let bias_data = bias.data_ptr::<f32>()?;
                
                for n in 0..batch_size {
                    for c in 0..out_channels {
                        for h in 0..out_height {
                            for w in 0..out_width {
                                let idx = ((n * out_channels + c) * out_height + h) * out_width + w;
                                output_data[idx] += bias_data[c];
                            }
                        }
                    }
                }
            }
            
            Ok(output)
        },
        DeviceType::CUDA => {
            #[cfg(feature = "cuda")]
            {
                Err(PhynexusError::UnsupportedOperation(
                    "CUDA conv_transpose2d implementation in progress".to_string()
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
            #[cfg(feature = "rocm")]
            {
                Err(PhynexusError::UnsupportedOperation(
                    "ROCm conv_transpose2d implementation in progress".to_string()
                ))
            }
            
            #[cfg(not(feature = "rocm"))]
            {
                Err(PhynexusError::UnsupportedOperation(
                    "ROCm support not enabled".to_string()
                ))
            }
        },
        DeviceType::WebGPU => {
            #[cfg(feature = "webgpu")]
            {
                Err(PhynexusError::UnsupportedOperation(
                    "WebGPU conv_transpose2d implementation in progress".to_string()
                ))
            }
            
            #[cfg(not(feature = "webgpu"))]
            {
                Err(PhynexusError::UnsupportedOperation(
                    "WebGPU support not enabled".to_string()
                ))
            }
        },
    }
}
