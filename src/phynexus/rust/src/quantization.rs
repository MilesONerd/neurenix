
use std::collections::HashMap;
use std::sync::Arc;

use crate::error::{PhynexusError, Result};
use crate::tensor::{Tensor, TensorData};
use crate::device::DeviceType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantizationType {
    INT8,
    
    FP16,
    
    FP8,
}

impl std::fmt::Display for QuantizationType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantizationType::INT8 => write!(f, "INT8"),
            QuantizationType::FP16 => write!(f, "FP16"),
            QuantizationType::FP8 => write!(f, "FP8"),
        }
    }
}

pub struct QuantizedTensor {
    pub tensor: Arc<Tensor>,
    
    pub scale: f32,
    
    pub zero_point: i32,
    
    pub dtype: QuantizationType,
}

impl QuantizedTensor {
    pub fn new(tensor: Arc<Tensor>, scale: f32, zero_point: i32, dtype: QuantizationType) -> Self {
        Self {
            tensor,
            scale,
            zero_point,
            dtype,
        }
    }
    
    pub fn dequantize(&self) -> Result<Arc<Tensor>> {
        match self.dtype {
            QuantizationType::INT8 => {
                let tensor_data = self.tensor.data()?;
                let mut dequantized_data = Vec::with_capacity(tensor_data.len());
                
                for &value in tensor_data.iter() {
                    let dequantized = (value as i32 - self.zero_point) as f32 * self.scale;
                    dequantized_data.push(dequantized);
                }
                
                let shape = self.tensor.shape()?;
                Tensor::from_data(dequantized_data, shape.to_vec(), self.tensor.device_type()?)
            },
            QuantizationType::FP16 => {
                let tensor_data = self.tensor.data()?;
                let mut dequantized_data = Vec::with_capacity(tensor_data.len() / 2);
                
                for i in (0..tensor_data.len()).step_by(2) {
                    if i + 1 < tensor_data.len() {
                        let bytes = [tensor_data[i], tensor_data[i + 1]];
                        let half = u16::from_le_bytes(bytes);
                        let float = half_to_float(half);
                        dequantized_data.push(float);
                    }
                }
                
                let shape = self.tensor.shape()?;
                Tensor::from_data(dequantized_data, shape.to_vec(), self.tensor.device_type()?)
            },
            QuantizationType::FP8 => {
                let tensor_data = self.tensor.data()?;
                let mut dequantized_data = Vec::with_capacity(tensor_data.len());
                
                for &value in tensor_data.iter() {
                    let dequantized = value as f32 / self.scale;
                    dequantized_data.push(dequantized);
                }
                
                let shape = self.tensor.shape()?;
                Tensor::from_data(dequantized_data, shape.to_vec(), self.tensor.device_type()?)
            },
        }
    }
}

pub fn quantize_tensor(tensor: &Arc<Tensor>, dtype: QuantizationType) -> Result<QuantizedTensor> {
    match dtype {
        QuantizationType::INT8 => {
            let tensor_data = tensor.data()?;
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            
            for &value in tensor_data.iter() {
                if value < min_val {
                    min_val = value;
                }
                if value > max_val {
                    max_val = value;
                }
            }
            
            let scale = if max_val > min_val { (max_val - min_val) / 255.0 } else { 1.0 };
            let zero_point = if scale != 0.0 { (127.0 - max_val / scale).round() as i32 } else { 0 };
            let zero_point = zero_point.max(0).min(255);
            
            let mut quantized_data = Vec::with_capacity(tensor_data.len());
            for &value in tensor_data.iter() {
                let quantized = ((value / scale) + zero_point as f32).round() as u8;
                quantized_data.push(quantized as u8);
            }
            
            let shape = tensor.shape()?;
            let quantized_tensor = Tensor::from_data(quantized_data, shape.to_vec(), tensor.device_type()?)?;
            
            Ok(QuantizedTensor::new(quantized_tensor, scale, zero_point, dtype))
        },
        QuantizationType::FP16 => {
            let tensor_data = tensor.data()?;
            let mut quantized_data = Vec::with_capacity(tensor_data.len() * 2);
            
            for &value in tensor_data.iter() {
                let half = float_to_half(value);
                let bytes = half.to_le_bytes();
                quantized_data.push(bytes[0]);
                quantized_data.push(bytes[1]);
            }
            
            let shape = tensor.shape()?;
            let quantized_tensor = Tensor::from_data(quantized_data, shape.to_vec(), tensor.device_type()?)?;
            
            Ok(QuantizedTensor::new(quantized_tensor, 1.0, 0, dtype))
        },
        QuantizationType::FP8 => {
            let tensor_data = tensor.data()?;
            let mut abs_max = 0.0;
            
            for &value in tensor_data.iter() {
                let abs_value = value.abs();
                if abs_value > abs_max {
                    abs_max = abs_value;
                }
            }
            
            let scale = if abs_max > 0.0 { 127.0 / abs_max } else { 1.0 };
            
            let mut quantized_data = Vec::with_capacity(tensor_data.len());
            for &value in tensor_data.iter() {
                let quantized = (value * scale).round() as i8;
                quantized_data.push(quantized as u8);
            }
            
            let shape = tensor.shape()?;
            let quantized_tensor = Tensor::from_data(quantized_data, shape.to_vec(), tensor.device_type()?)?;
            
            Ok(QuantizedTensor::new(quantized_tensor, 1.0 / scale, 0, dtype))
        },
    }
}

pub fn prune_tensor(tensor: &Arc<Tensor>, sparsity: f32, method: &str) -> Result<Arc<Tensor>> {
    if sparsity < 0.0 || sparsity > 1.0 {
        return Err(PhynexusError::InvalidArgument(format!(
            "Sparsity must be between 0.0 and 1.0, got {}", sparsity
        )));
    }
    
    let tensor_data = tensor.data()?;
    let mut pruned_data = tensor_data.clone();
    
    match method {
        "magnitude" => {
            let mut abs_values: Vec<(usize, f32)> = tensor_data.iter()
                .enumerate()
                .map(|(i, &v)| (i, v.abs()))
                .collect();
            
            abs_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            
            let threshold_idx = (tensor_data.len() as f32 * sparsity) as usize;
            for i in 0..threshold_idx {
                pruned_data[abs_values[i].0] = 0.0;
            }
        },
        "random" => {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            
            for i in 0..pruned_data.len() {
                if rng.gen::<f32>() < sparsity {
                    pruned_data[i] = 0.0;
                }
            }
        },
        _ => {
            return Err(PhynexusError::InvalidArgument(format!(
                "Unsupported pruning method: {}", method
            )));
        }
    }
    
    let shape = tensor.shape()?;
    Tensor::from_data(pruned_data, shape.to_vec(), tensor.device_type()?)
}

fn float_to_half(value: f32) -> u16 {
    //
    
    let bits = value.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;
    
    if exponent == 0 {
        return (sign << 15) as u16;
    } else if exponent == 0xFF {
        if mantissa == 0 {
            return ((sign << 15) | (0x1F << 10)) as u16;
        } else {
            return ((sign << 15) | (0x1F << 10) | 0x200) as u16;
        }
    }
    
    let adjusted_exponent = exponent - 127 + 15;
    
    if adjusted_exponent < 0 {
        return (sign << 15) as u16;
    } else if adjusted_exponent > 31 {
        return ((sign << 15) | (0x1F << 10)) as u16;
    }
    
    let half_mantissa = (mantissa >> 13) as u16;
    
    ((sign << 15) | ((adjusted_exponent as u16) << 10) | half_mantissa) as u16
}

fn half_to_float(half: u16) -> f32 {
    //
    
    let sign = ((half >> 15) & 0x1) as u32;
    let exponent = ((half >> 10) & 0x1F) as i32;
    let mantissa = (half & 0x3FF) as u32;
    
    if exponent == 0 {
        if mantissa == 0 {
            return f32::from_bits((sign << 31) as u32);
        } else {
            return f32::from_bits((sign << 31) as u32);
        }
    } else if exponent == 0x1F {
        if mantissa == 0 {
            return f32::from_bits((sign << 31) | (0xFF << 23)) as f32;
        } else {
            return f32::from_bits((sign << 31) | (0xFF << 23) | (mantissa << 13)) as f32;
        }
    }
    
    let adjusted_exponent = exponent - 15 + 127;
    
    let float_mantissa = mantissa << 13;
    
    f32::from_bits((sign << 31) | ((adjusted_exponent as u32) << 23) | float_mantissa)
}

pub struct CalibrationParams {
    pub scales: HashMap<String, f32>,
    
    pub zero_points: HashMap<String, i32>,
    
    pub dtype: QuantizationType,
}

impl CalibrationParams {
    pub fn new(dtype: QuantizationType) -> Self {
        Self {
            scales: HashMap::new(),
            zero_points: HashMap::new(),
            dtype,
        }
    }
    
    pub fn add_layer(&mut self, name: &str, scale: f32, zero_point: i32) {
        self.scales.insert(name.to_string(), scale);
        self.zero_points.insert(name.to_string(), zero_point);
    }
    
    pub fn get_scale(&self, name: &str) -> Option<f32> {
        self.scales.get(name).copied()
    }
    
    pub fn get_zero_point(&self, name: &str) -> Option<i32> {
        self.zero_points.get(name).copied()
    }
}

pub struct QuantizationConfig {
    pub dtype: QuantizationType,
    
    pub per_channel: bool,
    
    pub symmetric: bool,
    
    pub calibration_params: Option<CalibrationParams>,
}

impl QuantizationConfig {
    pub fn new(dtype: QuantizationType) -> Self {
        Self {
            dtype,
            per_channel: false,
            symmetric: false,
            calibration_params: None,
        }
    }
    
    pub fn with_per_channel(mut self, per_channel: bool) -> Self {
        self.per_channel = per_channel;
        self
    }
    
    pub fn with_symmetric(mut self, symmetric: bool) -> Self {
        self.symmetric = symmetric;
        self
    }
    
    pub fn with_calibration_params(mut self, calibration_params: CalibrationParams) -> Self {
        self.calibration_params = Some(calibration_params);
        self
    }
}

pub struct QATConfig {
    pub dtype: QuantizationType,
    
    pub per_channel: bool,
    
    pub symmetric: bool,
    
    pub quantize_weights: bool,
    
    pub quantize_activations: bool,
}

impl QATConfig {
    pub fn new(dtype: QuantizationType) -> Self {
        Self {
            dtype,
            per_channel: false,
            symmetric: false,
            quantize_weights: true,
            quantize_activations: true,
        }
    }
    
    pub fn with_per_channel(mut self, per_channel: bool) -> Self {
        self.per_channel = per_channel;
        self
    }
    
    pub fn with_symmetric(mut self, symmetric: bool) -> Self {
        self.symmetric = symmetric;
        self
    }
    
    pub fn with_quantize_weights(mut self, quantize_weights: bool) -> Self {
        self.quantize_weights = quantize_weights;
        self
    }
    
    pub fn with_quantize_activations(mut self, quantize_activations: bool) -> Self {
        self.quantize_activations = quantize_activations;
        self
    }
}

pub struct PruningConfig {
    pub sparsity: f32,
    
    pub method: String,
    
    pub structured: bool,
}

impl PruningConfig {
    pub fn new(sparsity: f32, method: &str) -> Self {
        Self {
            sparsity,
            method: method.to_string(),
            structured: false,
        }
    }
    
    pub fn with_structured(mut self, structured: bool) -> Self {
        self.structured = structured;
        self
    }
}
