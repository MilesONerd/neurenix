
use crate::error::PhynexusError;
use crate::tensor::Tensor;
use crate::nn::{Module, Conv2d, MaxPool2d, AvgPool2d, Linear, ReLU, BatchNorm2d};
use pyo3::prelude::*;
use std::collections::HashMap;

pub struct MultiScaleModel {
    input_channels: usize,
    num_scales: usize,
    scale_factor: f32,
    scale_branches: Vec<Module>,
    fusion: Module,
}

impl MultiScaleModel {
    pub fn new(input_channels: usize, num_scales: usize, scale_factor: f32) -> Self {
        Self {
            input_channels,
            num_scales,
            scale_factor,
            scale_branches: Vec::new(),
            fusion: Module::new(),
        }
    }
    
    fn generate_multi_scale_inputs(&self, x: &Tensor) -> Vec<Tensor> {
        let mut scale_inputs = vec![x.clone()];  // Original scale
        
        let mut current_input = x.clone();
        for i in 1..self.num_scales {
            let shape = current_input.shape();
            let h = shape[shape.len() - 2];
            let w = shape[shape.len() - 1];
            
            let new_h = (h as f32 * self.scale_factor) as usize;
            let new_w = (w as f32 * self.scale_factor) as usize;
            
            let pool = AvgPool2d::new((h / new_h, w / new_w));
            let downsampled = pool.forward(&current_input);
            
            scale_inputs.push(downsampled.clone());
            current_input = downsampled;
        }
        
        scale_inputs
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let scale_inputs = self.generate_multi_scale_inputs(x);
        
        let mut scale_features = Vec::new();
        for (i, scale_input) in scale_inputs.iter().enumerate() {
            scale_features.push(self.scale_branches[i].forward(scale_input));
        }
        
        self.fusion.forward(&scale_features)
    }
}

pub struct PyramidNetwork {
    model: MultiScaleModel,
    hidden_channels: Vec<usize>,
}

impl PyramidNetwork {
    pub fn new(input_channels: usize, hidden_channels: Vec<usize>, num_scales: usize, scale_factor: f32) -> Self {
        let mut model = MultiScaleModel::new(input_channels, num_scales, scale_factor);
        
        let mut branches = Vec::new();
        for i in 0..num_scales {
            let mut layers = Vec::new();
            let mut in_channels = input_channels;
            
            for &out_channels in &hidden_channels {
                layers.push(Conv2d::new(in_channels, out_channels, 3, 1, 1, false));
                layers.push(BatchNorm2d::new(out_channels));
                layers.push(ReLU::new());
                in_channels = out_channels;
            }
            
            branches.push(Module::from_layers(layers));
        }
        
        model.scale_branches = branches;
        
        
        Self {
            model,
            hidden_channels,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.model.forward(x)
    }
}

pub struct UNet {
    model: MultiScaleModel,
    hidden_channels: Vec<usize>,
    output_channels: usize,
}

impl UNet {
    pub fn new(input_channels: usize, hidden_channels: Vec<usize>, output_channels: usize, num_scales: usize, scale_factor: f32) -> Self {
        let mut model = MultiScaleModel::new(input_channels, num_scales, scale_factor);
        
        let mut encoders = Vec::new();
        
        for i in 0..num_scales {
            let mut layers = Vec::new();
            let in_channels = if i == 0 { input_channels } else { hidden_channels[i-1] };
            let out_channels = hidden_channels[i];
            
            layers.push(Conv2d::new(in_channels, out_channels, 3, 1, 1, false));
            layers.push(BatchNorm2d::new(out_channels));
            layers.push(ReLU::new());
            layers.push(Conv2d::new(out_channels, out_channels, 3, 1, 1, false));
            layers.push(BatchNorm2d::new(out_channels));
            layers.push(ReLU::new());
            
            encoders.push(Module::from_layers(layers));
        }
        
        model.scale_branches = encoders;
        
        
        Self {
            model,
            hidden_channels,
            output_channels,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.model.forward(x)
    }
}

pub fn register_models(py: Python, m: &PyModule) -> PyResult<()> {
    let multi_scale_model = PyModule::new(py, "multi_scale_model")?;
    m.add_submodule(multi_scale_model)?;
    
    let pyramid_network = PyModule::new(py, "pyramid_network")?;
    m.add_submodule(pyramid_network)?;
    
    let unet = PyModule::new(py, "unet")?;
    m.add_submodule(unet)?;
    
    Ok(())
}
