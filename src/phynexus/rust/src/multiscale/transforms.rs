
use crate::error::PhynexusError;
use crate::tensor::Tensor;
use crate::nn::{Module, Conv2d, ConvTranspose2d, AvgPool2d, Upsample};
use pyo3::prelude::*;
use std::collections::HashMap;

pub struct MultiScaleTransform {
}

impl MultiScaleTransform {
    pub fn new() -> Self {
        Self {}
    }
    
    pub fn forward(&self, x: &Tensor) -> Vec<Tensor> {
        vec![x.clone()]
    }
}

pub struct Rescale {
    transform: MultiScaleTransform,
    scales: Vec<f32>,
    mode: String,
    align_corners: bool,
    upsamplers: Vec<Option<Module>>,
}

impl Rescale {
    pub fn new(scales: Vec<f32>, mode: &str, align_corners: bool) -> Self {
        let transform = MultiScaleTransform::new();
        
        let mut upsamplers = Vec::new();
        for &scale in &scales {
            if scale != 1.0 {  // No need for upsampler if scale is 1.0
                let align = if mode != "nearest" { Some(align_corners) } else { None };
                let upsampler = Upsample::new(scale, mode, align);
                upsamplers.push(Some(Module::from(upsampler)));
            } else {
                upsamplers.push(None);
            }
        }
        
        Self {
            transform,
            scales,
            mode: mode.to_string(),
            align_corners,
            upsamplers,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Vec<Tensor> {
        let mut outputs = Vec::new();
        
        for (i, &scale) in self.scales.iter().enumerate() {
            if scale == 1.0 {
                outputs.push(x.clone());
            } else {
                let upsampler = self.upsamplers[i].as_ref().unwrap();
                outputs.push(upsampler.forward(x));
            }
        }
        
        outputs
    }
}

pub struct PyramidDownsample {
    transform: MultiScaleTransform,
    num_levels: usize,
    downsample_factor: f32,
    mode: String,
    downsamplers: Vec<Module>,
}

impl PyramidDownsample {
    pub fn new(num_levels: usize, downsample_factor: f32, mode: &str) -> Self {
        let transform = MultiScaleTransform::new();
        
        let mut downsamplers = Vec::new();
        
        for _ in 0..(num_levels - 1) {  // No downsampler needed for the original resolution
            let downsampler = match mode {
                "pool" => {
                    let kernel_size = (1.0 / downsample_factor) as usize;
                    Module::from(AvgPool2d::new((kernel_size, kernel_size), (kernel_size, kernel_size), (0, 0), false))
                },
                "conv" => {
                    let stride = (1.0 / downsample_factor) as usize;
                    Module::from(Conv2d::new(0, 0, 3, stride, 1, false))  // in_channels and out_channels set dynamically
                },
                _ => panic!("Unsupported mode: {}. Use 'pool' or 'conv'.", mode),
            };
            
            downsamplers.push(downsampler);
        }
        
        Self {
            transform,
            num_levels,
            downsample_factor,
            mode: mode.to_string(),
            downsamplers,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Vec<Tensor> {
        let mut outputs = vec![x.clone()];  // Start with the original resolution
        let mut current = x.clone();
        
        for i in 0..(self.num_levels - 1) {
            if self.mode == "conv" {
                let in_channels = current.shape()[1];
                
            }
            
            current = self.downsamplers[i].forward(&current);
            outputs.push(current.clone());
        }
        
        outputs
    }
}

pub struct GaussianPyramid {
    transform: MultiScaleTransform,
    num_levels: usize,
    sigma: f32,
    kernel_size: usize,
    smooth_down: Vec<Module>,
}

impl GaussianPyramid {
    pub fn new(num_levels: usize, sigma: f32, kernel_size: usize) -> Self {
        let transform = MultiScaleTransform::new();
        
        let mut smooth_down = Vec::new();
        
        for _ in 0..(num_levels - 1) {
            let mut layers = Vec::new();
            
            layers.push(Conv2d::new(0, 0, kernel_size as u32, 1, kernel_size as u32 / 2, false));  // in_channels and out_channels set dynamically
            layers.push(AvgPool2d::new((2, 2), (2, 2), (0, 0), false));
            
            smooth_down.push(Module::from_layers(layers));
        }
        
        Self {
            transform,
            num_levels,
            sigma,
            kernel_size,
            smooth_down,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Vec<Tensor> {
        let mut outputs = vec![x.clone()];  // Start with the original resolution
        let mut current = x.clone();
        
        for i in 0..(self.num_levels - 1) {
            let in_channels = current.shape()[1];
            
            
            current = self.smooth_down[i].forward(&current);
            outputs.push(current.clone());
        }
        
        outputs
    }
}

pub struct LaplacianPyramid {
    transform: MultiScaleTransform,
    num_levels: usize,
    gaussian_pyramid: GaussianPyramid,
    upsample: Vec<Module>,
}

impl LaplacianPyramid {
    pub fn new(num_levels: usize, sigma: f32, kernel_size: usize) -> Self {
        let transform = MultiScaleTransform::new();
        
        let gaussian_pyramid = GaussianPyramid::new(num_levels, sigma, kernel_size);
        
        let mut upsample = Vec::new();
        
        for _ in 0..(num_levels - 1) {
            let mut layers = Vec::new();
            
            layers.push(Upsample::new(2.0, "bilinear", Some(true)));
            layers.push(Conv2d::new(0, 0, kernel_size as u32, 1, kernel_size as u32 / 2, false));  // in_channels and out_channels set dynamically
            
            upsample.push(Module::from_layers(layers));
        }
        
        Self {
            transform,
            num_levels,
            gaussian_pyramid,
            upsample,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Vec<Tensor> {
        let gaussian_levels = self.gaussian_pyramid.forward(x);
        
        let mut laplacian_levels = Vec::new();
        
        for i in 0..(self.num_levels - 1) {
            let in_channels = gaussian_levels[i+1].shape()[1];
            
            
            let upsampled = self.upsample[i].forward(&gaussian_levels[i+1]);
            
            let laplacian = &gaussian_levels[i] - &upsampled;
            laplacian_levels.push(laplacian);
        }
        
        laplacian_levels.push(gaussian_levels[gaussian_levels.len() - 1].clone());
        
        laplacian_levels
    }
}

pub fn register_transforms(py: Python, m: &PyModule) -> PyResult<()> {
    let multi_scale_transform = PyModule::new(py, "multi_scale_transform")?;
    m.add_submodule(multi_scale_transform)?;
    
    let rescale = PyModule::new(py, "rescale")?;
    m.add_submodule(rescale)?;
    
    let pyramid_downsample = PyModule::new(py, "pyramid_downsample")?;
    m.add_submodule(pyramid_downsample)?;
    
    let gaussian_pyramid = PyModule::new(py, "gaussian_pyramid")?;
    m.add_submodule(gaussian_pyramid)?;
    
    let laplacian_pyramid = PyModule::new(py, "laplacian_pyramid")?;
    m.add_submodule(laplacian_pyramid)?;
    
    Ok(())
}
