
use crate::error::PhynexusError;
use crate::tensor::Tensor;
use crate::nn::{Module, Conv2d, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d};
use pyo3::prelude::*;
use std::collections::HashMap;

pub struct MultiScalePooling {
    output_size: (usize, usize),
    pool_type: String,
    pool: Box<dyn Module>,
}

impl MultiScalePooling {
    pub fn new(output_size: (usize, usize), pool_type: &str) -> Self {
        let pool = match pool_type {
            "avg" => Module::from(AdaptiveAvgPool2d::new(output_size)),
            "max" => Module::from(AdaptiveMaxPool2d::new(output_size)),
            _ => panic!("Unsupported pool_type: {}. Use 'avg' or 'max'.", pool_type),
        };
        
        Self {
            output_size,
            pool_type: pool_type.to_string(),
            pool,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.pool.forward(x)
    }
}

pub struct PyramidPooling {
    in_channels: usize,
    out_channels: usize,
    pool_sizes: Vec<usize>,
    pool_type: String,
    pyramid_levels: Vec<Box<dyn Module>>,
}

impl PyramidPooling {
    pub fn new(in_channels: usize, out_channels: usize, pool_sizes: Vec<usize>, pool_type: &str) -> Self {
        let mut pyramid_levels = Vec::new();
        
        for &pool_size in &pool_sizes {
            let mut level = Vec::new();
            
            if pool_type == "avg" {
                level.push(AdaptiveAvgPool2d::new((pool_size, pool_size)));
            } else {
                level.push(AdaptiveMaxPool2d::new((pool_size, pool_size)));
            }
            
            level.push(Conv2d::new(in_channels, out_channels, 1, 1, 0, false));
            
            pyramid_levels.push(Module::from_layers(level));
        }
        
        Self {
            in_channels,
            out_channels,
            pool_sizes,
            pool_type: pool_type.to_string(),
            pyramid_levels,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let h = shape[shape.len() - 2];
        let w = shape[shape.len() - 1];
        
        let mut features = vec![x.clone()];
        
        for level in &self.pyramid_levels {
            let pooled = level.forward(x);
            let upsampled = pooled.resize(&[h, w]);
            features.push(upsampled);
        }
        
        Tensor::cat(&features, 1)
    }
}

pub struct SpatialPyramidPooling {
    output_sizes: Vec<usize>,
    pool_type: String,
    pools: Vec<Box<dyn Module>>,
}

impl SpatialPyramidPooling {
    pub fn new(output_sizes: Vec<usize>, pool_type: &str) -> Self {
        let mut pools = Vec::new();
        
        for &size in &output_sizes {
            let pool = match pool_type {
                "avg" => Module::from(AdaptiveAvgPool2d::new((size, size))),
                "max" => Module::from(AdaptiveMaxPool2d::new((size, size))),
                _ => panic!("Unsupported pool_type: {}. Use 'avg' or 'max'.", pool_type),
            };
            
            pools.push(pool);
        }
        
        Self {
            output_sizes,
            pool_type: pool_type.to_string(),
            pools,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let batch_size = x.shape()[0];
        let mut features = Vec::new();
        
        for pool in &self.pools {
            let pooled = pool.forward(x);
            let flat = pooled.view(&[batch_size, -1]);
            features.push(flat);
        }
        
        Tensor::cat(&features, 1)
    }
}

pub fn register_pooling(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let multi_scale_pooling = PyModule::new(py, "multi_scale_pooling")?;
    m.add_submodule(&multi_scale_pooling)?;
    
    let pyramid_pooling = PyModule::new(py, "pyramid_pooling")?;
    m.add_submodule(&pyramid_pooling)?;
    
    let spatial_pyramid_pooling = PyModule::new(py, "spatial_pyramid_pooling")?;
    m.add_submodule(&spatial_pyramid_pooling)?;
    
    Ok(())
}
