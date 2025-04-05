
use crate::tensor::Tensor;
use crate::error::{PhynexusError, Result};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use std::f32;

pub fn zeros(shape: &[usize]) -> Result<Tensor> {
    Tensor::zeros(shape)
}

pub fn ones(shape: &[usize]) -> Result<Tensor> {
    Tensor::ones(shape)
}

pub fn uniform(shape: &[usize], low: f32, high: f32) -> Result<Tensor> {
    let size = shape.iter().product();
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(low, high);
    
    let mut tensor = Tensor::zeros(shape)?;
    
    unsafe {
        let ptr = tensor.data_mut() as *mut f32;
        for i in 0..size {
            *ptr.add(i) = dist.sample(&mut rng);
        }
    }
    
    Ok(tensor)
}

pub fn normal(shape: &[usize], mean: f32, std_dev: f32) -> Result<Tensor> {
    let size = shape.iter().product();
    let mut rng = rand::thread_rng();
    
    let mut tensor = Tensor::zeros(shape)?;
    
    unsafe {
        let ptr = tensor.data_mut() as *mut f32;
        
        let mut i = 0;
        while i < size {
            let u1 = rng.gen::<f32>();
            let u2 = rng.gen::<f32>();
            
            if u1 > 0.0 {  // Avoid log(0)
                let z0 = (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos();
                *ptr.add(i) = mean + std_dev * z0;
                
                if i + 1 < size {
                    let z1 = (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).sin();
                    *ptr.add(i + 1) = mean + std_dev * z1;
                    i += 2;
                } else {
                    i += 1;
                }
            }
        }
    }
    
    Ok(tensor)
}

pub fn kaiming_uniform(shape: &[usize], fan_in: usize) -> Result<Tensor> {
    let gain = 1.0;
    let bound = gain * (1.0 / (fan_in as f32)).sqrt();
    uniform(shape, -bound, bound)
}

pub fn kaiming_uniform_with_device(shape: &[usize], device: crate::device::Device) -> Result<Tensor> {
    let fan_in = if !shape.is_empty() { shape[0] } else { 1 };
    let mut tensor = kaiming_uniform(shape, fan_in)?;
    
    if device != crate::device::Device::cpu() {
        tensor = tensor.to_device(device)?;
    }
    
    Ok(tensor)
}

pub fn kaiming_normal(shape: &[usize], fan_in: usize) -> Result<Tensor> {
    let gain = 1.0;
    let std = gain / (fan_in as f32).sqrt();
    normal(shape, 0.0, std)
}

pub fn kaiming_normal_with_device(shape: &[usize], device: crate::device::Device) -> Result<Tensor> {
    let fan_in = if !shape.is_empty() { shape[0] } else { 1 };
    let mut tensor = kaiming_normal(shape, fan_in)?;
    
    if device != crate::device::Device::cpu() {
        tensor = tensor.to_device(device)?;
    }
    
    Ok(tensor)
}

pub fn xavier_uniform(shape: &[usize], fan_in: usize, fan_out: usize) -> Result<Tensor> {
    let gain = 1.0;
    let bound = gain * (6.0 / ((fan_in + fan_out) as f32)).sqrt();
    uniform(shape, -bound, bound)
}

pub fn xavier_uniform_with_device(shape: &[usize], device: crate::device::Device) -> Result<Tensor> {
    let fan_in = if !shape.is_empty() { shape[0] } else { 1 };
    let fan_out = if shape.len() > 1 { shape[shape.len() - 1] } else { 1 };
    
    let mut tensor = xavier_uniform(shape, fan_in, fan_out)?;
    
    if device != crate::device::Device::cpu() {
        tensor = tensor.to_device(device)?;
    }
    
    Ok(tensor)
}

pub fn xavier_normal(shape: &[usize], fan_in: usize, fan_out: usize) -> Result<Tensor> {
    let gain = 1.0;
    let std = gain * (2.0 / ((fan_in + fan_out) as f32)).sqrt();
    normal(shape, 0.0, std)
}

pub fn xavier_normal_with_device(shape: &[usize], device: crate::device::Device) -> Result<Tensor> {
    let fan_in = if !shape.is_empty() { shape[0] } else { 1 };
    let fan_out = if shape.len() > 1 { shape[shape.len() - 1] } else { 1 };
    
    let mut tensor = xavier_normal(shape, fan_in, fan_out)?;
    
    if device != crate::device::Device::cpu() {
        tensor = tensor.to_device(device)?;
    }
    
    Ok(tensor)
}
