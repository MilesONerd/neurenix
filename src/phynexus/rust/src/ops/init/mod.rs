
use crate::tensor::Tensor;
use crate::error::{PhynexusError, Result};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use std::f32;

pub fn zeros(shape: &[usize]) -> Result<Tensor> {
    let size = shape.iter().product();
    let data = vec![0.0; size];
    Tensor::new(data, shape.to_vec())
}

pub fn ones(shape: &[usize]) -> Result<Tensor> {
    let size = shape.iter().product();
    let data = vec![1.0; size];
    Tensor::new(data, shape.to_vec())
}

pub fn uniform(shape: &[usize], low: f32, high: f32) -> Result<Tensor> {
    let size = shape.iter().product();
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(low, high);
    let data: Vec<f32> = (0..size).map(|_| dist.sample(&mut rng)).collect();
    Tensor::new(data, shape.to_vec())
}

pub fn normal(shape: &[usize], mean: f32, std: f32) -> Result<Tensor> {
    let size = shape.iter().product();
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..size)
        .map(|_| {
            let u1 = rng.gen::<f32>();
            let u2 = rng.gen::<f32>();
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            mean + std * z0
        })
        .collect();
    Tensor::new(data, shape.to_vec())
}

pub fn kaiming_uniform(shape: &[usize], fan_in: usize) -> Result<Tensor> {
    let gain = 1.0;
    let bound = gain * (1.0 / (fan_in as f32)).sqrt();
    uniform(shape, -bound, bound)
}

pub fn kaiming_normal(shape: &[usize], fan_in: usize) -> Result<Tensor> {
    let gain = 1.0;
    let std = gain / (fan_in as f32).sqrt();
    normal(shape, 0.0, std)
}

pub fn xavier_uniform(shape: &[usize], fan_in: usize, fan_out: usize) -> Result<Tensor> {
    let gain = 1.0;
    let bound = gain * (6.0 / ((fan_in + fan_out) as f32)).sqrt();
    uniform(shape, -bound, bound)
}

pub fn xavier_normal(shape: &[usize], fan_in: usize, fan_out: usize) -> Result<Tensor> {
    let gain = 1.0;
    let std = gain * (2.0 / ((fan_in + fan_out) as f32)).sqrt();
    normal(shape, 0.0, std)
}
