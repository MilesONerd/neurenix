
use crate::error::Result;
use crate::tensor::Tensor;

pub trait Optimizer {
    fn step(&mut self) -> Result<()>;
    
    fn zero_grad(&mut self) -> Result<()>;
    
    fn add_param(&mut self, param: &mut Tensor) -> Result<()>;
}

pub struct SGD {
    lr: f32,
    
    momentum: f32,
    
    weight_decay: f32,
    
    params: Vec<*mut Tensor>,
    
    velocity: Vec<Vec<f32>>,
}

impl SGD {
    pub fn new(lr: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            momentum,
            weight_decay,
            params: Vec::new(),
            velocity: Vec::new(),
        }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) -> Result<()> {
        for (idx, param_ptr) in self.params.iter().enumerate() {
            let param = unsafe { &mut **param_ptr };
            
            if !param.requires_grad() {
                continue;
            }
            
            let grad = param.grad()
                .ok_or_else(|| crate::error::PhynexusError::UninitializedError(
                    "Parameter gradient is not initialized".to_string()
                ))?;
            
            let data = param.data_mut()?;
            let grad_data = grad.data()?;
            
            if self.weight_decay > 0.0 {
                for i in 0..data.len() {
                    grad_data[i] += self.weight_decay * data[i];
                }
            }
            
            if self.momentum > 0.0 {
                let velocity = &mut self.velocity[idx];
                
                for i in 0..data.len() {
                    velocity[i] = self.momentum * velocity[i] + grad_data[i];
                    data[i] -= self.lr * velocity[i];
                }
            } else {
                for i in 0..data.len() {
                    data[i] -= self.lr * grad_data[i];
                }
            }
        }
        
        Ok(())
    }
    
    fn zero_grad(&mut self) -> Result<()> {
        for param_ptr in &self.params {
            let param = unsafe { &mut **param_ptr };
            
            if !param.requires_grad() {
                continue;
            }
            
            param.set_grad(None);
        }
        
        Ok(())
    }
    
    fn add_param(&mut self, param: &mut Tensor) -> Result<()> {
        let param_size = param.data()?.len();
        self.params.push(param as *mut Tensor);
        self.velocity.push(vec![0.0; param_size]);
        Ok(())
    }
}

pub struct Adam {
    lr: f32,
    
    beta1: f32,
    
    beta2: f32,
    
    eps: f32,
    
    weight_decay: f32,
    
    params: Vec<*mut Tensor>,
    
    m: Vec<Vec<f32>>,
    
    v: Vec<Vec<f32>>,
    
    step: usize,
}

impl Adam {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            params: Vec::new(),
            m: Vec::new(),
            v: Vec::new(),
            step: 0,
        }
    }
}

impl Optimizer for Adam {
    fn step(&mut self) -> Result<()> {
        self.step += 1;
        
        for (idx, param_ptr) in self.params.iter().enumerate() {
            let param = unsafe { &mut **param_ptr };
            
            if !param.requires_grad() {
                continue;
            }
            
            let grad = param.grad()
                .ok_or_else(|| crate::error::PhynexusError::UninitializedError(
                    "Parameter gradient is not initialized".to_string()
                ))?;
            
            let data = param.data_mut()?;
            let grad_data = grad.data()?;
            
            if self.weight_decay > 0.0 {
                for i in 0..data.len() {
                    grad_data[i] += self.weight_decay * data[i];
                }
            }
            
            let m = &mut self.m[idx];
            let v = &mut self.v[idx];
            
            let beta1_t = self.beta1.powi(self.step as i32);
            let beta2_t = self.beta2.powi(self.step as i32);
            
            for i in 0..data.len() {
                m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad_data[i];
                v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad_data[i] * grad_data[i];
                
                let m_hat = m[i] / (1.0 - beta1_t);
                let v_hat = v[i] / (1.0 - beta2_t);
                
                data[i] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
        
        Ok(())
    }
    
    fn zero_grad(&mut self) -> Result<()> {
        for param_ptr in &self.params {
            let param = unsafe { &mut **param_ptr };
            
            if !param.requires_grad() {
                continue;
            }
            
            param.set_grad(None);
        }
        
        Ok(())
    }
    
    fn add_param(&mut self, param: &mut Tensor) -> Result<()> {
        let param_size = param.data()?.len();
        self.params.push(param as *mut Tensor);
        self.m.push(vec![0.0; param_size]);
        self.v.push(vec![0.0; param_size]);
        Ok(())
    }
}
