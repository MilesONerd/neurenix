
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
            
            let grad_data = if let Some(grad) = param.grad() {
                let grad_ptr = grad.data() as *const f32;
                let grad_size = grad.numel();
                let mut grad_vec = Vec::with_capacity(grad_size);
                
                unsafe {
                    for i in 0..grad_size {
                        grad_vec.push(*grad_ptr.add(i));
                    }
                }
                
                grad_vec
            } else {
                continue; // Skip parameters without gradients
            };
            
            let param_ptr = param.data_mut() as *mut f32;
            let param_size = param.numel();
            
            let velocity = &mut self.velocity[idx];
            
            for i in 0..param_size {
                unsafe {
                    let param_val = *param_ptr.add(i);
                    let mut grad_val = grad_data[i];
                    
                    if self.weight_decay > 0.0 {
                        grad_val += self.weight_decay * param_val;
                    }
                    
                    if self.momentum > 0.0 {
                        velocity[i] = self.momentum * velocity[i] + grad_val;
                        *param_ptr.add(i) -= self.lr * velocity[i];
                    } else {
                        *param_ptr.add(i) -= self.lr * grad_val;
                    }
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
        let param_size = param.numel();
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
            
            let grad_data = if let Some(grad) = param.grad() {
                let grad_ptr = grad.data() as *const f32;
                let grad_size = grad.numel();
                let mut grad_vec = Vec::with_capacity(grad_size);
                
                unsafe {
                    for i in 0..grad_size {
                        grad_vec.push(*grad_ptr.add(i));
                    }
                }
                
                grad_vec
            } else {
                continue; // Skip parameters without gradients
            };
            
            let param_ptr = param.data_mut() as *mut f32;
            let param_size = param.numel();
            
            let m = &mut self.m[idx];
            let v = &mut self.v[idx];
            
            let bias_correction1 = 1.0 - self.beta1.powi(self.step as i32);
            let bias_correction2 = 1.0 - self.beta2.powi(self.step as i32);
            
            for i in 0..param_size {
                unsafe {
                    let param_val = *param_ptr.add(i);
                    let mut grad_val = grad_data[i];
                    
                    if self.weight_decay > 0.0 {
                        grad_val += self.weight_decay * param_val;
                    }
                    
                    m[i] = self.beta1 * m[i] + (1.0 - self.beta1) * grad_val;
                    
                    v[i] = self.beta2 * v[i] + (1.0 - self.beta2) * grad_val * grad_val;
                    
                    let m_hat = m[i] / bias_correction1;
                    
                    let v_hat = v[i] / bias_correction2;
                    
                    *param_ptr.add(i) -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
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
        let param_size = param.numel();
        self.params.push(param as *mut Tensor);
        self.m.push(vec![0.0; param_size]);
        self.v.push(vec![0.0; param_size]);
        Ok(())
    }
}
