//! Meta-learning module for the Phynexus engine

use crate::error::Result;
use crate::nn::Module;
use crate::tensor::Tensor;

/// Meta-learning model
pub struct MetaLearningModel {
    /// Base model
    base_model: Box<dyn Module>,
    
    /// Adaptation model
    adaptation_model: Box<dyn Module>,
}

impl MetaLearningModel {
    /// Create a new meta-learning model
    pub fn new(base_model: Box<dyn Module>, adaptation_model: Box<dyn Module>) -> Self {
        Self {
            base_model,
            adaptation_model,
        }
    }
}

impl Module for MetaLearningModel {
    /// Forward pass
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let base_output = self.base_model.forward(input)?;
        self.adaptation_model.forward(&base_output)
    }
}

/// Train a meta-learning model
pub fn train_meta_learning_model(
    model: &mut MetaLearningModel,
    optimizer: &mut dyn crate::optimizer::Optimizer,
    task_inputs: &[Vec<Tensor>],
    task_targets: &[Vec<Tensor>],
    epochs: usize,
    tasks_per_batch: usize,
) -> Result<()> {
    use crate::nn::loss::{mse_loss, Loss};
    use crate::tensor::Tensor;
    use crate::utils::data::DataLoader;
    
    if task_inputs.len() != task_targets.len() {
        return Err(crate::error::PhynexusError::InvalidArgument(
            format!("Number of task inputs ({}) must match number of task targets ({})",
                    task_inputs.len(), task_targets.len())
        ));
    }
    
    let num_tasks = task_inputs.len();
    let device = model.base_model.parameters()[0].device().clone();
    
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut task_indices: Vec<usize> = (0..num_tasks).collect();
        
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        task_indices.shuffle(&mut rng);
        
        for batch_idx in 0..(num_tasks + tasks_per_batch - 1) / tasks_per_batch {
            let start_idx = batch_idx * tasks_per_batch;
            let end_idx = std::cmp::min(start_idx + tasks_per_batch, num_tasks);
            let batch_size = end_idx - start_idx;
            
            let mut batch_loss = Tensor::zeros(&[1], &device)?;
            
            for i in start_idx..end_idx {
                let task_idx = task_indices[i];
                let inputs = &task_inputs[task_idx];
                let targets = &task_targets[task_idx];
                
                if inputs.len() != targets.len() {
                    return Err(crate::error::PhynexusError::InvalidArgument(
                        format!("Number of inputs ({}) must match number of targets ({}) for task {}",
                                inputs.len(), targets.len(), task_idx)
                    ));
                }
                
                let support_size = inputs.len() / 2;
                let support_inputs = &inputs[0..support_size];
                let support_targets = &targets[0..support_size];
                let query_inputs = &inputs[support_size..];
                let query_targets = &targets[support_size..];
                
                for (input, target) in support_inputs.iter().zip(support_targets.iter()) {
                    let output = model.forward(input)?;
                    let loss = mse_loss(&output, target)?;
                    
                    optimizer.zero_grad();
                    loss.backward()?;
                    optimizer.step()?;
                }
                
                let mut task_loss = Tensor::zeros(&[1], &device)?;
                for (input, target) in query_inputs.iter().zip(query_targets.iter()) {
                    let output = model.forward(input)?;
                    let loss = mse_loss(&output, target)?;
                    task_loss = task_loss.add(&loss)?;
                }
                
                task_loss = task_loss.div(&Tensor::from_scalar(query_inputs.len() as f32, &device)?)?;
                batch_loss = batch_loss.add(&task_loss)?;
            }
            
            batch_loss = batch_loss.div(&Tensor::from_scalar(batch_size as f32, &device)?)?;
            
            optimizer.zero_grad();
            batch_loss.backward()?;
            optimizer.step()?;
            
            epoch_loss += batch_loss.item()?;
        }
        
        log::info!("Epoch {}/{}: Loss = {:.6}", epoch + 1, epochs, epoch_loss / (num_tasks as f32 / tasks_per_batch as f32));
    }
    
    Ok(())
}
