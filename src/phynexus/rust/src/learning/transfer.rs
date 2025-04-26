//! Transfer learning module for the Phynexus engine

use crate::error::Result;
use crate::nn::Module;
use crate::tensor::Tensor;

/// Transfer learning model
pub struct TransferModel {
    /// Base model
    base_model: Box<dyn Module>,
    
    /// Head model
    head_model: Box<dyn Module>,
    
    /// Whether to freeze the base model
    #[allow(dead_code)]
    freeze_base: bool,
}

impl TransferModel {
    /// Create a new transfer learning model
    pub fn new(base_model: Box<dyn Module>, head_model: Box<dyn Module>, freeze_base: bool) -> Self {
        Self {
            base_model,
            head_model,
            freeze_base,
        }
    }
}

impl Module for TransferModel {
    /// Forward pass
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let base_output = self.base_model.forward(input)?;
        self.head_model.forward(&base_output)
    }
}

/// Train a transfer learning model
pub fn train_transfer_model(
    model: &mut TransferModel,
    optimizer: &mut dyn crate::optimizer::Optimizer,
    inputs: &[Tensor],
    targets: &[Tensor],
    epochs: usize,
    batch_size: usize,
) -> Result<()> {
    use crate::nn::loss::{cross_entropy_loss, Loss};
    use crate::tensor::Tensor;
    use crate::utils::data::DataLoader;
    
    if inputs.len() != targets.len() {
        return Err(crate::error::PhynexusError::InvalidArgument(
            format!("Number of inputs ({}) must match number of targets ({})",
                    inputs.len(), targets.len())
        ));
    }
    
    if inputs.is_empty() {
        return Err(crate::error::PhynexusError::InvalidArgument(
            "Input data cannot be empty".to_string()
        ));
    }
    
    let device = model.head_model.parameters()[0].device().clone();
    let num_samples = inputs.len();
    
    let data_loader = DataLoader::new(inputs, Some(targets), batch_size, true)?;
    
    if model.freeze_base {
        for param in model.base_model.parameters_mut() {
            param.set_requires_grad(false);
        }
    }
    
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut correct = 0;
        let mut total = 0;
        let mut batch_count = 0;
        
        for batch in data_loader.iter() {
            let batch_inputs = batch.inputs();
            let batch_targets = batch.targets().unwrap();
            
            let mut batch_loss = Tensor::zeros(&[1], &device)?;
            
            for (input, target) in batch_inputs.iter().zip(batch_targets.iter()) {
                // Forward pass
                let output = model.forward(input)?;
                
                let loss = cross_entropy_loss(&output, target)?;
                batch_loss = batch_loss.add(&loss)?;
                
                let predicted = output.argmax(1)?;
                let target_class = target.argmax(1)?;
                if predicted.item()? == target_class.item()? {
                    correct += 1;
                }
                total += 1;
            }
            
            batch_loss = batch_loss.div(&Tensor::from_scalar(batch_inputs.len() as f32, &device)?)?;
            
            optimizer.zero_grad();
            batch_loss.backward()?;
            optimizer.step()?;
            
            epoch_loss += batch_loss.item()?;
            batch_count += 1;
        }
        
        let accuracy = (correct as f32) / (total as f32) * 100.0;
        log::info!("Epoch {}/{}: Loss = {:.6}, Accuracy = {:.2}%", 
                  epoch + 1, epochs, epoch_loss / batch_count as f32, accuracy);
    }
    
    if model.freeze_base {
        for param in model.base_model.parameters_mut() {
            param.set_requires_grad(true);
        }
    }
    
    Ok(())
}
