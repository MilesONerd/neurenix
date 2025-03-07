//! Transfer learning algorithms for the Phynexus engine

use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;
use crate::nn::Module;

/// Transfer learning configuration
pub struct TransferConfig {
    /// Whether to freeze the base model
    pub freeze_base: bool,
    
    /// Layers to fine-tune (if freeze_base is true)
    pub fine_tune_layers: Vec<String>,
    
    /// Learning rate for the base model
    pub base_lr: f32,
    
    /// Learning rate for the new layers
    pub new_layers_lr: f32,
}

impl Default for TransferConfig {
    fn default() -> Self {
        Self {
            freeze_base: true,
            fine_tune_layers: Vec::new(),
            base_lr: 0.0001,
            new_layers_lr: 0.001,
        }
    }
}

/// Transfer learning model
pub struct TransferModel {
    /// Base model
    base_model: Box<dyn Module>,
    
    /// New layers
    new_layers: Box<dyn Module>,
    
    /// Configuration
    config: TransferConfig,
}

impl TransferModel {
    /// Create a new transfer learning model
    pub fn new(base_model: Box<dyn Module>, new_layers: Box<dyn Module>, config: TransferConfig) -> Self {
        Self {
            base_model,
            new_layers,
            config,
        }
    }
    
    /// Forward pass
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let base_output = self.base_model.forward(input)?;
        self.new_layers.forward(&base_output)
    }
    
    /// Get the parameters of the model
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        
        // Add parameters from the base model if not frozen
        if !self.config.freeze_base {
            params.extend(self.base_model.parameters());
        }
        
        // Add parameters from the new layers
        params.extend(self.new_layers.parameters());
        
        params
    }
    
    /// Set the model to training mode
    pub fn train(&mut self, mode: bool) {
        self.base_model.train(mode);
        self.new_layers.train(mode);
    }
    
    /// Check if the model is in training mode
    pub fn is_training(&self) -> bool {
        self.new_layers.is_training()
    }
}

/// Fine-tune a pre-trained model on a new dataset
pub fn fine_tune(
    model: &mut TransferModel,
    optimizer: &mut dyn crate::optimizer::Optimizer,
    inputs: &[Tensor],
    targets: &[Tensor],
    epochs: usize,
    batch_size: usize,
) -> Result<Vec<f32>> {
    // TODO: Implement fine-tuning
    // For now, just return an error
    Err(PhynexusError::UnsupportedOperation(
        "Fine-tuning not yet implemented".to_string()
    ))
}
