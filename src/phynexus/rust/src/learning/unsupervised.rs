//! Unsupervised learning module for the Phynexus engine

use crate::error::Result;
use crate::nn::Module;
use crate::tensor::Tensor;

/// Autoencoder model
pub struct Autoencoder {
    /// Encoder model
    encoder: Box<dyn Module>,
    
    /// Decoder model
    decoder: Box<dyn Module>,
}

impl Autoencoder {
    /// Create a new autoencoder model
    pub fn new(encoder: Box<dyn Module>, decoder: Box<dyn Module>) -> Self {
        Self {
            encoder,
            decoder,
        }
    }
    
    /// Encode input
    pub fn encode(&self, input: &Tensor) -> Result<Tensor> {
        self.encoder.forward(input)
    }
    
    /// Decode latent representation
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        self.decoder.forward(latent)
    }
}

impl Module for Autoencoder {
    /// Forward pass
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let latent = self.encode(input)?;
        self.decode(&latent)
    }
}

/// Train an autoencoder model
pub fn train_autoencoder(
    model: &mut Autoencoder,
    optimizer: &mut dyn crate::optimizer::Optimizer,
    inputs: &[Tensor],
    epochs: usize,
    batch_size: usize,
) -> Result<()> {
    use crate::nn::loss::{mse_loss, Loss};
    use crate::tensor::Tensor;
    use crate::utils::data::DataLoader;
    
    if inputs.is_empty() {
        return Err(crate::error::PhynexusError::InvalidArgument(
            "Input data cannot be empty".to_string()
        ));
    }
    
    let device = model.encoder.parameters()[0].device().clone();
    let num_samples = inputs.len();
    
    let data_loader = DataLoader::new(inputs, None, batch_size, true)?;
    
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;
        let mut batch_count = 0;
        
        for batch in data_loader.iter() {
            let batch_inputs = batch.inputs();
            
            let mut batch_loss = Tensor::zeros(&[1], &device)?;
            
            for input in batch_inputs {
                // Forward pass
                let output = model.forward(input)?;
                
                let loss = mse_loss(&output, input)?;
                batch_loss = batch_loss.add(&loss)?;
            }
            
            batch_loss = batch_loss.div(&Tensor::from_scalar(batch_inputs.len() as f32, &device)?)?;
            
            optimizer.zero_grad();
            batch_loss.backward()?;
            optimizer.step()?;
            
            epoch_loss += batch_loss.item()?;
            batch_count += 1;
        }
        
        log::info!("Epoch {}/{}: Loss = {:.6}", epoch + 1, epochs, epoch_loss / batch_count as f32);
    }
    
    Ok(())
}
