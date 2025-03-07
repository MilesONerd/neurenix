//! Phynexus - High-performance tensor operations engine for the Neurenix AI framework
//!
//! This library provides the core tensor operations and hardware abstraction layer
//! for the Neurenix AI framework, optimized for various hardware platforms.

mod error;
mod tensor;
mod ops;
mod device;
mod memory;
mod graph;
mod optimizer;
mod nn;
mod hardware;
mod learning;
mod tensor_ops;

pub use error::{PhynexusError, Result};
pub use tensor::Tensor;
pub use device::{Device, DeviceType};
pub use ops::*;
pub use graph::{Graph, Node, Op};
pub use optimizer::{Optimizer, SGD, Adam};
pub use nn::{Module, Linear, Conv2d, LSTM, Sequential};
pub use hardware::{Backend, get_backend, MultiDeviceManager, DeviceInfo};
pub use learning::transfer::{TransferModel, TransferConfig, fine_tune};
pub use learning::meta::{MetaLearningModel, MetaLearningConfig, meta_train};
pub use learning::unsupervised::{Autoencoder, UnsupervisedLearningConfig, unsupervised_train};
pub use tensor_ops::{matmul, add, subtract, multiply, divide, reshape, transpose, conv};

/// Initialize the Phynexus engine with the given configuration
pub fn init() -> Result<()> {
    // Initialize hardware subsystem
    hardware::init()?;
    
    // Initialize logging
    env_logger::init();
    
    // Log initialization
    log::info!("Phynexus engine initialized");
    
    Ok(())
}

/// Shutdown the Phynexus engine
pub fn shutdown() -> Result<()> {
    // Shutdown subsystems in reverse order of initialization
    log::info!("Phynexus engine shutting down");
    
    Ok(())
}

/// Get version information for the Phynexus engine
pub fn version() -> &'static str {
    "0.1.0"
}

/// Get build information for the Phynexus engine
pub fn build_info() -> &'static str {
    "Phynexus engine 0.1.0"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(version(), "0.1.0");
    }
}
