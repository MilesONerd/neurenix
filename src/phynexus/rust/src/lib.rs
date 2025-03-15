//! Phynexus: A high-performance tensor library for embedded devices

// Re-export core modules
pub mod device;
pub mod error;
pub mod hardware;
pub mod ops;
pub mod tensor;
pub mod tensor_ops;
pub mod dtype;
pub mod learning;
pub mod nn;
pub mod optimizer;

// Re-export core types
pub use device::Device;
pub use error::{PhynexusError, Result};
pub use tensor::Tensor;

// Re-export tensor operations
pub use tensor_ops::{
    matmul,
    add, subtract, multiply,
};

/// Initialize the Phynexus library
pub fn init() -> error::Result<()> {
    // Initialize hardware backends
    Ok(())
}
