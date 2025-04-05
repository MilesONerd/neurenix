//! Error types for the Phynexus engine

use thiserror::Error;

/// Result type for the Phynexus engine
pub type Result<T> = std::result::Result<T, PhynexusError>;

/// Error type for the Phynexus engine
#[derive(Error, Debug)]
pub enum PhynexusError {
    /// Invalid argument
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    
    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
    
    /// Unsupported data type
    #[error("Unsupported data type: {0}")]
    UnsupportedDataType(String),
    
    /// Shape mismatch
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    
    #[error("Invalid shape: {0}")]
    InvalidShape(String),
    
    #[error("Invalid value: {0}")]
    InvalidValue(String),
    
    /// Device error
    #[error("Device error: {0}")]
    DeviceError(String),
    
    /// Device mismatch
    #[error("Device mismatch: {0}")]
    DeviceMismatch(String),
    
    /// Memory error
    #[error("Memory error: {0}")]
    MemoryError(String),
    
    /// Uninitialized error
    #[error("Uninitialized error: {0}")]
    UninitializedError(String),
    
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// Other error
    #[error("Other error: {0}")]
    Other(String),
}
