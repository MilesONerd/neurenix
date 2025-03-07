//! Error handling for the Phynexus engine

use thiserror::Error;
use std::result;

/// Custom error type for Phynexus operations
#[derive(Error, Debug)]
pub enum PhynexusError {
    #[error("Device error: {0}")]
    DeviceError(String),
    
    #[error("Memory allocation error: {0}")]
    MemoryError(String),
    
    #[error("Shape mismatch: {0}")]
    ShapeMismatch(String),
    
    #[error("Operation not supported: {0}")]
    UnsupportedOperation(String),
    
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Other error: {0}")]
    Other(String),
}

/// Result type for Phynexus operations
pub type Result<T> = result::Result<T, PhynexusError>;
