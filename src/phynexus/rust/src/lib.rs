//! Phynexus: A high-performance tensor library for embedded devices

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

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
pub mod quantization;
pub mod python;

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

#[pymodule]
fn _phynexus(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", "1.0.2")?;
    
    m.add_function(wrap_pyfunction!(py_init, m)?)?;
    
    
    Ok(())
}

#[pyfunction]
fn py_init() -> PyResult<()> {
    init().map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
}
