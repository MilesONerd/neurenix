
pub mod quantization;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub fn register_modules(py: Python, m: &PyModule) -> PyResult<()> {
    quantization::register_quantization(py, m)?;
    
    Ok(())
}
