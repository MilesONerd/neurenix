
pub mod quantization;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub fn register_modules(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    quantization::register_quantization(py, m)?;
    
    Ok(())
}
