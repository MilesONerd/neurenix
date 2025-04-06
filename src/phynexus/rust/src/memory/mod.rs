
pub mod unified;
pub mod hmm;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::error::PhynexusError;
use crate::tensor::Tensor;

pub fn register_memory(py: Python, m: &PyModule) -> PyResult<()> {
    let memory = PyModule::new(py, "memory")?;
    
    unified::register_unified(py, memory)?;
    hmm::register_hmm(py, memory)?;
    
    m.add_submodule(memory)?;
    
    Ok(())
}
