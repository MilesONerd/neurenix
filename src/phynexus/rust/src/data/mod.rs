
pub mod dataset_hub;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::error::PhynexusError;

pub fn register_data(py: Python, m: &PyModule) -> PyResult<()> {
    let data = PyModule::new(py, "data")?;
    
    dataset_hub::register_dataset_hub(py, data)?;
    
    m.add_submodule(data)?;
    
    Ok(())
}
