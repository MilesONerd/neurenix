
pub mod search;
pub mod nas;
pub mod model_selection;
pub mod pipeline;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::error::PhynexusError;
use crate::tensor::Tensor;

pub fn register_automl(py: Python, m: &PyModule) -> PyResult<()> {
    let automl = PyModule::new(py, "automl")?;
    
    search::register_search(py, automl)?;
    nas::register_nas(py, automl)?;
    model_selection::register_model_selection(py, automl)?;
    pipeline::register_pipeline(py, automl)?;
    
    m.add_submodule(automl)?;
    
    Ok(())
}
