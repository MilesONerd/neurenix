
use crate::error::PhynexusError;
use crate::tensor::Tensor;
use pyo3::prelude::*;

mod model;
mod embedding;
mod classifier;
mod utils;

pub use model::*;
pub use embedding::*;
pub use classifier::*;
pub use utils::*;

pub fn register_zeroshot(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "zeroshot")?;
    
    model::register_models(py, submodule)?;
    embedding::register_embeddings(py, submodule)?;
    classifier::register_classifiers(py, submodule)?;
    utils::register_utils(py, submodule)?;
    
    m.add_submodule(submodule)?;
    
    Ok(())
}
