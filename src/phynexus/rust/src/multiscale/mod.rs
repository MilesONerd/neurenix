
use crate::error::PhynexusError;
use crate::tensor::Tensor;
use pyo3::prelude::*;

mod models;
mod pooling;
mod fusion;
mod transforms;

pub use models::*;
pub use pooling::*;
pub use fusion::*;
pub use transforms::*;

pub fn register_multiscale(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py, "multiscale")?;
    
    models::register_models(py, submodule)?;
    
    pooling::register_pooling(py, submodule)?;
    
    fusion::register_fusion(py, submodule)?;
    
    transforms::register_transforms(py, submodule)?;
    
    m.add_submodule(&submodule)?;
    
    Ok(())
}
