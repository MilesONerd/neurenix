
pub mod mpi;
pub mod horovod;
pub mod deepspeed;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::error::PhynexusError;
use crate::tensor::Tensor;

pub fn register_distributed(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let distributed = PyModule::new(py, "distributed")?;
    
    mpi::register_mpi(py, distributed)?;
    horovod::register_horovod(py, distributed)?;
    deepspeed::register_deepspeed(py, distributed)?;
    
    m.add_submodule(&distributed)?;
    
    Ok(())
}
