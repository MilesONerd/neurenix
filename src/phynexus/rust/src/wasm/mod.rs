
mod simd;
mod multithreaded;

pub use simd::*;
pub use multithreaded::*;

use pyo3::prelude::*;

pub fn register_wasm(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py, "wasm")?;
    
    simd::register_simd(py, submodule)?;
    multithreaded::register_multithreaded(py, submodule)?;
    
    m.add_submodule(&submodule)?;
    
    Ok(())
}
