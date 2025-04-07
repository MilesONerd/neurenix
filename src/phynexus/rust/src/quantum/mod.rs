
mod qiskit;
mod cirq;
mod hybrid;
mod algorithms;
mod utils;

pub use qiskit::*;
pub use cirq::*;
pub use hybrid::*;
pub use algorithms::*;
pub use utils::*;

use pyo3::prelude::*;

pub fn register_quantum(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "quantum")?;
    
    qiskit::register_qiskit(py, submodule)?;
    cirq::register_cirq(py, submodule)?;
    hybrid::register_hybrid(py, submodule)?;
    algorithms::register_algorithms(py, submodule)?;
    utils::register_utils(py, submodule)?;
    
    m.add_submodule(submodule)?;
    
    Ok(())
}
