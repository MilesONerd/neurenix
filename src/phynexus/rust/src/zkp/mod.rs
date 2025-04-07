
mod proof_system;
mod zk_snark;
mod zk_stark;
mod bulletproofs;
mod sigma;
mod utils;

pub use proof_system::*;
pub use zk_snark::*;
pub use zk_stark::*;
pub use bulletproofs::*;
pub use sigma::*;
pub use utils::*;

use pyo3::prelude::*;

pub fn register_zkp(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "zkp")?;
    
    proof_system::register_proof_system(py, submodule)?;
    zk_snark::register_zk_snark(py, submodule)?;
    zk_stark::register_zk_stark(py, submodule)?;
    bulletproofs::register_bulletproofs(py, submodule)?;
    sigma::register_sigma(py, submodule)?;
    utils::register_utils(py, submodule)?;
    
    m.add_submodule(submodule)?;
    
    Ok(())
}
