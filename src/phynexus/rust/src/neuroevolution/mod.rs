
mod genetic;
mod neat;
mod hyperneat;
mod cmaes;
mod evolution_strategy;

pub use genetic::*;
pub use neat::*;
pub use hyperneat::*;
pub use cmaes::*;
pub use evolution_strategy::*;

use pyo3::prelude::*;

pub fn register_neuroevolution(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "neuroevolution")?;
    
    genetic::register_genetic(py, submodule)?;
    neat::register_neat(py, submodule)?;
    hyperneat::register_hyperneat(py, submodule)?;
    cmaes::register_cmaes(py, submodule)?;
    evolution_strategy::register_evolution_strategy(py, submodule)?;
    
    m.add_submodule(submodule)?;
    
    Ok(())
}
