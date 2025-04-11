
mod agent;
mod environment;
mod communication;
mod coordination;
mod learning;

pub use agent::*;
pub use environment::*;
pub use communication::*;
pub use coordination::*;
pub use learning::*;

use pyo3::prelude::*;

pub fn register_mas(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "mas")?;
    
    agent::register_agent(py, submodule)?;
    environment::register_environment(py, submodule)?;
    communication::register_communication(py, submodule)?;
    coordination::register_coordination(py, submodule)?;
    learning::register_learning(py, submodule)?;
    
    m.add_submodule(submodule)?;
    
    Ok(())
}
