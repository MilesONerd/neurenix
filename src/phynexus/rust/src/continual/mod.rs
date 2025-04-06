
mod ewc;
mod replay;
mod regularization;
mod distillation;
mod synaptic;

pub use ewc::*;
pub use replay::*;
pub use regularization::*;
pub use distillation::*;
pub use synaptic::*;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub fn register_continual(py: Python, m: &PyModule) -> PyResult<()> {
    let continual = PyModule::new(py, "continual")?;
    
    continual.add_function(wrap_pyfunction!(compute_ewc_importance, continual)?)?;
    continual.add_function(wrap_pyfunction!(compute_ewc_penalty, continual)?)?;
    
    continual.add_function(wrap_pyfunction!(update_replay_memory, continual)?)?;
    continual.add_function(wrap_pyfunction!(sample_replay_memory, continual)?)?;
    
    continual.add_function(wrap_pyfunction!(compute_l2_penalty, continual)?)?;
    continual.add_function(wrap_pyfunction!(compute_weight_importance, continual)?)?;
    
    continual.add_function(wrap_pyfunction!(compute_distillation_loss, continual)?)?;
    continual.add_function(wrap_pyfunction!(compute_combined_distillation_loss, continual)?)?;
    
    continual.add_function(wrap_pyfunction!(update_synaptic_importance, continual)?)?;
    continual.add_function(wrap_pyfunction!(compute_synaptic_penalty, continual)?)?;
    
    m.add_submodule(continual)?;
    
    Ok(())
}
