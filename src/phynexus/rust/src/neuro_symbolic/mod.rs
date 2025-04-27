
mod symbolic;
mod neural_symbolic;
mod differentiable_logic;
mod knowledge_distillation;
mod reasoning;

pub use symbolic::*;
pub use neural_symbolic::*;
pub use differentiable_logic::*;
pub use knowledge_distillation::*;
pub use reasoning::*;

use pyo3::prelude::*;

pub fn register_neuro_symbolic(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py, "neuro_symbolic")?;
    
    symbolic::register_symbolic(py, submodule)?;
    neural_symbolic::register_neural_symbolic(py, submodule)?;
    differentiable_logic::register_differentiable_logic(py, submodule)?;
    knowledge_distillation::register_knowledge_distillation(py, submodule)?;
    reasoning::register_reasoning(py, submodule)?;
    
    m.add_submodule(&submodule)?;
    
    Ok(())
}
