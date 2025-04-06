
pub mod shap;
pub mod lime;
pub mod feature_importance;
pub mod partial_dependence;
pub mod counterfactual;
pub mod activation;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use crate::error::PhynexusError;

pub fn register_explainable(py: Python, m: &PyModule) -> PyResult<()> {
    let explainable = PyModule::new(py, "explainable")?;
    
    shap::register_shap(py, explainable)?;
    lime::register_lime(py, explainable)?;
    feature_importance::register_feature_importance(py, explainable)?;
    partial_dependence::register_partial_dependence(py, explainable)?;
    counterfactual::register_counterfactual(py, explainable)?;
    activation::register_activation(py, explainable)?;
    
    m.add_submodule(explainable)?;
    
    Ok(())
}
