
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyValueError;

use crate::error::PhynexusError;
use crate::tensor::Tensor;
use crate::nn::Module;
use crate::optimizer::Optimizer;

#[pyfunction]
pub fn update_synaptic_importance(
    model: &PyAny,
    loss: &PyAny,
    omega: &PyDict,
    delta_theta: &PyDict,
) -> PyResult<()> {
    let py = model.py();
    
    loss.call_method1("backward", (py.eval("True", None, None)?,))?;
    
    for param_tuple in model.call_method0("named_parameters")?.iter()? {
        let (name, param) = param_tuple?.extract::<(String, PyObject)>()?;
        
        if !omega.contains(name)? || !delta_theta.contains(name)? {
            continue;
        }
        
        let grad = param.as_ref(py).getattr("grad")?;
        
        if grad.is_none() {
            continue;
        }
        
        let delta = delta_theta.get_item(name)?;
        
        let current_omega = omega.get_item(name)?;
        let grad_detached = grad.call_method0("detach")?.call_method0("clone")?;
        let update = grad_detached.call_method1("__mul__", (delta,))?;
        let update_neg = update.call_method1("__neg__", ())?;
        let new_omega = current_omega.call_method1("__add__", (update_neg,))?;
        
        omega.set_item(name, new_omega)?;
    }
    
    Ok(())
}

#[pyfunction]
pub fn compute_synaptic_penalty(
    model: &PyAny,
    importance: &PyDict,
    params_old: &PyDict,
    lambda_reg: f64,
) -> PyResult<PyObject> {
    let py = model.py();
    
    let penalty = py.import("neurenix")?.call_method1("tensor", (0.0,))?;
    
    let named_parameters = model.call_method0("named_parameters")?;
    
    for param_tuple in named_parameters.iter()? {
        let (name, param) = param_tuple?.extract::<(String, PyObject)>()?;
        
        if !importance.contains(name)? || !params_old.contains(name)? {
            continue;
        }
        
        let imp = importance.get_item(name)?;
        let old_param = params_old.get_item(name)?;
        
        let delta = param.as_ref(py).call_method1("__sub__", (old_param,))?;
        let delta_squared = delta.call_method1("pow", (2,))?;
        
        let weighted_delta = imp.call_method1("__mul__", (delta_squared,))?;
        
        let sum = weighted_delta.call_method0("sum")?;
        let new_penalty = penalty.call_method1("__add__", (sum,))?;
        
        std::mem::drop(penalty);
        penalty = new_penalty;
    }
    
    let scaled_penalty = penalty.call_method1("__mul__", (lambda_reg * 0.5,))?;
    
    Ok(scaled_penalty.to_object(py))
}
