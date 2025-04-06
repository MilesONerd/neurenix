
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyValueError;

use crate::error::PhynexusError;
use crate::tensor::Tensor;
use crate::nn::Module;
use crate::optimizer::Optimizer;

#[pyfunction]
pub fn compute_l2_penalty(
    model: &PyAny,
    params_old: &PyDict,
    lambda_reg: f64,
) -> PyResult<PyObject> {
    let py = model.py();
    
    let penalty = py.import("neurenix")?.call_method1("tensor", (0.0,))?;
    
    let named_parameters = model.call_method0("named_parameters")?;
    
    for param_tuple in named_parameters.iter()? {
        let (name, param) = param_tuple?.extract::<(String, PyObject)>()?;
        
        if !params_old.contains(name)? {
            continue;
        }
        
        let old_param = params_old.get_item(name)?;
        
        let delta = param.as_ref(py).call_method1("__sub__", (old_param,))?;
        let delta_squared = delta.call_method1("pow", (2,))?;
        
        let sum = delta_squared.call_method0("sum")?;
        let new_penalty = penalty.call_method1("__add__", (sum,))?;
        
        std::mem::drop(penalty);
        penalty = new_penalty;
    }
    
    let scaled_penalty = penalty.call_method1("__mul__", (lambda_reg * 0.5,))?;
    
    Ok(scaled_penalty.to_object(py))
}

#[pyfunction]
pub fn compute_weight_importance(
    model: &PyAny,
    importance_method: &str,
    importance_threshold: f64,
    dataloader: Option<&PyAny>,
    loss_fn: Option<&PyAny>,
) -> PyResult<PyObject> {
    let py = model.py();
    
    let mask = PyDict::new(py);
    
    match importance_method {
        "magnitude" => {
            let named_parameters = model.call_method0("named_parameters")?;
            
            for param_tuple in named_parameters.iter()? {
                let (name, param) = param_tuple?.extract::<(String, PyObject)>()?;
                
                let importance = param.as_ref(py).call_method0("abs")?;
                
                let threshold_tensor = py.import("neurenix")?.call_method1("tensor", (importance_threshold,))?;
                let mask_tensor = importance.call_method1("__gt__", (threshold_tensor,))?
                    .call_method1("float", ())?;
                
                mask.set_item(name, mask_tensor)?;
            }
        },
        "gradient" | "fisher" => {
            if dataloader.is_none() || loss_fn.is_none() {
                return Err(PyValueError::new_err(
                    format!("Dataloader and loss_fn are required for {} importance", importance_method)
                ));
            }
            
            let dataloader = dataloader.unwrap();
            let loss_fn = loss_fn.unwrap();
            
            let importance = PyDict::new(py);
            
            let named_parameters = model.call_method0("named_parameters")?;
            
            for param_tuple in named_parameters.iter()? {
                let (name, param) = param_tuple?.extract::<(String, PyObject)>()?;
                
                let zeros_like = py.import("neurenix")?.call_method1("zeros_like", (param,))?;
                importance.set_item(name, zeros_like)?;
            }
            
            model.call_method0("eval")?;
            
            for batch in dataloader.iter()? {
                let batch = batch?;
                let (inputs, targets) = batch.extract::<(PyObject, PyObject)>()?;
                
                model.call_method0("zero_grad")?;
                
                let outputs = model.call1((inputs,))?;
                
                let loss = loss_fn.call1((outputs, targets))?;
                
                loss.call_method0("backward")?;
                
                for param_tuple in named_parameters.iter()? {
                    let (name, param) = param_tuple?.extract::<(String, PyObject)>()?;
                    
                    let grad = param.as_ref(py).getattr("grad")?;
                    
                    if grad.is_none() {
                        continue;
                    }
                    
                    let grad_value = if importance_method == "gradient" {
                        grad.call_method0("abs")?
                    } else {  // fisher
                        grad.call_method1("pow", (2,))?
                    };
                    
                    let current_importance = importance.get_item(name)?;
                    let updated_importance = current_importance.call_method1("__add__", (grad_value,))?;
                    importance.set_item(name, updated_importance)?;
                }
            }
            
            for (name, imp) in importance.iter() {
                let threshold_tensor = py.import("neurenix")?.call_method1("tensor", (importance_threshold,))?;
                let mask_tensor = imp.call_method1("__gt__", (threshold_tensor,))?
                    .call_method1("float", ())?;
                
                mask.set_item(name, mask_tensor)?;
            }
        },
        _ => {
            return Err(PyValueError::new_err(
                format!("Unknown importance method: {}", importance_method)
            ));
        }
    }
    
    Ok(mask.to_object(py))
}
