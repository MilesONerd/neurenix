
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyValueError;

use crate::error::PhynexusError;
use crate::tensor::Tensor;
use crate::nn::Module;
use crate::optimizer::Optimizer;

#[pyfunction]
pub fn compute_ewc_importance(
    model: &PyAny,
    dataloader: &PyAny,
    loss_fn: &PyAny,
    optimizer: &PyAny,
    num_samples: Option<usize>,
) -> PyResult<PyObject> {
    let py = model.py();
    
    let importance = PyDict::new(py);
    
    let named_parameters = model.call_method0("named_parameters")?;
    
    for param_tuple in named_parameters.iter()? {
        let (name, param) = param_tuple?.extract::<(String, PyObject)>()?;
        
        let zeros_like = py.import("neurenix")?.call_method1("zeros_like", (param,))?;
        importance.set_item(name, zeros_like)?;
    }
    
    model.call_method0("eval")?;
    
    let max_samples = match num_samples {
        Some(n) => n,
        None => usize::MAX,
    };
    
    let mut sample_count = 0;
    
    for batch in dataloader.iter()? {
        if sample_count >= max_samples {
            break;
        }
        
        let batch = batch?;
        let (inputs, _) = batch.extract::<(PyObject, PyObject)>()?;
        
        let inputs_ref = inputs.as_ref(py);
        let batch_size = inputs_ref.getattr("shape")?.get_item(0)?.extract::<usize>()?;
        sample_count += batch_size;
        
        model.call_method0("zero_grad")?;
        
        let outputs = model.call1((inputs,))?;
        
        let log_probs = py.import("neurenix")?.call_method1("log_softmax", (outputs, "dim", 1))?;
        
        let probs = py.import("neurenix")?.call_method1("exp", (log_probs,))?.call_method0("detach")?;
        let samples = py.import("neurenix")?
            .call_method1("multinomial", (probs, 1))?
            .call_method0("flatten")?;
        
        let arange = py.import("neurenix")?.call_method1("arange", (batch_size,))?;
        let selected_log_probs = log_probs.call_method1("__getitem__", ((arange, samples),))?;
        
        let loss = selected_log_probs.call_method0("mean")?.call_method1("__neg__", ())?;
        
        loss.call_method0("backward")?;
        
        for param_tuple in named_parameters.iter()? {
            let (name, param) = param_tuple?.extract::<(String, PyObject)>()?;
            
            let grad = param.as_ref(py).getattr("grad")?;
            
            if grad.is_none() {
                continue;
            }
            
            let grad_squared = grad.call_method1("pow", (2,))?;
            let weighted_grad = grad_squared.call_method1("__mul__", (batch_size,))?;
            
            let current_importance = importance.get_item(name)?;
            let updated_importance = current_importance.call_method1("__add__", (weighted_grad,))?;
            importance.set_item(name, updated_importance)?;
        }
    }
    
    for (name, imp) in importance.iter() {
        let normalized_imp = imp.call_method1("__truediv__", (sample_count.max(1),))?;
        importance.set_item(name, normalized_imp)?;
    }
    
    Ok(importance.to_object(py))
}

#[pyfunction]
pub fn compute_ewc_penalty(
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
