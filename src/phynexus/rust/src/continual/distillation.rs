
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyValueError;

use crate::error::PhynexusError;
use crate::tensor::Tensor;
use crate::nn::Module;
use crate::optimizer::Optimizer;

#[pyfunction]
pub fn compute_distillation_loss(
    student_logits: &PyAny,
    teacher_logits: &PyAny,
    temperature: f64,
) -> PyResult<PyObject> {
    let py = student_logits.py();
    
    let temp_tensor = py.import("neurenix")?.call_method1("tensor", (temperature,))?;
    
    let teacher_logits_div = teacher_logits.call_method1("__truediv__", (temp_tensor,))?;
    let soft_targets = py.import("neurenix")?.call_method1("softmax", (teacher_logits_div, "dim", 1))?;
    
    let student_logits_div = student_logits.call_method1("__truediv__", (temp_tensor,))?;
    let soft_prob = py.import("neurenix")?.call_method1("log_softmax", (student_logits_div, "dim", 1))?;
    
    let mul = soft_targets.call_method1("__mul__", (soft_prob,))?;
    let sum = mul.call_method1("sum", ("dim", 1))?;
    let neg = sum.call_method1("__neg__", ())?;
    let loss = neg.call_method0("mean")?;
    
    let temp_squared = temperature * temperature;
    let scaled_loss = loss.call_method1("__mul__", (temp_squared,))?;
    
    Ok(scaled_loss.to_object(py))
}

#[pyfunction]
pub fn compute_combined_distillation_loss(
    student_logits: &PyAny,
    teacher_logits: &PyAny,
    targets: &PyAny,
    task_loss_fn: &PyAny,
    temperature: f64,
    alpha: f64,
) -> PyResult<PyObject> {
    let py = student_logits.py();
    
    let task_loss = task_loss_fn.call1((student_logits, targets))?;
    
    let dist_loss = compute_distillation_loss(student_logits, teacher_logits, temperature)?;
    
    let alpha_tensor = py.import("neurenix")?.call_method1("tensor", (alpha,))?;
    let one_minus_alpha = py.import("neurenix")?.call_method1("tensor", (1.0 - alpha,))?;
    
    let weighted_task_loss = task_loss.call_method1("__mul__", (alpha_tensor,))?;
    let weighted_dist_loss = dist_loss.call_method1("__mul__", (one_minus_alpha,))?;
    
    let combined_loss = weighted_task_loss.call_method1("__add__", (weighted_dist_loss,))?;
    
    Ok(combined_loss.to_object(py))
}
