
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::wrap_pyfunction;
use numpy::{PyArray, PyArrayDyn, IntoPyArray};
use ndarray::{Array, ArrayD};
use std::collections::HashMap;

use crate::error::PhynexusError;
use crate::tensor::Tensor;

#[pyfunction]
#[pyo3(signature = (model_fn, sample, target_class=None, target_pred=None, categorical_features=None, feature_ranges=None, max_iter=1000, random_state=0))]
fn counterfactual(
    py: Python,
    model_fn: PyObject,
    sample: &PyArrayDyn<f32>,
    target_class: Option<usize>,
    target_pred: Option<f32>,
    categorical_features: Option<Vec<usize>>,
    feature_ranges: Option<HashMap<usize, (f32, f32)>>,
    max_iter: usize,
    random_state: u64,
) -> PyResult<Py<PyDict>> {
    let sample_array = sample.to_owned_array();
    
    let mut rng = rand::thread_rng();
    
    let args = PyTuple::new(py, &[sample.to_object(py)]);
    let original_pred = model_fn.call(py, args, None)?;
    let original_pred_array = original_pred.extract::<&PyArrayDyn<f32>>(py)?;
    let original_pred_ndarray = original_pred_array.to_owned_array();
    
    let mut target = Array::zeros(original_pred_ndarray.shape());
    
    if let Some(class) = target_class {
        if original_pred_ndarray.shape().len() > 1 && original_pred_ndarray.shape()[0] > 1 {
            target[[class]] = 1.0;
        } else {
            target[0] = if class == 1 { 1.0 } else { 0.0 };
        }
    } else if let Some(pred) = target_pred {
        target[0] = pred;
    } else {
        if original_pred_ndarray.shape().len() > 1 && original_pred_ndarray.shape()[0] > 1 {
            let mut max_idx = 0;
            let mut max_val = original_pred_ndarray[[0]];
            
            for i in 1..original_pred_ndarray.shape()[0] {
                if original_pred_ndarray[[i]] > max_val {
                    max_val = original_pred_ndarray[[i]];
                    max_idx = i;
                }
            }
            
            let new_class = (max_idx + 1) % original_pred_ndarray.shape()[0];
            target[[new_class]] = 1.0;
        } else {
            target[0] = if original_pred_ndarray[0] < 0.5 { 1.0 } else { 0.0 };
        }
    }
    
    let mut counterfactual = sample_array.clone();
    
    let cat_features = categorical_features.unwrap_or_else(Vec::new);
    
    let feat_ranges = feature_ranges.unwrap_or_else(HashMap::new);
    
    let mut success = false;
    let mut changes = HashMap::new();
    
    for _ in 0..max_iter {
        let feature_idx = (rand::random::<f32>() * sample_array.len() as f32) as usize;
        
        if cat_features.contains(&feature_idx) {
            continue;
        }
        
        let (min_val, max_val) = feat_ranges.get(&feature_idx)
            .cloned()
            .unwrap_or((0.0, 1.0));
        
        if rand::random::<bool>() {
            counterfactual[feature_idx] = (counterfactual[feature_idx] + 0.1).min(max_val);
        } else {
            counterfactual[feature_idx] = (counterfactual[feature_idx] - 0.1).max(min_val);
        }
        
        let counterfactual_py = counterfactual.clone().into_pyarray(py);
        let args = PyTuple::new(py, &[counterfactual_py.to_object(py)]);
        let counterfactual_pred = model_fn.call(py, args, None)?;
        let counterfactual_pred_array = counterfactual_pred.extract::<&PyArrayDyn<f32>>(py)?;
        let counterfactual_pred_ndarray = counterfactual_pred_array.to_owned_array();
        
        if let Some(class) = target_class {
            if counterfactual_pred_ndarray.shape().len() > 1 && counterfactual_pred_ndarray.shape()[0] > 1 {
                let mut max_idx = 0;
                let mut max_val = counterfactual_pred_ndarray[[0]];
                
                for i in 1..counterfactual_pred_ndarray.shape()[0] {
                    if counterfactual_pred_ndarray[[i]] > max_val {
                        max_val = counterfactual_pred_ndarray[[i]];
                        max_idx = i;
                    }
                }
                
                if max_idx == class {
                    success = true;
                    break;
                }
            } else {
                let pred_class = if counterfactual_pred_ndarray[0] > 0.5 { 1 } else { 0 };
                if pred_class == class {
                    success = true;
                    break;
                }
            }
        } else if let Some(pred) = target_pred {
            if (counterfactual_pred_ndarray[0] - pred).abs() < 0.1 {
                success = true;
                break;
            }
        } else {
            if counterfactual_pred_ndarray.shape().len() > 1 && counterfactual_pred_ndarray.shape()[0] > 1 {
                let mut orig_max_idx = 0;
                let mut orig_max_val = original_pred_ndarray[[0]];
                
                for i in 1..original_pred_ndarray.shape()[0] {
                    if original_pred_ndarray[[i]] > orig_max_val {
                        orig_max_val = original_pred_ndarray[[i]];
                        orig_max_idx = i;
                    }
                }
                
                let mut cf_max_idx = 0;
                let mut cf_max_val = counterfactual_pred_ndarray[[0]];
                
                for i in 1..counterfactual_pred_ndarray.shape()[0] {
                    if counterfactual_pred_ndarray[[i]] > cf_max_val {
                        cf_max_val = counterfactual_pred_ndarray[[i]];
                        cf_max_idx = i;
                    }
                }
                
                if cf_max_idx != orig_max_idx {
                    success = true;
                    break;
                }
            } else {
                let orig_class = if original_pred_ndarray[0] > 0.5 { 1 } else { 0 };
                let cf_class = if counterfactual_pred_ndarray[0] > 0.5 { 1 } else { 0 };
                
                if cf_class != orig_class {
                    success = true;
                    break;
                }
            }
        }
    }
    
    for i in 0..sample_array.len() {
        if (counterfactual[i] - sample_array[i]).abs() > 1e-6 {
            changes.insert(i, (sample_array[i], counterfactual[i]));
        }
    }
    
    let result = PyDict::new(py);
    
    let counterfactual_py = counterfactual.into_pyarray(py);
    result.set_item("counterfactual", counterfactual_py)?;
    
    let original_pred_py = original_pred_ndarray.into_pyarray(py);
    result.set_item("original_prediction", original_pred_py)?;
    
    let counterfactual_array = result.get_item("counterfactual")?.extract::<&PyArrayDyn<f32>>()?;
    let args = PyTuple::new(py, &[counterfactual_array.to_object(py)]);
    let final_pred = model_fn.call(py, args, None)?;
    
    result.set_item("counterfactual_prediction", final_pred)?;
    
    let changes_py = PyDict::new(py);
    for (idx, (orig, cf)) in changes {
        let change_tuple = PyTuple::new(py, &[orig, cf]);
        changes_py.set_item(idx.to_string(), change_tuple)?;
    }
    
    result.set_item("changes", changes_py)?;
    
    result.set_item("success", success)?;
    
    Ok(result.into())
}

pub fn register_counterfactual(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let counterfactual_module = PyModule::new(py, "counterfactual")?;
    
    counterfactual_module.add_function(wrap_pyfunction!(counterfactual, counterfactual_module)?)?;
    
    m.add_submodule(&counterfactual_module)?;
    
    Ok(())
}
