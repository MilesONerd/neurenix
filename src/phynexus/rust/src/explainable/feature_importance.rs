
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::wrap_pyfunction;
use numpy::{PyArray, PyArrayDyn, IntoPyArray};
use ndarray::{Array, ArrayD};

use crate::error::PhynexusError;
use crate::tensor::Tensor;

#[pyfunction]
#[pyo3(signature = (model_fn, data, target=None, n_repeats=5, random_state=0))]
fn permutation_importance(
    py: Python,
    model_fn: PyObject,
    data: &PyArrayDyn<f32>,
    target: Option<&PyArrayDyn<f32>>,
    n_repeats: usize,
    random_state: u64,
) -> PyResult<Py<PyDict>> {
    let data_array = data.to_owned_array();
    
    let n_samples = data_array.shape()[0];
    let n_features = data_array.shape()[1];
    
    let mut rng = rand::thread_rng();
    
    let args = PyTuple::new(py, &[data.to_object(py)]);
    let baseline_pred = model_fn.call(py, args, None)?;
    let baseline_pred_array = baseline_pred.extract::<&PyArrayDyn<f32>>(py)?;
    let baseline_pred_ndarray = baseline_pred_array.to_owned_array();
    
    let mut baseline_score = 0.0;
    
    if let Some(target_data) = target {
        let target_array = target_data.to_owned_array();
        
        if baseline_pred_ndarray.shape().len() > 1 {
            let mut correct = 0;
            for i in 0..n_samples {
                let mut max_idx = 0;
                let mut max_val = baseline_pred_ndarray[[i, 0]];
                
                for j in 1..baseline_pred_ndarray.shape()[1] {
                    if baseline_pred_ndarray[[i, j]] > max_val {
                        max_val = baseline_pred_ndarray[[i, j]];
                        max_idx = j;
                    }
                }
                
                if (max_idx as f32 - target_array[i]).abs() < 1e-6 {
                    correct += 1;
                }
            }
            
            baseline_score = correct as f32 / n_samples as f32;
        } else {
            let mut mse = 0.0;
            for i in 0..n_samples {
                let diff = baseline_pred_ndarray[i] - target_array[i];
                mse += diff * diff;
            }
            
            baseline_score = -mse / n_samples as f32; // Negative MSE so higher is better
        }
    } else {
        for i in 0..n_samples {
            if baseline_pred_ndarray.shape().len() > 1 {
                for j in 0..baseline_pred_ndarray.shape()[1] {
                    baseline_score += baseline_pred_ndarray[[i, j]];
                }
            } else {
                baseline_score += baseline_pred_ndarray[i];
            }
        }
        
        baseline_score /= n_samples as f32;
    }
    
    let mut importances = Array::zeros((n_repeats, n_features));
    
    for i in 0..n_features {
        for j in 0..n_repeats {
            let mut permuted_data = data_array.clone();
            
            let mut perm_idx = Vec::new();
            for k in 0..n_samples {
                perm_idx.push(k);
            }
            
            for k in 0..n_samples {
                let idx = (rand::random::<f32>() * (n_samples - k) as f32) as usize;
                perm_idx.swap(k, k + idx);
            }
            
            for k in 0..n_samples {
                permuted_data[[k, i]] = data_array[[perm_idx[k], i]];
            }
            
            let permuted_data_py = permuted_data.into_pyarray(py);
            let args = PyTuple::new(py, &[permuted_data_py.to_object(py)]);
            let permuted_pred = model_fn.call(py, args, None)?;
            let permuted_pred_array = permuted_pred.extract::<&PyArrayDyn<f32>>(py)?;
            let permuted_pred_ndarray = permuted_pred_array.to_owned_array();
            
            let mut permuted_score = 0.0;
            
            if let Some(target_data) = target {
                let target_array = target_data.to_owned_array();
                
                if permuted_pred_ndarray.shape().len() > 1 {
                    let mut correct = 0;
                    for k in 0..n_samples {
                        let mut max_idx = 0;
                        let mut max_val = permuted_pred_ndarray[[k, 0]];
                        
                        for l in 1..permuted_pred_ndarray.shape()[1] {
                            if permuted_pred_ndarray[[k, l]] > max_val {
                                max_val = permuted_pred_ndarray[[k, l]];
                                max_idx = l;
                            }
                        }
                        
                        if (max_idx as f32 - target_array[k]).abs() < 1e-6 {
                            correct += 1;
                        }
                    }
                    
                    permuted_score = correct as f32 / n_samples as f32;
                } else {
                    let mut mse = 0.0;
                    for k in 0..n_samples {
                        let diff = permuted_pred_ndarray[k] - target_array[k];
                        mse += diff * diff;
                    }
                    
                    permuted_score = -mse / n_samples as f32; // Negative MSE so higher is better
                }
            } else {
                for k in 0..n_samples {
                    if permuted_pred_ndarray.shape().len() > 1 {
                        for l in 0..permuted_pred_ndarray.shape()[1] {
                            permuted_score += permuted_pred_ndarray[[k, l]];
                        }
                    } else {
                        permuted_score += permuted_pred_ndarray[k];
                    }
                }
                
                permuted_score /= n_samples as f32;
            }
            
            importances[[j, i]] = baseline_score - permuted_score;
        }
    }
    
    let mut importances_mean = Array::zeros(n_features);
    let mut importances_std = Array::zeros(n_features);
    
    for i in 0..n_features {
        let mut mean = 0.0;
        for j in 0..n_repeats {
            mean += importances[[j, i]];
        }
        mean /= n_repeats as f32;
        
        let mut std_dev = 0.0;
        for j in 0..n_repeats {
            let diff = importances[[j, i]] - mean;
            std_dev += diff * diff;
        }
        std_dev = (std_dev / n_repeats as f32).sqrt();
        
        importances_mean[i] = mean;
        importances_std[i] = std_dev;
    }
    
    let result = PyDict::new(py);
    
    let importances_mean_py = importances_mean.into_pyarray(py);
    result.set_item("importances_mean", importances_mean_py)?;
    
    let importances_std_py = importances_std.into_pyarray(py);
    result.set_item("importances_std", importances_std_py)?;
    
    let importances_py = importances.into_pyarray(py);
    result.set_item("importances", importances_py)?;
    
    Ok(result.into())
}

pub fn register_feature_importance(py: Python, m: &PyModule) -> PyResult<()> {
    let feature_importance = PyModule::new(py, "feature_importance")?;
    
    feature_importance.add_function(wrap_pyfunction!(permutation_importance, feature_importance)?)?;
    
    m.add_submodule(feature_importance)?;
    
    Ok(())
}
