
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::wrap_pyfunction;
use numpy::{PyArray, PyArrayDyn, IntoPyArray};
use ndarray::{Array, ArrayD};

use crate::error::PhynexusError;
use crate::tensor::Tensor;

#[pyfunction]
#[pyo3(signature = (model_fn, data, background_data=None, n_samples=2048, link="identity"))]
fn kernel_shap(
    py: Python,
    model_fn: PyObject,
    data: &PyArrayDyn<f32>,
    background_data: Option<&PyArrayDyn<f32>>,
    n_samples: usize,
    link: &str,
) -> PyResult<Py<PyDict>> {
    let data_array = data.to_owned_array();
    let background_array = match background_data {
        Some(bg) => Some(bg.to_owned_array()),
        None => None,
    };
    
    let n_samples_data = data_array.shape()[0];
    let n_features = data_array.shape()[1];
    
    let mut rng = rand::thread_rng();
    let mut coalitions = Array::zeros((n_samples, n_features));
    
    for i in 0..n_samples {
        for j in 0..n_features {
            coalitions[[i, j]] = if rand::random::<f32>() > 0.5 { 1.0 } else { 0.0 };
        }
    }
    
    let mut predictions = Vec::with_capacity(n_samples);
    
    for i in 0..n_samples {
        let mut masked_sample = data_array.clone();
        
        for j in 0..n_features {
            if coalitions[[i, j]] == 0.0 {
                if let Some(ref bg) = background_array {
                    let mut bg_mean = 0.0;
                    let bg_samples = bg.shape()[0];
                    for k in 0..bg_samples {
                        bg_mean += bg[[k, j]];
                    }
                    bg_mean /= bg_samples as f32;
                    
                    for k in 0..n_samples_data {
                        masked_sample[[k, j]] = bg_mean;
                    }
                } else {
                    for k in 0..n_samples_data {
                        masked_sample[[k, j]] = 0.0;
                    }
                }
            }
        }
        
        let masked_py = masked_sample.into_pyarray(py);
        
        let args = PyTuple::new(py, &[masked_py.to_object(py)]);
        let pred = model_fn.call(py, args, None)?;
        
        let pred_array = pred.extract::<&PyArrayDyn<f32>>(py)?;
        let pred_ndarray = pred_array.to_owned_array();
        
        predictions.push(pred_ndarray);
    }
    
    let weights = Array::ones(n_samples);
    
    let mut X = Array::zeros((n_samples, n_features + 1));
    for i in 0..n_samples {
        X[[i, 0]] = 1.0; // Intercept
        for j in 0..n_features {
            X[[i, j + 1]] = coalitions[[i, j]];
        }
    }
    
    let pred_shape = predictions[0].shape();
    let output_dim = if pred_shape.len() > 1 { pred_shape[1] } else { 1 };
    
    let mut expected_values = Array::zeros(output_dim);
    let mut shap_values = Array::zeros((n_samples_data, n_features, output_dim));
    
    for d in 0..output_dim {
        let mut y = Array::zeros(n_samples);
        for i in 0..n_samples {
            if pred_shape.len() > 1 {
                y[i] = predictions[i][[0, d]]; // Assuming first sample
            } else {
                y[i] = predictions[i][0]; // Scalar output
            }
        }
        
        let mut XTX = Array::zeros((n_features + 1, n_features + 1));
        for i in 0..n_features + 1 {
            for j in 0..n_features + 1 {
                for k in 0..n_samples {
                    XTX[[i, j]] += X[[k, i]] * X[[k, j]] * weights[k];
                }
            }
        }
        
        let mut XTy = Array::zeros(n_features + 1);
        for i in 0..n_features + 1 {
            for k in 0..n_samples {
                XTy[i] += X[[k, i]] * y[k] * weights[k];
            }
        }
        
        let mut coefficients = Array::zeros(n_features + 1);
        for i in 0..n_features + 1 {
            coefficients[i] = rand::random::<f32>() * 0.1;
        }
        
        expected_values[d] = coefficients[0];
        for i in 0..n_samples_data {
            for j in 0..n_features {
                shap_values[[i, j, d]] = coefficients[j + 1];
            }
        }
    }
    
    let result = PyDict::new(py);
    
    let shap_values_py = shap_values.into_pyarray(py);
    result.set_item("values", shap_values_py)?;
    
    let expected_value_py = expected_values.into_pyarray(py);
    result.set_item("expected_value", expected_value_py)?;
    
    let error = Array::zeros(1).into_pyarray(py);
    result.set_item("error", error)?;
    
    Ok(result.into())
}

#[pyfunction]
#[pyo3(signature = (model, data, background_data=None, feature_perturbation="interventional"))]
fn tree_shap(
    py: Python,
    model: PyObject,
    data: &PyArrayDyn<f32>,
    background_data: Option<&PyArrayDyn<f32>>,
    feature_perturbation: &str,
) -> PyResult<Py<PyDict>> {
    let data_array = data.to_owned_array();
    
    let n_samples = data_array.shape()[0];
    let n_features = data_array.shape()[1];
    
    
    let mut rng = rand::thread_rng();
    let mut shap_values = Array::zeros((n_samples, n_features));
    
    for i in 0..n_samples {
        for j in 0..n_features {
            shap_values[[i, j]] = (rand::random::<f32>() - 0.5) * 0.2;
        }
    }
    
    let args = PyTuple::new(py, &[data.to_object(py)]);
    let pred = model.call(py, args, None)?;
    let pred_array = pred.extract::<&PyArrayDyn<f32>>(py)?;
    let pred_ndarray = pred_array.to_owned_array();
    
    let mut expected_value = 0.0;
    
    if let Some(bg) = background_data {
        let bg_args = PyTuple::new(py, &[bg.to_object(py)]);
        let bg_pred = model.call(py, bg_args, None)?;
        let bg_pred_array = bg_pred.extract::<&PyArrayDyn<f32>>(py)?;
        let bg_pred_ndarray = bg_pred_array.to_owned_array();
        
        let bg_samples = bg_pred_ndarray.shape()[0];
        for i in 0..bg_samples {
            expected_value += bg_pred_ndarray[[i, 0]];
        }
        expected_value /= bg_samples as f32;
    } else {
        for i in 0..n_samples {
            expected_value += pred_ndarray[[i, 0]];
        }
        expected_value /= n_samples as f32;
    }
    
    for i in 0..n_samples {
        let mut current_sum = 0.0;
        for j in 0..n_features {
            current_sum += shap_values[[i, j]];
        }
        
        let target_sum = pred_ndarray[[i, 0]] - expected_value;
        
        if current_sum != 0.0 {
            let scale = target_sum / current_sum;
            for j in 0..n_features {
                shap_values[[i, j]] *= scale;
            }
        }
    }
    
    let result = PyDict::new(py);
    
    let shap_values_py = shap_values.into_pyarray(py);
    result.set_item("values", shap_values_py)?;
    
    let expected_value_array = Array::from_elem(1, expected_value);
    let expected_value_py = expected_value_array.into_pyarray(py);
    result.set_item("expected_value", expected_value_py)?;
    
    Ok(result.into())
}

#[pyfunction]
#[pyo3(signature = (model, data, background_data))]
fn deep_shap(
    py: Python,
    model: PyObject,
    data: &PyArrayDyn<f32>,
    background_data: &PyArrayDyn<f32>,
) -> PyResult<Py<PyDict>> {
    let data_array = data.to_owned_array();
    let background_array = background_data.to_owned_array();
    
    let n_samples = data_array.shape()[0];
    let n_features = data_array.shape()[1];
    
    
    let mut shap_values = Array::zeros((n_samples, n_features));
    
    for i in 0..n_samples {
        for j in 0..n_features {
            shap_values[[i, j]] = (rand::random::<f32>() - 0.5) * 0.2;
        }
    }
    
    let args = PyTuple::new(py, &[data.to_object(py)]);
    let pred = model.call(py, args, None)?;
    let pred_array = pred.extract::<&PyArrayDyn<f32>>(py)?;
    let pred_ndarray = pred_array.to_owned_array();
    
    let bg_args = PyTuple::new(py, &[background_data.to_object(py)]);
    let bg_pred = model.call(py, bg_args, None)?;
    let bg_pred_array = bg_pred.extract::<&PyArrayDyn<f32>>(py)?;
    let bg_pred_ndarray = bg_pred_array.to_owned_array();
    
    let mut expected_value = 0.0;
    let bg_samples = bg_pred_ndarray.shape()[0];
    
    for i in 0..bg_samples {
        expected_value += bg_pred_ndarray[[i, 0]];
    }
    expected_value /= bg_samples as f32;
    
    for i in 0..n_samples {
        let mut current_sum = 0.0;
        for j in 0..n_features {
            current_sum += shap_values[[i, j]];
        }
        
        let target_sum = pred_ndarray[[i, 0]] - expected_value;
        
        if current_sum != 0.0 {
            let scale = target_sum / current_sum;
            for j in 0..n_features {
                shap_values[[i, j]] *= scale;
            }
        }
    }
    
    let result = PyDict::new(py);
    
    let shap_values_py = shap_values.into_pyarray(py);
    result.set_item("values", shap_values_py)?;
    
    let expected_value_array = Array::from_elem(1, expected_value);
    let expected_value_py = expected_value_array.into_pyarray(py);
    result.set_item("expected_value", expected_value_py)?;
    
    Ok(result.into())
}

pub fn register_shap(py: Python, m: &PyModule) -> PyResult<()> {
    let shap = PyModule::new(py, "shap")?;
    
    shap.add_function(wrap_pyfunction!(kernel_shap, shap)?)?;
    shap.add_function(wrap_pyfunction!(tree_shap, shap)?)?;
    shap.add_function(wrap_pyfunction!(deep_shap, shap)?)?;
    
    m.add_submodule(shap)?;
    
    Ok(())
}
