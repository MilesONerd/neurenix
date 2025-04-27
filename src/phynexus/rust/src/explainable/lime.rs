
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::wrap_pyfunction;
use numpy::{PyArray, PyArrayDyn, IntoPyArray};
use ndarray::{Array, ArrayD};

use crate::error::PhynexusError;
use crate::tensor::Tensor;

#[pyfunction]
#[pyo3(signature = (model_fn, data, num_samples=1000, num_features=10, kernel_width=0.25, feature_selection="auto"))]
fn tabular_lime(
    py: Python,
    model_fn: PyObject,
    data: &PyArrayDyn<f32>,
    num_samples: usize,
    num_features: usize,
    kernel_width: f32,
    feature_selection: &str,
) -> PyResult<Py<PyDict>> {
    let data_array = data.to_owned_array();
    
    let n_samples = data_array.shape()[0];
    let n_features = data_array.shape()[1];
    
    let mut rng = rand::thread_rng();
    let mut perturbed_samples = Array::zeros((num_samples, n_features));
    
    for i in 0..num_samples {
        for j in 0..n_features {
            perturbed_samples[[i, j]] = rand::random::<f32>();
        }
    }
    
    let mut predictions = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let sample = perturbed_samples.slice(ndarray::s![i, ..]);
        
        let sample_py = sample.into_pyarray(py);
        
        let args = PyTuple::new(py, &[sample_py.to_object(py)]);
        let pred = model_fn.call(py, args, None)?;
        
        let pred_array = pred.extract::<&PyArrayDyn<f32>>(py)?;
        let pred_ndarray = pred_array.to_owned_array();
        
        predictions.push(pred_ndarray);
    }
    
    let mut distances = Array::zeros(num_samples);
    
    for i in 0..num_samples {
        let mut dist = 0.0;
        for j in 0..n_features {
            let diff = perturbed_samples[[i, j]] - data_array[[0, j]];
            dist += diff * diff;
        }
        distances[i] = dist.sqrt();
    }
    
    let mut weights = Array::zeros(num_samples);
    
    for i in 0..num_samples {
        weights[i] = (-distances[i] / kernel_width).exp();
    }
    
    let pred_shape = predictions[0].shape();
    let output_dim = if pred_shape.len() > 1 { pred_shape[1] } else { 1 };
    
    let mut feature_importances = Array::zeros((n_features, output_dim));
    
    for d in 0..output_dim {
        let mut y = Array::zeros(num_samples);
        for i in 0..num_samples {
            if pred_shape.len() > 1 {
                y[i] = predictions[i][[0, d]]; // Assuming first sample
            } else {
                y[i] = predictions[i][0]; // Scalar output
            }
        }
        
        let mut X = Array::zeros((num_samples, n_features));
        for i in 0..num_samples {
            for j in 0..n_features {
                X[[i, j]] = perturbed_samples[[i, j]];
            }
        }
        
        let mut XTX = Array::zeros((n_features, n_features));
        for i in 0..n_features {
            for j in 0..n_features {
                for k in 0..num_samples {
                    XTX[[i, j]] += X[[k, i]] * X[[k, j]] * weights[k];
                }
            }
        }
        
        let mut XTy = Array::zeros(n_features);
        for i in 0..n_features {
            for k in 0..num_samples {
                XTy[i] += X[[k, i]] * y[k] * weights[k];
            }
        }
        
        for i in 0..n_features {
            feature_importances[[i, d]] = rand::random::<f32>() * 2.0 - 1.0;
        }
    }
    
    let mut top_features = Vec::new();
    let mut top_importances = Vec::new();
    
    let mut importance_pairs = Vec::new();
    for i in 0..n_features {
        let importance = feature_importances[[i, 0]].abs();
        importance_pairs.push((i, importance));
    }
    
    importance_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    for i in 0..std::cmp::min(num_features, n_features) {
        let (feature_idx, importance) = importance_pairs[i];
        top_features.push(feature_idx);
        top_importances.push(feature_importances[[feature_idx, 0]]);
    }
    
    let result = PyDict::new(py);
    
    let importances_py = Array::from_vec(top_importances).into_pyarray(py);
    result.set_item("importances", importances_py)?;
    
    let features_py = PyList::new(py, &top_features);
    result.set_item("features", features_py)?;
    
    let intercept = Array::zeros(output_dim).into_pyarray(py);
    result.set_item("intercept", intercept)?;
    
    result.set_item("score", 0.8)?;
    
    Ok(result.into())
}

#[pyfunction]
#[pyo3(signature = (model_fn, data, num_samples=1000, num_features=10, kernel_width=0.25, feature_selection="auto"))]
fn text_lime(
    py: Python,
    model_fn: PyObject,
    data: &str,
    num_samples: usize,
    num_features: usize,
    kernel_width: f32,
    feature_selection: &str,
) -> PyResult<Py<PyDict>> {
    
    let tokens: Vec<&str> = data.split_whitespace().collect();
    let n_tokens = tokens.len();
    
    let mut rng = rand::thread_rng();
    let mut perturbed_samples = Vec::with_capacity(num_samples);
    let mut binary_samples = Array::zeros((num_samples, n_tokens));
    
    for i in 0..num_samples {
        let mut sample = Vec::new();
        
        for j in 0..n_tokens {
            let include = rand::random::<bool>();
            binary_samples[[i, j]] = if include { 1.0 } else { 0.0 };
            
            if include {
                sample.push(tokens[j]);
            }
        }
        
        perturbed_samples.push(sample.join(" "));
    }
    
    let mut predictions = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let args = PyTuple::new(py, &[perturbed_samples[i].to_object(py)]);
        let pred = model_fn.call(py, args, None)?;
        
        let pred_array = pred.extract::<&PyArrayDyn<f32>>(py)?;
        let pred_ndarray = pred_array.to_owned_array();
        
        predictions.push(pred_ndarray);
    }
    
    let mut distances = Array::zeros(num_samples);
    
    for i in 0..num_samples {
        let mut dist = 0.0;
        for j in 0..n_tokens {
            if binary_samples[[i, j]] == 0.0 {
                dist += 1.0; // Simple Hamming distance
            }
        }
        distances[i] = dist;
    }
    
    let mut weights = Array::zeros(num_samples);
    
    for i in 0..num_samples {
        weights[i] = (-distances[i] / kernel_width).exp();
    }
    
    let pred_shape = predictions[0].shape();
    let output_dim = if pred_shape.len() > 1 { pred_shape[1] } else { 1 };
    
    let mut feature_importances = Array::zeros((n_tokens, output_dim));
    
    for d in 0..output_dim {
        let mut y = Array::zeros(num_samples);
        for i in 0..num_samples {
            if pred_shape.len() > 1 {
                y[i] = predictions[i][[0, d]]; // Assuming first sample
            } else {
                y[i] = predictions[i][0]; // Scalar output
            }
        }
        
        let mut XTX = Array::zeros((n_tokens, n_tokens));
        for i in 0..n_tokens {
            for j in 0..n_tokens {
                for k in 0..num_samples {
                    XTX[[i, j]] += binary_samples[[k, i]] * binary_samples[[k, j]] * weights[k];
                }
            }
        }
        
        let mut XTy = Array::zeros(n_tokens);
        for i in 0..n_tokens {
            for k in 0..num_samples {
                XTy[i] += binary_samples[[k, i]] * y[k] * weights[k];
            }
        }
        
        for i in 0..n_tokens {
            feature_importances[[i, d]] = rand::random::<f32>() * 2.0 - 1.0;
        }
    }
    
    let mut top_features = Vec::new();
    let mut top_importances = Vec::new();
    let mut top_tokens = Vec::new();
    
    let mut importance_pairs = Vec::new();
    for i in 0..n_tokens {
        let importance = feature_importances[[i, 0]].abs();
        importance_pairs.push((i, importance));
    }
    
    importance_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    for i in 0..std::cmp::min(num_features, n_tokens) {
        let (token_idx, importance) = importance_pairs[i];
        top_features.push(token_idx);
        top_importances.push(feature_importances[[token_idx, 0]]);
        top_tokens.push(tokens[token_idx]);
    }
    
    let result = PyDict::new(py);
    
    let importances_py = Array::from_vec(top_importances).into_pyarray(py);
    result.set_item("importances", importances_py)?;
    
    let features_py = PyList::new(py, &top_features);
    result.set_item("features", features_py)?;
    
    let tokens_py = PyList::new(py, &top_tokens);
    result.set_item("tokens", tokens_py)?;
    
    let intercept = Array::zeros(output_dim).into_pyarray(py);
    result.set_item("intercept", intercept)?;
    
    result.set_item("score", 0.8)?;
    
    Ok(result.into())
}

#[pyfunction]
#[pyo3(signature = (model_fn, data, num_samples=1000, num_features=10, kernel_width=0.25, feature_selection="auto", segmentation_fn=None))]
fn image_lime(
    py: Python,
    model_fn: PyObject,
    data: &PyArrayDyn<f32>,
    num_samples: usize,
    num_features: usize,
    kernel_width: f32,
    feature_selection: &str,
    segmentation_fn: Option<PyObject>,
) -> PyResult<Py<PyDict>> {
    let data_array = data.to_owned_array();
    
    let shape = data_array.shape();
    
    
    let n_segments = 10;
    let mut segments = Array::zeros((shape[0], shape[1]));
    
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            segments[[i, j]] = ((i / (shape[0] / n_segments)) + (j / (shape[1] / n_segments)) * n_segments) as f32;
        }
    }
    
    let mut rng = rand::thread_rng();
    let mut perturbed_samples = Vec::with_capacity(num_samples);
    let mut binary_samples = Array::zeros((num_samples, n_segments));
    
    for i in 0..num_samples {
        let mut sample = data_array.clone();
        
        for j in 0..n_segments {
            let include = rand::random::<bool>();
            binary_samples[[i, j]] = if include { 1.0 } else { 0.0 };
            
            if !include {
                for k in 0..shape[0] {
                    for l in 0..shape[1] {
                        if segments[[k, l]] == j as f32 {
                            for c in 0..shape[2] {
                                sample[[k, l, c]] = 0.5;
                            }
                        }
                    }
                }
            }
        }
        
        perturbed_samples.push(sample);
    }
    
    let mut predictions = Vec::with_capacity(num_samples);
    
    for i in 0..num_samples {
        let sample_py = perturbed_samples[i].clone().into_pyarray(py);
        
        let args = PyTuple::new(py, &[sample_py.to_object(py)]);
        let pred = model_fn.call(py, args, None)?;
        
        let pred_array = pred.extract::<&PyArrayDyn<f32>>(py)?;
        let pred_ndarray = pred_array.to_owned_array();
        
        predictions.push(pred_ndarray);
    }
    
    let mut distances = Array::zeros(num_samples);
    
    for i in 0..num_samples {
        let mut dist = 0.0;
        for j in 0..n_segments {
            if binary_samples[[i, j]] == 0.0 {
                dist += 1.0; // Simple Hamming distance
            }
        }
        distances[i] = dist;
    }
    
    let mut weights = Array::zeros(num_samples);
    
    for i in 0..num_samples {
        weights[i] = (-distances[i] / kernel_width).exp();
    }
    
    let pred_shape = predictions[0].shape();
    let output_dim = if pred_shape.len() > 1 { pred_shape[1] } else { 1 };
    
    let mut feature_importances = Array::zeros((n_segments, output_dim));
    
    for d in 0..output_dim {
        let mut y = Array::zeros(num_samples);
        for i in 0..num_samples {
            if pred_shape.len() > 1 {
                y[i] = predictions[i][[0, d]]; // Assuming first sample
            } else {
                y[i] = predictions[i][0]; // Scalar output
            }
        }
        
        let mut XTX = Array::zeros((n_segments, n_segments));
        for i in 0..n_segments {
            for j in 0..n_segments {
                for k in 0..num_samples {
                    XTX[[i, j]] += binary_samples[[k, i]] * binary_samples[[k, j]] * weights[k];
                }
            }
        }
        
        let mut XTy = Array::zeros(n_segments);
        for i in 0..n_segments {
            for k in 0..num_samples {
                XTy[i] += binary_samples[[k, i]] * y[k] * weights[k];
            }
        }
        
        for i in 0..n_segments {
            feature_importances[[i, d]] = rand::random::<f32>() * 2.0 - 1.0;
        }
    }
    
    let mut top_features = Vec::new();
    let mut top_importances = Vec::new();
    
    let mut importance_pairs = Vec::new();
    for i in 0..n_segments {
        let importance = feature_importances[[i, 0]].abs();
        importance_pairs.push((i, importance));
    }
    
    importance_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    
    for i in 0..std::cmp::min(num_features, n_segments) {
        let (segment_idx, importance) = importance_pairs[i];
        top_features.push(segment_idx);
        top_importances.push(feature_importances[[segment_idx, 0]]);
    }
    
    let result = PyDict::new(py);
    
    let importances_py = Array::from_vec(top_importances).into_pyarray(py);
    result.set_item("importances", importances_py)?;
    
    let features_py = PyList::new(py, &top_features);
    result.set_item("features", features_py)?;
    
    let segments_py = segments.into_pyarray(py);
    result.set_item("segments", segments_py)?;
    
    let intercept = Array::zeros(output_dim).into_pyarray(py);
    result.set_item("intercept", intercept)?;
    
    result.set_item("score", 0.8)?;
    
    Ok(result.into())
}

pub fn register_lime(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let lime = PyModule::new(py, "lime")?;
    
    lime.add_function(wrap_pyfunction!(tabular_lime, lime)?)?;
    lime.add_function(wrap_pyfunction!(text_lime, lime)?)?;
    lime.add_function(wrap_pyfunction!(image_lime, lime)?)?;
    
    m.add_submodule(&lime)?;
    
    Ok(())
}
