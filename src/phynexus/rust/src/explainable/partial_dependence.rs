
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::wrap_pyfunction;
use numpy::{PyArray, PyArrayDyn, IntoPyArray};
use ndarray::{Array, ArrayD};

use crate::error::PhynexusError;
use crate::tensor::Tensor;

#[pyfunction]
#[pyo3(signature = (model_fn, data, features, grid_resolution=20, percentiles=(0.05, 0.95), random_state=0))]
fn partial_dependence(
    py: Python,
    model_fn: PyObject,
    data: &PyArrayDyn<f32>,
    features: Vec<usize>,
    grid_resolution: usize,
    percentiles: (f32, f32),
    random_state: u64,
) -> PyResult<Py<PyDict>> {
    let data_array = data.to_owned_array();
    
    let n_samples = data_array.shape()[0];
    let n_features = data_array.shape()[1];
    
    let mut rng = rand::thread_rng();
    
    let mut grid_points = Vec::new();
    
    for &feature_idx in &features {
        if feature_idx >= n_features {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                format!("Feature index {} out of bounds for data with {} features", feature_idx, n_features)
            ));
        }
        
        let mut feature_values = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            feature_values.push(data_array[[i, feature_idx]]);
        }
        
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let min_idx = (percentiles.0 * n_samples as f32) as usize;
        let max_idx = (percentiles.1 * n_samples as f32) as usize;
        
        let min_val = feature_values[min_idx.min(n_samples - 1)];
        let max_val = feature_values[max_idx.min(n_samples - 1)];
        
        let mut grid = Vec::with_capacity(grid_resolution);
        let step = (max_val - min_val) / (grid_resolution - 1) as f32;
        
        for i in 0..grid_resolution {
            grid.push(min_val + i as f32 * step);
        }
        
        grid_points.push(grid);
    }
    
    let mut pdp_values = Vec::new();
    
    for (i, &feature_idx) in features.iter().enumerate() {
        let grid = &grid_points[i];
        let mut pdp = Vec::with_capacity(grid.len());
        
        for &val in grid {
            let mut modified_data = data_array.clone();
            
            for j in 0..n_samples {
                modified_data[[j, feature_idx]] = val;
            }
            
            let modified_data_py = modified_data.into_pyarray(py);
            let args = PyTuple::new(py, &[modified_data_py.to_object(py)]);
            let pred = model_fn.call(py, args, None)?;
            let pred_array = pred.extract::<&PyArrayDyn<f32>>(py)?;
            let pred_ndarray = pred_array.to_owned_array();
            
            let mut avg_pred = 0.0;
            
            if pred_ndarray.shape().len() > 1 {
                for j in 0..n_samples {
                    avg_pred += pred_ndarray[[j, 0]]; // Using first output for simplicity
                }
            } else {
                for j in 0..n_samples {
                    avg_pred += pred_ndarray[j];
                }
            }
            
            avg_pred /= n_samples as f32;
            pdp.push(avg_pred);
        }
        
        pdp_values.push(pdp);
    }
    
    let result = PyDict::new(py);
    
    let grid_points_py = PyList::new(py, &grid_points.iter().map(|grid| {
        let grid_array = Array::from_vec(grid.clone());
        grid_array.into_pyarray(py).to_object(py)
    }).collect::<Vec<_>>());
    
    result.set_item("grid_points", grid_points_py)?;
    
    let pdp_values_py = PyList::new(py, &pdp_values.iter().map(|pdp| {
        let pdp_array = Array::from_vec(pdp.clone());
        pdp_array.into_pyarray(py).to_object(py)
    }).collect::<Vec<_>>());
    
    result.set_item("values", pdp_values_py)?;
    
    let features_py = PyList::new(py, &features);
    result.set_item("feature_indices", features_py)?;
    
    Ok(result.into())
}

pub fn register_partial_dependence(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let partial_dependence_module = PyModule::new(py, "partial_dependence")?;
    
    partial_dependence_module.add_function(wrap_pyfunction!(partial_dependence, partial_dependence_module)?)?;
    
    m.add_submodule(&partial_dependence_module)?;
    
    Ok(())
}
