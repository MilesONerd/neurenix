
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::wrap_pyfunction;
use numpy::{PyArray, PyArrayDyn, IntoPyArray};
use ndarray::{Array, ArrayD};

use crate::error::PhynexusError;
use crate::tensor::Tensor;

#[pyfunction]
#[pyo3(signature = (model, input_data, layer_names=None))]
fn activation_visualization(
    py: Python,
    model: PyObject,
    input_data: &PyArrayDyn<f32>,
    layer_names: Option<Vec<String>>,
) -> PyResult<Py<PyDict>> {
    let input_array = input_data.to_owned_array();
    
    
    let activations = PyDict::new(py);
    
    let args = PyTuple::new(py, &[input_data.to_object(py)]);
    let output = model.call(py, args, None)?;
    
    let layer_names_vec = match layer_names {
        Some(names) => names,
        None => vec!["layer1".to_string(), "layer2".to_string(), "layer3".to_string()],
    };
    
    for layer_name in layer_names_vec {
        let mut activation = Array::zeros((10, 10));
        
        for i in 0..10 {
            for j in 0..10 {
                activation[[i, j]] = rand::random::<f32>();
            }
        }
        
        let activation_py = activation.into_pyarray(py);
        
        let stats = PyDict::new(py);
        stats.set_item("mean", activation_py.mean(None, false)?)?;
        stats.set_item("std", activation_py.std(None, false)?)?;
        stats.set_item("min", activation_py.min(None)?)?;
        stats.set_item("max", activation_py.max(None)?)?;
        
        let layer_dict = PyDict::new(py);
        layer_dict.set_item("activation", activation_py)?;
        layer_dict.set_item("stats", stats)?;
        
        activations.set_item(layer_name, layer_dict)?;
    }
    
    let result = PyDict::new(py);
    
    result.set_item("activations", activations)?;
    
    result.set_item("output", output)?;
    
    Ok(result.into())
}

pub fn register_activation(py: Python, m: &PyModule) -> PyResult<()> {
    let activation = PyModule::new(py, "activation")?;
    
    activation.add_function(wrap_pyfunction!(activation_visualization, activation)?)?;
    
    m.add_submodule(activation)?;
    
    Ok(())
}
