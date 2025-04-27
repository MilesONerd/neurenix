
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyValueError;

use std::collections::HashMap;
use rand::Rng;
use rand::seq::SliceRandom;

use crate::error::PhynexusError;
use crate::tensor::Tensor;

fn convert_param_space(param_space: &PyDict) -> PyResult<HashMap<String, Vec<PyObject>>> {
    let mut param_space_map = HashMap::new();
    
    Python::with_gil(|py| {
        for (key, value) in param_space.iter() {
            let key_str = key.extract::<String>()?;
            let values = value.extract::<Vec<PyObject>>()?;
            param_space_map.insert(key_str, values);
        }
        Ok(())
    })?;
    
    Ok(param_space_map)
}

pub fn register_search(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let search = PyModule::new(py, "search")?;
    
    search.add_class::<PyGridSearch>()?;
    search.add_class::<PyRandomSearch>()?;
    search.add_class::<PyBayesianOptimization>()?;
    search.add_class::<PyEvolutionarySearch>()?;
    
    m.add_submodule(&search)?;
    
    Ok(())
}
