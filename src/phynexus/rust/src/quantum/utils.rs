
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use numpy::{PyArray, PyArray1, PyArray2};
use numpy::IntoPyArray;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

use crate::error::PhynexusError;

#[pyfunction]
fn state_to_tensor(py: Python, state_vector: &PyArray1<f64>) -> PyResult<PyObject> {
    let state_array = unsafe { state_vector.as_array() };
    let n_states = state_array.len();
    let n_qubits = (n_states as f64).log2() as usize;
    
    if n_states != 2usize.pow(n_qubits as u32) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "State vector length {} is not a power of 2",
            n_states
        )));
    }
    
    
    Ok(state_vector.to_object(py))
}

#[pyfunction]
fn tensor_to_state(py: Python, tensor: &PyArray1<f64>) -> PyResult<PyObject> {
    let tensor_array = unsafe { tensor.as_array() };
    
    
    Ok(tensor.to_object(py))
}

#[pyfunction]
fn measure_expectation(
    py: Python,
    state_vector: &PyArray1<f64>,
    observable: &PyDict,
) -> PyResult<f64> {
    let state_array = unsafe { state_vector.as_array() };
    let n_states = state_array.len();
    
    
    let mut expectation = 0.0;
    
    if let Some(pauli_string) = observable.get_item("pauli_string") {
        let pauli_str = pauli_string.extract::<String>()?;
        let coeffs = observable.get_item("coefficients")
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Observable missing 'coefficients'"))?
            .extract::<Vec<f64>>()?;
        
        for (i, coeff) in coeffs.iter().enumerate() {
            expectation += coeff * 0.5;  // Placeholder computation
        }
    } else if let Some(matrix) = observable.get_item("matrix") {
        let matrix_array = matrix.extract::<&PyArray2<f64>>()?;
        let matrix_ndarray = unsafe { matrix_array.as_array() };
        
        for i in 0..n_states {
            for j in 0..n_states {
                if i < matrix_ndarray.shape()[0] && j < matrix_ndarray.shape()[1] {
                    expectation += state_array[i] * matrix_ndarray[[i, j]] * state_array[j];
                }
            }
        }
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Observable must contain either 'pauli_string' or 'matrix'"
        ));
    }
    
    Ok(expectation)
}

#[pyfunction]
fn quantum_gradient(
    py: Python,
    circuit_fn: PyObject,
    parameters: &PyArray1<f64>,
    observable: &PyDict,
    method: Option<&str>,
) -> PyResult<PyObject> {
    let params_array = unsafe { parameters.as_array() };
    let n_params = params_array.len();
    
    
    let mut gradient = Array1::<f64>::zeros(n_params);
    
    let method_str = method.unwrap_or("parameter_shift");
    
    match method_str {
        "parameter_shift" => {
            for i in 0..n_params {
                gradient[i] = 0.5;  // Placeholder gradient value
            }
        }
        "finite_difference" => {
            for i in 0..n_params {
                gradient[i] = 0.1;  // Placeholder gradient value
            }
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported gradient method: {}",
                method_str
            )));
        }
    }
    
    Ok(gradient.into_pyarray(py).to_object(py))
}

#[pyfunction]
fn quantum_fidelity(
    py: Python,
    state1: &PyArray1<f64>,
    state2: &PyArray1<f64>,
) -> PyResult<f64> {
    let state1_array = unsafe { state1.as_array() };
    let state2_array = unsafe { state2.as_array() };
    
    if state1_array.len() != state2_array.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "State vectors must have the same length"
        ));
    }
    
    let mut overlap = 0.0;
    for i in 0..state1_array.len() {
        overlap += state1_array[i] * state2_array[i];
    }
    
    Ok(overlap.powi(2))
}

#[pyfunction]
fn apply_gate(
    py: Python,
    state: &PyArray1<f64>,
    gate: &PyDict,
    target_qubits: Vec<usize>,
) -> PyResult<PyObject> {
    let state_array = unsafe { state.as_array() };
    let n_states = state_array.len();
    let n_qubits = (n_states as f64).log2() as usize;
    
    
    let mut result = state_array.to_owned();
    
    if let Some(gate_type) = gate.get_item("type") {
        let gate_type_str = gate_type.extract::<String>()?;
        
        match gate_type_str.as_str() {
            "X" => {
                for target in &target_qubits {
                    for i in 0..n_states {
                        if i & (1 << target) == 0 {
                            let j = i | (1 << target);
                            let temp = result[i];
                            result[i] = result[j];
                            result[j] = temp;
                        }
                    }
                }
            }
            "H" => {
                for target in &target_qubits {
                    for i in 0..n_states {
                        if i & (1 << target) == 0 {
                            let j = i | (1 << target);
                            let temp_i = result[i];
                            let temp_j = result[j];
                            result[i] = (temp_i + temp_j) / std::f64::consts::SQRT_2;
                            result[j] = (temp_i - temp_j) / std::f64::consts::SQRT_2;
                        }
                    }
                }
            }
            _ => {
            }
        }
    }
    
    Ok(Array1::from_vec(result.to_vec()).into_pyarray(py).to_object(py))
}

pub fn register_utils(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "utils")?;
    
    submodule.add_function(wrap_pyfunction!(state_to_tensor, submodule)?)?;
    submodule.add_function(wrap_pyfunction!(tensor_to_state, submodule)?)?;
    submodule.add_function(wrap_pyfunction!(measure_expectation, submodule)?)?;
    submodule.add_function(wrap_pyfunction!(quantum_gradient, submodule)?)?;
    submodule.add_function(wrap_pyfunction!(quantum_fidelity, submodule)?)?;
    submodule.add_function(wrap_pyfunction!(apply_gate, submodule)?)?;
    
    m.add_submodule(submodule)?;
    
    Ok(())
}
