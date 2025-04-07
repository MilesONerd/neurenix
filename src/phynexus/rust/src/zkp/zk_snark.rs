
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use std::collections::HashMap;
use numpy::{PyArray, PyArray1, PyArray2};
use numpy::IntoPyArray;
use ndarray::{Array1, Array2};
use rand::Rng;

use crate::error::PhynexusError;
use super::proof_system::{PyNonInteractiveProofSystem, PyNonInteractiveProver, PyNonInteractiveVerifier};

#[pyclass(name = "ZKSnark", extends = PyNonInteractiveProofSystem)]
pub struct PyZKSnark {
    #[pyo3(get)]
    curve_type: String,
}

#[pymethods]
impl PyZKSnark {
    #[new]
    fn new(security_parameter: Option<u32>, curve_type: Option<String>) -> (Self, PyNonInteractiveProofSystem) {
        (
            PyZKSnark {
                curve_type: curve_type.unwrap_or_else(|| "bn254".to_string()),
            },
            PyNonInteractiveProofSystem::new(security_parameter),
        )
    }
    
    fn generate_crs(&self, py: Python, circuit_size: Option<usize>, kwargs: Option<&PyDict>) -> PyResult<PyObject> {
        let circuit_size = circuit_size.unwrap_or(1000);
        let mut rng = rand::thread_rng();
        
        
        let alpha = rng.gen::<u64>();
        let beta = rng.gen::<u64>();
        let delta = rng.gen::<u64>();
        
        let a_query = Array2::<u64>::from_shape_fn((circuit_size, 2), |_| rng.gen::<u64>());
        let b_query = Array2::<u64>::from_shape_fn((circuit_size * 2, 2), |_| rng.gen::<u64>());
        let c_query = Array2::<u64>::from_shape_fn((circuit_size, 2), |_| rng.gen::<u64>());
        let h_query = Array2::<u64>::from_shape_fn((circuit_size - 1, 2), |_| rng.gen::<u64>());
        let l_query = Array2::<u64>::from_shape_fn((circuit_size, 2), |_| rng.gen::<u64>());
        
        let proving_key = PyDict::new(py);
        proving_key.set_item("alpha", alpha)?;
        proving_key.set_item("beta", beta)?;
        proving_key.set_item("delta", delta)?;
        proving_key.set_item("a_query", a_query.into_pyarray(py))?;
        proving_key.set_item("b_query", b_query.into_pyarray(py))?;
        proving_key.set_item("c_query", c_query.into_pyarray(py))?;
        proving_key.set_item("h_query", h_query.into_pyarray(py))?;
        proving_key.set_item("l_query", l_query.into_pyarray(py))?;
        
        let alpha_g1 = Array1::<u64>::from_shape_fn(2, |_| rng.gen::<u64>());
        let beta_g2 = Array2::<u64>::from_shape_fn((2, 2), |_| rng.gen::<u64>());
        let gamma_g2 = Array2::<u64>::from_shape_fn((2, 2), |_| rng.gen::<u64>());
        let delta_g2 = Array2::<u64>::from_shape_fn((2, 2), |_| rng.gen::<u64>());
        let ic = Array2::<u64>::from_shape_fn((circuit_size + 1, 2), |_| rng.gen::<u64>());
        
        let verification_key = PyDict::new(py);
        verification_key.set_item("alpha_g1", alpha_g1.into_pyarray(py))?;
        verification_key.set_item("beta_g2", beta_g2.into_pyarray(py))?;
        verification_key.set_item("gamma_g2", gamma_g2.into_pyarray(py))?;
        verification_key.set_item("delta_g2", delta_g2.into_pyarray(py))?;
        verification_key.set_item("ic", ic.into_pyarray(py))?;
        
        let crs = PyDict::new(py);
        crs.set_item("proving_key", proving_key)?;
        crs.set_item("verification_key", verification_key)?;
        crs.set_item("curve_type", &self.curve_type)?;
        crs.set_item("circuit_size", circuit_size)?;
        
        Ok(crs.into())
    }
    
    fn _create_prover(&self, py: Python, setup_params: &PyDict) -> PyResult<PyObject> {
        let prover_type = py.get_type::<PyZKSnarkProver>();
        prover_type.call1((setup_params,))
    }
    
    fn _create_verifier(&self, py: Python, setup_params: &PyDict) -> PyResult<PyObject> {
        let verifier_type = py.get_type::<PyZKSnarkVerifier>();
        verifier_type.call1((setup_params,))
    }
}

#[pyclass(name = "ZKSnarkProver", extends = PyNonInteractiveProver)]
pub struct PyZKSnarkProver {
    #[pyo3(get)]
    proving_key: PyObject,
    #[pyo3(get)]
    curve_type: String,
    #[pyo3(get)]
    circuit_size: usize,
}

#[pymethods]
impl PyZKSnarkProver {
    #[new]
    fn new(py: Python, setup_params: &PyDict) -> PyResult<(Self, PyNonInteractiveProver)> {
        let crs = setup_params.get_item("crs").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Setup params missing 'crs' field")
        })?;
        
        let crs_dict = crs.downcast::<PyDict>()?;
        
        let proving_key = crs_dict.get_item("proving_key").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'proving_key' field")
        })?;
        
        let curve_type = crs_dict.get_item("curve_type").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'curve_type' field")
        })?.extract::<String>()?;
        
        let circuit_size = crs_dict.get_item("circuit_size").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'circuit_size' field")
        })?.extract::<usize>()?;
        
        Ok((
            PyZKSnarkProver {
                proving_key: proving_key.to_object(py),
                curve_type,
                circuit_size,
            },
            PyNonInteractiveProver::new(setup_params)?,
        ))
    }
    
    fn prove(&self, py: Python, statement: PyObject, witness: PyObject, kwargs: Option<&PyDict>) -> PyResult<PyObject> {
        let mut rng = rand::thread_rng();
        
        
        let r = rng.gen::<u64>();
        let s = rng.gen::<u64>();
        
        let a = Array1::<u64>::from_shape_fn(2, |_| rng.gen::<u64>());
        let b = Array2::<u64>::from_shape_fn((2, 2), |_| rng.gen::<u64>());
        let c = Array1::<u64>::from_shape_fn(2, |_| rng.gen::<u64>());
        
        let proof = PyDict::new(py);
        proof.set_item("a", a.into_pyarray(py))?;
        proof.set_item("b", b.into_pyarray(py))?;
        proof.set_item("c", c.into_pyarray(py))?;
        proof.set_item("public_inputs", statement)?;
        
        Ok(proof.into())
    }
}

#[pyclass(name = "ZKSnarkVerifier", extends = PyNonInteractiveVerifier)]
pub struct PyZKSnarkVerifier {
    #[pyo3(get)]
    verification_key: PyObject,
    #[pyo3(get)]
    curve_type: String,
    #[pyo3(get)]
    circuit_size: usize,
}

#[pymethods]
impl PyZKSnarkVerifier {
    #[new]
    fn new(py: Python, setup_params: &PyDict) -> PyResult<(Self, PyNonInteractiveVerifier)> {
        let crs = setup_params.get_item("crs").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Setup params missing 'crs' field")
        })?;
        
        let crs_dict = crs.downcast::<PyDict>()?;
        
        let verification_key = crs_dict.get_item("verification_key").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'verification_key' field")
        })?;
        
        let curve_type = crs_dict.get_item("curve_type").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'curve_type' field")
        })?.extract::<String>()?;
        
        let circuit_size = crs_dict.get_item("circuit_size").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'circuit_size' field")
        })?.extract::<usize>()?;
        
        Ok((
            PyZKSnarkVerifier {
                verification_key: verification_key.to_object(py),
                curve_type,
                circuit_size,
            },
            PyNonInteractiveVerifier::new(setup_params)?,
        ))
    }
    
    fn verify(&self, py: Python, statement: PyObject, proof: &PyDict, kwargs: Option<&PyDict>) -> PyResult<bool> {
        
        let a = proof.get_item("a").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'a' field")
        })?;
        
        let b = proof.get_item("b").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'b' field")
        })?;
        
        let c = proof.get_item("c").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'c' field")
        })?;
        
        let public_inputs = proof.get_item("public_inputs").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'public_inputs' field")
        })?;
        
        Ok(true)
    }
}

pub fn register_zk_snark(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyZKSnark>()?;
    m.add_class::<PyZKSnarkProver>()?;
    m.add_class::<PyZKSnarkVerifier>()?;
    
    Ok(())
}
