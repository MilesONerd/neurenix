
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType, PyBytes};
use std::collections::HashMap;
use rand::Rng;

use crate::error::PhynexusError;
use super::proof_system::{PyInteractiveProofSystem, PyInteractiveProver, PyInteractiveVerifier};

#[pyclass(name = "SigmaProtocol", extends = PyInteractiveProofSystem)]
pub struct PySigmaProtocol {}

#[pymethods]
impl PySigmaProtocol {
    #[new]
    fn new(security_parameter: Option<u32>) -> (Self, PyInteractiveProofSystem) {
        (
            PySigmaProtocol {},
            PyInteractiveProofSystem::new(security_parameter),
        )
    }
    
    fn setup(&mut self, py: Python, kwargs: Option<&PyDict>) -> PyResult<PyObject> {
        let result = PyDict::new(py);
        result.set_item("security_parameter", self.as_ref().security_parameter)?;
        result.set_item("hash_function", kwargs.and_then(|k| k.get_item("hash_function")).unwrap_or_else(|| "sha256".into_py(py)))?;
        
        let parent = self.as_ref();
        parent.setup_params = Some(
            result
                .iter()
                .map(|(k, v)| (k.extract::<String>().unwrap(), v.to_object(py)))
                .collect(),
        );
        
        Ok(result.into())
    }
    
    fn generate_challenge(&self, py: Python, commitment: PyObject) -> PyResult<PyObject> {
        
        let hash_function = self.as_ref().setup_params.as_ref().map_or(
            "sha256".to_string(),
            |params| params.get("hash_function").map_or(
                "sha256".to_string(),
                |v| v.extract::<String>(py).unwrap_or_else(|_| "sha256".to_string())
            )
        );
        
        let commitment_bytes = if let Ok(bytes) = commitment.extract::<&PyBytes>(py) {
            bytes.as_bytes().to_vec()
        } else {
            let commitment_str = commitment.extract::<String>(py).unwrap_or_default();
            commitment_str.as_bytes().to_vec()
        };
        
        let hash_module = py.import("hashlib")?;
        let hash_obj = hash_module.call_method1(hash_function.as_str(), (PyBytes::new(py, &commitment_bytes),))?;
        let digest = hash_obj.call_method0("digest")?;
        
        Ok(digest)
    }
    
    fn _create_prover(&self, py: Python, setup_params: &PyDict) -> PyResult<PyObject> {
        let prover_type = py.get_type::<PySigmaProver>();
        prover_type.call1((setup_params,))
    }
    
    fn _create_verifier(&self, py: Python, setup_params: &PyDict) -> PyResult<PyObject> {
        let verifier_type = py.get_type::<PySigmaVerifier>();
        verifier_type.call1((setup_params,))
    }
}

#[pyclass(name = "SigmaProver", extends = PyInteractiveProver)]
pub struct PySigmaProver {
    #[pyo3(get)]
    hash_function: String,
}

#[pymethods]
impl PySigmaProver {
    #[new]
    fn new(py: Python, setup_params: &PyDict) -> PyResult<(Self, PyInteractiveProver)> {
        let hash_function = setup_params.get_item("hash_function").map_or(
            "sha256".to_string(),
            |v| v.extract::<String>().unwrap_or_else(|_| "sha256".to_string())
        );
        
        Ok((
            PySigmaProver {
                hash_function,
            },
            PyInteractiveProver::new(setup_params)?,
        ))
    }
    
    fn commit(&self, py: Python, _statement: PyObject, _witness: PyObject) -> PyResult<PyObject> {
        
        let mut rng = rand::thread_rng();
        let random_bytes: Vec<u8> = (0..32).map(|_| rng.gen::<u8>()).collect();
        
        Ok(PyBytes::new(py, &random_bytes).into())
    }
    
    fn respond(
        &self,
        py: Python,
        _statement: PyObject,
        _witness: PyObject,
        _commitment: PyObject,
        _challenge: PyObject,
    ) -> PyResult<PyObject> {
        
        let mut rng = rand::thread_rng();
        let random_bytes: Vec<u8> = (0..32).map(|_| rng.gen::<u8>()).collect();
        
        Ok(PyBytes::new(py, &random_bytes).into())
    }
    
    fn prove_non_interactive(&self, py: Python, statement: PyObject, witness: PyObject) -> PyResult<PyObject> {
        let commitment = self.commit(py, statement.clone(), witness.clone())?;
        
        let hash_module = py.import("hashlib")?;
        
        let commitment_bytes = if let Ok(bytes) = commitment.extract::<&PyBytes>(py) {
            bytes.as_bytes().to_vec()
        } else {
            let commitment_str = commitment.extract::<String>(py).unwrap_or_default();
            commitment_str.as_bytes().to_vec()
        };
        
        let statement_bytes = if let Ok(bytes) = statement.extract::<&PyBytes>(py) {
            bytes.as_bytes().to_vec()
        } else {
            let statement_str = statement.extract::<String>(py).unwrap_or_default();
            statement_str.as_bytes().to_vec()
        };
        
        let mut combined_bytes = commitment_bytes;
        combined_bytes.extend_from_slice(&statement_bytes);
        
        let hash_obj = hash_module.call_method1(self.hash_function.as_str(), (PyBytes::new(py, &combined_bytes),))?;
        let challenge = hash_obj.call_method0("digest")?;
        
        let response = self.respond(py, statement, witness, commitment.clone(), challenge.clone())?;
        
        let proof = PyDict::new(py);
        proof.set_item("commitment", commitment)?;
        proof.set_item("challenge", challenge)?;
        proof.set_item("response", response)?;
        
        Ok(proof.into())
    }
}

#[pyclass(name = "SigmaVerifier", extends = PyInteractiveVerifier)]
pub struct PySigmaVerifier {
    #[pyo3(get)]
    hash_function: String,
}

#[pymethods]
impl PySigmaVerifier {
    #[new]
    fn new(py: Python, setup_params: &PyDict) -> PyResult<(Self, PyInteractiveVerifier)> {
        let hash_function = setup_params.get_item("hash_function").map_or(
            "sha256".to_string(),
            |v| v.extract::<String>().unwrap_or_else(|_| "sha256".to_string())
        );
        
        Ok((
            PySigmaVerifier {
                hash_function,
            },
            PyInteractiveVerifier::new(setup_params)?,
        ))
    }
    
    fn challenge(&self, py: Python, commitment: PyObject) -> PyResult<PyObject> {
        
        let mut rng = rand::thread_rng();
        let random_bytes: Vec<u8> = (0..32).map(|_| rng.gen::<u8>()).collect();
        
        Ok(PyBytes::new(py, &random_bytes).into())
    }
    
    fn check(
        &self,
        _py: Python,
        _statement: PyObject,
        _commitment: PyObject,
        _challenge: PyObject,
        _response: PyObject,
    ) -> PyResult<bool> {
        
        Ok(true)
    }
    
    fn verify_non_interactive(&self, py: Python, statement: PyObject, proof: &PyDict) -> PyResult<bool> {
        let commitment = proof.get_item("commitment").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'commitment' field")
        })?;
        
        let challenge = proof.get_item("challenge").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'challenge' field")
        })?;
        
        let response = proof.get_item("response").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'response' field")
        })?;
        
        let hash_module = py.import("hashlib")?;
        
        let commitment_bytes = if let Ok(bytes) = commitment.extract::<&PyBytes>(py) {
            bytes.as_bytes().to_vec()
        } else {
            let commitment_str = commitment.extract::<String>(py).unwrap_or_default();
            commitment_str.as_bytes().to_vec()
        };
        
        let statement_bytes = if let Ok(bytes) = statement.extract::<&PyBytes>(py) {
            bytes.as_bytes().to_vec()
        } else {
            let statement_str = statement.extract::<String>(py).unwrap_or_default();
            statement_str.as_bytes().to_vec()
        };
        
        let mut combined_bytes = commitment_bytes;
        combined_bytes.extend_from_slice(&statement_bytes);
        
        let hash_obj = hash_module.call_method1(self.hash_function.as_str(), (PyBytes::new(py, &combined_bytes),))?;
        let expected_challenge = hash_obj.call_method0("digest")?;
        
        let challenges_match = challenge.eq(expected_challenge)?;
        
        if !challenges_match {
            return Ok(false);
        }
        
        self.check(py, statement, commitment, challenge, response)
    }
}

pub fn register_sigma(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySigmaProtocol>()?;
    m.add_class::<PySigmaProver>()?;
    m.add_class::<PySigmaVerifier>()?;
    
    Ok(())
}
