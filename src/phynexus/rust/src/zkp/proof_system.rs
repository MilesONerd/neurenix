
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use std::collections::HashMap;
use crate::error::PhynexusError;

pub trait ProofSystem {
    fn setup(&self, params: HashMap<String, PyObject>) -> Result<HashMap<String, PyObject>, PhynexusError>;
    
    fn create_prover(&self, setup_params: HashMap<String, PyObject>) -> Result<Box<dyn Prover>, PhynexusError>;
    
    fn create_verifier(&self, setup_params: HashMap<String, PyObject>) -> Result<Box<dyn Verifier>, PhynexusError>;
}

pub trait Prover {
    fn prove(
        &self,
        statement: PyObject,
        witness: PyObject,
        params: HashMap<String, PyObject>,
    ) -> Result<HashMap<String, PyObject>, PhynexusError>;
}

pub trait Verifier {
    fn verify(
        &self,
        statement: PyObject,
        proof: HashMap<String, PyObject>,
        params: HashMap<String, PyObject>,
    ) -> Result<bool, PhynexusError>;
}

pub trait InteractiveProofSystem: ProofSystem {
    fn generate_challenge(&self, commitment: PyObject) -> Result<PyObject, PhynexusError>;
}

pub trait InteractiveProver: Prover {
    fn commit(
        &self,
        statement: PyObject,
        witness: PyObject,
    ) -> Result<PyObject, PhynexusError>;
    
    fn respond(
        &self,
        statement: PyObject,
        witness: PyObject,
        commitment: PyObject,
        challenge: PyObject,
    ) -> Result<PyObject, PhynexusError>;
}

pub trait InteractiveVerifier: Verifier {
    fn challenge(&self, commitment: PyObject) -> Result<PyObject, PhynexusError>;
    
    fn check(
        &self,
        statement: PyObject,
        commitment: PyObject,
        challenge: PyObject,
        response: PyObject,
    ) -> Result<bool, PhynexusError>;
}

pub trait NonInteractiveProofSystem: ProofSystem {
    fn generate_crs(&self, params: HashMap<String, PyObject>) -> Result<HashMap<String, PyObject>, PhynexusError>;
}

#[pyclass(name = "ProofSystem")]
pub struct PyProofSystem {
    #[pyo3(get)]
    security_parameter: u32,
    setup_params: Option<HashMap<String, PyObject>>,
}

#[pymethods]
impl PyProofSystem {
    #[new]
    fn new(security_parameter: Option<u32>) -> Self {
        PyProofSystem {
            security_parameter: security_parameter.unwrap_or(128),
            setup_params: None,
        }
    }
    
    fn setup(&mut self, py: Python, kwargs: Option<&PyDict>) -> PyResult<PyObject> {
        let params = kwargs.map_or_else(
            || HashMap::new(),
            |dict| {
                dict.iter()
                    .map(|(k, v)| (k.extract::<String>().unwrap(), v.to_object(py)))
                    .collect()
            },
        );
        
        let result = HashMap::new();
        self.setup_params = Some(result.clone());
        
        Ok(result.into_py_dict(py).into())
    }
    
    fn get_prover(&self, py: Python) -> PyResult<PyObject> {
        let prover_type = py.get_type::<PyProver>();
        let args = (self.setup_params.clone().unwrap_or_default().into_py_dict(py),);
        prover_type.call1(args)
    }
    
    fn get_verifier(&self, py: Python) -> PyResult<PyObject> {
        let verifier_type = py.get_type::<PyVerifier>();
        let args = (self.setup_params.clone().unwrap_or_default().into_py_dict(py),);
        verifier_type.call1(args)
    }
    
    fn _create_prover(&self, _py: Python, _setup_params: &PyDict) -> PyResult<PyObject> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "ProofSystem._create_prover must be implemented by subclasses",
        ))
    }
    
    fn _create_verifier(&self, _py: Python, _setup_params: &PyDict) -> PyResult<PyObject> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "ProofSystem._create_verifier must be implemented by subclasses",
        ))
    }
}

#[pyclass(name = "Prover")]
pub struct PyProver {
    #[pyo3(get)]
    setup_params: HashMap<String, PyObject>,
}

#[pymethods]
impl PyProver {
    #[new]
    fn new(setup_params: &PyDict) -> PyResult<Self> {
        let py = setup_params.py();
        let params = setup_params
            .iter()
            .map(|(k, v)| (k.extract::<String>().unwrap(), v.to_object(py)))
            .collect();
        
        Ok(PyProver { setup_params: params })
    }
    
    fn prove(&self, _py: Python, _statement: PyObject, _witness: PyObject, _kwargs: Option<&PyDict>) -> PyResult<PyObject> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Prover.prove must be implemented by subclasses",
        ))
    }
}

#[pyclass(name = "Verifier")]
pub struct PyVerifier {
    #[pyo3(get)]
    setup_params: HashMap<String, PyObject>,
}

#[pymethods]
impl PyVerifier {
    #[new]
    fn new(setup_params: &PyDict) -> PyResult<Self> {
        let py = setup_params.py();
        let params = setup_params
            .iter()
            .map(|(k, v)| (k.extract::<String>().unwrap(), v.to_object(py)))
            .collect();
        
        Ok(PyVerifier { setup_params: params })
    }
    
    fn verify(&self, _py: Python, _statement: PyObject, _proof: &PyDict, _kwargs: Option<&PyDict>) -> PyResult<bool> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "Verifier.verify must be implemented by subclasses",
        ))
    }
}

#[pyclass(name = "InteractiveProofSystem", extends = PyProofSystem)]
pub struct PyInteractiveProofSystem {}

#[pymethods]
impl PyInteractiveProofSystem {
    #[new]
    fn new(security_parameter: Option<u32>) -> (Self, PyProofSystem) {
        (
            PyInteractiveProofSystem {},
            PyProofSystem::new(security_parameter),
        )
    }
    
    fn generate_challenge(&self, _py: Python, _commitment: PyObject) -> PyResult<PyObject> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "InteractiveProofSystem.generate_challenge must be implemented by subclasses",
        ))
    }
}

#[pyclass(name = "InteractiveProver", extends = PyProver)]
pub struct PyInteractiveProver {}

#[pymethods]
impl PyInteractiveProver {
    #[new]
    fn new(setup_params: &PyDict) -> PyResult<(Self, PyProver)> {
        Ok((PyInteractiveProver {}, PyProver::new(setup_params)?))
    }
    
    fn commit(&self, _py: Python, _statement: PyObject, _witness: PyObject) -> PyResult<PyObject> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "InteractiveProver.commit must be implemented by subclasses",
        ))
    }
    
    fn respond(
        &self,
        _py: Python,
        _statement: PyObject,
        _witness: PyObject,
        _commitment: PyObject,
        _challenge: PyObject,
    ) -> PyResult<PyObject> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "InteractiveProver.respond must be implemented by subclasses",
        ))
    }
    
    fn prove(
        &self,
        py: Python,
        statement: PyObject,
        witness: PyObject,
        verifier: PyObject,
        kwargs: Option<&PyDict>,
    ) -> PyResult<PyObject> {
        let commitment = self.commit(py, statement.clone(), witness.clone())?;
        
        let challenge = verifier
            .call_method1(py, "challenge", (commitment.clone(),))?;
        
        let response = self.respond(
            py,
            statement.clone(),
            witness.clone(),
            commitment.clone(),
            challenge.clone(),
        )?;
        
        let result = PyDict::new(py);
        result.set_item("commitment", commitment)?;
        result.set_item("challenge", challenge)?;
        result.set_item("response", response)?;
        
        Ok(result.into())
    }
}

#[pyclass(name = "InteractiveVerifier", extends = PyVerifier)]
pub struct PyInteractiveVerifier {}

#[pymethods]
impl PyInteractiveVerifier {
    #[new]
    fn new(setup_params: &PyDict) -> PyResult<(Self, PyVerifier)> {
        Ok((PyInteractiveVerifier {}, PyVerifier::new(setup_params)?))
    }
    
    fn challenge(&self, _py: Python, _commitment: PyObject) -> PyResult<PyObject> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "InteractiveVerifier.challenge must be implemented by subclasses",
        ))
    }
    
    fn check(
        &self,
        _py: Python,
        _statement: PyObject,
        _commitment: PyObject,
        _challenge: PyObject,
        _response: PyObject,
    ) -> PyResult<bool> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "InteractiveVerifier.check must be implemented by subclasses",
        ))
    }
    
    fn verify(&self, py: Python, statement: PyObject, proof: &PyDict, _kwargs: Option<&PyDict>) -> PyResult<bool> {
        let commitment = proof.get_item("commitment").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'commitment' field")
        })?;
        
        let challenge = proof.get_item("challenge").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'challenge' field")
        })?;
        
        let response = proof.get_item("response").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'response' field")
        })?;
        
        self.check(py, statement, commitment, challenge, response)
    }
}

#[pyclass(name = "NonInteractiveProofSystem", extends = PyProofSystem)]
pub struct PyNonInteractiveProofSystem {}

#[pymethods]
impl PyNonInteractiveProofSystem {
    #[new]
    fn new(security_parameter: Option<u32>) -> (Self, PyProofSystem) {
        (
            PyNonInteractiveProofSystem {},
            PyProofSystem::new(security_parameter),
        )
    }
    
    fn generate_crs(&self, _py: Python, _kwargs: Option<&PyDict>) -> PyResult<PyObject> {
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "NonInteractiveProofSystem.generate_crs must be implemented by subclasses",
        ))
    }
    
    fn setup(&mut self, py: Python, kwargs: Option<&PyDict>) -> PyResult<PyObject> {
        let crs = self.generate_crs(py, kwargs)?;
        let result = PyDict::new(py);
        result.set_item("crs", crs)?;
        
        let parent = self.as_ref();
        parent.setup_params = Some(
            result
                .iter()
                .map(|(k, v)| (k.extract::<String>().unwrap(), v.to_object(py)))
                .collect(),
        );
        
        Ok(result.into())
    }
}

#[pyclass(name = "NonInteractiveProver", extends = PyProver)]
pub struct PyNonInteractiveProver {}

#[pymethods]
impl PyNonInteractiveProver {
    #[new]
    fn new(setup_params: &PyDict) -> PyResult<(Self, PyProver)> {
        Ok((PyNonInteractiveProver {}, PyProver::new(setup_params)?))
    }
    
    #[getter]
    fn crs(&self, py: Python) -> PyResult<PyObject> {
        let setup_params = &self.as_ref().setup_params;
        match setup_params.get("crs") {
            Some(crs) => Ok(crs.clone()),
            None => {
                let empty_dict = PyDict::new(py);
                Ok(empty_dict.into())
            }
        }
    }
}

#[pyclass(name = "NonInteractiveVerifier", extends = PyVerifier)]
pub struct PyNonInteractiveVerifier {}

#[pymethods]
impl PyNonInteractiveVerifier {
    #[new]
    fn new(setup_params: &PyDict) -> PyResult<(Self, PyVerifier)> {
        Ok((PyNonInteractiveVerifier {}, PyVerifier::new(setup_params)?))
    }
    
    #[getter]
    fn crs(&self, py: Python) -> PyResult<PyObject> {
        let setup_params = &self.as_ref().setup_params;
        match setup_params.get("crs") {
            Some(crs) => Ok(crs.clone()),
            None => {
                let empty_dict = PyDict::new(py);
                Ok(empty_dict.into())
            }
        }
    }
}

pub fn register_proof_system(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyProofSystem>()?;
    m.add_class::<PyProver>()?;
    m.add_class::<PyVerifier>()?;
    m.add_class::<PyInteractiveProofSystem>()?;
    m.add_class::<PyInteractiveProver>()?;
    m.add_class::<PyInteractiveVerifier>()?;
    m.add_class::<PyNonInteractiveProofSystem>()?;
    m.add_class::<PyNonInteractiveProver>()?;
    m.add_class::<PyNonInteractiveVerifier>()?;
    
    Ok(())
}
