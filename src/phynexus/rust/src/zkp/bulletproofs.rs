
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyType};
use std::collections::HashMap;
use numpy::{PyArray, PyArray1, PyArray2};
use numpy::IntoPyArray;
use ndarray::{Array1, Array2};
use rand::Rng;

use crate::error::PhynexusError;
use super::proof_system::{PyNonInteractiveProofSystem, PyNonInteractiveProver, PyNonInteractiveVerifier};

#[pyclass(name = "Bulletproofs", extends = PyNonInteractiveProofSystem)]
pub struct PyBulletproofs {
    #[pyo3(get)]
    curve_type: String,
}

#[pymethods]
impl PyBulletproofs {
    #[new]
    fn new(security_parameter: Option<u32>, curve_type: Option<String>) -> (Self, PyNonInteractiveProofSystem) {
        (
            PyBulletproofs {
                curve_type: curve_type.unwrap_or_else(|| "ristretto".to_string()),
            },
            PyNonInteractiveProofSystem::new(security_parameter),
        )
    }
    
    fn generate_crs(&self, py: Python, max_bit_size: Option<usize>, kwargs: Option<&PyDict>) -> PyResult<PyObject> {
        let max_bit_size = max_bit_size.unwrap_or(64);
        let mut rng = rand::thread_rng();
        
        
        let g_vec = Array2::<u64>::from_shape_fn((max_bit_size, 2), |_| rng.gen::<u64>());
        let h_vec = Array2::<u64>::from_shape_fn((max_bit_size, 2), |_| rng.gen::<u64>());
        
        let g = Array1::<u64>::from_shape_fn(2, |_| rng.gen::<u64>());
        let h = Array1::<u64>::from_shape_fn(2, |_| rng.gen::<u64>());
        let u = Array1::<u64>::from_shape_fn(2, |_| rng.gen::<u64>());
        
        let crs = PyDict::new(py);
        crs.set_item("g", g.into_pyarray(py))?;
        crs.set_item("h", h.into_pyarray(py))?;
        crs.set_item("u", u.into_pyarray(py))?;
        crs.set_item("g_vec", g_vec.into_pyarray(py))?;
        crs.set_item("h_vec", h_vec.into_pyarray(py))?;
        crs.set_item("curve_type", &self.curve_type)?;
        crs.set_item("max_bit_size", max_bit_size)?;
        
        Ok(crs.into())
    }
    
    fn _create_prover(&self, py: Python, setup_params: &PyDict) -> PyResult<PyObject> {
        let prover_type = py.get_type::<PyBulletproofsProver>();
        prover_type.call1((setup_params,))
    }
    
    fn _create_verifier(&self, py: Python, setup_params: &PyDict) -> PyResult<PyObject> {
        let verifier_type = py.get_type::<PyBulletproofsVerifier>();
        verifier_type.call1((setup_params,))
    }
}

#[pyclass(name = "BulletproofsProver", extends = PyNonInteractiveProver)]
pub struct PyBulletproofsProver {
    #[pyo3(get)]
    g: PyObject,
    #[pyo3(get)]
    h: PyObject,
    #[pyo3(get)]
    u: PyObject,
    #[pyo3(get)]
    g_vec: PyObject,
    #[pyo3(get)]
    h_vec: PyObject,
    #[pyo3(get)]
    curve_type: String,
    #[pyo3(get)]
    max_bit_size: usize,
}

#[pymethods]
impl PyBulletproofsProver {
    #[new]
    fn new(py: Python, setup_params: &PyDict) -> PyResult<(Self, PyNonInteractiveProver)> {
        let crs = setup_params.get_item("crs").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Setup params missing 'crs' field")
        })?;
        
        let crs_dict = crs.downcast::<PyDict>()?;
        
        let g = crs_dict.get_item("g").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'g' field")
        })?;
        
        let h = crs_dict.get_item("h").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'h' field")
        })?;
        
        let u = crs_dict.get_item("u").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'u' field")
        })?;
        
        let g_vec = crs_dict.get_item("g_vec").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'g_vec' field")
        })?;
        
        let h_vec = crs_dict.get_item("h_vec").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'h_vec' field")
        })?;
        
        let curve_type = crs_dict.get_item("curve_type").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'curve_type' field")
        })?.extract::<String>()?;
        
        let max_bit_size = crs_dict.get_item("max_bit_size").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'max_bit_size' field")
        })?.extract::<usize>()?;
        
        Ok((
            PyBulletproofsProver {
                g: g.to_object(py),
                h: h.to_object(py),
                u: u.to_object(py),
                g_vec: g_vec.to_object(py),
                h_vec: h_vec.to_object(py),
                curve_type,
                max_bit_size,
            },
            PyNonInteractiveProver::new(setup_params)?,
        ))
    }
    
    fn _inner_product_proof(&self, py: Python, a: &PyArray1<u64>, b: &PyArray1<u64>) -> PyResult<PyObject> {
        let mut rng = rand::thread_rng();
        
        
        let n = a.shape()[0];
        let log_n = (n as f64).log2() as usize;
        
        let l_vec = PyList::new(py);
        let r_vec = PyList::new(py);
        
        for _ in 0..log_n {
            let l = Array1::<u64>::from_shape_fn(2, |_| rng.gen::<u64>());
            let r = Array1::<u64>::from_shape_fn(2, |_| rng.gen::<u64>());
            
            l_vec.append(l.into_pyarray(py))?;
            r_vec.append(r.into_pyarray(py))?;
        }
        
        let proof = PyDict::new(py);
        proof.set_item("l_vec", l_vec)?;
        proof.set_item("r_vec", r_vec)?;
        proof.set_item("a", a[0])?;
        proof.set_item("b", b[0])?;
        
        Ok(proof.into())
    }
    
    fn prove(&self, py: Python, statement: PyObject, witness: PyObject, kwargs: Option<&PyDict>) -> PyResult<PyObject> {
        let mut rng = rand::thread_rng();
        
        
        let witness_dict = witness.extract::<&PyDict>(py)?;
        
        let value = witness_dict.get_item("value").map_or(0, |v| v.extract::<u64>().unwrap_or(0));
        let blinding_factor = witness_dict.get_item("blinding_factor").map_or(0, |v| v.extract::<u64>().unwrap_or(0));
        
        let statement_dict = statement.extract::<&PyDict>(py)?;
        
        let commitment = statement_dict.get_item("commitment").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Statement missing 'commitment' field")
        })?;
        
        let bit_size = kwargs.and_then(|k| k.get_item("bit_size")).map_or(64, |v| v.extract::<usize>().unwrap_or(64));
        
        let binary = (0..bit_size).map(|i| ((value >> i) & 1) as u64).collect::<Vec<u64>>();
        let a_l = Array1::from_vec(binary);
        
        let a_r = Array1::from_vec(a_l.iter().map(|&b| 1 - b).collect::<Vec<u64>>());
        
        let alpha = rng.gen::<u64>();
        
        let A = Array1::<u64>::from_shape_fn(2, |_| rng.gen::<u64>());
        
        let y = rng.gen::<u64>();
        let z = rng.gen::<u64>();
        
        let ip_proof = self._inner_product_proof(py, &a_l.into_pyarray(py), &a_r.into_pyarray(py))?;
        
        let proof = PyDict::new(py);
        proof.set_item("A", A.into_pyarray(py))?;
        proof.set_item("S", Array1::<u64>::from_shape_fn(2, |_| rng.gen::<u64>()).into_pyarray(py))?;
        proof.set_item("T1", Array1::<u64>::from_shape_fn(2, |_| rng.gen::<u64>()).into_pyarray(py))?;
        proof.set_item("T2", Array1::<u64>::from_shape_fn(2, |_| rng.gen::<u64>()).into_pyarray(py))?;
        proof.set_item("tau_x", rng.gen::<u64>())?;
        proof.set_item("mu", rng.gen::<u64>())?;
        proof.set_item("t_hat", rng.gen::<u64>())?;
        proof.set_item("inner_product_proof", ip_proof)?;
        proof.set_item("commitment", commitment)?;
        
        Ok(proof.into())
    }
}

#[pyclass(name = "BulletproofsVerifier", extends = PyNonInteractiveVerifier)]
pub struct PyBulletproofsVerifier {
    #[pyo3(get)]
    g: PyObject,
    #[pyo3(get)]
    h: PyObject,
    #[pyo3(get)]
    u: PyObject,
    #[pyo3(get)]
    g_vec: PyObject,
    #[pyo3(get)]
    h_vec: PyObject,
    #[pyo3(get)]
    curve_type: String,
    #[pyo3(get)]
    max_bit_size: usize,
}

#[pymethods]
impl PyBulletproofsVerifier {
    #[new]
    fn new(py: Python, setup_params: &PyDict) -> PyResult<(Self, PyNonInteractiveVerifier)> {
        let crs = setup_params.get_item("crs").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Setup params missing 'crs' field")
        })?;
        
        let crs_dict = crs.downcast::<PyDict>()?;
        
        let g = crs_dict.get_item("g").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'g' field")
        })?;
        
        let h = crs_dict.get_item("h").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'h' field")
        })?;
        
        let u = crs_dict.get_item("u").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'u' field")
        })?;
        
        let g_vec = crs_dict.get_item("g_vec").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'g_vec' field")
        })?;
        
        let h_vec = crs_dict.get_item("h_vec").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'h_vec' field")
        })?;
        
        let curve_type = crs_dict.get_item("curve_type").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'curve_type' field")
        })?.extract::<String>()?;
        
        let max_bit_size = crs_dict.get_item("max_bit_size").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("CRS missing 'max_bit_size' field")
        })?.extract::<usize>()?;
        
        Ok((
            PyBulletproofsVerifier {
                g: g.to_object(py),
                h: h.to_object(py),
                u: u.to_object(py),
                g_vec: g_vec.to_object(py),
                h_vec: h_vec.to_object(py),
                curve_type,
                max_bit_size,
            },
            PyNonInteractiveVerifier::new(setup_params)?,
        ))
    }
    
    fn _verify_inner_product(&self, _py: Python, _proof: &PyDict, _P: &PyArray1<u64>, _g_vec: &PyArray2<u64>, _h_vec: &PyArray2<u64>) -> PyResult<bool> {
        
        Ok(true)
    }
    
    fn verify(&self, py: Python, statement: PyObject, proof: &PyDict, _kwargs: Option<&PyDict>) -> PyResult<bool> {
        
        let A = proof.get_item("A").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'A' field")
        })?;
        
        let S = proof.get_item("S").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'S' field")
        })?;
        
        let T1 = proof.get_item("T1").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'T1' field")
        })?;
        
        let T2 = proof.get_item("T2").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'T2' field")
        })?;
        
        let tau_x = proof.get_item("tau_x").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'tau_x' field")
        })?;
        
        let mu = proof.get_item("mu").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'mu' field")
        })?;
        
        let t_hat = proof.get_item("t_hat").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 't_hat' field")
        })?;
        
        let inner_product_proof = proof.get_item("inner_product_proof").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'inner_product_proof' field")
        })?;
        
        let commitment = proof.get_item("commitment").ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Proof missing 'commitment' field")
        })?;
        
        Ok(true)
    }
}

pub fn register_bulletproofs(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyBulletproofs>()?;
    m.add_class::<PyBulletproofsProver>()?;
    m.add_class::<PyBulletproofsVerifier>()?;
    
    Ok(())
}
