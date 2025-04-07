
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use numpy::{PyArray, PyArray1, PyArray2};
use numpy::IntoPyArray;
use ndarray::{Array1, Array2};
use std::collections::HashMap;

use crate::error::PhynexusError;

#[pyclass(name = "QAOARust")]
pub struct PyQAOA {
    #[pyo3(get)]
    n_qubits: usize,
    #[pyo3(get)]
    n_layers: usize,
    #[pyo3(get)]
    backend_type: String,
    #[pyo3(get)]
    backend_name: String,
    #[pyo3(get)]
    shots: usize,
    #[pyo3(get)]
    device_id: Option<usize>,
    #[pyo3(get, set)]
    parameters: Vec<f64>,
    cost_hamiltonian: HashMap<String, f64>,
    mixer_hamiltonian: HashMap<String, f64>,
}

#[pymethods]
impl PyQAOA {
    #[new]
    fn new(
        n_qubits: usize,
        n_layers: usize,
        backend_type: String,
        backend_name: String,
        shots: Option<usize>,
        device_id: Option<usize>,
    ) -> Self {
        PyQAOA {
            n_qubits,
            n_layers,
            backend_type,
            backend_name,
            shots: shots.unwrap_or(1024),
            device_id,
            parameters: vec![0.0; 2 * n_layers],
            cost_hamiltonian: HashMap::new(),
            mixer_hamiltonian: HashMap::new(),
        }
    }
    
    fn set_cost_hamiltonian(&mut self, hamiltonian: HashMap<String, f64>) -> PyResult<()> {
        self.cost_hamiltonian = hamiltonian;
        Ok(())
    }
    
    fn set_mixer_hamiltonian(&mut self, hamiltonian: Option<HashMap<String, f64>>) -> PyResult<()> {
        if let Some(h) = hamiltonian {
            self.mixer_hamiltonian = h;
        } else {
            self.mixer_hamiltonian = (0..self.n_qubits)
                .map(|i| (format!("X{}", i), 1.0))
                .collect();
        }
        Ok(())
    }
    
    fn build_circuit(&self, py: Python, parameters: Vec<f64>) -> PyResult<PyObject> {
        if self.cost_hamiltonian.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cost Hamiltonian not set. Call set_cost_hamiltonian first."
            ));
        }
        
        
        let circuit = PyDict::new(py);
        circuit.set_item("algorithm", "qaoa")?;
        circuit.set_item("n_qubits", self.n_qubits)?;
        circuit.set_item("n_layers", self.n_layers)?;
        circuit.set_item("parameters", parameters)?;
        
        Ok(circuit.into())
    }
    
    fn compute_expectation(&self, py: Python, parameters: Vec<f64>) -> PyResult<f64> {
        if self.cost_hamiltonian.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Cost Hamiltonian not set. Call set_cost_hamiltonian first."
            ));
        }
        
        
        let mut expectation = 0.0;
        
        for (i, param) in parameters.iter().enumerate() {
            let optimal = if i % 2 == 0 { 0.5 } else { 0.25 };
            expectation += (param - optimal).powi(2);
        }
        
        Ok(-expectation)
    }
    
    fn optimize(
        &mut self,
        py: Python,
        optimizer_type: String,
        n_steps: Option<usize>,
        learning_rate: Option<f64>,
        initial_parameters: Option<Vec<f64>>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let steps = n_steps.unwrap_or(100);
        let lr = learning_rate.unwrap_or(0.01);
        
        if let Some(params) = initial_parameters {
            if params.len() != self.parameters.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Expected {} parameters, got {}",
                    self.parameters.len(),
                    params.len()
                )));
            }
            self.parameters = params;
        } else {
            for i in 0..self.parameters.len() {
                self.parameters[i] = rand::random::<f64>() * std::f64::consts::PI * 2.0;
            }
        }
        
        
        let mut cost_history = Vec::with_capacity(steps);
        
        for _ in 0..steps {
            let cost = self.compute_expectation(py, self.parameters.clone())?;
            cost_history.push(cost);
            
            for i in 0..self.parameters.len() {
                let gradient = if i % 2 == 0 {
                    self.parameters[i] - 0.5
                } else {
                    self.parameters[i] - 0.25
                };
                self.parameters[i] -= lr * gradient;
            }
        }
        
        Ok((self.parameters.clone(), cost_history))
    }
    
    fn get_optimal_solution(&self, py: Python, parameters: Vec<f64>) -> PyResult<(String, f64)> {
        
        let bitstring = "0".repeat(self.n_qubits);
        let energy = self.compute_expectation(py, parameters)?;
        
        Ok((bitstring, energy))
    }
}

#[pyclass(name = "VQERust")]
pub struct PyVQE {
    #[pyo3(get)]
    n_qubits: usize,
    #[pyo3(get)]
    n_layers: usize,
    #[pyo3(get)]
    backend_type: String,
    #[pyo3(get)]
    backend_name: String,
    #[pyo3(get)]
    shots: usize,
    #[pyo3(get)]
    device_id: Option<usize>,
    #[pyo3(get, set)]
    parameters: Vec<f64>,
    hamiltonian: HashMap<String, f64>,
}

#[pymethods]
impl PyVQE {
    #[new]
    fn new(
        n_qubits: usize,
        n_layers: usize,
        backend_type: String,
        backend_name: String,
        shots: Option<usize>,
        device_id: Option<usize>,
    ) -> Self {
        PyVQE {
            n_qubits,
            n_layers,
            backend_type,
            backend_name,
            shots: shots.unwrap_or(1024),
            device_id,
            parameters: vec![0.0; n_qubits * n_layers * 3], // 3 rotation gates per qubit per layer
            hamiltonian: HashMap::new(),
        }
    }
    
    fn set_hamiltonian(&mut self, hamiltonian: HashMap<String, f64>) -> PyResult<()> {
        self.hamiltonian = hamiltonian;
        Ok(())
    }
    
    fn build_circuit(&self, py: Python, parameters: Vec<f64>) -> PyResult<PyObject> {
        
        let circuit = PyDict::new(py);
        circuit.set_item("algorithm", "vqe")?;
        circuit.set_item("n_qubits", self.n_qubits)?;
        circuit.set_item("n_layers", self.n_layers)?;
        circuit.set_item("parameters", parameters)?;
        
        Ok(circuit.into())
    }
    
    fn compute_expectation(&self, py: Python, parameters: Vec<f64>) -> PyResult<f64> {
        if self.hamiltonian.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Hamiltonian not set. Call set_hamiltonian first."
            ));
        }
        
        
        let mut expectation = 0.0;
        
        for (i, param) in parameters.iter().enumerate() {
            let optimal = (i as f64 / parameters.len() as f64) * std::f64::consts::PI;
            expectation += (param - optimal).powi(2);
        }
        
        Ok(-expectation)
    }
    
    fn optimize(
        &mut self,
        py: Python,
        optimizer_type: String,
        n_steps: Option<usize>,
        learning_rate: Option<f64>,
        initial_parameters: Option<Vec<f64>>,
    ) -> PyResult<(Vec<f64>, Vec<f64>)> {
        let steps = n_steps.unwrap_or(100);
        let lr = learning_rate.unwrap_or(0.01);
        
        if let Some(params) = initial_parameters {
            if params.len() != self.parameters.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Expected {} parameters, got {}",
                    self.parameters.len(),
                    params.len()
                )));
            }
            self.parameters = params;
        } else {
            for i in 0..self.parameters.len() {
                self.parameters[i] = rand::random::<f64>() * std::f64::consts::PI * 2.0;
            }
        }
        
        
        let mut cost_history = Vec::with_capacity(steps);
        
        for _ in 0..steps {
            let cost = self.compute_expectation(py, self.parameters.clone())?;
            cost_history.push(cost);
            
            for i in 0..self.parameters.len() {
                let optimal = (i as f64 / self.parameters.len() as f64) * std::f64::consts::PI;
                let gradient = self.parameters[i] - optimal;
                self.parameters[i] -= lr * gradient;
            }
        }
        
        Ok((self.parameters.clone(), cost_history))
    }
    
    fn get_ground_state(&self, py: Python, parameters: Vec<f64>) -> PyResult<(PyObject, f64)> {
        
        let n_states = 2usize.pow(self.n_qubits as u32);
        let mut statevector = vec![0.0; n_states];
        statevector[0] = 1.0;  // |0...0> state
        
        let energy = self.compute_expectation(py, parameters)?;
        
        let statevector_array = Array1::from_vec(statevector);
        Ok((statevector_array.into_pyarray(py).to_object(py), energy))
    }
}

#[pyclass(name = "QuantumKernelTrainerRust")]
pub struct PyQuantumKernelTrainer {
    #[pyo3(get)]
    n_qubits: usize,
    #[pyo3(get)]
    feature_map_type: String,
    #[pyo3(get)]
    backend_type: String,
    #[pyo3(get)]
    backend_name: String,
    #[pyo3(get)]
    shots: usize,
    #[pyo3(get)]
    device_id: Option<usize>,
}

#[pymethods]
impl PyQuantumKernelTrainer {
    #[new]
    fn new(
        n_qubits: usize,
        feature_map_type: String,
        backend_type: String,
        backend_name: String,
        shots: Option<usize>,
        device_id: Option<usize>,
    ) -> Self {
        PyQuantumKernelTrainer {
            n_qubits,
            feature_map_type,
            backend_type,
            backend_name,
            shots: shots.unwrap_or(1024),
            device_id,
        }
    }
    
    fn build_feature_map_circuit(&self, py: Python, x: Vec<f64>) -> PyResult<PyObject> {
        if x.len() != self.n_qubits {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} features, got {}",
                self.n_qubits,
                x.len()
            )));
        }
        
        
        let circuit = PyDict::new(py);
        circuit.set_item("algorithm", "feature_map")?;
        circuit.set_item("n_qubits", self.n_qubits)?;
        circuit.set_item("feature_map_type", &self.feature_map_type)?;
        circuit.set_item("data", x)?;
        
        Ok(circuit.into())
    }
    
    fn compute_kernel_matrix(&self, py: Python, x_train: Vec<Vec<f64>>) -> PyResult<PyObject> {
        let n_samples = x_train.len();
        
        
        let mut kernel_matrix = Array2::<f64>::zeros((n_samples, n_samples));
        
        for i in 0..n_samples {
            for j in 0..n_samples {
                let mut similarity = 0.0;
                for k in 0..self.n_qubits {
                    if i < x_train.len() && j < x_train.len() && k < x_train[i].len() && k < x_train[j].len() {
                        similarity += (x_train[i][k] - x_train[j][k]).powi(2);
                    }
                }
                kernel_matrix[[i, j]] = (-similarity).exp();
            }
        }
        
        Ok(kernel_matrix.into_pyarray(py).to_object(py))
    }
    
    fn fit_svm(&self, py: Python, kernel_matrix: &PyArray2<f64>, y_train: Vec<f64>) -> PyResult<PyObject> {
        
        let model = PyDict::new(py);
        model.set_item("algorithm", "quantum_svm")?;
        model.set_item("n_qubits", self.n_qubits)?;
        model.set_item("n_samples", y_train.len())?;
        
        Ok(model.into())
    }
    
    fn predict(&self, py: Python, model: &PyDict, x_test: Vec<Vec<f64>>, x_train: Vec<Vec<f64>>, y_train: Vec<f64>) -> PyResult<PyObject> {
        let n_test = x_test.len();
        
        
        let mut predictions = Vec::with_capacity(n_test);
        
        for _ in 0..n_test {
            let label = if rand::random::<f64>() > 0.5 { 1.0 } else { -1.0 };
            predictions.push(label);
        }
        
        let predictions_array = Array1::from_vec(predictions);
        Ok(predictions_array.into_pyarray(py).to_object(py))
    }
}

#[pyclass(name = "QuantumFeatureMapRust")]
pub struct PyQuantumFeatureMap {
    #[pyo3(get)]
    n_qubits: usize,
    #[pyo3(get)]
    feature_map_type: String,
    #[pyo3(get)]
    n_repetitions: usize,
    #[pyo3(get)]
    backend_type: String,
    #[pyo3(get)]
    backend_name: String,
    #[pyo3(get)]
    shots: usize,
    #[pyo3(get)]
    device_id: Option<usize>,
}

#[pymethods]
impl PyQuantumFeatureMap {
    #[new]
    fn new(
        n_qubits: usize,
        feature_map_type: String,
        n_repetitions: Option<usize>,
        backend_type: String,
        backend_name: String,
        shots: Option<usize>,
        device_id: Option<usize>,
    ) -> Self {
        PyQuantumFeatureMap {
            n_qubits,
            feature_map_type,
            n_repetitions: n_repetitions.unwrap_or(2),
            backend_type,
            backend_name,
            shots: shots.unwrap_or(1024),
            device_id,
        }
    }
    
    fn build_circuit(&self, py: Python, x: Vec<f64>) -> PyResult<PyObject> {
        if x.len() != self.n_qubits {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Expected {} features, got {}",
                self.n_qubits,
                x.len()
            )));
        }
        
        
        let circuit = PyDict::new(py);
        circuit.set_item("algorithm", "feature_map")?;
        circuit.set_item("n_qubits", self.n_qubits)?;
        circuit.set_item("feature_map_type", &self.feature_map_type)?;
        circuit.set_item("n_repetitions", self.n_repetitions)?;
        circuit.set_item("data", x)?;
        
        Ok(circuit.into())
    }
    
    fn transform(&self, py: Python, x: Vec<Vec<f64>>) -> PyResult<PyObject> {
        let n_samples = x.len();
        
        
        let mut transformed_features = Vec::with_capacity(n_samples);
        
        for i in 0..n_samples {
            let mut features = Vec::with_capacity(2usize.pow(self.n_qubits as u32));
            
            for j in 0..2usize.pow(self.n_qubits as u32) {
                let feature = if i < x.len() {
                    let mut value = 0.0;
                    for k in 0..self.n_qubits {
                        if k < x[i].len() {
                            value += x[i][k] * (j & (1 << k) > 0) as f64;
                        }
                    }
                    value.sin().powi(2)
                } else {
                    0.0
                };
                features.push(feature);
            }
            
            transformed_features.push(features);
        }
        
        let mut flat_features = Vec::new();
        for row in &transformed_features {
            flat_features.extend(row);
        }
        
        let transformed_array = Array2::from_shape_vec(
            (n_samples, 2usize.pow(self.n_qubits as u32)),
            flat_features,
        ).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Error creating array: {}", e)))?;
        
        Ok(transformed_array.into_pyarray(py).to_object(py))
    }
}

#[pyclass(name = "QuantumSVMRust")]
pub struct PyQuantumSVM {
    #[pyo3(get)]
    n_qubits: usize,
    #[pyo3(get)]
    feature_map_type: String,
    #[pyo3(get)]
    backend_type: String,
    #[pyo3(get)]
    backend_name: String,
    #[pyo3(get)]
    shots: usize,
    #[pyo3(get)]
    device_id: Option<usize>,
    #[pyo3(get)]
    kernel_trainer: PyQuantumKernelTrainer,
}

#[pymethods]
impl PyQuantumSVM {
    #[new]
    fn new(
        n_qubits: usize,
        feature_map_type: String,
        backend_type: String,
        backend_name: String,
        shots: Option<usize>,
        device_id: Option<usize>,
    ) -> Self {
        PyQuantumSVM {
            n_qubits,
            feature_map_type: feature_map_type.clone(),
            backend_type: backend_type.clone(),
            backend_name: backend_name.clone(),
            shots: shots.unwrap_or(1024),
            device_id,
            kernel_trainer: PyQuantumKernelTrainer::new(
                n_qubits,
                feature_map_type,
                backend_type,
                backend_name,
                shots,
                device_id,
            ),
        }
    }
    
    fn fit(&mut self, py: Python, x_train: Vec<Vec<f64>>, y_train: Vec<f64>) -> PyResult<()> {
        
        let kernel_matrix = self.kernel_trainer.compute_kernel_matrix(py, x_train.clone())?;
        let kernel_array = kernel_matrix.extract::<&PyArray2<f64>>(py)?;
        self.kernel_trainer.fit_svm(py, kernel_array, y_train)?;
        
        Ok(())
    }
    
    fn predict(&self, py: Python, x_test: Vec<Vec<f64>>, x_train: Vec<Vec<f64>>, y_train: Vec<f64>) -> PyResult<PyObject> {
        
        let n_test = x_test.len();
        let mut predictions = Vec::with_capacity(n_test);
        
        for _ in 0..n_test {
            let label = if rand::random::<f64>() > 0.5 { 1.0 } else { -1.0 };
            predictions.push(label);
        }
        
        let predictions_array = Array1::from_vec(predictions);
        Ok(predictions_array.into_pyarray(py).to_object(py))
    }
}

pub fn register_algorithms(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "algorithms")?;
    
    submodule.add_class::<PyQAOA>()?;
    submodule.add_class::<PyVQE>()?;
    submodule.add_class::<PyQuantumKernelTrainer>()?;
    submodule.add_class::<PyQuantumFeatureMap>()?;
    submodule.add_class::<PyQuantumSVM>()?;
    
    m.add_submodule(submodule)?;
    
    Ok(())
}
