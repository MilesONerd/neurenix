
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyValueError;

use std::collections::HashMap;
use rand::Rng;
use rand::seq::SliceRandom;

use crate::error::PhynexusError;
use crate::tensor::Tensor;

pub fn register_nas(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let nas = PyModule::new(py, "nas")?;
    
    nas.add_class::<PyNeuralArchitectureSearch>()?;
    nas.add_class::<PyENAS>()?;
    nas.add_class::<PyDARTS>()?;
    nas.add_class::<PyPNAS>()?;
    
    m.add_submodule(&nas)?;
    
    Ok(())
}

#[pyclass]
struct PyNeuralArchitectureSearch {
    max_epochs: usize,
    search_space: PyObject,
    best_architecture: Option<PyObject>,
    best_score: f64,
}

#[pymethods]
impl PyNeuralArchitectureSearch {
    #[new]
    fn new(search_space: PyObject, max_epochs: Option<usize>) -> Self {
        Self {
            max_epochs: max_epochs.unwrap_or(100),
            search_space,
            best_architecture: None,
            best_score: std::f64::NEG_INFINITY,
        }
    }
    
    fn search(&mut self, _train_fn: &PyAny) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            Err(PyValueError::new_err("Not implemented").into_py(py))
        })
    }
    
    fn get_best_architecture(&self) -> PyResult<Option<PyObject>> {
        match &self.best_architecture {
            Some(arch) => Ok(Some(arch.clone())),
            None => Ok(None),
        }
    }
}

#[pyclass]
struct PyENAS {
    base: PyNeuralArchitectureSearch,
    controller_hidden_size: usize,
    controller_temperature: f64,
    controller_tanh_constant: f64,
    controller_entropy_weight: f64,
}

#[pymethods]
impl PyENAS {
    #[new]
    fn new(
        search_space: PyObject,
        max_epochs: Option<usize>,
        controller_hidden_size: Option<usize>,
        controller_temperature: Option<f64>,
        controller_tanh_constant: Option<f64>,
        controller_entropy_weight: Option<f64>,
    ) -> Self {
        Self {
            base: PyNeuralArchitectureSearch::new(search_space, max_epochs),
            controller_hidden_size: controller_hidden_size.unwrap_or(64),
            controller_temperature: controller_temperature.unwrap_or(5.0),
            controller_tanh_constant: controller_tanh_constant.unwrap_or(2.5),
            controller_entropy_weight: controller_entropy_weight.unwrap_or(0.0001),
        }
    }
    
    fn search(&mut self, train_fn: &PyAny) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let controller = self._build_controller(py)?;
            let mut architectures = Vec::new();
            let mut rewards = Vec::new();
            
            for epoch in 0..self.base.max_epochs {
                let architecture = self._sample_architecture(py, &controller)?;
                
                let kwargs = PyDict::new(py);
                kwargs.set_item("architecture", &architecture)?;
                let reward = train_fn.call((), Some(kwargs))?.extract::<f64>()?;
                
                architectures.push(architecture.clone());
                rewards.push(reward);
                
                self._update_controller(py, &controller, &architectures, &rewards)?;
                
                if reward > self.base.best_score {
                    self.base.best_score = reward;
                    self.base.best_architecture = Some(architecture);
                }
                
                if epoch % 10 == 0 {
                    println!("ENAS Epoch {}: Best reward = {}", epoch, self.base.best_score);
                }
            }
            
            match &self.base.best_architecture {
                Some(arch) => Ok(arch.clone()),
                None => Err(PyValueError::new_err("No architecture found").into_py(py)),
            }
        })
    }
    
    fn get_best_architecture(&self) -> PyResult<Option<PyObject>> {
        self.base.get_best_architecture()
    }
}

impl PyENAS {
    fn _build_controller(&self, py: Python) -> PyResult<PyObject> {
        let controller = PyDict::new(py);
        controller.set_item("hidden_size", self.controller_hidden_size)?;
        controller.set_item("temperature", self.controller_temperature)?;
        controller.set_item("tanh_constant", self.controller_tanh_constant)?;
        controller.set_item("entropy_weight", self.controller_entropy_weight)?;
        
        Ok(controller.into())
    }
    
    fn _sample_architecture(&self, py: Python, _controller: &PyObject) -> PyResult<PyObject> {
        let mut rng = rand::thread_rng();
        
        let search_space = self.base.search_space.extract::<&PyDict>(py)?;
        let architecture = PyDict::new(py);
        
        for (key, value) in search_space.iter() {
            let key_str = key.extract::<String>()?;
            let options = value.extract::<Vec<PyObject>>()?;
            
            if !options.is_empty() {
                let idx = rng.gen_range(0..options.len());
                architecture.set_item(key_str, &options[idx])?;
            }
        }
        
        Ok(architecture.into())
    }
    
    fn _update_controller(&self, py: Python, _controller: &PyObject, 
                         _architectures: &[PyObject], _rewards: &[f64]) -> PyResult<()> {
        
        let rewards = _rewards.to_vec();
        let mean_reward = rewards.iter().sum::<f64>() / rewards.len() as f64;
        let std_reward = (rewards.iter().map(|r| (r - mean_reward).powi(2)).sum::<f64>() / rewards.len() as f64).sqrt();
        
        
        Ok(())
    }
}

#[pyclass]
struct PyDARTS {
    base: PyNeuralArchitectureSearch,
    unrolled: bool,
    alpha_lr: f64,
    alpha_weight_decay: f64,
}

#[pymethods]
impl PyDARTS {
    #[new]
    fn new(
        search_space: PyObject,
        max_epochs: Option<usize>,
        unrolled: Option<bool>,
        alpha_lr: Option<f64>,
        alpha_weight_decay: Option<f64>,
    ) -> Self {
        Self {
            base: PyNeuralArchitectureSearch::new(search_space, max_epochs),
            unrolled: unrolled.unwrap_or(true),
            alpha_lr: alpha_lr.unwrap_or(0.0003),
            alpha_weight_decay: alpha_weight_decay.unwrap_or(0.001),
        }
    }
    
    fn search(&mut self, train_fn: &PyAny) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let alphas = self._initialize_alphas(py)?;
            
            for epoch in 0..self.base.max_epochs {
                let kwargs = PyDict::new(py);
                kwargs.set_item("alphas", &alphas)?;
                kwargs.set_item("unrolled", self.unrolled)?;
                
                let result = train_fn.call((), Some(kwargs))?;
                let (trained_alphas, reward) = result.extract::<(PyObject, f64)>()?;
                
                let alphas = trained_alphas;
                
                let architecture = self._derive_architecture(py, &alphas)?;
                
                if reward > self.base.best_score {
                    self.base.best_score = reward;
                    self.base.best_architecture = Some(architecture.clone());
                }
                
                if epoch % 10 == 0 {
                    println!("DARTS Epoch {}: Best reward = {}", epoch, self.base.best_score);
                }
            }
            
            match &self.base.best_architecture {
                Some(arch) => Ok(arch.clone()),
                None => Err(PyValueError::new_err("No architecture found").into_py(py)),
            }
        })
    }
    
    fn get_best_architecture(&self) -> PyResult<Option<PyObject>> {
        self.base.get_best_architecture()
    }
}

impl PyDARTS {
    fn _initialize_alphas(&self, py: Python) -> PyResult<PyObject> {
        let search_space = self.base.search_space.extract::<&PyDict>(py)?;
        let alphas = PyDict::new(py);
        
        for (key, value) in search_space.iter() {
            let key_str = key.extract::<String>()?;
            let options = value.extract::<Vec<PyObject>>()?;
            
            if !options.is_empty() {
                let alpha_values = PyList::empty(py);
                for _ in 0..options.len() {
                    alpha_values.append(0.0)?;
                }
                alphas.set_item(key_str, alpha_values)?;
            }
        }
        
        Ok(alphas.into())
    }
    
    fn _derive_architecture(&self, py: Python, alphas: &PyObject) -> PyResult<PyObject> {
        let alphas_dict = alphas.extract::<&PyDict>(py)?;
        let search_space = self.base.search_space.extract::<&PyDict>(py)?;
        let architecture = PyDict::new(py);
        
        for (key, value) in alphas_dict.iter() {
            let key_str = key.extract::<String>()?;
            let alpha_values = value.extract::<Vec<f64>>()?;
            
            if !alpha_values.is_empty() {
                let max_idx = alpha_values.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                
                let options = search_space.get_item(key_str.clone())?.extract::<Vec<PyObject>>()?;
                if max_idx < options.len() {
                    architecture.set_item(key_str, &options[max_idx])?;
                }
            }
        }
        
        Ok(architecture.into())
    }
}

#[pyclass]
struct PyPNAS {
    base: PyNeuralArchitectureSearch,
    num_init_architectures: usize,
    num_expansions: usize,
    k_best: usize,
}

#[pymethods]
impl PyPNAS {
    #[new]
    fn new(
        search_space: PyObject,
        max_epochs: Option<usize>,
        num_init_architectures: Option<usize>,
        num_expansions: Option<usize>,
        k_best: Option<usize>,
    ) -> Self {
        Self {
            base: PyNeuralArchitectureSearch::new(search_space, max_epochs),
            num_init_architectures: num_init_architectures.unwrap_or(10),
            num_expansions: num_expansions.unwrap_or(5),
            k_best: k_best.unwrap_or(5),
        }
    }
    
    fn search(&mut self, train_fn: &PyAny) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut architectures = self._initialize_architectures(py)?;
            let mut rewards = Vec::new();
            
            for architecture in &architectures {
                let kwargs = PyDict::new(py);
                kwargs.set_item("architecture", architecture)?;
                let reward = train_fn.call((), Some(kwargs))?.extract::<f64>()?;
                rewards.push(reward);
                
                if reward > self.base.best_score {
                    self.base.best_score = reward;
                    self.base.best_architecture = Some(architecture.clone());
                }
            }
            
            for expansion in 0..self.num_expansions {
                let mut arch_rewards: Vec<(usize, f64)> = rewards.iter().enumerate().map(|(i, r)| (i, *r)).collect();
                arch_rewards.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
                
                let top_k_indices: Vec<usize> = arch_rewards.iter().take(self.k_best).map(|(i, _)| *i).collect();
                let top_k_architectures: Vec<PyObject> = top_k_indices.iter().map(|&i| architectures[i].clone()).collect();
                
                let expanded_architectures = self._expand_architectures(py, &top_k_architectures)?;
                
                architectures = expanded_architectures;
                rewards = Vec::new();
                
                for architecture in &architectures {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("architecture", architecture)?;
                    let reward = train_fn.call((), Some(kwargs))?.extract::<f64>()?;
                    rewards.push(reward);
                    
                    if reward > self.base.best_score {
                        self.base.best_score = reward;
                        self.base.best_architecture = Some(architecture.clone());
                    }
                }
                
                println!("PNAS Expansion {}: Best reward = {}", expansion, self.base.best_score);
            }
            
            match &self.base.best_architecture {
                Some(arch) => Ok(arch.clone()),
                None => Err(PyValueError::new_err("No architecture found").into_py(py)),
            }
        })
    }
    
    fn get_best_architecture(&self) -> PyResult<Option<PyObject>> {
        self.base.get_best_architecture()
    }
}

impl PyPNAS {
    fn _initialize_architectures(&self, py: Python) -> PyResult<Vec<PyObject>> {
        let mut rng = rand::thread_rng();
        let mut architectures = Vec::new();
        
        let search_space = self.base.search_space.extract::<&PyDict>(py)?;
        
        for _ in 0..self.num_init_architectures {
            let architecture = PyDict::new(py);
            
            for (key, value) in search_space.iter() {
                let key_str = key.extract::<String>()?;
                let options = value.extract::<Vec<PyObject>>()?;
                
                if !options.is_empty() {
                    let idx = rng.gen_range(0..options.len());
                    architecture.set_item(key_str, &options[idx])?;
                }
            }
            
            architectures.push(architecture.into());
        }
        
        Ok(architectures)
    }
    
    fn _expand_architectures(&self, py: Python, architectures: &[PyObject]) -> PyResult<Vec<PyObject>> {
        let mut rng = rand::thread_rng();
        let mut expanded_architectures = Vec::new();
        
        let search_space = self.base.search_space.extract::<&PyDict>(py)?;
        
        for architecture in architectures {
            let arch_dict = architecture.extract::<&PyDict>(py)?;
            
            expanded_architectures.push(architecture.clone());
            
            for _ in 0..2 {
                let new_arch = PyDict::new(py);
                
                for (key, value) in arch_dict.iter() {
                    new_arch.set_item(key, value)?;
                }
                
                let keys: Vec<String> = search_space.keys().map(|k| k.extract::<String>().unwrap()).collect();
                if !keys.is_empty() {
                    let key_idx = rng.gen_range(0..keys.len());
                    let key = &keys[key_idx];
                    
                    let options = search_space.get_item(key)?.extract::<Vec<PyObject>>()?;
                    if !options.is_empty() {
                        let idx = rng.gen_range(0..options.len());
                        new_arch.set_item(key, &options[idx])?;
                    }
                }
                
                expanded_architectures.push(new_arch.into());
            }
        }
        
        Ok(expanded_architectures)
    }
}
