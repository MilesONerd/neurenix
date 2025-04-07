
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::{HashMap, HashSet};
use rand::{Rng, thread_rng};
use crate::tensor::Tensor;
use crate::error::PhynexusError;
use super::neat::{NodeType, NodeGene, ConnectionGene, NEATGenome, NEATConfig, NEAT};

#[pyclass]
#[derive(Debug, Clone)]
pub struct Substrate {
    #[pyo3(get, set)]
    pub input_coords: Vec<Vec<f64>>,
    #[pyo3(get, set)]
    pub output_coords: Vec<Vec<f64>>,
    #[pyo3(get, set)]
    pub hidden_coords: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub dimensionality: usize,
}

#[pymethods]
impl Substrate {
    #[new]
    pub fn new(input_coords: Vec<Vec<f64>>, output_coords: Vec<Vec<f64>>, 
               hidden_coords: Option<Vec<Vec<f64>>>) -> PyResult<Self> {
        let hidden_coords = hidden_coords.unwrap_or_default();
        
        if input_coords.is_empty() || output_coords.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Input and output coordinates cannot be empty",
            ));
        }
        
        let dim = input_coords[0].len();
        
        for coords in &[&input_coords, &output_coords, &hidden_coords] {
            for coord in *coords {
                if coord.len() != dim {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        format!("All coordinates must have the same dimensionality: {}", dim),
                    ));
                }
            }
        }
        
        Ok(Self {
            input_coords,
            output_coords,
            hidden_coords,
            dimensionality: dim,
        })
    }
    
    pub fn get_connection_inputs(&self, source_idx: usize, target_idx: usize,
                                source_coords: Vec<f64>, target_coords: Vec<f64>) -> Vec<f64> {
        let mut inputs = Vec::new();
        inputs.extend_from_slice(&source_coords);
        inputs.extend_from_slice(&target_coords);
        
        inputs
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct CPPN {
    #[pyo3(get)]
    pub genome: NEATGenome,
}

#[pymethods]
impl CPPN {
    #[new]
    pub fn new(genome: NEATGenome) -> Self {
        Self { genome }
    }
    
    pub fn query_connection(&self, inputs: Vec<f64>) -> f64 {
        
        let mut sum = 0.0;
        let mut count = 0;
        
        for conn in self.genome.connections.values() {
            if conn.enabled {
                sum += conn.weight;
                count += 1;
            }
        }
        
        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }
}

#[pyclass]
#[derive(Debug)]
pub struct HyperNEATNetwork {
    #[pyo3(get)]
    pub substrate: Substrate,
    #[pyo3(get)]
    pub cppn: CPPN,
    #[pyo3(get, set)]
    pub weight_threshold: f64,
    #[pyo3(get, set)]
    pub activation: String,
    #[pyo3(get)]
    pub weights: HashMap<(usize, usize), f64>,
}

#[pymethods]
impl HyperNEATNetwork {
    #[new]
    pub fn new(substrate: Substrate, cppn: CPPN, 
              weight_threshold: Option<f64>, activation: Option<String>) -> PyResult<Self> {
        let weight_threshold = weight_threshold.unwrap_or(0.2);
        let activation = activation.unwrap_or_else(|| "sigmoid".to_string());
        
        let mut network = Self {
            substrate,
            cppn,
            weight_threshold,
            activation,
            weights: HashMap::new(),
        };
        
        network.generate_network()?;
        
        Ok(network)
    }
    
    fn generate_network(&mut self) -> PyResult<()> {
        self.weights.clear();
        
        if !self.substrate.hidden_coords.is_empty() {
            for (i, source_coord) in self.substrate.input_coords.iter().enumerate() {
                for (j, target_coord) in self.substrate.hidden_coords.iter().enumerate() {
                    let inputs = self.substrate.get_connection_inputs(
                        i, j + self.substrate.input_coords.len(),
                        source_coord.clone(), target_coord.clone(),
                    );
                    
                    let weight = self.cppn.query_connection(inputs);
                    
                    if weight.abs() > self.weight_threshold {
                        self.weights.insert((i, j + self.substrate.input_coords.len()), weight);
                    }
                }
            }
            
            for (i, source_coord) in self.substrate.hidden_coords.iter().enumerate() {
                for (j, target_coord) in self.substrate.output_coords.iter().enumerate() {
                    let inputs = self.substrate.get_connection_inputs(
                        i + self.substrate.input_coords.len(),
                        j + self.substrate.input_coords.len() + self.substrate.hidden_coords.len(),
                        source_coord.clone(), target_coord.clone(),
                    );
                    
                    let weight = self.cppn.query_connection(inputs);
                    
                    if weight.abs() > self.weight_threshold {
                        self.weights.insert((
                            i + self.substrate.input_coords.len(),
                            j + self.substrate.input_coords.len() + self.substrate.hidden_coords.len(),
                        ), weight);
                    }
                }
            }
        } else {
            for (i, source_coord) in self.substrate.input_coords.iter().enumerate() {
                for (j, target_coord) in self.substrate.output_coords.iter().enumerate() {
                    let inputs = self.substrate.get_connection_inputs(
                        i, j + self.substrate.input_coords.len(),
                        source_coord.clone(), target_coord.clone(),
                    );
                    
                    let weight = self.cppn.query_connection(inputs);
                    
                    if weight.abs() > self.weight_threshold {
                        self.weights.insert((i, j + self.substrate.input_coords.len()), weight);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    pub fn forward(&self, x: &Tensor) -> PyResult<Tensor> {
        
        let batch_size = x.shape()[0];
        let output_size = self.substrate.output_coords.len();
        
        let mut output = Tensor::zeros(&[batch_size, output_size]);
        
        for ((source, target), weight) in &self.weights {
            if *source < x.shape()[1] && *target >= self.substrate.input_coords.len() {
                let target_idx = target - self.substrate.input_coords.len();
                if target_idx < output_size {
                    for b in 0..batch_size {
                        let val = output.get(&[b, target_idx]) + x.get(&[b, *source]) * *weight;
                        output.set(&[b, target_idx], val);
                    }
                }
            }
        }
        
        match self.activation.as_str() {
            "sigmoid" => {
                for b in 0..batch_size {
                    for o in 0..output_size {
                        let val = 1.0 / (1.0 + (-output.get(&[b, o])).exp());
                        output.set(&[b, o], val);
                    }
                }
            }
            "tanh" => {
                for b in 0..batch_size {
                    for o in 0..output_size {
                        let val = output.get(&[b, o]).tanh();
                        output.set(&[b, o], val);
                    }
                }
            }
            "relu" => {
                for b in 0..batch_size {
                    for o in 0..output_size {
                        let val = output.get(&[b, o]).max(0.0);
                        output.set(&[b, o], val);
                    }
                }
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Unsupported activation function: {}", self.activation),
                ));
            }
        }
        
        Ok(output)
    }
}

#[pyclass]
pub struct HyperNEAT {
    #[pyo3(get)]
    pub substrate: Substrate,
    #[pyo3(get)]
    pub neat: NEAT,
    #[pyo3(get, set)]
    pub weight_threshold: f64,
    #[pyo3(get, set)]
    pub activation: String,
    #[pyo3(get, set)]
    pub best_network: Option<HyperNEATNetwork>,
}

#[pymethods]
impl HyperNEAT {
    #[new]
    pub fn new(substrate: Substrate, neat_config: Option<NEATConfig>,
              weight_threshold: Option<f64>, activation: Option<String>) -> Self {
        Self {
            substrate,
            neat: NEAT::new(neat_config),
            weight_threshold: weight_threshold.unwrap_or(0.2),
            activation: activation.unwrap_or_else(|| "sigmoid".to_string()),
            best_network: None,
        }
    }
    
    pub fn initialize(&mut self, population_size: usize) -> PyResult<()> {
        let cppn_inputs = self.substrate.dimensionality * 2;  // Source and target coordinates
        let cppn_outputs = 1;  // Weight
        
        self.neat.initialize(population_size, cppn_inputs, cppn_outputs)
    }
    
    pub fn evolve(&mut self, py: Python, fitness_function: PyObject, 
                 generations: Option<usize>, callback: Option<PyObject>) -> PyResult<()> {
        let generations = generations.unwrap_or(100);
        
        for generation in 0..generations {
            let mut genomes = Vec::new();
            for genome in &self.neat.genomes {
                genomes.push(genome.clone());
            }
            
            let mut networks = Vec::new();
            for genome in &genomes {
                let cppn = CPPN::new(genome.clone());
                let network = HyperNEATNetwork::new(
                    self.substrate.clone(),
                    cppn,
                    Some(self.weight_threshold),
                    Some(self.activation.clone()),
                )?;
                networks.push(network);
            }
            
            let networks_list = PyList::new(py, &networks);
            let fitness_values = fitness_function.call1(py, (networks_list,))?;
            let fitness_values: Vec<f64> = fitness_values.extract(py)?;
            
            if fitness_values.len() != genomes.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Fitness function must return a list of fitness values with the same length as the population",
                ));
            }
            
            for (i, genome) in genomes.iter_mut().enumerate() {
                genome.fitness = fitness_values[i];
            }
            
            self.neat.genomes = genomes;
            
            let neat_fitness_function = |genome: &NEATGenome| -> f64 {
                genome.fitness
            };
            
            self.neat.evolve(neat_fitness_function)?;
            
            if let Some(ref callback) = callback {
                let networks_list = PyList::new(py, &networks);
                callback.call1(py, (generation, networks_list))?;
            }
        }
        
        if let Some(best_genome) = self.neat.get_best_genome() {
            let cppn = CPPN::new(best_genome);
            self.best_network = Some(HyperNEATNetwork::new(
                self.substrate.clone(),
                cppn,
                Some(self.weight_threshold),
                Some(self.activation.clone()),
            )?);
        }
        
        Ok(())
    }
    
    pub fn get_best_network(&self) -> Option<HyperNEATNetwork> {
        self.best_network.clone()
    }
}

pub fn register_hyperneat(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "hyperneat")?;
    
    submodule.add_class::<Substrate>()?;
    submodule.add_class::<CPPN>()?;
    submodule.add_class::<HyperNEATNetwork>()?;
    submodule.add_class::<HyperNEAT>()?;
    
    m.add_submodule(submodule)?;
    
    Ok(())
}
