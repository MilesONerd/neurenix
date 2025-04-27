
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::{HashMap, HashSet};
use rand::{Rng, thread_rng};
use crate::tensor::Tensor;
use crate::error::PhynexusError;

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Input = 0,
    Hidden = 1,
    Output = 2,
    Bias = 3,
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct NodeGene {
    #[pyo3(get, set)]
    pub id: usize,
    #[pyo3(get, set)]
    pub node_type: NodeType,
    #[pyo3(get, set)]
    pub activation: String,
    #[pyo3(get, set)]
    pub response: f64,
    #[pyo3(get, set)]
    pub bias: f64,
}

#[pymethods]
impl NodeGene {
    #[new]
    pub fn new(id: usize, node_type: NodeType, activation: &str) -> Self {
        let bias = if node_type == NodeType::Bias { 1.0 } else { 0.0 };
        
        Self {
            id,
            node_type,
            activation: activation.to_string(),
            response: 1.0,
            bias,
        }
    }
    
    pub fn copy(&self) -> Self {
        Self {
            id: self.id,
            node_type: self.node_type,
            activation: self.activation.clone(),
            response: self.response,
            bias: self.bias,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct ConnectionGene {
    #[pyo3(get, set)]
    pub input: usize,
    #[pyo3(get, set)]
    pub output: usize,
    #[pyo3(get, set)]
    pub weight: f64,
    #[pyo3(get, set)]
    pub innovation: usize,
    #[pyo3(get, set)]
    pub enabled: bool,
}

#[pymethods]
impl ConnectionGene {
    #[new]
    pub fn new(input: usize, output: usize, weight: f64, innovation: usize, enabled: bool) -> Self {
        Self {
            input,
            output,
            weight,
            innovation,
            enabled,
        }
    }
    
    pub fn copy(&self) -> Self {
        Self {
            input: self.input,
            output: self.output,
            weight: self.weight,
            innovation: self.innovation,
            enabled: self.enabled,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct NEATGenome {
    #[pyo3(get)]
    pub nodes: HashMap<usize, NodeGene>,
    #[pyo3(get)]
    pub connections: HashMap<(usize, usize), ConnectionGene>,
    #[pyo3(get, set)]
    pub fitness: f64,
    #[pyo3(get, set)]
    pub adjusted_fitness: f64,
    #[pyo3(get, set)]
    pub species_id: Option<usize>,
}

#[pymethods]
impl NEATGenome {
    #[new]
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            connections: HashMap::new(),
            fitness: 0.0,
            adjusted_fitness: 0.0,
            species_id: None,
        }
    }
    
    pub fn add_node(&mut self, node: NodeGene) {
        self.nodes.insert(node.id, node);
    }
    
    pub fn add_connection(&mut self, conn: ConnectionGene) {
        self.connections.insert((conn.input, conn.output), conn);
    }
    
    pub fn mutate_weight(&mut self, perturb_prob: f64, perturb_amount: f64) {
        let mut rng = thread_rng();
        
        for conn in self.connections.values_mut() {
            if rng.gen::<f64>() < perturb_prob {
                conn.weight += rng.gen_range(-perturb_amount..perturb_amount);
                conn.weight = conn.weight.max(-8.0).min(8.0);
            } else {
                conn.weight = rng.gen_range(-4.0..4.0);
            }
        }
    }
    
    pub fn copy(&self) -> Self {
        let mut new_genome = NEATGenome::new();
        
        for (node_id, node) in &self.nodes {
            new_genome.nodes.insert(*node_id, node.copy());
        }
        
        for (conn_key, conn) in &self.connections {
            new_genome.connections.insert(*conn_key, conn.copy());
        }
        
        new_genome.fitness = self.fitness;
        new_genome.adjusted_fitness = self.adjusted_fitness;
        new_genome.species_id = self.species_id;
        
        new_genome
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct NEATConfig {
    #[pyo3(get, set)]
    pub compatibility_threshold: f64,
    #[pyo3(get, set)]
    pub excess_coefficient: f64,
    #[pyo3(get, set)]
    pub weight_coefficient: f64,
    #[pyo3(get, set)]
    pub species_elitism: f64,
    #[pyo3(get, set)]
    pub species_stagnation_threshold: usize,
    #[pyo3(get, set)]
    pub min_species_count: usize,
    #[pyo3(get, set)]
    pub weight_mutation_rate: f64,
    #[pyo3(get, set)]
    pub weight_perturb_prob: f64,
    #[pyo3(get, set)]
    pub weight_perturb_amount: f64,
    #[pyo3(get, set)]
    pub connection_toggle_rate: f64,
    #[pyo3(get, set)]
    pub node_activation_mutation_rate: f64,
    #[pyo3(get, set)]
    pub node_bias_mutation_rate: f64,
    #[pyo3(get, set)]
    pub bias_perturb_amount: f64,
    #[pyo3(get, set)]
    pub add_node_mutation_rate: f64,
    #[pyo3(get, set)]
    pub add_connection_mutation_rate: f64,
    #[pyo3(get, set)]
    pub asexual_reproduction_rate: f64,
    #[pyo3(get, set)]
    pub activation_functions: Vec<String>,
}

#[pymethods]
impl NEATConfig {
    #[new]
    fn new() -> Self {
        Self {
            compatibility_threshold: 3.0,
            excess_coefficient: 1.0,
            weight_coefficient: 0.4,
            species_elitism: 0.2,
            species_stagnation_threshold: 15,
            min_species_count: 2,
            weight_mutation_rate: 0.8,
            weight_perturb_prob: 0.9,
            weight_perturb_amount: 0.5,
            connection_toggle_rate: 0.1,
            node_activation_mutation_rate: 0.1,
            node_bias_mutation_rate: 0.1,
            bias_perturb_amount: 0.5,
            add_node_mutation_rate: 0.03,
            add_connection_mutation_rate: 0.05,
            asexual_reproduction_rate: 0.25,
            activation_functions: vec!["sigmoid".to_string(), "tanh".to_string(), "relu".to_string()],
        }
    }
}

#[pyclass]
#[derive(Debug)]
pub struct NEATSpecies {
    #[pyo3(get)]
    pub id: usize,
    #[pyo3(get)]
    pub members: Vec<NEATGenome>,
    #[pyo3(get, set)]
    pub representative: Option<NEATGenome>,
    #[pyo3(get)]
    pub fitness_history: Vec<f64>,
    #[pyo3(get, set)]
    pub stagnation: usize,
    #[pyo3(get, set)]
    pub max_fitness: f64,
    #[pyo3(get, set)]
    pub adjusted_fitness_sum: f64,
}

#[pymethods]
impl NEATSpecies {
    #[new]
    pub fn new(id: usize) -> Self {
        Self {
            id,
            members: Vec::new(),
            representative: None,
            fitness_history: Vec::new(),
            stagnation: 0,
            max_fitness: f64::NEG_INFINITY,
            adjusted_fitness_sum: 0.0,
        }
    }
    
    pub fn add(&mut self, mut genome: NEATGenome) {
        genome.species_id = Some(self.id);
        self.members.push(genome);
    }
    
    pub fn update_representative(&mut self) {
        if !self.members.is_empty() {
            let mut rng = thread_rng();
            let idx = rng.gen_range(0..self.members.len());
            self.representative = Some(self.members[idx].copy());
        }
    }
}

#[pyclass]
pub struct NEAT {
    #[pyo3(get)]
    pub config: NEATConfig,
    #[pyo3(get)]
    pub species: HashMap<usize, NEATSpecies>,
    #[pyo3(get)]
    pub genomes: Vec<NEATGenome>,
    #[pyo3(get)]
    pub generation: usize,
    #[pyo3(get)]
    pub innovation_history: HashMap<(usize, usize), usize>,
    #[pyo3(get, set)]
    pub node_innovation: usize,
    #[pyo3(get, set)]
    pub connection_innovation: usize,
    #[pyo3(get, set)]
    pub best_genome: Option<NEATGenome>,
    #[pyo3(get, set)]
    pub best_fitness: f64,
    #[pyo3(get, set)]
    pub species_counter: usize,
}

#[pymethods]
impl NEAT {
    #[new]
    pub fn new(config: Option<NEATConfig>) -> Self {
        Self {
            config: config.unwrap_or_else(NEATConfig::new),
            species: HashMap::new(),
            genomes: Vec::new(),
            generation: 0,
            innovation_history: HashMap::new(),
            node_innovation: 0,
            connection_innovation: 0,
            best_genome: None,
            best_fitness: f64::NEG_INFINITY,
            species_counter: 0,
        }
    }
    
    pub fn initialize(&mut self, population_size: usize, num_inputs: usize, num_outputs: usize) -> PyResult<()> {
        self.genomes.clear();
        
        for _ in 0..population_size {
            let genome = self.create_minimal_genome(num_inputs, num_outputs)?;
            self.genomes.push(genome);
        }
        
        self.speciate()?;
        
        Ok(())
    }
    
    fn create_minimal_genome(&mut self, num_inputs: usize, num_outputs: usize) -> PyResult<NEATGenome> {
        let mut rng = thread_rng();
        let mut genome = NEATGenome::new();
        
        for i in 0..num_inputs {
            let node = NodeGene::new(i, NodeType::Input, "sigmoid");
            genome.add_node(node);
            self.node_innovation = self.node_innovation.max(i);
        }
        
        let bias_id = self.node_innovation + 1;
        let bias_node = NodeGene::new(bias_id, NodeType::Bias, "sigmoid");
        genome.add_node(bias_node);
        self.node_innovation = bias_id;
        
        let mut output_ids = Vec::new();
        for i in 0..num_outputs {
            let node_id = self.node_innovation + 1 + i;
            let node = NodeGene::new(node_id, NodeType::Output, "sigmoid");
            genome.add_node(node);
            output_ids.push(node_id);
            self.node_innovation = node_id;
        }
        
        for input_id in 0..num_inputs {
            for &output_id in &output_ids {
                let key = (input_id, output_id);
                let innovation = if let Some(&innovation) = self.innovation_history.get(&key) {
                    innovation
                } else {
                    self.connection_innovation += 1;
                    let innovation = self.connection_innovation;
                    self.innovation_history.insert(key, innovation);
                    innovation
                };
                
                let weight = rng.gen_range(-2.0..2.0);
                let conn = ConnectionGene::new(input_id, output_id, weight, innovation, true);
                genome.add_connection(conn);
            }
        }
        
        for &output_id in &output_ids {
            let key = (bias_id, output_id);
            let innovation = if let Some(&innovation) = self.innovation_history.get(&key) {
                innovation
            } else {
                self.connection_innovation += 1;
                let innovation = self.connection_innovation;
                self.innovation_history.insert(key, innovation);
                innovation
            };
            
            let weight = rng.gen_range(-2.0..2.0);
            let conn = ConnectionGene::new(bias_id, output_id, weight, innovation, true);
            genome.add_connection(conn);
        }
        
        Ok(genome)
    }
    
    fn speciate(&mut self) -> PyResult<()> {
        for species in self.species.values_mut() {
            species.members.clear();
        }
        
        for genome in self.genomes.iter().cloned() {
            let mut found_species = false;
            
            for species in self.species.values_mut() {
                if let Some(ref representative) = species.representative {
                    if self.is_compatible(&genome, representative)? {
                        species.add(genome.clone());
                        found_species = true;
                        break;
                    }
                }
            }
            
            if !found_species {
                self.species_counter += 1;
                let mut new_species = NEATSpecies::new(self.species_counter);
                new_species.add(genome.clone());
                new_species.representative = Some(genome);
                self.species.insert(self.species_counter, new_species);
            }
        }
        
        self.species.retain(|_, species| !species.members.is_empty());
        
        Ok(())
    }
    
    fn is_compatible(&self, genome1: &NEATGenome, genome2: &NEATGenome) -> PyResult<bool> {
        let mut disjoint_excess = 0;
        let mut matching = 0;
        let mut weight_diff = 0.0;
        
        let all_innovations_1: HashSet<_> = genome1.connections.values().map(|c| c.innovation).collect();
        let all_innovations_2: HashSet<_> = genome2.connections.values().map(|c| c.innovation).collect();
        
        let max_innovation_1 = all_innovations_1.iter().max().copied().unwrap_or(0);
        let max_innovation_2 = all_innovations_2.iter().max().copied().unwrap_or(0);
        
        for conn in genome1.connections.values() {
            if all_innovations_2.contains(&conn.innovation) {
                matching += 1;
                let other_conn = genome2.connections.values()
                    .find(|c| c.innovation == conn.innovation)
                    .unwrap();
                weight_diff += (conn.weight - other_conn.weight).abs();
            } else if conn.innovation <= max_innovation_2 {
                disjoint_excess += 1;
            } else {
                disjoint_excess += 1;
            }
        }
        
        for conn in genome2.connections.values() {
            if !all_innovations_1.contains(&conn.innovation) {
                if conn.innovation <= max_innovation_1 {
                    disjoint_excess += 1;
                } else {
                    disjoint_excess += 1;
                }
            }
        }
        
        let n = genome1.connections.len().max(genome2.connections.len());
        let n = if n < 1 { 1 } else { n };  // Avoid division by zero
        
        let weight_diff = if matching > 0 { weight_diff / matching as f64 } else { 0.0 };
        
        let compatibility = self.config.excess_coefficient * disjoint_excess as f64 / n as f64 +
                           self.config.weight_coefficient * weight_diff;
        
        Ok(compatibility < self.config.compatibility_threshold)
    }
    
    pub fn get_best_genome(&self) -> Option<NEATGenome> {
        self.best_genome.clone()
    }
}

pub fn register_neat(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py, "neat")?;
    
    submodule.add_class::<NodeType>()?;
    submodule.add_class::<NodeGene>()?;
    submodule.add_class::<ConnectionGene>()?;
    submodule.add_class::<NEATGenome>()?;
    submodule.add_class::<NEATConfig>()?;
    submodule.add_class::<NEATSpecies>()?;
    submodule.add_class::<NEAT>()?;
    
    m.add_submodule(&submodule)?;
    
    Ok(())
}
