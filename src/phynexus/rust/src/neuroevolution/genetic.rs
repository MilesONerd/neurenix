
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;
use rand::{Rng, thread_rng};
use rand::distributions::{Distribution, Uniform};
use crate::tensor::Tensor;
use crate::error::PhynexusError;

pub struct GeneticAlgorithm {
    population_size: usize,
    genome_size: usize,
    mutation_rate: f64,
    crossover_rate: f64,
    selection_pressure: f64,
    elitism_count: usize,
    population: Vec<Individual>,
    best_fitness: f64,
    best_genome: Vec<f64>,
    generation: usize,
}

pub struct Individual {
    genome: Vec<f64>,
    fitness: f64,
}

impl Individual {
    pub fn new(genome_size: usize) -> Self {
        let mut rng = thread_rng();
        let dist = Uniform::from(-1.0..1.0);
        
        let genome = (0..genome_size)
            .map(|_| dist.sample(&mut rng))
            .collect();
            
        Self {
            genome,
            fitness: 0.0,
        }
    }
    
    pub fn with_genome(genome: Vec<f64>) -> Self {
        Self {
            genome,
            fitness: 0.0,
        }
    }
    
    pub fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }
    
    pub fn fitness(&self) -> f64 {
        self.fitness
    }
    
    pub fn genome(&self) -> &Vec<f64> {
        &self.genome
    }
    
    pub fn mutate(&mut self, mutation_rate: f64, mutation_range: f64) {
        let mut rng = thread_rng();
        
        for gene in &mut self.genome {
            if rng.gen::<f64>() < mutation_rate {
                *gene += rng.gen_range(-mutation_range..mutation_range);
                
                *gene = gene.max(-5.0).min(5.0);
            }
        }
    }
    
    pub fn crossover(&self, other: &Individual) -> (Individual, Individual) {
        let mut rng = thread_rng();
        let crossover_point = rng.gen_range(0..self.genome.len());
        
        let mut child1_genome = Vec::with_capacity(self.genome.len());
        let mut child2_genome = Vec::with_capacity(self.genome.len());
        
        for i in 0..self.genome.len() {
            if i < crossover_point {
                child1_genome.push(self.genome[i]);
                child2_genome.push(other.genome[i]);
            } else {
                child1_genome.push(other.genome[i]);
                child2_genome.push(self.genome[i]);
            }
        }
        
        (
            Individual::with_genome(child1_genome),
            Individual::with_genome(child2_genome),
        )
    }
}

impl GeneticAlgorithm {
    pub fn new(
        population_size: usize,
        genome_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        selection_pressure: f64,
        elitism_count: usize,
    ) -> Self {
        let population = (0..population_size)
            .map(|_| Individual::new(genome_size))
            .collect();
            
        Self {
            population_size,
            genome_size,
            mutation_rate,
            crossover_rate,
            selection_pressure,
            elitism_count,
            population,
            best_fitness: f64::NEG_INFINITY,
            best_genome: vec![0.0; genome_size],
            generation: 0,
        }
    }
    
    pub fn initialize(&mut self) {
        self.population = (0..self.population_size)
            .map(|_| Individual::new(self.genome_size))
            .collect();
    }
    
    pub fn evaluate<F>(&mut self, fitness_function: F)
    where
        F: Fn(&Vec<f64>) -> f64,
    {
        for individual in &mut self.population {
            let fitness = fitness_function(&individual.genome());
            individual.set_fitness(fitness);
            
            if fitness > self.best_fitness {
                self.best_fitness = fitness;
                self.best_genome = individual.genome().clone();
            }
        }
    }
    
    pub fn select(&self) -> &Individual {
        let mut rng = thread_rng();
        let tournament_size = (self.selection_pressure * self.population_size as f64) as usize;
        let tournament_size = tournament_size.max(2).min(self.population_size);
        
        let mut best_individual = &self.population[rng.gen_range(0..self.population_size)];
        
        for _ in 1..tournament_size {
            let individual = &self.population[rng.gen_range(0..self.population_size)];
            if individual.fitness() > best_individual.fitness() {
                best_individual = individual;
            }
        }
        
        best_individual
    }
    
    pub fn evolve<F>(&mut self, fitness_function: F)
    where
        F: Fn(&Vec<f64>) -> f64,
    {
        self.evaluate(&fitness_function);
        
        self.population.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());
        
        let mut new_population = Vec::with_capacity(self.population_size);
        
        for i in 0..self.elitism_count.min(self.population_size) {
            new_population.push(Individual::with_genome(self.population[i].genome().clone()));
        }
        
        while new_population.len() < self.population_size {
            let parent1 = self.select();
            let parent2 = self.select();
            
            let mut rng = thread_rng();
            
            if rng.gen::<f64>() < self.crossover_rate {
                let (mut child1, mut child2) = parent1.crossover(parent2);
                
                child1.mutate(self.mutation_rate, 0.5);
                child2.mutate(self.mutation_rate, 0.5);
                
                new_population.push(child1);
                if new_population.len() < self.population_size {
                    new_population.push(child2);
                }
            } else {
                let mut child1 = Individual::with_genome(parent1.genome().clone());
                let mut child2 = Individual::with_genome(parent2.genome().clone());
                
                child1.mutate(self.mutation_rate, 0.5);
                child2.mutate(self.mutation_rate, 0.5);
                
                new_population.push(child1);
                if new_population.len() < self.population_size {
                    new_population.push(child2);
                }
            }
        }
        
        self.population = new_population;
        self.generation += 1;
    }
    
    pub fn best_individual(&self) -> Option<&Individual> {
        self.population.iter().max_by(|a, b| a.fitness().partial_cmp(&b.fitness()).unwrap())
    }
    
    pub fn best_genome(&self) -> &Vec<f64> {
        &self.best_genome
    }
    
    pub fn best_fitness(&self) -> f64 {
        self.best_fitness
    }
    
    pub fn generation(&self) -> usize {
        self.generation
    }
}

#[pymodule]
pub fn genetic(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGeneticAlgorithm>()?;
    m.add_class::<PyIndividual>()?;
    
    Ok(())
}

#[pyclass]
struct PyIndividual {
    inner: Individual,
}

#[pymethods]
impl PyIndividual {
    #[new]
    fn new(genome: Vec<f64>) -> Self {
        Self {
            inner: Individual::with_genome(genome),
        }
    }
    
    #[staticmethod]
    fn random(genome_size: usize) -> Self {
        Self {
            inner: Individual::new(genome_size),
        }
    }
    
    #[getter]
    fn genome(&self) -> Vec<f64> {
        self.inner.genome().clone()
    }
    
    #[getter]
    fn fitness(&self) -> f64 {
        self.inner.fitness()
    }
    
    #[setter]
    fn set_fitness(&mut self, fitness: f64) {
        self.inner.set_fitness(fitness);
    }
    
    fn mutate(&mut self, mutation_rate: f64, mutation_range: f64) {
        self.inner.mutate(mutation_rate, mutation_range);
    }
    
    fn crossover(&self, other: &PyIndividual) -> (PyIndividual, PyIndividual) {
        let (child1, child2) = self.inner.crossover(&other.inner);
        (
            PyIndividual { inner: child1 },
            PyIndividual { inner: child2 },
        )
    }
}

#[pyclass]
struct PyGeneticAlgorithm {
    inner: GeneticAlgorithm,
}

#[pymethods]
impl PyGeneticAlgorithm {
    #[new]
    fn new(
        population_size: usize,
        genome_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        selection_pressure: f64,
        elitism_count: usize,
    ) -> Self {
        Self {
            inner: GeneticAlgorithm::new(
                population_size,
                genome_size,
                mutation_rate,
                crossover_rate,
                selection_pressure,
                elitism_count,
            ),
        }
    }
    
    fn initialize(&mut self) {
        self.inner.initialize();
    }
    
    fn evaluate(&mut self, py: Python, fitness_function: PyObject) -> PyResult<()> {
        let population_genomes: Vec<Vec<f64>> = self.inner.population
            .iter()
            .map(|ind| ind.genome().clone())
            .collect();
            
        let population_list = PyList::new(py, &population_genomes);
        let fitness_values = fitness_function.call1(py, (population_list,))?;
        let fitness_values: Vec<f64> = fitness_values.extract(py)?;
        
        if fitness_values.len() != self.inner.population.len() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Fitness function must return a list of fitness values with the same length as the population",
            ));
        }
        
        for (individual, fitness) in self.inner.population.iter_mut().zip(fitness_values) {
            individual.set_fitness(fitness);
            
            if fitness > self.inner.best_fitness {
                self.inner.best_fitness = fitness;
                self.inner.best_genome = individual.genome().clone();
            }
        }
        
        Ok(())
    }
    
    fn evolve(&mut self, py: Python, fitness_function: PyObject) -> PyResult<()> {
        self.evaluate(py, fitness_function.clone())?;
        
        self.inner.population.sort_by(|a, b| b.fitness().partial_cmp(&a.fitness()).unwrap());
        
        let mut new_population = Vec::with_capacity(self.inner.population_size);
        
        for i in 0..self.inner.elitism_count.min(self.inner.population_size) {
            new_population.push(Individual::with_genome(self.inner.population[i].genome().clone()));
        }
        
        let mut rng = thread_rng();
        while new_population.len() < self.inner.population_size {
            let parent1_idx = self.tournament_selection();
            let parent2_idx = self.tournament_selection();
            
            let parent1 = &self.inner.population[parent1_idx];
            let parent2 = &self.inner.population[parent2_idx];
            
            if rng.gen::<f64>() < self.inner.crossover_rate {
                let (mut child1, mut child2) = parent1.crossover(parent2);
                
                child1.mutate(self.inner.mutation_rate, 0.5);
                child2.mutate(self.inner.mutation_rate, 0.5);
                
                new_population.push(child1);
                if new_population.len() < self.inner.population_size {
                    new_population.push(child2);
                }
            } else {
                let mut child1 = Individual::with_genome(parent1.genome().clone());
                let mut child2 = Individual::with_genome(parent2.genome().clone());
                
                child1.mutate(self.inner.mutation_rate, 0.5);
                child2.mutate(self.inner.mutation_rate, 0.5);
                
                new_population.push(child1);
                if new_population.len() < self.inner.population_size {
                    new_population.push(child2);
                }
            }
        }
        
        self.inner.population = new_population;
        self.inner.generation += 1;
        
        Ok(())
    }
    
    fn tournament_selection(&self) -> usize {
        let mut rng = thread_rng();
        let tournament_size = (self.inner.selection_pressure * self.inner.population_size as f64) as usize;
        let tournament_size = tournament_size.max(2).min(self.inner.population_size);
        
        let mut best_idx = rng.gen_range(0..self.inner.population_size);
        let mut best_fitness = self.inner.population[best_idx].fitness();
        
        for _ in 1..tournament_size {
            let idx = rng.gen_range(0..self.inner.population_size);
            let fitness = self.inner.population[idx].fitness();
            
            if fitness > best_fitness {
                best_idx = idx;
                best_fitness = fitness;
            }
        }
        
        best_idx
    }
    
    #[getter]
    fn best_genome(&self) -> Vec<f64> {
        self.inner.best_genome().clone()
    }
    
    #[getter]
    fn best_fitness(&self) -> f64 {
        self.inner.best_fitness()
    }
    
    #[getter]
    fn generation(&self) -> usize {
        self.inner.generation()
    }
    
    #[getter]
    fn population(&self, py: Python) -> PyResult<PyObject> {
        let population = PyList::empty(py);
        
        for individual in &self.inner.population {
            let py_individual = PyIndividual {
                inner: Individual::with_genome(individual.genome().clone()),
            };
            py_individual.inner.set_fitness(individual.fitness());
            
            population.append(Py::new(py, py_individual)?.into_py(py))?;
        }
        
        Ok(population.into())
    }
}

pub fn register_genetic(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py, "genetic")?;
    
    submodule.add_class::<PyGeneticAlgorithm>()?;
    submodule.add_class::<PyIndividual>()?;
    
    m.add_submodule(&submodule)?;
    
    Ok(())
}
