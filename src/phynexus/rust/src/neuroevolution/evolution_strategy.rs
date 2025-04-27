
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;
use rand::{Rng, thread_rng};
use ndarray::{Array1, Array2};
use crate::tensor::Tensor;
use crate::error::PhynexusError;

#[pyclass]
#[derive(Debug, Clone)]
pub struct ESConfig {
    #[pyo3(get, set)]
    pub population_size: usize,
    #[pyo3(get, set)]
    pub sigma: f64,
    #[pyo3(get, set)]
    pub learning_rate: f64,
    #[pyo3(get, set)]
    pub decay: f64,
    #[pyo3(get, set)]
    pub noise_std: f64,
    #[pyo3(get, set)]
    pub weight_decay: f64,
    #[pyo3(get, set)]
    pub antithetic: bool,
    #[pyo3(get, set)]
    pub rank_based: bool,
    #[pyo3(get, set)]
    pub normalize_observations: bool,
    #[pyo3(get, set)]
    pub normalize_updates: bool,
    #[pyo3(get, set)]
    pub adam: bool,
    #[pyo3(get, set)]
    pub adam_beta1: f64,
    #[pyo3(get, set)]
    pub adam_beta2: f64,
    #[pyo3(get, set)]
    pub adam_epsilon: f64,
}

#[pymethods]
impl ESConfig {
    #[new]
    pub fn new(
        population_size: Option<usize>,
        sigma: Option<f64>,
        learning_rate: Option<f64>,
        decay: Option<f64>,
        noise_std: Option<f64>,
        weight_decay: Option<f64>,
        antithetic: Option<bool>,
        rank_based: Option<bool>,
        normalize_observations: Option<bool>,
        normalize_updates: Option<bool>,
        adam: Option<bool>,
        adam_beta1: Option<f64>,
        adam_beta2: Option<f64>,
        adam_epsilon: Option<f64>,
    ) -> Self {
        Self {
            population_size: population_size.unwrap_or(100),
            sigma: sigma.unwrap_or(0.1),
            learning_rate: learning_rate.unwrap_or(0.01),
            decay: decay.unwrap_or(0.999),
            noise_std: noise_std.unwrap_or(0.01),
            weight_decay: weight_decay.unwrap_or(0.0),
            antithetic: antithetic.unwrap_or(true),
            rank_based: rank_based.unwrap_or(true),
            normalize_observations: normalize_observations.unwrap_or(true),
            normalize_updates: normalize_updates.unwrap_or(true),
            adam: adam.unwrap_or(true),
            adam_beta1: adam_beta1.unwrap_or(0.9),
            adam_beta2: adam_beta2.unwrap_or(0.999),
            adam_epsilon: adam_epsilon.unwrap_or(1e-8),
        }
    }
}

#[pyclass]
#[derive(Debug)]
pub struct ESPopulation {
    #[pyo3(get)]
    pub config: ESConfig,
    #[pyo3(get)]
    pub dimension: usize,
    #[pyo3(get)]
    pub population_size: usize,
    #[pyo3(get)]
    pub half_popsize: usize,
    #[pyo3(get)]
    pub solutions: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub fitnesses: Vec<f64>,
    #[pyo3(get, set)]
    pub best_fitness: f64,
    #[pyo3(get)]
    pub best_solution: Vec<f64>,
}

#[pymethods]
impl ESPopulation {
    #[new]
    pub fn new(config: ESConfig, dimension: usize) -> Self {
        let population_size = config.population_size;
        let half_popsize = if config.antithetic {
            assert!(population_size % 2 == 0, "Population size must be even for antithetic sampling");
            population_size / 2
        } else {
            population_size
        };
        
        Self {
            config,
            dimension,
            population_size,
            half_popsize,
            solutions: Vec::new(),
            fitnesses: Vec::new(),
            best_fitness: f64::NEG_INFINITY,
            best_solution: vec![0.0; dimension],
        }
    }
    
    pub fn sample(&mut self, mean: Vec<f64>) -> Vec<Vec<f64>> {
        let mut rng = thread_rng();
        self.solutions.clear();
        
        if self.config.antithetic {
            for _ in 0..self.half_popsize {
                let mut noise = Vec::with_capacity(self.dimension);
                for _ in 0..self.dimension {
                    noise.push(rng.sample(rand_distr::StandardNormal));
                }
                
                let mut solution_pos = Vec::with_capacity(self.dimension);
                for i in 0..self.dimension {
                    solution_pos.push(mean[i] + self.config.sigma * noise[i]);
                }
                self.solutions.push(solution_pos);
                
                let mut solution_neg = Vec::with_capacity(self.dimension);
                for i in 0..self.dimension {
                    solution_neg.push(mean[i] - self.config.sigma * noise[i]);
                }
                self.solutions.push(solution_neg);
            }
        } else {
            for _ in 0..self.population_size {
                let mut solution = Vec::with_capacity(self.dimension);
                for i in 0..self.dimension {
                    let noise: f64 = rng.sample(rand_distr::StandardNormal);
                    solution.push(mean[i] + self.config.sigma * noise);
                }
                self.solutions.push(solution);
            }
        }
        
        self.solutions.clone()
    }
    
    pub fn update(&mut self, fitnesses: Vec<f64>) -> (Vec<f64>, f64) {
        self.fitnesses = fitnesses;
        
        let mut best_idx = 0;
        let mut best_fitness = self.fitnesses[0];
        
        for (i, &fitness) in self.fitnesses.iter().enumerate().skip(1) {
            if fitness > best_fitness {
                best_idx = i;
                best_fitness = fitness;
            }
        }
        
        if best_fitness > self.best_fitness {
            self.best_fitness = best_fitness;
            self.best_solution = self.solutions[best_idx].clone();
        }
        
        (self.best_solution.clone(), self.best_fitness)
    }
    
    pub fn get_best(&self) -> (Vec<f64>, f64) {
        (self.best_solution.clone(), self.best_fitness)
    }
}

#[pyclass]
pub struct EvolutionStrategy {
    #[pyo3(get)]
    pub dimension: usize,
    #[pyo3(get, set)]
    pub mean: Vec<f64>,
    #[pyo3(get)]
    pub config: ESConfig,
    #[pyo3(get)]
    pub population: ESPopulation,
    #[pyo3(get, set)]
    pub m: Vec<f64>,
    #[pyo3(get, set)]
    pub v: Vec<f64>,
    #[pyo3(get, set)]
    pub t: usize,
    #[pyo3(get, set)]
    pub obs_mean: Vec<f64>,
    #[pyo3(get, set)]
    pub obs_std: Vec<f64>,
    #[pyo3(get, set)]
    pub obs_count: usize,
}

#[pymethods]
impl EvolutionStrategy {
    #[new]
    pub fn new(dimension: usize, mean: Option<Vec<f64>>, config: Option<ESConfig>) -> Self {
        let config = config.unwrap_or_else(|| ESConfig::new(
            None, None, None, None, None, None, None, None, None, None, None, None, None, None
        ));
        let mean = mean.unwrap_or_else(|| vec![0.0; dimension]);
        
        let m = vec![0.0; dimension];
        let v = vec![0.0; dimension];
        let obs_mean = vec![0.0; dimension];
        let obs_std = vec![1.0; dimension];
        
        Self {
            dimension,
            mean,
            config: config.clone(),
            population: ESPopulation::new(config, dimension),
            m,
            v,
            t: 0,
            obs_mean,
            obs_std,
            obs_count: 0,
        }
    }
    
    pub fn ask(&mut self) -> Vec<Vec<f64>> {
        self.population.sample(self.mean.clone())
    }
    
    pub fn tell(&mut self, fitnesses: Vec<f64>) -> PyResult<()> {
        if fitnesses.len() != self.population.population_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Expected {} fitness values, got {}", self.population.population_size, fitnesses.len()),
            ));
        }
        
        self.population.update(fitnesses.clone());
        
        let mut fitnesses = fitnesses;
        if self.config.rank_based {
            let mut indices: Vec<usize> = (0..fitnesses.len()).collect();
            indices.sort_by(|&i, &j| fitnesses[j].partial_cmp(&fitnesses[i]).unwrap());
            
            let mut ranks = vec![0.0; fitnesses.len()];
            for (i, &idx) in indices.iter().enumerate() {
                ranks[idx] = i as f64 / (fitnesses.len() - 1) as f64 - 0.5;
            }
            
            fitnesses = ranks;
        }
        
        let mut weighted_noise = vec![0.0; self.dimension];
        
        if self.config.antithetic {
            let half_popsize = self.population.half_popsize;
            
            for i in 0..half_popsize {
                let mut noise = Vec::with_capacity(self.dimension);
                for j in 0..self.dimension {
                    noise.push((self.population.solutions[i][j] - self.mean[j]) / self.config.sigma);
                }
                
                for j in 0..self.dimension {
                    weighted_noise[j] += fitnesses[i] * noise[j] - fitnesses[i + half_popsize] * noise[j];
                }
            }
            
            for j in 0..self.dimension {
                weighted_noise[j] /= half_popsize as f64;
            }
        } else {
            for i in 0..self.population.population_size {
                for j in 0..self.dimension {
                    let noise = (self.population.solutions[i][j] - self.mean[j]) / self.config.sigma;
                    weighted_noise[j] += fitnesses[i] * noise;
                }
            }
            
            for j in 0..self.dimension {
                weighted_noise[j] /= self.population.population_size as f64;
            }
        }
        
        if self.config.normalize_updates {
            let mut std = 0.0;
            for &val in &weighted_noise {
                std += val * val;
            }
            std = (std / self.dimension as f64).sqrt();
            
            if std > 1e-8 {
                for j in 0..self.dimension {
                    weighted_noise[j] /= std;
                }
            }
        }
        
        if self.config.weight_decay > 0.0 {
            for j in 0..self.dimension {
                weighted_noise[j] -= self.config.weight_decay * self.mean[j];
            }
        }
        
        if self.config.adam {
            self.t += 1;
            
            for j in 0..self.dimension {
                self.m[j] = self.config.adam_beta1 * self.m[j] + (1.0 - self.config.adam_beta1) * weighted_noise[j];
                self.v[j] = self.config.adam_beta2 * self.v[j] + (1.0 - self.config.adam_beta2) * (weighted_noise[j] * weighted_noise[j]);
                
                let m_hat = self.m[j] / (1.0 - self.config.adam_beta1.powi(self.t as i32));
                let v_hat = self.v[j] / (1.0 - self.config.adam_beta2.powi(self.t as i32));
                
                self.mean[j] += self.config.learning_rate * m_hat / (v_hat.sqrt() + self.config.adam_epsilon);
            }
        } else {
            for j in 0..self.dimension {
                self.mean[j] += self.config.learning_rate * weighted_noise[j];
            }
        }
        
        self.config.learning_rate *= self.config.decay;
        self.config.sigma *= self.config.decay;
        
        Ok(())
    }
    
    pub fn optimize(&mut self, py: Python, objective_function: PyObject, 
                   iterations: Option<usize>, verbose: Option<bool>) -> PyResult<(Vec<f64>, f64)> {
        let iterations = iterations.unwrap_or(100);
        let verbose = verbose.unwrap_or(false);
        
        for i in 0..iterations {
            let solutions = self.ask();
            
            let solutions_list = PyList::new(py, &solutions);
            let fitnesses = objective_function.call1(py, (solutions_list,))?;
            let fitnesses: Vec<f64> = fitnesses.extract(py)?;
            
            self.tell(fitnesses)?;
            
            if verbose && (i % 10 == 0 || i == iterations - 1) {
                println!("Iteration {}: Best fitness = {}", i, self.population.best_fitness);
            }
        }
        
        Ok(self.population.get_best())
    }
    
    pub fn get_best(&self) -> (Vec<f64>, f64) {
        self.population.get_best()
    }
}

#[pyclass]
pub struct ESModel {
    #[pyo3(get)]
    pub model: PyObject,
    #[pyo3(get)]
    pub es: Option<EvolutionStrategy>,
    #[pyo3(get)]
    pub config: ESConfig,
    #[pyo3(get)]
    pub best_params: Option<Vec<f64>>,
}

#[pymethods]
impl ESModel {
    #[new]
    pub fn new(model: PyObject, config: Option<ESConfig>) -> Self {
        Self {
            model,
            es: None,
            config: config.unwrap_or_else(|| ESConfig::new(
                None, None, None, None, None, None, None, None, None, None, None, None, None, None
            )),
            best_params: None,
        }
    }
    
    fn _count_parameters(&self, py: Python) -> PyResult<usize> {
        let result = self.model.call_method0(py, "parameters")?;
        let params: Vec<PyObject> = result.extract(py)?;
        
        let mut count = 0;
        for param in params {
            let numel = param.call_method0(py, "numel")?;
            count += numel.extract::<usize>(py)?;
        }
        
        Ok(count)
    }
    
    fn _model_params_to_vector(&self, py: Python) -> PyResult<Vec<f64>> {
        let result = self.model.call_method0(py, "parameters")?;
        let params: Vec<PyObject> = result.extract(py)?;
        
        let mut vector = Vec::new();
        for param in params {
            let data = param.call_method0(py, "detach")?;
            let numpy = data.call_method0(py, "numpy")?;
            let flat = numpy.call_method0(py, "flatten")?;
            let values: Vec<f64> = flat.extract(py)?;
            vector.extend(values);
        }
        
        Ok(vector)
    }
    
    fn _vector_to_model_params(&self, py: Python, vector: Vec<f64>) -> PyResult<()> {
        let result = self.model.call_method0(py, "parameters")?;
        let params: Vec<PyObject> = result.extract(py)?;
        
        let mut start = 0;
        for param in params {
            let shape = param.getattr(py, "shape")?;
            let shape: Vec<usize> = shape.extract(py)?;
            
            let size = shape.iter().product::<usize>();
            let end = start + size;
            
            if end > vector.len() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Vector size ({}) is too small for model parameters", vector.len()),
                ));
            }
            
            let slice = &vector[start..end];
            let numpy = py.import("numpy")?;
            let array = numpy.call_method1("array", (slice,))?;
            let reshaped = array.call_method1("reshape", (shape,))?;
            
            let tensor = py.import("neurenix.tensor")?.call_method1("Tensor", (reshaped,))?;
            param.setattr(py, "data", tensor)?;
            
            start = end;
        }
        
        Ok(())
    }
    
    fn _evaluate(&self, py: Python, params: Vec<f64>, x: PyObject, y: PyObject, 
                loss_fn: PyObject) -> PyResult<f64> {
        self._vector_to_model_params(py, params)?;
        
        let y_pred = self.model.call1(py, (x,))?;
        let loss = loss_fn.call1(py, (y_pred, y))?;
        let loss_value: f64 = loss.extract(py)?;
        
        Ok(-loss_value)  // Negate loss for maximization
    }
    
    pub fn fit(&mut self, py: Python, x: PyObject, y: PyObject, loss_fn: PyObject, 
              iterations: Option<usize>, verbose: Option<bool>) -> PyResult<f64> {
        let iterations = iterations.unwrap_or(100);
        let verbose = verbose.unwrap_or(false);
        
        let dimension = self._count_parameters(py)?;
        let initial_params = self._model_params_to_vector(py)?;
        
        let mut es = EvolutionStrategy::new(dimension, Some(initial_params), Some(self.config.clone()));
        
        let model = self.model.clone_ref(py);
        let x_clone = x.clone_ref(py);
        let y_clone = y.clone_ref(py);
        let loss_fn_clone = loss_fn.clone_ref(py);
        
        let objective = move |py: Python, params: Vec<Vec<f64>>| -> PyResult<Vec<f64>> {
            let mut fitnesses = Vec::with_capacity(params.len());
            
            for param_set in params {
                let fitness = self._evaluate(py, param_set, x_clone.clone_ref(py), y_clone.clone_ref(py), loss_fn_clone.clone_ref(py))?;
                fitnesses.push(fitness);
            }
            
            Ok(fitnesses)
        };
        
        let (best_params, best_fitness) = es.optimize(py, PyObject::from(pyo3::PyFunction::new(
            py,
            move |py, params: Vec<Vec<f64>>| objective(py, params),
        )), Some(iterations), Some(verbose))?;
        
        self._vector_to_model_params(py, best_params.clone())?;
        self.best_params = Some(best_params);
        self.es = Some(es);
        
        Ok(-best_fitness)  // Return loss (not negative)
    }
    
    pub fn forward(&self, py: Python, x: PyObject) -> PyResult<PyObject> {
        let result = self.model.call1(py, (x,))?;
        Ok(result)
    }
}

pub fn register_evolution_strategy(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py, "evolution_strategy")?;
    
    submodule.add_class::<ESConfig>()?;
    submodule.add_class::<ESPopulation>()?;
    submodule.add_class::<EvolutionStrategy>()?;
    submodule.add_class::<ESModel>()?;
    
    m.add_submodule(&submodule)?;
    
    Ok(())
}
