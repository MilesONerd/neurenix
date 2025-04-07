
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;
use rand::{Rng, thread_rng};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use ndarray_linalg::{Eigh, SVD};
use crate::tensor::Tensor;
use crate::error::PhynexusError;

#[pyclass]
#[derive(Debug, Clone)]
pub struct CMAESConfig {
    #[pyo3(get, set)]
    pub population_size: usize,
    #[pyo3(get, set)]
    pub sigma: f64,
    #[pyo3(get, set)]
    pub weights_option: String,
    #[pyo3(get, set)]
    pub c1: Option<f64>,
    #[pyo3(get, set)]
    pub cmu: Option<f64>,
    #[pyo3(get, set)]
    pub cs: Option<f64>,
    #[pyo3(get, set)]
    pub damps: Option<f64>,
    #[pyo3(get, set)]
    pub active: bool,
    #[pyo3(get, set)]
    pub diagonal_covariance: bool,
    #[pyo3(get, set)]
    pub max_iter: usize,
    #[pyo3(get, set)]
    pub tol: f64,
    #[pyo3(get, set)]
    pub ftol: f64,
    #[pyo3(get, set)]
    pub xtol: f64,
    #[pyo3(get, set)]
    pub bounds: bool,
    #[pyo3(get, set)]
    pub lower_bounds: Option<Vec<f64>>,
    #[pyo3(get, set)]
    pub upper_bounds: Option<Vec<f64>>,
}

#[pymethods]
impl CMAESConfig {
    #[new]
    pub fn new(
        population_size: Option<usize>,
        sigma: Option<f64>,
        weights_option: Option<String>,
        c1: Option<f64>,
        cmu: Option<f64>,
        cs: Option<f64>,
        damps: Option<f64>,
        active: Option<bool>,
        diagonal_covariance: Option<bool>,
        max_iter: Option<usize>,
        tol: Option<f64>,
        ftol: Option<f64>,
        xtol: Option<f64>,
        bounds: Option<bool>,
        lower_bounds: Option<Vec<f64>>,
        upper_bounds: Option<Vec<f64>>,
    ) -> Self {
        Self {
            population_size: population_size.unwrap_or(0),  // Will be set based on dimension
            sigma: sigma.unwrap_or(0.5),
            weights_option: weights_option.unwrap_or_else(|| "default".to_string()),
            c1,
            cmu,
            cs,
            damps,
            active: active.unwrap_or(true),
            diagonal_covariance: diagonal_covariance.unwrap_or(false),
            max_iter: max_iter.unwrap_or(1000),
            tol: tol.unwrap_or(1e-8),
            ftol: ftol.unwrap_or(1e-11),
            xtol: xtol.unwrap_or(1e-11),
            bounds: bounds.unwrap_or(false),
            lower_bounds,
            upper_bounds,
        }
    }
}

#[pyclass]
pub struct CMAES {
    #[pyo3(get)]
    pub config: CMAESConfig,
    #[pyo3(get)]
    pub dimension: usize,
    #[pyo3(get)]
    pub mean: Vec<f64>,
    #[pyo3(get, set)]
    pub sigma: f64,
    #[pyo3(get)]
    pub population_size: usize,
    #[pyo3(get)]
    pub mu: usize,
    #[pyo3(get)]
    pub weights: Vec<f64>,
    #[pyo3(get)]
    pub mueff: f64,
    #[pyo3(get)]
    pub c1: f64,
    #[pyo3(get)]
    pub cmu: f64,
    #[pyo3(get)]
    pub cs: f64,
    #[pyo3(get)]
    pub damps: f64,
    #[pyo3(get)]
    pub chiN: f64,
    #[pyo3(get)]
    pub pc: Vec<f64>,
    #[pyo3(get)]
    pub ps: Vec<f64>,
    #[pyo3(get)]
    pub C: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub eigenvalues: Vec<f64>,
    #[pyo3(get)]
    pub eigenvectors: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub BD: Vec<Vec<f64>>,
    #[pyo3(get)]
    pub iteration: usize,
    #[pyo3(get)]
    pub best_solution: Vec<f64>,
    #[pyo3(get)]
    pub best_fitness: f64,
    #[pyo3(get)]
    pub evaluations: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub convergence_message: String,
}

#[pymethods]
impl CMAES {
    #[new]
    pub fn new(dimension: usize, mean: Option<Vec<f64>>, config: Option<CMAESConfig>) -> PyResult<Self> {
        let mut config = config.unwrap_or_else(|| CMAESConfig::new(
            None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None
        ));
        
        let mean = mean.unwrap_or_else(|| vec![0.0; dimension]);
        
        if mean.len() != dimension {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Mean vector dimension ({}) does not match the specified dimension ({})", mean.len(), dimension),
            ));
        }
        
        if config.population_size == 0 {
            config.population_size = 4 + (3.0 * (dimension as f64).ln()).floor() as usize;
        }
        
        let mu = config.population_size / 2;
        
        let weights = if config.weights_option == "equal" {
            vec![1.0 / mu as f64; mu]
        } else {
            let mut weights = Vec::with_capacity(mu);
            for i in 0..mu {
                weights.push((mu as f64 + 0.5) - (i as f64));
            }
            
            let sum: f64 = weights.iter().sum();
            for w in &mut weights {
                *w /= sum;
            }
            
            weights
        };
        
        let mueff = {
            let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
            1.0 / sum_sq
        };
        
        let c1 = config.c1.unwrap_or_else(|| {
            2.0 / ((dimension as f64 + 1.3).powi(2) + mueff)
        });
        
        let cmu = config.cmu.unwrap_or_else(|| {
            (2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dimension as f64 + 2.0).powi(2) + 2.0 * mueff / 2.0)).min(1.0 - c1)
        });
        
        let cs = config.cs.unwrap_or_else(|| {
            (mueff + 2.0) / (dimension as f64 + mueff + 5.0)
        });
        
        let damps = config.damps.unwrap_or_else(|| {
            1.0 + 2.0 * ((mueff - 1.0) / (dimension as f64 + 1.0)).max(0.0)
        });
        
        let chiN = (dimension as f64).sqrt() * (1.0 - 1.0 / (4.0 * dimension as f64) + 1.0 / (21.0 * (dimension as f64).powi(2)));
        
        let pc = vec![0.0; dimension];
        let ps = vec![0.0; dimension];
        
        let mut C = vec![vec![0.0; dimension]; dimension];
        for i in 0..dimension {
            C[i][i] = 1.0;
        }
        
        let eigenvalues = vec![1.0; dimension];
        let mut eigenvectors = vec![vec![0.0; dimension]; dimension];
        for i in 0..dimension {
            eigenvectors[i][i] = 1.0;
        }
        
        let mut BD = vec![vec![0.0; dimension]; dimension];
        for i in 0..dimension {
            BD[i][i] = 1.0;
        }
        
        Ok(Self {
            config,
            dimension,
            mean,
            sigma: config.sigma,
            population_size: config.population_size,
            mu,
            weights,
            mueff,
            c1,
            cmu,
            cs,
            damps,
            chiN,
            pc,
            ps,
            C,
            eigenvalues,
            eigenvectors,
            BD,
            iteration: 0,
            best_solution: mean.clone(),
            best_fitness: f64::INFINITY,
            evaluations: 0,
            converged: false,
            convergence_message: String::new(),
        })
    }
    
    pub fn ask(&mut self, py: Python) -> PyResult<Vec<Vec<f64>>> {
        let mut rng = thread_rng();
        let mut solutions = Vec::with_capacity(self.population_size);
        
        if self.iteration == 0 || !self.config.diagonal_covariance {
            self.update_eigendecomposition()?;
        }
        
        for _ in 0..self.population_size {
            let mut z = Vec::with_capacity(self.dimension);
            for _ in 0..self.dimension {
                z.push(rng.sample(rand_distr::StandardNormal));
            }
            
            let mut y = vec![0.0; self.dimension];
            
            if self.config.diagonal_covariance {
                for j in 0..self.dimension {
                    y[j] = self.eigenvalues[j].sqrt() * z[j];
                }
            } else {
                for i in 0..self.dimension {
                    for j in 0..self.dimension {
                        y[i] += self.BD[i][j] * z[j];
                    }
                }
            }
            
            let mut x = vec![0.0; self.dimension];
            for i in 0..self.dimension {
                x[i] = self.mean[i] + self.sigma * y[i];
            }
            
            if self.config.bounds {
                if let (Some(ref lower), Some(ref upper)) = (&self.config.lower_bounds, &self.config.upper_bounds) {
                    for i in 0..self.dimension {
                        x[i] = x[i].max(lower[i]).min(upper[i]);
                    }
                }
            }
            
            solutions.push(x);
        }
        
        Ok(solutions)
    }
    
    pub fn tell(&mut self, solutions: Vec<Vec<f64>>, fitnesses: Vec<f64>) -> PyResult<()> {
        if solutions.len() != self.population_size || fitnesses.len() != self.population_size {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Expected {} solutions and fitnesses, got {} and {}", 
                        self.population_size, solutions.len(), fitnesses.len()),
            ));
        }
        
        let mut idx: Vec<usize> = (0..self.population_size).collect();
        idx.sort_by(|&i, &j| fitnesses[i].partial_cmp(&fitnesses[j]).unwrap());
        
        if fitnesses[idx[0]] < self.best_fitness {
            self.best_fitness = fitnesses[idx[0]];
            self.best_solution = solutions[idx[0]].clone();
        }
        
        let old_mean = self.mean.clone();
        self.mean = vec![0.0; self.dimension];
        
        for i in 0..self.mu {
            let idx_i = idx[i];
            for j in 0..self.dimension {
                self.mean[j] += self.weights[i] * solutions[idx_i][j];
            }
        }
        
        let mut y = vec![0.0; self.dimension];
        for i in 0..self.dimension {
            y[i] = (self.mean[i] - old_mean[i]) / self.sigma;
        }
        
        let mut ps_norm = 0.0;
        for i in 0..self.dimension {
            let mut sum = 0.0;
            for j in 0..self.dimension {
                sum += self.eigenvectors[i][j] * self.ps[j];
            }
            ps_norm += sum * sum;
        }
        ps_norm = ps_norm.sqrt();
        
        for i in 0..self.dimension {
            self.pc[i] = (1.0 - self.c1).sqrt() * self.pc[i] + 
                        (self.c1 * (2.0 - self.c1) * self.mueff).sqrt() * y[i];
        }
        
        if self.config.diagonal_covariance {
            for i in 0..self.dimension {
                self.C[i][i] = (1.0 - self.c1 - self.cmu) * self.C[i][i] + 
                              self.c1 * self.pc[i] * self.pc[i];
                
                for j in 0..self.mu {
                    let idx_j = idx[j];
                    let yj = (solutions[idx_j][i] - old_mean[i]) / self.sigma;
                    self.C[i][i] += self.cmu * self.weights[j] * yj * yj;
                }
            }
        } else {
            for i in 0..self.dimension {
                for j in 0..i+1 {
                    self.C[i][j] = (1.0 - self.c1 - self.cmu) * self.C[i][j] + 
                                  self.c1 * self.pc[i] * self.pc[j];
                    
                    for k in 0..self.mu {
                        let idx_k = idx[k];
                        let yi = (solutions[idx_k][i] - old_mean[i]) / self.sigma;
                        let yj = (solutions[idx_k][j] - old_mean[j]) / self.sigma;
                        self.C[i][j] += self.cmu * self.weights[k] * yi * yj;
                    }
                    
                    self.C[j][i] = self.C[i][j];
                }
            }
            
            if self.config.active {
                let mu_minus = self.population_size - self.mu;
                if mu_minus > 0 {
                    let alpha_mu_minus = 1.0 + self.c1 / self.cmu;
                    let alpha_mu_eff_minus = 1.0 + 2.0 * mu_minus / (self.mueff + 2.0);
                    let alpha_pos_def = (1.0 - self.c1 - self.cmu) / (self.dimension as f64 * self.cmu);
                    
                    let alpha = (alpha_mu_minus * alpha_mu_eff_minus * alpha_pos_def).min(1.0);
                    
                    for k in self.mu..self.population_size.min(2 * self.mu) {
                        let idx_k = idx[k];
                        let weight = -alpha * self.weights[k - self.mu];
                        
                        for i in 0..self.dimension {
                            for j in 0..i+1 {
                                let yi = (solutions[idx_k][i] - old_mean[i]) / self.sigma;
                                let yj = (solutions[idx_k][j] - old_mean[j]) / self.sigma;
                                self.C[i][j] += self.cmu * weight * yi * yj;
                                self.C[j][i] = self.C[i][j];
                            }
                        }
                    }
                }
            }
        }
        
        let cs_factor = (self.cs * (2.0 - self.cs) * self.mueff).sqrt();
        for i in 0..self.dimension {
            let mut sum = 0.0;
            for j in 0..self.dimension {
                sum += self.eigenvectors[j][i] * y[j];
            }
            self.ps[i] = (1.0 - self.cs) * self.ps[i] + cs_factor * sum;
        }
        
        let ps_norm = {
            let mut sum = 0.0;
            for i in 0..self.dimension {
                sum += self.ps[i] * self.ps[i];
            }
            sum.sqrt()
        };
        
        self.sigma *= (self.cs / self.damps) * (ps_norm / self.chiN - 1.0).exp();
        
        self.iteration += 1;
        self.evaluations += self.population_size;
        
        self.check_convergence();
        
        Ok(())
    }
    
    fn update_eigendecomposition(&mut self) -> PyResult<()> {
        if self.config.diagonal_covariance {
            for i in 0..self.dimension {
                self.eigenvalues[i] = self.C[i][i];
                for j in 0..self.dimension {
                    self.eigenvectors[i][j] = if i == j { 1.0 } else { 0.0 };
                    self.BD[i][j] = if i == j { self.eigenvalues[i].sqrt() } else { 0.0 };
                }
            }
        } else {
            let mut c_array = Array2::zeros((self.dimension, self.dimension));
            for i in 0..self.dimension {
                for j in 0..self.dimension {
                    c_array[[i, j]] = self.C[i][j];
                }
            }
            
            let (eigvals, eigvecs) = match c_array.eigh(ndarray_linalg::UPLO::Upper) {
                Ok(result) => result,
                Err(_) => {
                    return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        "Failed to compute eigendecomposition of covariance matrix",
                    ));
                }
            };
            
            for i in 0..self.dimension {
                self.eigenvalues[i] = eigvals[i];
                for j in 0..self.dimension {
                    self.eigenvectors[i][j] = eigvecs[[i, j]];
                }
            }
            
            for i in 0..self.dimension {
                for j in 0..self.dimension {
                    self.BD[i][j] = self.eigenvectors[i][j] * self.eigenvalues[j].sqrt();
                }
            }
        }
        
        Ok(())
    }
    
    fn check_convergence(&mut self) {
        if self.iteration >= self.config.max_iter {
            self.converged = true;
            self.convergence_message = format!("Maximum number of iterations ({}) reached", self.config.max_iter);
            return;
        }
        
        let condition = if self.eigenvalues.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() <= 0.0 {
            f64::INFINITY
        } else {
            self.eigenvalues.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() / 
            self.eigenvalues.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
        };
        
        if condition > 1e14 {
            self.converged = true;
            self.convergence_message = format!("Condition of covariance matrix too large: {}", condition);
            return;
        }
        
        if self.sigma < 1e-20 || self.sigma > 1e20 {
            self.converged = true;
            self.convergence_message = format!("Sigma too small or too large: {}", self.sigma);
            return;
        }
        
        if self.iteration > 0 && self.best_fitness < self.config.ftol {
            self.converged = true;
            self.convergence_message = format!("Function value below tolerance: {}", self.best_fitness);
            return;
        }
        
        let x_tol = self.sigma * self.eigenvalues.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap().sqrt();
        if x_tol < self.config.xtol {
            self.converged = true;
            self.convergence_message = format!("Search space size below tolerance: {}", x_tol);
            return;
        }
    }
    
    pub fn optimize(&mut self, py: Python, objective_function: PyObject, 
                   max_iter: Option<usize>, verbose: Option<bool>) -> PyResult<(Vec<f64>, f64)> {
        let max_iter = max_iter.unwrap_or(self.config.max_iter);
        let verbose = verbose.unwrap_or(false);
        
        for i in 0..max_iter {
            if self.converged {
                break;
            }
            
            let solutions = self.ask(py)?;
            
            let solutions_list = PyList::new(py, &solutions);
            let fitnesses = objective_function.call1(py, (solutions_list,))?;
            let fitnesses: Vec<f64> = fitnesses.extract(py)?;
            
            self.tell(solutions, fitnesses)?;
            
            if verbose && (i % 10 == 0 || i == max_iter - 1) {
                println!("Iteration {}: Best fitness = {}", i, self.best_fitness);
            }
        }
        
        Ok((self.best_solution.clone(), self.best_fitness))
    }
    
    pub fn get_best(&self) -> (Vec<f64>, f64) {
        (self.best_solution.clone(), self.best_fitness)
    }
}

pub fn register_cmaes(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "cmaes")?;
    
    submodule.add_class::<CMAESConfig>()?;
    submodule.add_class::<CMAES>()?;
    
    m.add_submodule(submodule)?;
    
    Ok(())
}
