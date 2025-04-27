
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyValueError;

use std::collections::HashMap;
use rand::Rng;
use rand::seq::SliceRandom;

use crate::error::PhynexusError;
use crate::tensor::Tensor;

pub fn register_model_selection(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let model_selection = PyModule::new(py, "model_selection")?;
    
    model_selection.add_class::<PyAutoModelSelection>()?;
    model_selection.add_class::<PyCrossValidation>()?;
    model_selection.add_class::<PyNestedCrossValidation>()?;
    
    m.add_submodule(&model_selection)?;
    
    Ok(())
}

#[pyclass]
struct PyAutoModelSelection {
    model_classes: Vec<PyObject>,
    hyperparams: HashMap<PyObject, HashMap<String, Vec<PyObject>>>,
    max_trials: usize,
    results: Vec<(PyObject, f64)>,
    best_model: Option<PyObject>,
    best_score: f64,
}

#[pymethods]
impl PyAutoModelSelection {
    #[new]
    fn new(model_classes: Vec<PyObject>, hyperparams: &PyDict, max_trials: Option<usize>) -> PyResult<Self> {
        let mut hyperparams_map = HashMap::new();
        
        Python::with_gil(|py| {
            for (key, value) in hyperparams.iter() {
                let model_class = key.extract::<PyObject>()?;
                let params_dict = value.extract::<&PyDict>()?;
                let mut params_map = HashMap::new();
                
                for (param_name, param_values) in params_dict.iter() {
                    let name = param_name.extract::<String>()?;
                    let values = param_values.extract::<Vec<PyObject>>()?;
                    params_map.insert(name, values);
                }
                
                hyperparams_map.insert(model_class, params_map);
            }
            Ok(())
        })?;
        
        Ok(Self {
            model_classes,
            hyperparams: hyperparams_map,
            max_trials: max_trials.unwrap_or(10),
            results: Vec::new(),
            best_model: None,
            best_score: std::f64::NEG_INFINITY,
        })
    }
    
    fn select(&mut self, x_train: &PyAny, y_train: &PyAny, x_val: &PyAny, y_val: &PyAny) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut trials = 0;
            
            for model_class in &self.model_classes {
                if let Some(params_map) = self.hyperparams.get(model_class) {
                    let param_names: Vec<String> = params_map.keys().cloned().collect();
                    let param_values: Vec<Vec<PyObject>> = param_names.iter()
                        .map(|name| params_map.get(name).unwrap().clone())
                        .collect();
                    
                    let mut indices = vec![0; param_names.len()];
                    let mut done = false;
                    
                    while !done && trials < self.max_trials {
                        let model = model_class.call(py, (), None)?;
                        
                        for i in 0..param_names.len() {
                            let name = &param_names[i];
                            let value = &param_values[i][indices[i]];
                            
                            let setattr = py.import("builtins")?.getattr("setattr")?;
                            setattr.call1((model, name, value))?;
                        }
                        
                        model.call_method1("fit", (x_train, y_train))?;
                        
                        let score = model.call_method1("evaluate", (x_val, y_val))?.extract::<f64>()?;
                        
                        self.results.push((model.extract::<PyObject>()?, score));
                        
                        if score > self.best_score {
                            self.best_score = score;
                            self.best_model = Some(model.extract::<PyObject>()?);
                        }
                        
                        let mut j = param_names.len() - 1;
                        indices[j] += 1;
                        while indices[j] >= param_values[j].len() {
                            indices[j] = 0;
                            if j == 0 {
                                done = true;
                                break;
                            }
                            j -= 1;
                            indices[j] += 1;
                        }
                        
                        trials += 1;
                    }
                }
            }
            
            match &self.best_model {
                Some(model) => Ok(model.clone()),
                None => Err(PyValueError::new_err("No model found").into_py(py)),
            }
        })
    }
    
    fn get_best_model(&self) -> PyResult<Option<PyObject>> {
        match &self.best_model {
            Some(model) => Ok(Some(model.clone())),
            None => Ok(None),
        }
    }
    
    fn get_results(&self) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let results = PyList::empty(py);
            for (model, score) in &self.results {
                let tuple = PyTuple::new(py, &[model.clone(), (*score).into_py(py)]);
                results.append(tuple)?;
            }
            Ok(results.into())
        })
    }
}

#[pyclass]
struct PyCrossValidation {
    n_splits: usize,
    shuffle: bool,
    random_seed: Option<u64>,
}

#[pymethods]
impl PyCrossValidation {
    #[new]
    fn new(n_splits: Option<usize>, shuffle: Option<bool>, random_seed: Option<u64>) -> Self {
        Self {
            n_splits: n_splits.unwrap_or(5),
            shuffle: shuffle.unwrap_or(true),
            random_seed,
        }
    }
    
    fn split(&self, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let np = py.import("numpy")?;
            
            let n_samples = x.getattr("shape")?.get_item(0)?.extract::<usize>()?;
            let indices = np.call_method1("arange", (n_samples,))?;
            
            let shuffled_indices = if self.shuffle {
                let mut rng = match self.random_seed {
                    Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
                    None => rand::rngs::StdRng::from_entropy(),
                };
                
                let mut indices_vec = (0..n_samples).collect::<Vec<usize>>();
                indices_vec.shuffle(&mut rng);
                
                let indices_array = np.call_method1("array", (indices_vec,))?;
                indices_array
            } else {
                indices
            };
            
            let fold_size = n_samples / self.n_splits;
            let folds = PyList::empty(py);
            
            for i in 0..self.n_splits {
                let start = i * fold_size;
                let end = if i < self.n_splits - 1 { (i + 1) * fold_size } else { n_samples };
                
                let val_indices = shuffled_indices.call_method("__getitem__", (py.eval(&format!("slice({}, {})", start, end), None, None)?,), None)?;
                
                let train_indices_1 = if start > 0 {
                    shuffled_indices.call_method("__getitem__", (py.eval(&format!("slice(0, {})", start), None, None)?,), None)?
                } else {
                    np.call_method1("array", (Vec::<usize>::new(),))?
                };
                
                let train_indices_2 = if end < n_samples {
                    shuffled_indices.call_method("__getitem__", (py.eval(&format!("slice({}, {})", end, n_samples), None, None)?,), None)?
                } else {
                    np.call_method1("array", (Vec::<usize>::new(),))?
                };
                
                let train_indices = np.call_method1("concatenate", ((train_indices_1, train_indices_2),))?;
                
                let fold = PyTuple::new(py, &[train_indices, val_indices]);
                folds.append(fold)?;
            }
            
            Ok(folds.into())
        })
    }
    
    fn evaluate(&self, model_fn: &PyAny, x: &PyAny, y: &PyAny, fit_params: Option<&PyDict>) -> PyResult<f64> {
        Python::with_gil(|py| {
            let np = py.import("numpy")?;
            let logging = py.import("logging")?;
            
            let fit_params = match fit_params {
                Some(params) => params.clone(),
                None => PyDict::new(py),
            };
            
            let folds = self.split(x, y)?;
            let folds_list = folds.extract::<&PyList>()?;
            
            let mut scores = Vec::new();
            
            for (fold, fold_tuple) in folds_list.iter().enumerate() {
                let (train_indices, val_indices) = fold_tuple.extract::<(&PyAny, &PyAny)>()?;
                
                let x_train = x.call_method("__getitem__", (train_indices,), None)?;
                let y_train = y.call_method("__getitem__", (train_indices,), None)?;
                let x_val = x.call_method("__getitem__", (val_indices,), None)?;
                let y_val = y.call_method("__getitem__", (val_indices,), None)?;
                
                let model = model_fn.call((), None)?;
                
                let x_train_tensor = py.import("neurenix")?.call_method1("Tensor", (x_train,))?;
                let y_train_tensor = py.import("neurenix")?.call_method1("Tensor", (y_train,))?;
                let x_val_tensor = py.import("neurenix")?.call_method1("Tensor", (x_val,))?;
                let y_val_tensor = py.import("neurenix")?.call_method1("Tensor", (y_val,))?;
                
                logging.call_method1("info", (format!("Training fold {}/{}", fold + 1, self.n_splits),))?;
                
                model.call_method("fit", (x_train_tensor, y_train_tensor), Some(&fit_params))?;
                
                let score = model.call_method1("evaluate", (x_val_tensor, y_val_tensor))?.extract::<f64>()?;
                scores.push(score);
                
                logging.call_method1("info", (format!("Fold {} score: {:.4}", fold + 1, score),))?;
            }
            
            let mean_score = np.call_method1("mean", (scores,))?.extract::<f64>()?;
            let std_score = np.call_method1("std", (scores,))?.extract::<f64>()?;
            
            logging.call_method1("info", (format!("Cross-validation results: {:.4} ± {:.4}", mean_score, std_score),))?;
            
            Ok(mean_score)
        })
    }
}

#[pyclass]
struct PyNestedCrossValidation {
    outer_cv: PyCrossValidation,
    inner_cv: PyCrossValidation,
}

#[pymethods]
impl PyNestedCrossValidation {
    #[new]
    fn new(outer_splits: Option<usize>, inner_splits: Option<usize>, 
           shuffle: Option<bool>, random_seed: Option<u64>) -> Self {
        Self {
            outer_cv: PyCrossValidation::new(outer_splits, shuffle, random_seed),
            inner_cv: PyCrossValidation::new(inner_splits, shuffle, random_seed),
        }
    }
    
    fn evaluate(&self, model_fn: &PyAny, param_grid: &PyDict, x: &PyAny, y: &PyAny, 
                fit_params: Option<&PyDict>) -> PyResult<(f64, PyObject)> {
        Python::with_gil(|py| {
            let np = py.import("numpy")?;
            let itertools = py.import("itertools")?;
            let logging = py.import("logging")?;
            
            let fit_params = match fit_params {
                Some(params) => params.clone(),
                None => PyDict::new(py),
            };
            
            let outer_folds = self.outer_cv.split(x, y)?;
            let outer_folds_list = outer_folds.extract::<&PyList>()?;
            
            let mut outer_scores = Vec::new();
            let best_params_list = PyList::empty(py);
            
            for (fold, fold_tuple) in outer_folds_list.iter().enumerate() {
                let (train_indices, val_indices) = fold_tuple.extract::<(&PyAny, &PyAny)>()?;
                
                let x_train = x.call_method("__getitem__", (train_indices,), None)?;
                let y_train = y.call_method("__getitem__", (train_indices,), None)?;
                let x_val = x.call_method("__getitem__", (val_indices,), None)?;
                let y_val = y.call_method("__getitem__", (val_indices,), None)?;
                
                let mut best_score = std::f64::NEG_INFINITY;
                let mut best_params = None;
                
                let param_names = param_grid.keys().collect::<Vec<_>>();
                let param_values = param_names.iter()
                    .map(|&name| param_grid.get_item(name).unwrap())
                    .collect::<Vec<_>>();
                
                let param_combinations = itertools.call_method1("product", (param_values,))?;
                
                for values in param_combinations.iter()? {
                    let values_tuple = values.extract::<&PyTuple>()?;
                    let params = PyDict::new(py);
                    
                    for (i, &name) in param_names.iter().enumerate() {
                        params.set_item(name, values_tuple.get_item(i)?)?;
                    }
                    
                    let create_model = py.eval(
                        "lambda params: lambda: model_fn(params)",
                        Some(&PyDict::new(py).set_item("model_fn", model_fn)?.set_item("params", params)?),
                        None
                    )?;
                    
                    let score = self.inner_cv.evaluate(create_model.call1((params,))?, x_train, y_train, Some(&fit_params))?;
                    
                    if score > best_score {
                        best_score = score;
                        best_params = Some(params.clone());
                    }
                }
                
                let best_params = match best_params {
                    Some(params) => params,
                    None => PyDict::new(py),
                };
                
                let model = model_fn.call1((best_params,))?;
                
                let x_train_tensor = py.import("neurenix")?.call_method1("Tensor", (x_train,))?;
                let y_train_tensor = py.import("neurenix")?.call_method1("Tensor", (y_train,))?;
                let x_val_tensor = py.import("neurenix")?.call_method1("Tensor", (x_val,))?;
                let y_val_tensor = py.import("neurenix")?.call_method1("Tensor", (y_val,))?;
                
                logging.call_method1("info", (format!("Training outer fold {}/{} with best parameters: {:?}", 
                                                    fold + 1, self.outer_cv.n_splits, best_params),))?;
                
                model.call_method("fit", (x_train_tensor, y_train_tensor), Some(&fit_params))?;
                
                let score = model.call_method1("evaluate", (x_val_tensor, y_val_tensor))?.extract::<f64>()?;
                outer_scores.push(score);
                best_params_list.append(best_params)?;
                
                logging.call_method1("info", (format!("Outer fold {} score: {:.4}", fold + 1, score),))?;
            }
            
            let mean_score = np.call_method1("mean", (outer_scores,))?.extract::<f64>()?;
            let std_score = np.call_method1("std", (outer_scores,))?.extract::<f64>()?;
            
            logging.call_method1("info", (format!("Nested cross-validation results: {:.4} ± {:.4}", mean_score, std_score),))?;
            
            Ok((mean_score, best_params_list.into()))
        })
    }
}
