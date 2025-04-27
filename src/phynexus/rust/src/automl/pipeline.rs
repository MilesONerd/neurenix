
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyValueError;

use std::collections::HashMap;
use rand::Rng;

use crate::error::PhynexusError;
use crate::tensor::Tensor;

pub fn register_pipeline(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let pipeline = PyModule::new(py, "pipeline")?;
    
    pipeline.add_class::<PyFeatureSelection>()?;
    pipeline.add_class::<PyVarianceThreshold>()?;
    pipeline.add_class::<PySelectKBest>()?;
    pipeline.add_class::<PyDataPreprocessing>()?;
    pipeline.add_class::<PyStandardScaler>()?;
    pipeline.add_class::<PyMinMaxScaler>()?;
    pipeline.add_class::<PyAutoPipeline>()?;
    
    m.add_submodule(&pipeline)?;
    
    Ok(())
}

#[pyclass]
struct PyFeatureSelection {
    n_features_to_select: Option<usize>,
    selected_features: Option<Vec<usize>>,
}

#[pymethods]
impl PyFeatureSelection {
    #[new]
    fn new(n_features_to_select: Option<usize>) -> Self {
        Self {
            n_features_to_select,
            selected_features: None,
        }
    }
    
    fn fit(&mut self, _x: &PyAny, _y: &PyAny) -> PyResult<PyRef<Self>> {
        Err(PyValueError::new_err("Not implemented"))
    }
    
    fn transform(&self, x: &PyAny) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            match &self.selected_features {
                Some(features) => {
                    let np = py.import("numpy")?;
                    let features_array = np.call_method1("array", (features,))?;
                    let result = x.call_method("__getitem__", (py.eval("slice(None)", None, None)?, features_array), None)?;
                    Ok(result.extract()?)
                },
                None => Err(PyValueError::new_err("Feature selector has not been fitted").into_py(py)),
            }
        })
    }
    
    fn fit_transform(&mut self, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
        self.fit(x, y)?;
        self.transform(x)
    }
}

#[pyclass]
struct PyVarianceThreshold {
    base: PyFeatureSelection,
    threshold: f64,
}

#[pymethods]
impl PyVarianceThreshold {
    #[new]
    fn new(threshold: Option<f64>) -> Self {
        Self {
            base: PyFeatureSelection::new(None),
            threshold: threshold.unwrap_or(0.0),
        }
    }
    
    fn fit(&mut self, x: &PyAny, _y: &PyAny) -> PyResult<PyRef<Self>> {
        Python::with_gil(|py| {
            let np = py.import("numpy")?;
            
            let variances = np.call_method1("var", (x, 0))?;
            let mask = variances.call_method1("__gt__", (self.threshold,))?;
            let indices = np.call_method1("where", (mask,))?;
            let selected_features = indices.get_item(0)?.extract::<Vec<usize>>()?;
            
            self.base.selected_features = Some(selected_features);
            
            Ok(PyRef::new(py, self)?)
        })
    }
    
    fn transform(&self, x: &PyAny) -> PyResult<PyObject> {
        self.base.transform(x)
    }
    
    fn fit_transform(&mut self, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
        self.fit(x, y)?;
        self.transform(x)
    }
}

#[pyclass]
struct PySelectKBest {
    base: PyFeatureSelection,
    score_func: PyObject,
    scores: Option<PyObject>,
}

#[pymethods]
impl PySelectKBest {
    #[new]
    fn new(score_func: PyObject, k: Option<usize>) -> Self {
        Self {
            base: PyFeatureSelection::new(k),
            score_func,
            scores: None,
        }
    }
    
    fn fit(&mut self, x: &PyAny, y: &PyAny) -> PyResult<PyRef<Self>> {
        Python::with_gil(|py| {
            let np = py.import("numpy")?;
            
            let scores = self.score_func.call1(py, (x, y))?;
            self.scores = Some(scores.clone());
            
            let n_features = x.getattr("shape")?.get_item(1)?.extract::<usize>()?;
            let k = self.base.n_features_to_select.unwrap_or(n_features / 2);
            
            let indices = np.call_method1("argsort", (scores,))?;
            let selected_indices = indices.call_method("__getitem__", (py.eval(&format!("slice(-{}, None)", k), None, None)?,), None)?;
            let selected_features = selected_indices.extract::<Vec<usize>>()?;
            
            self.base.selected_features = Some(selected_features);
            
            Ok(PyRef::new(py, self)?)
        })
    }
    
    fn transform(&self, x: &PyAny) -> PyResult<PyObject> {
        self.base.transform(x)
    }
    
    fn fit_transform(&mut self, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
        self.fit(x, y)?;
        self.transform(x)
    }
}

#[pyclass]
struct PyDataPreprocessing {}

#[pymethods]
impl PyDataPreprocessing {
    #[new]
    fn new() -> Self {
        Self {}
    }
    
    fn fit(&self, _x: &PyAny) -> PyResult<PyRef<Self>> {
        Err(PyValueError::new_err("Not implemented"))
    }
    
    fn transform(&self, _x: &PyAny) -> PyResult<PyObject> {
        Err(PyValueError::new_err("Not implemented"))
    }
    
    fn fit_transform(&mut self, x: &PyAny) -> PyResult<PyObject> {
        self.fit(x)?;
        self.transform(x)
    }
}

#[pyclass]
struct PyStandardScaler {
    mean: Option<PyObject>,
    std: Option<PyObject>,
}

#[pymethods]
impl PyStandardScaler {
    #[new]
    fn new() -> Self {
        Self {
            mean: None,
            std: None,
        }
    }
    
    fn fit(&mut self, x: &PyAny) -> PyResult<PyRef<Self>> {
        Python::with_gil(|py| {
            let np = py.import("numpy")?;
            
            let mean = np.call_method1("mean", (x, 0))?;
            let std = np.call_method1("std", (x, 0))?;
            
            let zeros_mask = std.call_method1("__eq__", (0.0,))?;
            let std = np.call_method("where", (zeros_mask, 1.0, std), None)?;
            
            self.mean = Some(mean.extract()?);
            self.std = Some(std.extract()?);
            
            Ok(PyRef::new(py, self)?)
        })
    }
    
    fn transform(&self, x: &PyAny) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            match (&self.mean, &self.std) {
                (Some(mean), Some(std)) => {
                    let np = py.import("numpy")?;
                    let result = np.call_method("divide", (np.call_method("subtract", (x, mean), None)?, std), None)?;
                    Ok(result.extract()?)
                },
                _ => Err(PyValueError::new_err("Scaler has not been fitted").into_py(py)),
            }
        })
    }
    
    fn fit_transform(&mut self, x: &PyAny) -> PyResult<PyObject> {
        self.fit(x)?;
        self.transform(x)
    }
}

#[pyclass]
struct PyMinMaxScaler {
    feature_range: (f64, f64),
    min: Option<PyObject>,
    max: Option<PyObject>,
    scale: Option<PyObject>,
}

#[pymethods]
impl PyMinMaxScaler {
    #[new]
    fn new(feature_range: Option<(f64, f64)>) -> Self {
        Self {
            feature_range: feature_range.unwrap_or((0.0, 1.0)),
            min: None,
            max: None,
            scale: None,
        }
    }
    
    fn fit(&mut self, x: &PyAny) -> PyResult<PyRef<Self>> {
        Python::with_gil(|py| {
            let np = py.import("numpy")?;
            
            let min = np.call_method1("min", (x, 0))?;
            let max = np.call_method1("max", (x, 0))?;
            
            let min_eq_max = np.call_method("equal", (min, max), None)?;
            let max = np.call_method("where", (min_eq_max, np.call_method("add", (min, 1.0), None)?, max), None)?;
            
            let data_range = np.call_method("subtract", (max, min), None)?;
            let scale = np.call_method("divide", ((self.feature_range.1 - self.feature_range.0), data_range), None)?;
            
            self.min = Some(min.extract()?);
            self.max = Some(max.extract()?);
            self.scale = Some(scale.extract()?);
            
            Ok(PyRef::new(py, self)?)
        })
    }
    
    fn transform(&self, x: &PyAny) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            match (&self.min, &self.max, &self.scale) {
                (Some(min), Some(_), Some(scale)) => {
                    let np = py.import("numpy")?;
                    let x_scaled = np.call_method("multiply", (np.call_method("subtract", (x, min), None)?, scale), None)?;
                    let result = np.call_method("add", (x_scaled, self.feature_range.0), None)?;
                    Ok(result.extract()?)
                },
                _ => Err(PyValueError::new_err("Scaler has not been fitted").into_py(py)),
            }
        })
    }
    
    fn fit_transform(&mut self, x: &PyAny) -> PyResult<PyObject> {
        self.fit(x)?;
        self.transform(x)
    }
}

#[pyclass]
struct PyAutoPipeline {
    steps: Vec<(String, PyObject)>,
}

#[pymethods]
impl PyAutoPipeline {
    #[new]
    fn new(steps: Option<Vec<(String, PyObject)>>) -> Self {
        Self {
            steps: steps.unwrap_or_default(),
        }
    }
    
    fn add_step(&mut self, name: String, transform: PyObject) -> PyRef<Self> {
        self.steps.push((name, transform));
        Python::with_gil(|py| PyRef::new(py, self).unwrap())
    }
    
    fn fit(&mut self, x: &PyAny, y: Option<&PyAny>) -> PyResult<PyRef<Self>> {
        Python::with_gil(|py| {
            let mut x_transformed = x.extract::<PyObject>()?;
            
            for (_, transform) in &self.steps {
                if transform.hasattr(py, "fit")? {
                    if let Some(y_val) = y {
                        if transform.hasattr(py, "fit_transform")? && 
                           transform.getattr(py, "fit")?.getattr(py, "__code__")?.getattr(py, "co_varnames")?.contains(py, "y")? {
                            transform.call_method1(py, "fit", (x_transformed.extract::<&PyAny>(py)?, y_val))?;
                        } else {
                            transform.call_method1(py, "fit", (x_transformed.extract::<&PyAny>(py)?,))?;
                        }
                    } else {
                        transform.call_method1(py, "fit", (x_transformed.extract::<&PyAny>(py)?,))?;
                    }
                }
                
                if transform.hasattr(py, "transform")? {
                    x_transformed = transform.call_method1(py, "transform", (x_transformed.extract::<&PyAny>(py)?,))?.extract()?;
                }
            }
            
            Ok(PyRef::new(py, self)?)
        })
    }
    
    fn transform(&self, x: &PyAny) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut x_transformed = x.extract::<PyObject>()?;
            
            for (_, transform) in &self.steps {
                if transform.hasattr(py, "transform")? {
                    x_transformed = transform.call_method1(py, "transform", (x_transformed.extract::<&PyAny>(py)?,))?.extract()?;
                }
            }
            
            Ok(x_transformed)
        })
    }
    
    fn fit_transform(&mut self, x: &PyAny, y: Option<&PyAny>) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            let mut x_transformed = x.extract::<PyObject>()?;
            
            for (_, transform) in &self.steps {
                if transform.hasattr(py, "fit_transform")? && y.is_some() && 
                   transform.getattr(py, "fit")?.getattr(py, "__code__")?.getattr(py, "co_varnames")?.contains(py, "y")? {
                    x_transformed = transform.call_method1(py, "fit_transform", (x_transformed.extract::<&PyAny>(py)?, y.unwrap()))?.extract()?;
                } else if transform.hasattr(py, "fit_transform")? {
                    x_transformed = transform.call_method1(py, "fit_transform", (x_transformed.extract::<&PyAny>(py)?,))?.extract()?;
                } else {
                    if transform.hasattr(py, "fit")? {
                        if let Some(y_val) = y {
                            if transform.getattr(py, "fit")?.getattr(py, "__code__")?.getattr(py, "co_varnames")?.contains(py, "y")? {
                                transform.call_method1(py, "fit", (x_transformed.extract::<&PyAny>(py)?, y_val))?;
                            } else {
                                transform.call_method1(py, "fit", (x_transformed.extract::<&PyAny>(py)?,))?;
                            }
                        } else {
                            transform.call_method1(py, "fit", (x_transformed.extract::<&PyAny>(py)?,))?;
                        }
                    }
                    
                    if transform.hasattr(py, "transform")? {
                        x_transformed = transform.call_method1(py, "transform", (x_transformed.extract::<&PyAny>(py)?,))?.extract()?;
                    }
                }
            }
            
            Ok(x_transformed)
        })
    }
}
