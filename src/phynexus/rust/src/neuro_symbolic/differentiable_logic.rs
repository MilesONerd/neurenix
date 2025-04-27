
use std::collections::{HashMap, HashSet};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyString, PyTuple};
use crate::tensor::Tensor;
use crate::error::PhynexusError;

#[pyclass(unsendable)]
#[derive(Clone)]
pub struct LogicTensor {
    tensor: Tensor,
}

#[pymethods]
impl LogicTensor {
    #[new]
    fn new(tensor: Tensor) -> Self {
        LogicTensor { tensor }
    }

    #[staticmethod]
    fn fuzzy_and(py: Python, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
        let result = x.call_method1("__mul__", (y,))?;
        Ok(result.into())
    }

    #[staticmethod]
    fn fuzzy_or(py: Python, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
        let product = x.call_method1("__mul__", (y,))?;
        let sum = x.call_method1("__add__", (y,))?;
        let result = sum.call_method1("__sub__", (product,))?;
        Ok(result.into())
    }

    #[staticmethod]
    fn fuzzy_not(py: Python, x: &PyAny) -> PyResult<PyObject> {
        let ones = py.import("neurenix.tensor")?.getattr("Tensor")?.call_method1("ones_like", (x,))?;
        let result = ones.call_method1("__sub__", (x,))?;
        Ok(result.into())
    }

    #[staticmethod]
    fn fuzzy_implies(py: Python, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
        let not_x = LogicTensor::fuzzy_not(py, x)?;
        let result = LogicTensor::fuzzy_or(py, &not_x, y)?;
        Ok(result.into())
    }

    #[staticmethod]
    fn probabilistic_and(py: Python, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
        let result = x.call_method1("__mul__", (y,))?;
        Ok(result.into())
    }

    #[staticmethod]
    fn probabilistic_or(py: Python, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
        let product = x.call_method1("__mul__", (y,))?;
        let sum = x.call_method1("__add__", (y,))?;
        let result = sum.call_method1("__sub__", (product,))?;
        Ok(result.into())
    }

    #[staticmethod]
    fn probabilistic_not(py: Python, x: &PyAny) -> PyResult<PyObject> {
        let ones = py.import("neurenix.tensor")?.getattr("Tensor")?.call_method1("ones_like", (x,))?;
        let result = ones.call_method1("__sub__", (x,))?;
        Ok(result.into())
    }

    #[staticmethod]
    fn lukasiewicz_and(py: Python, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
        let sum = x.call_method1("__add__", (y,))?;
        let ones = py.import("neurenix.tensor")?.getattr("Tensor")?.call_method1("ones_like", (x,))?;
        let diff = sum.call_method1("__sub__", (ones,))?;
        let zeros = py.import("neurenix.tensor")?.getattr("Tensor")?.call_method1("zeros_like", (x,))?;
        let result = py.import("neurenix.tensor")?.getattr("Tensor")?.call_method1("maximum", (diff, zeros))?;
        Ok(result.into())
    }

    #[staticmethod]
    fn lukasiewicz_or(py: Python, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
        let sum = x.call_method1("__add__", (y,))?;
        let ones = py.import("neurenix.tensor")?.getattr("Tensor")?.call_method1("ones_like", (x,))?;
        let result = py.import("neurenix.tensor")?.getattr("Tensor")?.call_method1("minimum", (sum, ones))?;
        Ok(result.into())
    }

    #[staticmethod]
    fn godel_and(py: Python, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
        let result = py.import("neurenix.tensor")?.getattr("Tensor")?.call_method1("minimum", (x, y))?;
        Ok(result.into())
    }

    #[staticmethod]
    fn godel_or(py: Python, x: &PyAny, y: &PyAny) -> PyResult<PyObject> {
        let result = py.import("neurenix.tensor")?.getattr("Tensor")?.call_method1("maximum", (x, y))?;
        Ok(result.into())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct DifferentiableLogic {
    logic_type: String,
    #[pyo3(get)]
    and_op: PyObject,
    #[pyo3(get)]
    or_op: PyObject,
    #[pyo3(get)]
    not_op: PyObject,
    #[pyo3(get)]
    implies_op: PyObject,
}

#[pymethods]
impl DifferentiableLogic {
    #[new]
    fn new(py: Python, logic_type: Option<String>) -> PyResult<Self> {
        let logic = logic_type.unwrap_or_else(|| "fuzzy".to_string());
        
        let logic_tensor = py.import("neurenix.neuro_symbolic.differentiable_logic")?.getattr("LogicTensor")?;
        
        let (and_op, or_op, not_op, implies_op) = match logic.as_str() {
            "fuzzy" => (
                logic_tensor.getattr("fuzzy_and")?,
                logic_tensor.getattr("fuzzy_or")?,
                logic_tensor.getattr("fuzzy_not")?,
                logic_tensor.getattr("fuzzy_implies")?,
            ),
            "probabilistic" => (
                logic_tensor.getattr("probabilistic_and")?,
                logic_tensor.getattr("probabilistic_or")?,
                logic_tensor.getattr("probabilistic_not")?,
                py.eval(
                    "lambda x, y: LogicTensor.probabilistic_or(LogicTensor.probabilistic_not(x), y)",
                    None,
                    Some(PyDict::new(py).set_item("LogicTensor", logic_tensor)?),
                )?,
            ),
            "lukasiewicz" => (
                logic_tensor.getattr("lukasiewicz_and")?,
                logic_tensor.getattr("lukasiewicz_or")?,
                logic_tensor.getattr("fuzzy_not")?,
                py.eval(
                    "lambda x, y: LogicTensor.lukasiewicz_or(LogicTensor.fuzzy_not(x), y)",
                    None,
                    Some(PyDict::new(py).set_item("LogicTensor", logic_tensor)?),
                )?,
            ),
            "godel" => (
                logic_tensor.getattr("godel_and")?,
                logic_tensor.getattr("godel_or")?,
                logic_tensor.getattr("fuzzy_not")?,
                py.eval(
                    "lambda x, y: Tensor.ones_like(x) if (x <= y).all() else y",
                    None,
                    Some(PyDict::new(py).set_item("Tensor", py.import("neurenix.tensor")?.getattr("Tensor")?)?),
                )?,
            ),
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Invalid logic type: {}", logic)
                ));
            }
        };
        
        Ok(DifferentiableLogic {
            logic_type: logic,
            and_op: and_op.into(),
            or_op: or_op.into(),
            not_op: not_op.into(),
            implies_op: implies_op.into(),
        })
    }

    fn forward(&self, py: Python, x: &PyAny) -> PyResult<PyObject> {
        Ok(x.into_py(py))
    }
}

#[pyclass]
#[derive(Clone)]
pub struct FuzzyLogic {
    differentiable_logic: DifferentiableLogic,
}

#[pymethods]
impl FuzzyLogic {
    #[new]
    fn new(py: Python, logic_type: Option<String>) -> PyResult<Self> {
        let logic = logic_type.unwrap_or_else(|| "fuzzy".to_string());
        
        if !["fuzzy", "lukasiewicz", "godel"].contains(&logic.as_str()) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid fuzzy logic type: {}", logic)
            ));
        }
        
        Ok(FuzzyLogic {
            differentiable_logic: DifferentiableLogic::new(py, Some(logic))?,
        })
    }

    fn forward(&self, py: Python, x: &PyAny) -> PyResult<PyObject> {
        Ok(x.into_py(py))
    }

    fn evaluate_rule(&self, py: Python, antecedent: &PyAny, consequent: &PyAny) -> PyResult<PyObject> {
        let result = self.differentiable_logic.implies_op.call1(py, (antecedent, consequent))?;
        Ok(result.into())
    }

    fn evaluate_conjunction(&self, py: Python, tensors: &PyAny) -> PyResult<PyObject> {
        let len = tensors.len()?;
        if len == 0 {
            let result = py.import("neurenix.tensor")?.getattr("Tensor")?.call1((vec![1.0],))?;
            return Ok(result.into());
        }
        
        let mut result = tensors.get_item(0)?.into_py(py);
        
        for i in 1..len {
            let tensor = tensors.get_item(i)?;
            result = self.differentiable_logic.and_op.call1(py, (result, tensor))?;
        }
        
        Ok(result.into())
    }

    fn evaluate_disjunction(&self, py: Python, tensors: &PyAny) -> PyResult<PyObject> {
        let len = tensors.len()?;
        if len == 0 {
            let result = py.import("neurenix.tensor")?.getattr("Tensor")?.call1((vec![0.0],))?;
            return Ok(result.into());
        }
        
        let mut result = tensors.get_item(0)?.into_py(py);
        
        for i in 1..len {
            let tensor = tensors.get_item(i)?;
            result = self.differentiable_logic.or_op.call1(py, (result, tensor))?;
        }
        
        Ok(result.into())
    }

    fn evaluate_negation(&self, py: Python, tensor: &PyAny) -> PyResult<PyObject> {
        let result = self.differentiable_logic.not_op.call1(py, (tensor,))?;
        Ok(result.into())
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ProbabilisticLogic {
    differentiable_logic: DifferentiableLogic,
}

#[pymethods]
impl ProbabilisticLogic {
    #[new]
    fn new(py: Python) -> PyResult<Self> {
        Ok(ProbabilisticLogic {
            differentiable_logic: DifferentiableLogic::new(py, Some("probabilistic".to_string()))?,
        })
    }

    fn forward(&self, py: Python, x: &PyAny) -> PyResult<PyObject> {
        Ok(x.into_py(py))
    }

    fn joint_probability(&self, py: Python, x: &PyAny, y: &PyAny, 
                       conditional_prob: Option<&PyAny>) -> PyResult<PyObject> {
        if let Some(cond_prob) = conditional_prob {
            let result = x.call_method1("__mul__", (cond_prob,))?;
            Ok(result.into())
        } else {
            let result = x.call_method1("__mul__", (y,))?;
            Ok(result.into())
        }
    }

    fn conditional_probability(&self, py: Python, joint_prob: &PyAny, 
                             marginal_prob: &PyAny) -> PyResult<PyObject> {
        let epsilon = py.import("neurenix.tensor")?.getattr("Tensor")?.call1((vec![1e-8],))?;
        let denominator = marginal_prob.call_method1("__add__", (epsilon,))?;
        let result = joint_prob.call_method1("__truediv__", (denominator,))?;
        Ok(result.into())
    }

    fn marginal_probability(&self, py: Python, joint_probs: &PyAny) -> PyResult<PyObject> {
        let len = joint_probs.len()?;
        if len == 0 {
            let result = py.import("neurenix.tensor")?.getattr("Tensor")?.call1((vec![0.0],))?;
            return Ok(result.into());
        }
        
        let mut result = joint_probs.get_item(0)?.into_py(py);
        
        for i in 1..len {
            let joint_prob = joint_probs.get_item(i)?;
            result = result.call_method1(py, "__add__", (joint_prob,))?;
        }
        
        Ok(result.into())
    }

    fn bayes_rule(&self, py: Python, likelihood: &PyAny, prior: &PyAny, 
                evidence: &PyAny) -> PyResult<PyObject> {
        let epsilon = py.import("neurenix.tensor")?.getattr("Tensor")?.call1((vec![1e-8],))?;
        let denominator = evidence.call_method1("__add__", (epsilon,))?;
        
        let numerator = likelihood.call_method1("__mul__", (prior,))?;
        let result = numerator.call_method1("__truediv__", (denominator,))?;
        
        Ok(result.into())
    }
}

pub fn register_differentiable_logic(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let submodule = PyModule::new(py, "differentiable_logic")?;
    
    submodule.add_class::<LogicTensor>()?;
    submodule.add_class::<DifferentiableLogic>()?;
    submodule.add_class::<FuzzyLogic>()?;
    submodule.add_class::<ProbabilisticLogic>()?;
    
    m.add_submodule(&submodule)?;
    
    Ok(())
}
