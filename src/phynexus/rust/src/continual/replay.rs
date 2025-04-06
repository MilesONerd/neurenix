
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyValueError;

use crate::error::PhynexusError;
use crate::tensor::Tensor;

#[pyfunction]
pub fn update_replay_memory(
    replay: &PyAny,
    inputs: &PyAny,
    targets: &PyAny,
    task_id: Option<usize>,
) -> PyResult<()> {
    let py = replay.py();
    
    let per_class = replay.getattr("per_class")?.extract::<bool>()?;
    let strategy = replay.getattr("strategy")?.extract::<String>()?;
    let memory = replay.getattr("memory")?;
    
    let inputs_ref = inputs.as_ref(py);
    let num_examples = inputs_ref.getattr("shape")?.get_item(0)?.extract::<usize>()?;
    
    for i in 0..num_examples {
        let x = inputs_ref.call_method1("__getitem__", (py.eval("slice(None, None, None)", None, None)?, i))?;
        let y = targets.call_method1("__getitem__", (py.eval("slice(None, None, None)", None, None)?, i))?;
        
        let x_clone = x.call_method0("clone")?.call_method0("detach")?;
        let y_clone = y.call_method0("clone")?.call_method0("detach")?;
        
        if per_class {
            let class_idx = if y.call_method0("dim")?.extract::<usize>()? > 0 && 
                             y.call_method0("size")?.call_method1("__getitem__", (0,))?.extract::<usize>()? > 1 {
                py.import("neurenix")?.call_method1("argmax", (y,))?.call_method0("item")?.extract::<usize>()?
            } else {
                y.call_method0("item")?.extract::<usize>()?
            };
            
            let class_memory = if memory.contains(class_idx)? {
                memory.get_item(class_idx)?
            } else {
                let new_list = PyList::empty(py);
                memory.set_item(class_idx, new_list)?;
                memory.get_item(class_idx)?
            };
            
            if strategy == "reservoir" {
                update_reservoir_sampling(py, class_memory, x_clone, y_clone, replay)?;
            } else {
                let example = PyTuple::new(py, &[x_clone, y_clone]);
                class_memory.call_method1("append", (example,))?;
                
                let memory_size = replay.getattr("memory_size")?.extract::<usize>()?;
                let num_classes = memory.len()?;
                let max_per_class = memory_size / num_classes;
                
                if class_memory.len()? > max_per_class {
                    if strategy == "random" {
                        let idx = py.import("random")?.call_method1("randint", (0, class_memory.len()? - 2))?;
                        class_memory.call_method1("pop", (idx,))?;
                    } else {
                        class_memory.call_method1("pop", (0,))?;
                    }
                }
            }
        } else {
            if strategy == "reservoir" {
                update_reservoir_sampling(py, memory, x_clone, y_clone, replay)?;
            } else {
                let example = PyTuple::new(py, &[x_clone, y_clone]);
                memory.call_method1("append", (example,))?;
                
                let memory_size = replay.getattr("memory_size")?.extract::<usize>()?;
                
                if memory.len()? > memory_size {
                    if strategy == "random" {
                        let idx = py.import("random")?.call_method1("randint", (0, memory.len()? - 2))?;
                        memory.call_method1("pop", (idx,))?;
                    } else {
                        memory.call_method1("pop", (0,))?;
                    }
                }
            }
        }
    }
    
    Ok(())
}

fn update_reservoir_sampling(
    py: Python,
    memory: &PyAny,
    x: PyObject,
    y: PyObject,
    replay: &PyAny,
) -> PyResult<()> {
    let memory_size = replay.getattr("memory_size")?.extract::<usize>()?;
    let current_size = memory.len()?;
    
    if current_size < memory_size {
        let example = PyTuple::new(py, &[x, y]);
        memory.call_method1("append", (example,))?;
    } else {
        let t = current_size + 1;
        let random = py.import("random")?;
        let random_val = random.call_method0("random")?.extract::<f64>()?;
        
        if random_val < (memory_size as f64) / (t as f64) {
            let idx = random.call_method1("randint", (0, memory_size - 1))?;
            let example = PyTuple::new(py, &[x, y]);
            memory.set_item(idx, example)?;
        }
    }
    
    Ok(())
}

#[pyfunction]
pub fn sample_replay_memory(
    replay: &PyAny,
    batch_size: usize,
) -> PyResult<(PyObject, PyObject)> {
    let py = replay.py();
    
    let per_class = replay.getattr("per_class")?.extract::<bool>()?;
    let memory = replay.getattr("memory")?;
    
    let random = py.import("random")?;
    let neurenix = py.import("neurenix")?;
    
    let mut inputs_list = Vec::new();
    let mut targets_list = Vec::new();
    
    if per_class {
        let num_classes = memory.len()?;
        if num_classes == 0 {
            return Err(PyValueError::new_err("Replay memory is empty"));
        }
        
        let samples_per_class = batch_size / num_classes;
        let mut remainder = batch_size % num_classes;
        
        for (class_idx, class_memory) in memory.iter()? {
            let class_memory = class_memory?;
            let class_size = class_memory.len()?;
            
            if class_size == 0 {
                continue;
            }
            
            let mut n_samples = samples_per_class;
            if remainder > 0 {
                n_samples += 1;
                remainder -= 1;
            }
            
            if n_samples == 0 {
                continue;
            }
            
            let indices = if n_samples > class_size {
                let mut indices = Vec::new();
                for _ in 0..n_samples {
                    let idx = random.call_method1("randint", (0, class_size - 1))?.extract::<usize>()?;
                    indices.push(idx);
                }
                indices
            } else {
                let indices_list = random.call_method1("sample", (py.eval(&format!("range({})", class_size), None, None)?, n_samples))?;
                let mut indices = Vec::new();
                for idx in indices_list.iter()? {
                    indices.push(idx?.extract::<usize>()?);
                }
                indices
            };
            
            for idx in indices {
                let example = class_memory.get_item(idx)?;
                let (x, y) = example.extract::<(PyObject, PyObject)>()?;
                inputs_list.push(x);
                targets_list.push(y);
            }
        }
    } else {
        let memory_size = memory.len()?;
        if memory_size == 0 {
            return Err(PyValueError::new_err("Replay memory is empty"));
        }
        
        let indices = if batch_size > memory_size {
            let mut indices = Vec::new();
            for _ in 0..batch_size {
                let idx = random.call_method1("randint", (0, memory_size - 1))?.extract::<usize>()?;
                indices.push(idx);
            }
            indices
        } else {
            let indices_list = random.call_method1("sample", (py.eval(&format!("range({})", memory_size), None, None)?, batch_size))?;
            let mut indices = Vec::new();
            for idx in indices_list.iter()? {
                indices.push(idx?.extract::<usize>()?);
            }
            indices
        };
        
        for idx in indices {
            let example = memory.get_item(idx)?;
            let (x, y) = example.extract::<(PyObject, PyObject)>()?;
            inputs_list.push(x);
            targets_list.push(y);
        }
    }
    
    let inputs = neurenix.call_method1("cat", (PyList::new(py, &inputs_list), 0))?;
    let targets = neurenix.call_method1("cat", (PyList::new(py, &targets_list), 0))?;
    
    Ok((inputs.to_object(py), targets.to_object(py)))
}
