
use crate::error::PhynexusError;
use crate::tensor::Tensor;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::{Arc, Mutex};
use std::thread;

#[pyfunction]
fn is_multithreading_supported() -> PyResult<bool> {
    #[cfg(target_arch = "wasm32")]
    {
        Ok(true)
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        Ok(false)
    }
}

#[pyfunction]
fn enable_multithreading() -> PyResult<bool> {
    #[cfg(target_arch = "wasm32")]
    {
        Ok(true)
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        Ok(false)
    }
}

#[pyfunction]
fn get_num_workers() -> PyResult<usize> {
    #[cfg(target_arch = "wasm32")]
    {
        Ok(4)
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        Ok(num_cpus::get())
    }
}

#[pyfunction]
fn parallel_matmul(py: Python, a: &PyAny, b: &PyAny) -> PyResult<PyObject> {
    let a_tensor = Tensor::from_pyany(a)?;
    let b_tensor = Tensor::from_pyany(b)?;
    
    #[cfg(target_arch = "wasm32")]
    {
        let result = a_tensor.matmul(&b_tensor)?;
        Ok(result.to_pyobject(py))
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        let result = a_tensor.matmul(&b_tensor)?;
        Ok(result.to_pyobject(py))
    }
}

#[pyfunction]
#[pyo3(signature = (input, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1))]
fn parallel_conv2d(
    py: Python,
    input: &PyAny,
    weight: &PyAny,
    bias: Option<&PyAny>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> PyResult<PyObject> {
    let input_tensor = Tensor::from_pyany(input)?;
    let weight_tensor = Tensor::from_pyany(weight)?;
    let bias_tensor = match bias {
        Some(b) => Some(Tensor::from_pyany(b)?),
        None => None,
    };
    
    #[cfg(target_arch = "wasm32")]
    {
        let nn = py.import("neurenix.nn.functional")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("stride", stride)?;
        kwargs.set_item("padding", padding)?;
        kwargs.set_item("dilation", dilation)?;
        kwargs.set_item("groups", groups)?;
        
        if let Some(b) = bias {
            nn.getattr("conv2d")?.call((input, weight, b), Some(kwargs))
        } else {
            nn.getattr("conv2d")?.call((input, weight), Some(kwargs))
        }
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        let nn = py.import("neurenix.nn.functional")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("stride", stride)?;
        kwargs.set_item("padding", padding)?;
        kwargs.set_item("dilation", dilation)?;
        kwargs.set_item("groups", groups)?;
        
        if let Some(b) = bias {
            nn.getattr("conv2d")?.call((input, weight, b), Some(kwargs))
        } else {
            nn.getattr("conv2d")?.call((input, weight), Some(kwargs))
        }
    }
}

#[pyfunction]
fn parallel_map(py: Python, func: &PyAny, tensors: &PyList) -> PyResult<PyObject> {
    #[cfg(target_arch = "wasm32")]
    {
        let result = PyList::empty(py);
        for tensor in tensors.iter() {
            let processed = func.call1((tensor,))?;
            result.append(processed)?;
        }
        Ok(result.into())
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        if tensors.len() <= 1 {
            let result = PyList::empty(py);
            for tensor in tensors.iter() {
                let processed = func.call1((tensor,))?;
                result.append(processed)?;
            }
            Ok(result.into())
        } else {
            let num_threads = std::cmp::min(tensors.len(), num_cpus::get());
            let chunk_size = (tensors.len() + num_threads - 1) / num_threads;
            
            let results = Arc::new(Mutex::new(vec![None; tensors.len()]));
            let mut handles = vec![];
            
            for i in 0..num_threads {
                let start = i * chunk_size;
                let end = std::cmp::min(start + chunk_size, tensors.len());
                
                if start >= end {
                    break;
                }
                
                let py_gil = Python::acquire_gil();
                let py = py_gil.python();
                let func = func.to_object(py);
                let tensors_slice = tensors.get_slice(start as isize, end as isize)?.to_object(py);
                let results_clone = Arc::clone(&results);
                
                let handle = thread::spawn(move || {
                    let py_gil = Python::acquire_gil();
                    let py = py_gil.python();
                    let func = func.extract::<&PyAny>(py).unwrap();
                    let tensors_slice = tensors_slice.extract::<&PyList>(py).unwrap();
                    
                    for (i, tensor) in tensors_slice.iter().enumerate() {
                        let processed = func.call1((tensor,)).unwrap();
                        let mut results = results_clone.lock().unwrap();
                        results[start + i] = Some(processed.to_object(py));
                    }
                });
                
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Thread panicked")
                })?;
            }
            
            let result = PyList::empty(py);
            let results = results.lock().unwrap();
            for processed in results.iter() {
                if let Some(obj) = processed {
                    result.append(obj.extract::<&PyAny>(py)?)?;
                }
            }
            
            Ok(result.into())
        }
    }
}

#[pyfunction]
fn parallel_batch_processing(py: Python, model: &PyAny, batches: &PyList) -> PyResult<PyObject> {
    #[cfg(target_arch = "wasm32")]
    {
        let result = PyList::empty(py);
        for batch in batches.iter() {
            let processed = model.call1((batch,))?;
            result.append(processed)?;
        }
        Ok(result.into())
    }
    
    #[cfg(not(target_arch = "wasm32"))]
    {
        if batches.len() <= 1 {
            let result = PyList::empty(py);
            for batch in batches.iter() {
                let processed = model.call1((batch,))?;
                result.append(processed)?;
            }
            Ok(result.into())
        } else {
            let num_threads = std::cmp::min(batches.len(), num_cpus::get());
            let chunk_size = (batches.len() + num_threads - 1) / num_threads;
            
            let results = Arc::new(Mutex::new(vec![None; batches.len()]));
            let mut handles = vec![];
            
            for i in 0..num_threads {
                let start = i * chunk_size;
                let end = std::cmp::min(start + chunk_size, batches.len());
                
                if start >= end {
                    break;
                }
                
                let py_gil = Python::acquire_gil();
                let py = py_gil.python();
                let model = model.to_object(py);
                let batches_slice = batches.get_slice(start as isize, end as isize)?.to_object(py);
                let results_clone = Arc::clone(&results);
                
                let handle = thread::spawn(move || {
                    let py_gil = Python::acquire_gil();
                    let py = py_gil.python();
                    let model = model.extract::<&PyAny>(py).unwrap();
                    let batches_slice = batches_slice.extract::<&PyList>(py).unwrap();
                    
                    for (i, batch) in batches_slice.iter().enumerate() {
                        let processed = model.call1((batch,)).unwrap();
                        let mut results = results_clone.lock().unwrap();
                        results[start + i] = Some(processed.to_object(py));
                    }
                });
                
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("Thread panicked")
                })?;
            }
            
            let result = PyList::empty(py);
            let results = results.lock().unwrap();
            for processed in results.iter() {
                if let Some(obj) = processed {
                    result.append(obj.extract::<&PyAny>(py)?)?;
                }
            }
            
            Ok(result.into())
        }
    }
}

pub fn register_multithreaded(py: Python, m: &PyModule) -> PyResult<()> {
    let submodule = PyModule::new(py, "multithreaded")?;
    
    submodule.add_function(wrap_pyfunction!(is_multithreading_supported, submodule)?)?;
    submodule.add_function(wrap_pyfunction!(enable_multithreading, submodule)?)?;
    submodule.add_function(wrap_pyfunction!(get_num_workers, submodule)?)?;
    submodule.add_function(wrap_pyfunction!(parallel_matmul, submodule)?)?;
    submodule.add_function(wrap_pyfunction!(parallel_conv2d, submodule)?)?;
    submodule.add_function(wrap_pyfunction!(parallel_map, submodule)?)?;
    submodule.add_function(wrap_pyfunction!(parallel_batch_processing, submodule)?)?;
    
    m.add_submodule(submodule)?;
    
    Ok(())
}
