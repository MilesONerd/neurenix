
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyRuntimeError;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;

lazy_static::lazy_static! {
    static ref MPI_INITIALIZED: AtomicBool = AtomicBool::new(false);
    static ref MPI_WORLD_SIZE: RwLock<i32> = RwLock::new(1);
    static ref MPI_RANK: RwLock<i32> = RwLock::new(0);
    static ref MPI_LOCAL_RANK: RwLock<i32> = RwLock::new(0);
}

#[pyfunction]
fn mpi_initialize(
    py: Python,
    backend: Option<String>,
    init_method: Option<String>,
    world_size: Option<i32>,
    rank: Option<i32>,
    local_rank: Option<i32>,
    timeout: Option<f64>,
) -> PyResult<PyObject> {
    if MPI_INITIALIZED.load(Ordering::SeqCst) {
        let result = PyDict::new(py);
        result.set_item("world_size", *MPI_WORLD_SIZE.read().unwrap())?;
        result.set_item("rank", *MPI_RANK.read().unwrap())?;
        result.set_item("local_rank", *MPI_LOCAL_RANK.read().unwrap())?;
        return Ok(result.into());
    }

    let backend = backend.unwrap_or_else(|| "openmpi".to_string());
    let init_method = init_method.unwrap_or_else(|| "env".to_string());
    let timeout = timeout.unwrap_or(1800.0);

    let world_size = match world_size {
        Some(ws) => ws,
        None => {
            std::env::var("OMPI_COMM_WORLD_SIZE")
                .or_else(|_| std::env::var("PMI_SIZE"))
                .or_else(|_| std::env::var("MPI_WORLD_SIZE"))
                .map(|s| s.parse::<i32>().unwrap_or(1))
                .unwrap_or(1)
        }
    };

    let rank = match rank {
        Some(r) => r,
        None => {
            std::env::var("OMPI_COMM_WORLD_RANK")
                .or_else(|_| std::env::var("PMI_RANK"))
                .or_else(|_| std::env::var("MPI_RANK"))
                .map(|s| s.parse::<i32>().unwrap_or(0))
                .unwrap_or(0)
        }
    };

    let local_rank = match local_rank {
        Some(lr) => lr,
        None => {
            std::env::var("OMPI_COMM_WORLD_LOCAL_RANK")
                .or_else(|_| std::env::var("MPI_LOCAL_RANK"))
                .or_else(|_| std::env::var("LOCAL_RANK"))
                .map(|s| s.parse::<i32>().unwrap_or(0))
                .unwrap_or(0)
        }
    };

    //     
    //     
    
    *MPI_WORLD_SIZE.write().unwrap() = world_size;
    *MPI_RANK.write().unwrap() = rank;
    *MPI_LOCAL_RANK.write().unwrap() = local_rank;
    
    MPI_INITIALIZED.store(true, Ordering::SeqCst);
    
    let result = PyDict::new(py);
    result.set_item("world_size", world_size)?;
    result.set_item("rank", rank)?;
    result.set_item("local_rank", local_rank)?;
    
    Ok(result.into())
}

#[pyfunction]
fn mpi_finalize(py: Python) -> PyResult<()> {
    if !MPI_INITIALIZED.load(Ordering::SeqCst) {
        return Err(PyRuntimeError::new_err("MPI is not initialized"));
    }
    
    
    MPI_INITIALIZED.store(false, Ordering::SeqCst);
    
    Ok(())
}

#[pyfunction]
fn mpi_barrier(py: Python) -> PyResult<()> {
    if !MPI_INITIALIZED.load(Ordering::SeqCst) {
        return Err(PyRuntimeError::new_err("MPI is not initialized"));
    }
    
    
    thread::sleep(Duration::from_millis(10));
    
    Ok(())
}

#[pyfunction]
fn mpi_broadcast(py: Python, data: PyObject, src: Option<i32>) -> PyResult<PyObject> {
    if !MPI_INITIALIZED.load(Ordering::SeqCst) {
        return Err(PyRuntimeError::new_err("MPI is not initialized"));
    }
    
    let src = src.unwrap_or(0);
    let rank = *MPI_RANK.read().unwrap();
    
    
    
    Ok(data)
}

#[pyfunction]
fn mpi_all_reduce(py: Python, data: PyObject, op: Option<String>) -> PyResult<PyObject> {
    if !MPI_INITIALIZED.load(Ordering::SeqCst) {
        return Err(PyRuntimeError::new_err("MPI is not initialized"));
    }
    
    let op = op.unwrap_or_else(|| "sum".to_string());
    
    
    
    Ok(data)
}

#[pyfunction]
fn mpi_all_gather(py: Python, data: PyObject) -> PyResult<PyObject> {
    if !MPI_INITIALIZED.load(Ordering::SeqCst) {
        return Err(PyRuntimeError::new_err("MPI is not initialized"));
    }
    
    
    
    let world_size = *MPI_WORLD_SIZE.read().unwrap();
    let result = PyList::new(py, &vec![data.clone_ref(py); world_size as usize]);
    
    Ok(result.into())
}

#[pyfunction]
fn mpi_scatter(py: Python, data: PyObject, src: Option<i32>) -> PyResult<PyObject> {
    if !MPI_INITIALIZED.load(Ordering::SeqCst) {
        return Err(PyRuntimeError::new_err("MPI is not initialized"));
    }
    
    let src = src.unwrap_or(0);
    let rank = *MPI_RANK.read().unwrap();
    
    
    
    if let Ok(data_list) = data.extract::<&PyList>(py) {
        let world_size = *MPI_WORLD_SIZE.read().unwrap();
        if data_list.len() == world_size as usize {
            return Ok(data_list.get_item(rank as usize)?.into());
        }
    }
    
    Ok(data)
}

pub fn register_mpi(py: Python, m: &PyModule) -> PyResult<()> {
    let mpi = PyModule::new(py, "mpi")?;
    
    mpi.add_function(wrap_pyfunction!(mpi_initialize, mpi)?)?;
    mpi.add_function(wrap_pyfunction!(mpi_finalize, mpi)?)?;
    mpi.add_function(wrap_pyfunction!(mpi_barrier, mpi)?)?;
    mpi.add_function(wrap_pyfunction!(mpi_broadcast, mpi)?)?;
    mpi.add_function(wrap_pyfunction!(mpi_all_reduce, mpi)?)?;
    mpi.add_function(wrap_pyfunction!(mpi_all_gather, mpi)?)?;
    mpi.add_function(wrap_pyfunction!(mpi_scatter, mpi)?)?;
    
    m.add_submodule(mpi)?;
    
    Ok(())
}
