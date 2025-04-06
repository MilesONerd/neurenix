
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyValueError;
use std::path::Path;
use std::fs;
use std::io;

use crate::error::PhynexusError;
use crate::tensor::Tensor;
use crate::nn::Module;
use crate::optimizer::Optimizer;

#[pyfunction]
pub fn save_checkpoint(
    checkpoint_dir: &str,
    checkpoint_name: &str,
    model: &PyAny,
    optimizer: Option<&PyAny>,
    step: Option<usize>,
    metrics: Option<&PyDict>,
    metadata: Option<&PyDict>,
    atomic: Option<bool>,
) -> PyResult<String> {
    let py = model.py();
    
    let dir_path = Path::new(checkpoint_dir);
    if !dir_path.exists() {
        fs::create_dir_all(dir_path).map_err(|e| {
            PyValueError::new_err(format!("Failed to create checkpoint directory: {}", e))
        })?;
    }
    
    let step_str = match step {
        Some(s) => format!("_{}", s),
        None => String::new(),
    };
    
    let checkpoint_path = format!("{}/{}{}_{}.pt", checkpoint_dir, checkpoint_name, step_str, timestamp());
    let temp_path = format!("{}.tmp", checkpoint_path);
    
    let state_dict = model.call_method0("state_dict")?;
    
    let checkpoint = PyDict::new(py);
    checkpoint.set_item("model_state_dict", state_dict)?;
    
    if let Some(opt) = optimizer {
        let opt_state_dict = opt.call_method0("state_dict")?;
        checkpoint.set_item("optimizer_state_dict", opt_state_dict)?;
    }
    
    if let Some(s) = step {
        checkpoint.set_item("step", s)?;
    }
    
    if let Some(m) = metrics {
        checkpoint.set_item("metrics", m)?;
    }
    
    if let Some(m) = metadata {
        checkpoint.set_item("metadata", m)?;
    }
    
    checkpoint.set_item("timestamp", timestamp())?;
    
    let save_path = if atomic.unwrap_or(true) {
        py.import("torch")?.call_method1("save", (checkpoint, &temp_path))?;
        
        fs::rename(&temp_path, &checkpoint_path).map_err(|e| {
            let _ = fs::remove_file(&temp_path);
            PyValueError::new_err(format!("Failed to save checkpoint: {}", e))
        })?;
        
        checkpoint_path
    } else {
        py.import("torch")?.call_method1("save", (checkpoint, &checkpoint_path))?;
        checkpoint_path
    };
    
    Ok(save_path)
}

#[pyfunction]
pub fn load_checkpoint(
    checkpoint_dir: &str,
    checkpoint_name: &str,
    model: &PyAny,
    optimizer: Option<&PyAny>,
    step: Option<usize>,
    latest: Option<bool>,
) -> PyResult<PyObject> {
    let py = model.py();
    
    let checkpoint_path = if let Some(s) = step {
        let pattern = format!("{}/{}_{}_*.pt", checkpoint_dir, checkpoint_name, s);
        find_checkpoint(&pattern, false)?
    } else if latest.unwrap_or(true) {
        let pattern = format!("{}/{}_*.pt", checkpoint_dir, checkpoint_name);
        find_checkpoint(&pattern, true)?
    } else {
        return Err(PyValueError::new_err(
            "Either step or latest must be specified"
        ));
    };
    
    let checkpoint = py.import("torch")?.call_method1("load", (checkpoint_path,))?;
    
    let model_state_dict = checkpoint.get_item("model_state_dict")?;
    model.call_method1("load_state_dict", (model_state_dict,))?;
    
    if let Some(opt) = optimizer {
        if checkpoint.contains("optimizer_state_dict")? {
            let opt_state_dict = checkpoint.get_item("optimizer_state_dict")?;
            opt.call_method1("load_state_dict", (opt_state_dict,))?;
        }
    }
    
    Ok(checkpoint.to_object(py))
}

#[pyfunction]
pub fn list_checkpoints(
    checkpoint_dir: &str,
    checkpoint_name: &str,
) -> PyResult<PyObject> {
    let py = Python::acquire_gil().python();
    
    let pattern = format!("{}/{}_*.pt", checkpoint_dir, checkpoint_name);
    let checkpoint_files = glob_files(&pattern)?;
    
    let mut checkpoint_files: Vec<String> = checkpoint_files.into_iter().collect();
    checkpoint_files.sort_by(|a, b| {
        let a_time = extract_timestamp(a).unwrap_or(0);
        let b_time = extract_timestamp(b).unwrap_or(0);
        b_time.cmp(&a_time)
    });
    
    let checkpoint_list = PyList::empty(py);
    
    for file in checkpoint_files {
        let checkpoint = py.import("torch")?.call_method1("load", (file.clone(),))?;
        
        let info = PyDict::new(py);
        
        info.set_item("path", file)?;
        
        if checkpoint.contains("step")? {
            let step = checkpoint.get_item("step")?;
            info.set_item("step", step)?;
        }
        
        if checkpoint.contains("timestamp")? {
            let timestamp = checkpoint.get_item("timestamp")?;
            info.set_item("timestamp", timestamp)?;
        }
        
        if checkpoint.contains("metadata")? {
            let metadata = checkpoint.get_item("metadata")?;
            info.set_item("metadata", metadata)?;
        }
        
        checkpoint_list.append(info)?;
    }
    
    Ok(checkpoint_list.to_object(py))
}

#[pyfunction]
pub fn delete_checkpoint(
    checkpoint_path: &str,
) -> PyResult<bool> {
    fs::remove_file(checkpoint_path).map_err(|e| {
        PyValueError::new_err(format!("Failed to delete checkpoint: {}", e))
    })?;
    
    Ok(true)
}

fn find_checkpoint(pattern: &str, latest: bool) -> PyResult<String> {
    let checkpoint_files = glob_files(pattern)?;
    
    if checkpoint_files.is_empty() {
        return Err(PyValueError::new_err(
            format!("No checkpoint found matching pattern: {}", pattern)
        ));
    }
    
    if latest {
        let mut latest_file = checkpoint_files[0].clone();
        let mut latest_time = extract_timestamp(&latest_file).unwrap_or(0);
        
        for file in &checkpoint_files[1..] {
            if let Some(time) = extract_timestamp(file) {
                if time > latest_time {
                    latest_time = time;
                    latest_file = file.clone();
                }
            }
        }
        
        Ok(latest_file)
    } else {
        Ok(checkpoint_files[0].clone())
    }
}

fn glob_files(pattern: &str) -> PyResult<Vec<String>> {
    let py = Python::acquire_gil().python();
    
    let glob_module = py.import("glob")?;
    let files = glob_module.call_method1("glob", (pattern,))?;
    
    let mut result = Vec::new();
    for file in files.iter()? {
        result.push(file?.extract::<String>()?);
    }
    
    Ok(result)
}

fn extract_timestamp(filename: &str) -> Option<u64> {
    let path = Path::new(filename);
    let stem = path.file_stem()?.to_str()?;
    
    let parts: Vec<&str> = stem.split('_').collect();
    if parts.len() < 2 {
        return None;
    }
    
    parts.last()?.parse::<u64>().ok()
}

fn timestamp() -> u64 {
    let py = Python::acquire_gil().python();
    
    py.import("time")
        .and_then(|time| time.call_method0("time"))
        .and_then(|time_float| time_float.call_method0("__int__"))
        .and_then(|time_int| time_int.extract::<u64>())
        .unwrap_or(0)
}
