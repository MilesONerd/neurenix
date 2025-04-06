
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyValueError;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use crate::error::PhynexusError;
use crate::tensor::Tensor;
use crate::nn::Module;
use crate::optimizer::Optimizer;

#[pyfunction]
pub fn start_async_training(
    trainer: &PyAny,
    model: &PyAny,
    dataloader: &PyAny,
    loss_fn: &PyAny,
    optimizer: &PyAny,
    checkpoint_manager: &PyAny,
    max_steps: Option<usize>,
    checkpoint_interval: Option<f64>,
    metrics_fn: Option<&PyAny>,
) -> PyResult<PyObject> {
    let py = trainer.py();
    
    let running = trainer.getattr("_running")?;
    
    if running.extract::<bool>()? {
        return Err(PyValueError::new_err("Async trainer is already running"));
    }
    
    trainer.setattr("_running", true)?;
    
    let interval = match checkpoint_interval {
        Some(i) => i,
        None => trainer.getattr("checkpoint_interval")?.extract::<f64>()?,
    };
    
    let steps = match max_steps {
        Some(s) => s,
        None => trainer.getattr("max_steps")?.extract::<usize>()?,
    };
    
    let status = PyDict::new(py);
    status.set_item("step", 0)?;
    status.set_item("running", true)?;
    status.set_item("last_checkpoint", py.None())?;
    status.set_item("metrics", PyDict::new(py))?;
    
    trainer.setattr("_status", status.clone())?;
    
    let thread_builder = thread::Builder::new().name("async_trainer".to_string());
    
    thread_builder.spawn(move || {
        let gil = Python::acquire_gil();
        let py = gil.python();
        
        let locals = PyDict::new(py);
        locals.set_item("trainer", trainer)?;
        locals.set_item("model", model)?;
        locals.set_item("dataloader", dataloader)?;
        locals.set_item("loss_fn", loss_fn)?;
        locals.set_item("optimizer", optimizer)?;
        locals.set_item("checkpoint_manager", checkpoint_manager)?;
        locals.set_item("metrics_fn", metrics_fn.unwrap_or(py.None()))?;
        locals.set_item("max_steps", steps)?;
        locals.set_item("checkpoint_interval", interval)?;
        locals.set_item("status", status)?;
        
        let code = r#"
import time
import traceback

try:
    # Initialize
    model.train()
    step = status.get('step', 0)
    last_checkpoint_time = time.time()
    
    # Training loop
    while status['running'] and step < max_steps:
        for batch in dataloader:
            if not status['running']:
                break
                
            # Get batch data
            inputs, targets = batch
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update step
            step += 1
            status['step'] = step
            
            # Compute metrics if provided
            if metrics_fn is not None:
                metrics = metrics_fn(outputs, targets)
                if isinstance(metrics, dict):
                    for k, v in metrics.items():
                        if k not in status['metrics']:
                            status['metrics'][k] = []
                        status['metrics'][k].append(v)
            
            # Check if checkpoint interval has elapsed
            current_time = time.time()
            if current_time - last_checkpoint_time >= checkpoint_interval:
                # Save checkpoint
                checkpoint_path = checkpoint_manager.save(
                    'main',
                    model,
                    optimizer,
                    step=step,
                    metrics=status['metrics'],
                    metadata={'async_trainer': True}
                )
                
                status['last_checkpoint'] = checkpoint_path
                last_checkpoint_time = current_time
            
            # Check if max steps reached
            if step >= max_steps:
                break
    
    # Final checkpoint
    if status['running']:
        checkpoint_path = checkpoint_manager.save(
            'main',
            model,
            optimizer,
            step=step,
            metrics=status['metrics'],
            metadata={'async_trainer': True, 'final': True}
        )
        
        status['last_checkpoint'] = checkpoint_path
    
    # Update status
    status['running'] = False

except Exception as e:
    status['error'] = str(e)
    status['traceback'] = traceback.format_exc()
    status['running'] = False
    raise
"#;
        
        py.run(code, None, Some(locals))?;
        
        Ok(())
    }).map_err(|e| {
        let _ = trainer.setattr("_running", false);
        PyValueError::new_err(format!("Failed to start async training thread: {}", e))
    })?;
    
    Ok(status.to_object(py))
}

#[pyfunction]
pub fn stop_async_training(
    trainer: &PyAny,
    wait: Option<bool>,
    timeout: Option<f64>,
) -> PyResult<PyObject> {
    let py = trainer.py();
    
    let running = trainer.getattr("_running")?;
    let status = trainer.getattr("_status")?;
    
    if !running.extract::<bool>()? {
        return Err(PyValueError::new_err("Async trainer is not running"));
    }
    
    status.set_item("running", false)?;
    
    if wait.unwrap_or(false) {
        let timeout_secs = timeout.unwrap_or(60.0);
        let start_time = Instant::now();
        
        while running.extract::<bool>()? {
            if start_time.elapsed().as_secs_f64() > timeout_secs {
                return Err(PyValueError::new_err("Timeout waiting for async training to stop"));
            }
            
            py.allow_threads(|| {
                thread::sleep(Duration::from_millis(100));
            });
        }
    }
    
    trainer.setattr("_running", false)?;
    
    Ok(status.to_object(py))
}

#[pyfunction]
pub fn get_training_status(
    trainer: &PyAny,
) -> PyResult<PyObject> {
    let py = trainer.py();
    
    let status = trainer.getattr("_status")?;
    
    Ok(status.to_object(py))
}
