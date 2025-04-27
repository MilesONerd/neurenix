
mod checkpoint;
mod async_trainer;
mod resume;
mod distributed;

pub use checkpoint::*;
pub use async_trainer::*;
pub use resume::*;
pub use distributed::*;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub fn register_async_train(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let async_train = PyModule::new(py, "async_train")?;
    
    async_train.add_function(wrap_pyfunction!(save_checkpoint, async_train)?)?;
    async_train.add_function(wrap_pyfunction!(load_checkpoint, async_train)?)?;
    async_train.add_function(wrap_pyfunction!(list_checkpoints, async_train)?)?;
    async_train.add_function(wrap_pyfunction!(delete_checkpoint, async_train)?)?;
    
    async_train.add_function(wrap_pyfunction!(start_async_training, async_train)?)?;
    async_train.add_function(wrap_pyfunction!(stop_async_training, async_train)?)?;
    async_train.add_function(wrap_pyfunction!(get_training_status, async_train)?)?;
    
    async_train.add_function(wrap_pyfunction!(init_auto_resume, async_train)?)?;
    async_train.add_function(wrap_pyfunction!(check_system_resources, async_train)?)?;
    async_train.add_function(wrap_pyfunction!(register_signal_handlers, async_train)?)?;
    
    async_train.add_function(wrap_pyfunction!(init_distributed_network, async_train)?)?;
    async_train.add_function(wrap_pyfunction!(distributed_checkpoint_sync_loop, async_train)?)?;
    async_train.add_function(wrap_pyfunction!(distributed_checkpoint_barrier, async_train)?)?;
    async_train.add_function(wrap_pyfunction!(apply_differential_privacy, async_train)?)?;
    
    m.add_submodule(&async_train)?;
    
    Ok(())
}
