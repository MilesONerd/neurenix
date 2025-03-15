//! Hardware abstraction layer for the Phynexus engine

mod cpu;
mod cuda;
mod rocm;
mod webgpu;
mod tpu;
mod multi_device;

pub use cpu::CpuBackend;
pub use cuda::CudaBackend;
pub use rocm::RocmBackend;
pub use webgpu::WebGpuBackend;
pub use tpu::TpuBackend;
pub use multi_device::{MultiDeviceManager, DeviceInfo};

use crate::error::Result;
use crate::tensor::Tensor;

/// Trait for hardware backends
pub trait Backend: Send + Sync {
    /// Get the name of the backend
    fn name(&self) -> &str;
    
    /// Check if the backend is available
    fn is_available(&self) -> bool;
    
    /// Get the number of devices available for this backend
    fn device_count(&self) -> usize;
    
    /// Allocate memory on the device
    fn allocate(&self, size: usize, device_index: usize) -> Result<*mut u8>;
    
    /// Free memory on the device
    fn free(&self, ptr: *mut u8, device_index: usize) -> Result<()>;
    
    /// Copy memory from host to device
    fn copy_host_to_device(&self, host_ptr: *const u8, device_ptr: *mut u8, size: usize, device_index: usize) -> Result<()>;
    
    /// Copy memory from device to host
    fn copy_device_to_host(&self, device_ptr: *const u8, host_ptr: *mut u8, size: usize, device_index: usize) -> Result<()>;
    
    /// Copy memory from device to device
    fn copy_device_to_device(&self, src_ptr: *const u8, dst_ptr: *mut u8, size: usize, src_device_index: usize, dst_device_index: usize) -> Result<()>;
    
    /// Synchronize the device
    fn synchronize(&self, device_index: usize) -> Result<()>;
}

/// Get the backend for the given device type
pub fn get_backend(device_type: crate::device::DeviceType) -> Result<Box<dyn Backend>> {
    match device_type {
        crate::device::DeviceType::CPU => Ok(Box::new(CpuBackend::new())),
        crate::device::DeviceType::CUDA => {
            let backend = CudaBackend::new()?;
            Ok(Box::new(backend))
        },
        crate::device::DeviceType::ROCm => {
            let backend = RocmBackend::new()?;
            Ok(Box::new(backend))
        },
        crate::device::DeviceType::WebGPU => {
            let backend = WebGpuBackend::new()?;
            Ok(Box::new(backend))
        },
        crate::device::DeviceType::TPU => {
            let backend = TpuBackend::new()?;
            Ok(Box::new(backend))
        },
    }
}

/// Global multi-device manager
static mut MULTI_DEVICE_MANAGER: Option<MultiDeviceManager> = None;

/// Initialize the hardware subsystem
pub fn init() -> Result<()> {
    unsafe {
        if MULTI_DEVICE_MANAGER.is_none() {
            MULTI_DEVICE_MANAGER = Some(MultiDeviceManager::new());
        }
    }
    
    Ok(())
}

/// Get the global multi-device manager
pub fn get_multi_device_manager() -> Result<&'static MultiDeviceManager> {
    unsafe {
        MULTI_DEVICE_MANAGER.as_ref().ok_or_else(|| {
            crate::error::PhynexusError::UninitializedError(
                "Hardware subsystem not initialized".to_string()
            )
        })
    }
}
