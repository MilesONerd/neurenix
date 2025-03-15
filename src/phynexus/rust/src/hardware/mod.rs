//! Hardware backends for the Phynexus engine

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

/// Backend trait for hardware-specific operations
pub trait Backend: Send + Sync {
    /// Get the number of devices for this backend
    fn get_device_count(&self) -> Result<usize>;
    
    /// Allocate memory on a device
    fn allocate(&self, size: usize, device_index: usize) -> Result<*mut u8>;
    
    /// Free memory on a device
    fn free(&self, ptr: *mut u8, device_index: usize) -> Result<()>;
    
    /// Copy data from host to device
    fn copy_host_to_device(&self, host_ptr: *const u8, device_ptr: *mut u8, size: usize, device_index: usize) -> Result<()>;
    
    /// Copy data from device to host
    fn copy_device_to_host(&self, device_ptr: *const u8, host_ptr: *mut u8, size: usize, device_index: usize) -> Result<()>;
    
    /// Copy data from device to device
    fn copy_device_to_device(&self, src_ptr: *const u8, dst_ptr: *mut u8, size: usize, src_device_index: usize, dst_device_index: usize) -> Result<()>;
    
    /// Synchronize a device
    fn synchronize(&self, device_index: usize) -> Result<()>;
}

/// Get the backend for the given device type
pub fn get_backend(device_type: crate::device::DeviceType) -> Result<Box<dyn Backend + 'static>> {
    match device_type {
        crate::device::DeviceType::CPU => Ok(Box::new(CpuBackend::new())),
        crate::device::DeviceType::CUDA => Ok(Box::new(CudaBackend::new()?)),
        crate::device::DeviceType::ROCm => Ok(Box::new(RocmBackend::new()?)),
        crate::device::DeviceType::WebGPU => Ok(Box::new(WebGpuBackend::new()?)),
        crate::device::DeviceType::TPU => Ok(Box::new(TpuBackend::new()?)),
    }
}
