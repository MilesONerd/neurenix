//! Device abstraction for the Phynexus engine

use crate::error::{PhynexusError, Result};
use crate::memory::{Memory, CpuMemory};
use std::fmt;
use std::sync::Arc;

/// Represents a computational device (CPU, GPU, etc.)
#[derive(Clone)]
pub struct Device {
    /// The type of the device
    device_type: DeviceType,
    
    /// The index of the device (for multiple devices of the same type)
    index: usize,
}

/// Supported device types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    CPU,
    CUDA,
    ROCm,
    WebGPU,
}

impl Device {
    /// Create a new device with the given type and index
    pub fn new(device_type: DeviceType, index: usize) -> Self {
        Self {
            device_type,
            index,
        }
    }
    
    /// Get the default CPU device
    pub fn cpu() -> Self {
        Self::new(DeviceType::CPU, 0)
    }
    
    /// Get a CUDA device with the given index
    pub fn cuda(index: usize) -> Self {
        Self::new(DeviceType::CUDA, index)
    }
    
    /// Get a ROCm device with the given index
    pub fn rocm(index: usize) -> Self {
        Self::new(DeviceType::ROCm, index)
    }
    
    /// Get a WebGPU device with the given index
    pub fn webgpu(index: usize) -> Self {
        Self::new(DeviceType::WebGPU, index)
    }
    
    /// Get the type of the device
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }
    
    /// Get the index of the device
    pub fn index(&self) -> usize {
        self.index
    }
    
    /// Allocate memory on the device
    pub fn allocate_memory(&self, size: usize) -> Result<Arc<dyn Memory>> {
        match self.device_type {
            DeviceType::CPU => Ok(Arc::new(CpuMemory::new(size)?)),
            DeviceType::CUDA => Err(PhynexusError::UnsupportedOperation(
                "CUDA memory allocation not yet implemented".to_string()
            )),
            DeviceType::ROCm => Err(PhynexusError::UnsupportedOperation(
                "ROCm memory allocation not yet implemented".to_string()
            )),
            DeviceType::WebGPU => Err(PhynexusError::UnsupportedOperation(
                "WebGPU memory allocation not yet implemented".to_string()
            )),
        }
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}({})", self.device_type, self.index)
    }
}
