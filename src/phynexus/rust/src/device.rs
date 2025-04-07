//! Device abstraction for the Phynexus engine

use crate::error::{PhynexusError, Result};
use crate::hardware::MultiDeviceManager;
use std::fmt;
use std::sync::{Arc, Once};

/// Device type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU device
    CPU,
    
    /// CUDA device
    CUDA,
    
    /// ROCm device
    ROCm,
    
    /// WebGPU device
    WebGPU,
    
    /// TPU device
    TPU,
    
    Vulkan,
    
    OpenCL,
    
    OneAPI,
    
    DirectML,
    
    OneDNN,
    
    MKLDNN,
    
    TensorRT,
    
    GraphCore,
    
    FPGA,
    
    TensorCores,
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::CPU => write!(f, "CPU"),
            DeviceType::CUDA => write!(f, "CUDA"),
            DeviceType::ROCm => write!(f, "ROCm"),
            DeviceType::WebGPU => write!(f, "WebGPU"),
            DeviceType::TPU => write!(f, "TPU"),
            DeviceType::Vulkan => write!(f, "Vulkan"),
            DeviceType::OpenCL => write!(f, "OpenCL"),
            DeviceType::OneAPI => write!(f, "oneAPI"),
            DeviceType::DirectML => write!(f, "DirectML"),
            DeviceType::OneDNN => write!(f, "oneDNN"),
            DeviceType::MKLDNN => write!(f, "MKL-DNN"),
            DeviceType::TensorRT => write!(f, "TensorRT"),
            DeviceType::GraphCore => write!(f, "GraphCore"),
            DeviceType::FPGA => write!(f, "FPGA"),
            DeviceType::TensorCores => write!(f, "TensorCores"),
        }
    }
}

/// Device abstraction for tensor operations
#[derive(Clone)]
pub struct Device {
    /// Device type
    device_type: DeviceType,
    
    /// Device index
    device_index: usize,
    
    /// Multi-device manager
    manager: Arc<MultiDeviceManager>,
}

// Global multi-device manager is now handled inside get_manager()

impl Device {
    /// Create a new device
    pub fn new(device_type: DeviceType, device_index: usize) -> Result<Self> {
        let manager = Self::get_manager()?;
        
        // Check if the device exists
        let device_count = manager.device_count();
        let mut found = false;
        
        for i in 0..device_count {
            let device_info = manager.device_info(i)?;
            if device_info.device_type == device_type && device_info.device_index == device_index {
                found = true;
                break;
            }
        }
        
        if !found {
            return Err(PhynexusError::InvalidArgument(format!(
                "Device {}:{} not found", device_type, device_index
            )));
        }
        
        Ok(Self {
            device_type,
            device_index,
            manager,
        })
    }
    
    /// Create a CPU device
    pub fn cpu() -> Self {
        Self::new(DeviceType::CPU, 0).unwrap()
    }
    
    /// Create a CUDA device
    pub fn cuda(device_index: usize) -> Self {
        Self::new(DeviceType::CUDA, device_index).unwrap_or_else(|_| Self::cpu())
    }
    
    /// Create a ROCm device
    pub fn rocm(device_index: usize) -> Self {
        Self::new(DeviceType::ROCm, device_index).unwrap_or_else(|_| Self::cpu())
    }
    
    /// Create a WebGPU device
    pub fn webgpu() -> Self {
        Self::new(DeviceType::WebGPU, 0).unwrap_or_else(|_| Self::cpu())
    }
    
    /// Create a TPU device
    pub fn tpu(device_index: usize) -> Self {
        Self::new(DeviceType::TPU, device_index).unwrap_or_else(|_| Self::cpu())
    }
    
    /// Get the device type
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }
    
    /// Get the device index
    pub fn device_index(&self) -> usize {
        self.device_index
    }
    
    /// Allocate memory on the device
    pub fn allocate(&self, size: usize) -> Result<*mut u8> {
        let device_id = self.get_device_id()?;
        self.manager.allocate(size, device_id)
    }
    
    /// Free memory on the device
    pub fn free(&self, ptr: *mut u8) -> Result<()> {
        let device_id = self.get_device_id()?;
        self.manager.free(ptr, device_id)
    }
    
    /// Copy data from host to device
    pub fn copy_from_host(&self, host_ptr: *const u8, device_ptr: *mut u8, size: usize) -> Result<()> {
        let device_id = self.get_device_id()?;
        self.manager.copy_host_to_device(host_ptr, device_ptr, size, device_id)
    }
    
    /// Copy data from device to host
    pub fn copy_to_host(&self, device_ptr: *const u8, host_ptr: *mut u8, size: usize) -> Result<()> {
        let device_id = self.get_device_id()?;
        self.manager.copy_device_to_host(device_ptr, host_ptr, size, device_id)
    }
    
    /// Copy data from another device
    pub fn copy_from_device(&self, src_ptr: *const u8, src_device: Device, dst_ptr: *mut u8) -> Result<()> {
        let src_device_id = src_device.get_device_id()?;
        let dst_device_id = self.get_device_id()?;
        
        self.manager.copy_device_to_device(src_ptr, dst_ptr, 0, src_device_id, dst_device_id)
    }
    
    /// Synchronize the device
    pub fn synchronize(&self) -> Result<()> {
        let device_id = self.get_device_id()?;
        self.manager.synchronize(device_id)
    }
    
    /// Get the global multi-device manager
    fn get_manager() -> Result<Arc<MultiDeviceManager>> {
        // Thread-safe singleton pattern using Once
        static INIT: Once = Once::new();
        static mut MANAGER: Option<Arc<MultiDeviceManager>> = None;
        
        unsafe {
            INIT.call_once(|| {
                // This code runs exactly once across all threads
                match MultiDeviceManager::new() {
                    Ok(manager) => {
                        MANAGER = Some(Arc::new(manager));
                    },
                    Err(e) => {
                        // We can't return an error from call_once, so we'll just log it
                        eprintln!("Failed to initialize MultiDeviceManager: {:?}", e);
                    }
                }
            });
            
            // After initialization, MANAGER is either Some or None
            // Using a direct access to avoid the shared reference to mutable static warning
            if let Some(ref manager) = MANAGER {
                Ok(manager.clone())
            } else {
                Err(PhynexusError::UninitializedError(
                    "Failed to initialize MultiDeviceManager".to_string()
                ))
            }
        }
    }
    
    /// Get the device ID in the multi-device manager
    fn get_device_id(&self) -> Result<usize> {
        let device_count = self.manager.device_count();
        
        for i in 0..device_count {
            let device_info = self.manager.device_info(i)?;
            if device_info.device_type == self.device_type && device_info.device_index == self.device_index {
                return Ok(i);
            }
        }
        
        Err(PhynexusError::InvalidArgument(format!(
            "Device {}:{} not found", self.device_type, self.device_index
        )))
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("device_type", &self.device_type)
            .field("device_index", &self.device_index)
            .finish()
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        self.device_type == other.device_type && self.device_index == other.device_index
    }
}

impl Eq for Device {}
