//! Multi-device management for the Phynexus engine

use crate::device::DeviceType;
use crate::error::{PhynexusError, Result};
use crate::hardware::{Backend, CpuBackend, CudaBackend, RocmBackend, WebGpuBackend, TpuBackend};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Multi-device manager for coordinating operations across multiple devices
pub struct MultiDeviceManager {
    /// Available backends
    backends: HashMap<DeviceType, Arc<dyn Backend + 'static>>,
    
    /// Device information
    devices: Vec<DeviceInfo>,
    
    /// Active device ID
    active_device: Mutex<usize>,
}

/// Information about a device
pub struct DeviceInfo {
    /// Device type
    pub device_type: DeviceType,
    
    /// Device index
    pub device_index: usize,
    
    /// Device name
    pub name: String,
    
    /// Device memory capacity in bytes
    pub memory_capacity: usize,
    
    /// Backend reference
    pub backend: Arc<dyn Backend + 'static>,
}

impl MultiDeviceManager {
    /// Create a new multi-device manager
    pub fn new() -> Result<Self> {
        let mut backends: HashMap<DeviceType, Arc<dyn Backend + 'static>> = HashMap::new();
        let mut devices = Vec::new();
        
        // Add CPU backend
        let cpu_backend = Arc::new(CpuBackend::new()) as Arc<dyn Backend + 'static>;
        backends.insert(DeviceType::CPU, cpu_backend.clone());
        
        // Add CPU device
        devices.push(DeviceInfo {
            device_type: DeviceType::CPU,
            device_index: 0,
            name: "CPU".to_string(),
            memory_capacity: 0, // Unknown
            backend: cpu_backend.clone(),
        });
        
        // Try to initialize CUDA backend
        if let Ok(cuda_backend) = CudaBackend::new() {
            let cuda_backend = Arc::new(cuda_backend) as Arc<dyn Backend + 'static>;
            backends.insert(DeviceType::CUDA, cuda_backend.clone());
            
            // Add CUDA devices
            if let Ok(device_count) = cuda_backend.get_device_count() {
                for i in 0..device_count {
                    devices.push(DeviceInfo {
                        device_type: DeviceType::CUDA,
                        device_index: i,
                        name: format!("CUDA Device {}", i),
                        memory_capacity: 0, // Unknown
                        backend: cuda_backend.clone(),
                    });
                }
            }
        }
        
        // Try to initialize ROCm backend
        if let Ok(rocm_backend) = RocmBackend::new() {
            let rocm_backend = Arc::new(rocm_backend) as Arc<dyn Backend + 'static>;
            backends.insert(DeviceType::ROCm, rocm_backend.clone());
            
            // Add ROCm devices
            if let Ok(device_count) = rocm_backend.get_device_count() {
                for i in 0..device_count {
                    devices.push(DeviceInfo {
                        device_type: DeviceType::ROCm,
                        device_index: i,
                        name: format!("ROCm Device {}", i),
                        memory_capacity: 0, // Unknown
                        backend: rocm_backend.clone(),
                    });
                }
            }
        }
        
        // Try to initialize WebGPU backend
        if let Ok(webgpu_backend) = WebGpuBackend::new() {
            let webgpu_backend = Arc::new(webgpu_backend) as Arc<dyn Backend + 'static>;
            backends.insert(DeviceType::WebGPU, webgpu_backend.clone());
            
            // Add WebGPU devices
            if let Ok(device_count) = webgpu_backend.get_device_count() {
                for i in 0..device_count {
                    devices.push(DeviceInfo {
                        device_type: DeviceType::WebGPU,
                        device_index: i,
                        name: format!("WebGPU Device {}", i),
                        memory_capacity: 0, // Unknown
                        backend: webgpu_backend.clone(),
                    });
                }
            }
        }
        
        // Try to initialize TPU backend
        if let Ok(tpu_backend) = TpuBackend::new() {
            let tpu_backend = Arc::new(tpu_backend) as Arc<dyn Backend + 'static>;
            backends.insert(DeviceType::TPU, tpu_backend.clone());
            
            // Add TPU devices
            if let Ok(device_count) = tpu_backend.get_device_count() {
                for i in 0..device_count {
                    devices.push(DeviceInfo {
                        device_type: DeviceType::TPU,
                        device_index: i,
                        name: format!("TPU Device {}", i),
                        memory_capacity: 0, // Unknown
                        backend: tpu_backend.clone(),
                    });
                }
            }
        }
        
        Ok(Self {
            backends,
            devices,
            active_device: Mutex::new(0),
        })
    }
    
    /// Get the number of devices
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }
    
    /// Get information about a device
    pub fn device_info(&self, device_id: usize) -> Result<&DeviceInfo> {
        self.devices.get(device_id).ok_or_else(|| {
            PhynexusError::InvalidArgument(format!("Invalid device ID: {}", device_id))
        })
    }
    
    /// Set the active device
    pub fn set_active_device(&self, device_id: usize) -> Result<()> {
        if device_id >= self.devices.len() {
            return Err(PhynexusError::InvalidArgument(format!(
                "Invalid device ID: {}", device_id
            )));
        }
        
        *self.active_device.lock().unwrap() = device_id;
        Ok(())
    }
    
    /// Get the active device
    pub fn get_active_device(&self) -> usize {
        *self.active_device.lock().unwrap()
    }
    
    /// Get the backend for a device
    pub fn get_backend(&self, device_id: usize) -> Result<Arc<dyn Backend + 'static>> {
        let device_info = self.device_info(device_id)?;
        Ok(device_info.backend.clone())
    }
    
    /// Allocate memory on a device
    pub fn allocate(&self, size: usize, device_id: usize) -> Result<*mut u8> {
        let device_info = self.device_info(device_id)?;
        device_info.backend.allocate(size, device_info.device_index)
    }
    
    /// Free memory on a device
    pub fn free(&self, ptr: *mut u8, device_id: usize) -> Result<()> {
        let device_info = self.device_info(device_id)?;
        device_info.backend.free(ptr, device_info.device_index)
    }
    
    /// Copy data from host to device
    pub fn copy_host_to_device(&self, host_ptr: *const u8, device_ptr: *mut u8, size: usize, device_id: usize) -> Result<()> {
        let device_info = self.device_info(device_id)?;
        device_info.backend.copy_host_to_device(host_ptr, device_ptr, size, device_info.device_index)
    }
    
    /// Copy data from device to host
    pub fn copy_device_to_host(&self, device_ptr: *const u8, host_ptr: *mut u8, size: usize, device_id: usize) -> Result<()> {
        let device_info = self.device_info(device_id)?;
        device_info.backend.copy_device_to_host(device_ptr, host_ptr, size, device_info.device_index)
    }
    
    /// Copy data from device to device
    pub fn copy_device_to_device(&self, src_ptr: *const u8, dst_ptr: *mut u8, size: usize, src_device_id: usize, dst_device_id: usize) -> Result<()> {
        let src_device_info = self.device_info(src_device_id)?;
        let dst_device_info = self.device_info(dst_device_id)?;
        
        src_device_info.backend.copy_device_to_device(
            src_ptr,
            dst_ptr,
            size,
            src_device_info.device_index,
            dst_device_info.device_index,
        )
    }
    
    /// Synchronize a device
    pub fn synchronize(&self, device_id: usize) -> Result<()> {
        let device_info = self.device_info(device_id)?;
        device_info.backend.synchronize(device_info.device_index)
    }
}
