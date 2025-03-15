//! Multi-device support for the Phynexus engine

use std::sync::{Arc, Mutex};
use std::collections::HashMap;

use crate::error::{PhynexusError, Result};
use crate::hardware::Backend;
use crate::device::DeviceType;

/// Multi-device manager for coordinating operations across multiple devices
pub struct MultiDeviceManager {
    /// Available backends
    backends: HashMap<DeviceType, Arc<dyn Backend>>,
    
    /// Device information
    devices: Vec<DeviceInfo>,
    
    /// Current active device
    active_device: Mutex<usize>,
}

/// Information about a device
pub struct DeviceInfo {
    /// Device type
    device_type: DeviceType,
    
    /// Device index within its backend
    device_index: usize,
    
    /// Device name
    name: String,
    
    /// Device memory capacity in bytes
    memory_capacity: usize,
    
    /// Backend reference
    backend: Arc<dyn Backend>,
}

impl MultiDeviceManager {
    /// Create a new multi-device manager
    pub fn new() -> Self {
        let mut backends = HashMap::new();
        let mut devices = Vec::new();
        
        // Initialize CPU backend
        let cpu_backend = Arc::new(crate::hardware::CpuBackend::new());
        backends.insert(DeviceType::CPU, cpu_backend.clone());
        
        // Add CPU device
        devices.push(DeviceInfo {
            device_type: DeviceType::CPU,
            device_index: 0,
            name: "CPU".to_string(),
            memory_capacity: 0, // Unknown
            backend: cpu_backend,
        });
        
        // Try to initialize CUDA backend
        if let Ok(cuda_backend) = crate::hardware::CudaBackend::new() {
            let cuda_backend = Arc::new(cuda_backend);
            backends.insert(DeviceType::CUDA, cuda_backend.clone());
            
            // Add CUDA devices
            let device_count = cuda_backend.device_count();
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
        
        // Try to initialize ROCm backend
        if let Ok(rocm_backend) = crate::hardware::RocmBackend::new() {
            let rocm_backend = Arc::new(rocm_backend);
            backends.insert(DeviceType::ROCm, rocm_backend.clone());
            
            // Add ROCm devices
            let device_count = rocm_backend.device_count();
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
        
        // Try to initialize WebGPU backend
        if let Ok(webgpu_backend) = crate::hardware::WebGpuBackend::new() {
            let webgpu_backend = Arc::new(webgpu_backend);
            backends.insert(DeviceType::WebGPU, webgpu_backend.clone());
            
            // Add WebGPU devices
            let device_count = webgpu_backend.device_count();
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
        
        // Try to initialize TPU backend
        if let Ok(tpu_backend) = crate::hardware::TpuBackend::new() {
            let tpu_backend = Arc::new(tpu_backend);
            backends.insert(DeviceType::TPU, tpu_backend.clone());
            
            // Add TPU devices
            let device_count = tpu_backend.device_count();
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
        
        Self {
            backends,
            devices,
            active_device: Mutex::new(0), // Default to CPU
        }
    }
    
    /// Get the number of available devices
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }
    
    /// Get information about a device
    pub fn device_info(&self, device_id: usize) -> Result<&DeviceInfo> {
        self.devices.get(device_id).ok_or_else(|| {
            PhynexusError::InvalidArgument(format!(
                "Invalid device ID: {}", device_id
            ))
        })
    }
    
    /// Set the active device
    pub fn set_active_device(&self, device_id: usize) -> Result<()> {
        if device_id >= self.devices.len() {
            return Err(PhynexusError::InvalidArgument(format!(
                "Invalid device ID: {}", device_id
            )));
        }
        
        let mut active_device = self.active_device.lock().unwrap();
        *active_device = device_id;
        
        Ok(())
    }
    
    /// Get the active device ID
    pub fn get_active_device(&self) -> usize {
        *self.active_device.lock().unwrap()
    }
    
    /// Get the backend for a device
    pub fn get_backend(&self, device_id: usize) -> Result<Arc<dyn Backend>> {
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
    
    /// Copy memory from host to device
    pub fn copy_host_to_device(&self, host_ptr: *const u8, device_ptr: *mut u8, size: usize, device_id: usize) -> Result<()> {
        let device_info = self.device_info(device_id)?;
        device_info.backend.copy_host_to_device(host_ptr, device_ptr, size, device_info.device_index)
    }
    
    /// Copy memory from device to host
    pub fn copy_device_to_host(&self, device_ptr: *const u8, host_ptr: *mut u8, size: usize, device_id: usize) -> Result<()> {
        let device_info = self.device_info(device_id)?;
        device_info.backend.copy_device_to_host(device_ptr, host_ptr, size, device_info.device_index)
    }
    
    /// Copy memory from device to device
    pub fn copy_device_to_device(&self, src_ptr: *const u8, dst_ptr: *mut u8, size: usize, src_device_id: usize, dst_device_id: usize) -> Result<()> {
        let src_device_info = self.device_info(src_device_id)?;
        let dst_device_info = self.device_info(dst_device_id)?;
        
        // If the devices are of the same type and backend, use the backend's copy function
        if src_device_info.device_type == dst_device_info.device_type {
            return src_device_info.backend.copy_device_to_device(
                src_ptr, dst_ptr, size, 
                src_device_info.device_index, dst_device_info.device_index
            );
        }
        
        // Otherwise, we need to copy through the host
        let mut host_buffer = vec![0u8; size];
        
        // Copy from source device to host
        src_device_info.backend.copy_device_to_host(
            src_ptr, host_buffer.as_mut_ptr(), size, src_device_info.device_index
        )?;
        
        // Copy from host to destination device
        dst_device_info.backend.copy_host_to_device(
            host_buffer.as_ptr(), dst_ptr, size, dst_device_info.device_index
        )?;
        
        Ok(())
    }
    
    /// Synchronize a device
    pub fn synchronize(&self, device_id: usize) -> Result<()> {
        let device_info = self.device_info(device_id)?;
        device_info.backend.synchronize(device_info.device_index)
    }
    
    /// Synchronize all devices
    pub fn synchronize_all(&self) -> Result<()> {
        for (i, _) in self.devices.iter().enumerate() {
            self.synchronize(i)?;
        }
        
        Ok(())
    }
}

impl Default for MultiDeviceManager {
    fn default() -> Self {
        Self::new()
    }
}
