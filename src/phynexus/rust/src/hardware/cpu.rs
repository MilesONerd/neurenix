//! CPU backend implementation for the Phynexus engine

use std::alloc::{self, Layout};
use std::ptr::NonNull;

use crate::error::{PhynexusError, Result};
use crate::hardware::Backend;

/// CPU backend implementation
pub struct CpuBackend;

impl CpuBackend {
    /// Create a new CPU backend
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "CPU"
    }
    
    fn is_available(&self) -> bool {
        true
    }
    
    fn device_count(&self) -> usize {
        1
    }
    
    fn allocate(&self, size: usize, _device_index: usize) -> Result<*mut u8> {
        let layout = Layout::array::<u8>(size)
            .map_err(|e| PhynexusError::MemoryError(e.to_string()))?;
        
        let ptr = unsafe {
            let ptr = alloc::alloc(layout);
            if ptr.is_null() {
                return Err(PhynexusError::MemoryError(
                    "Failed to allocate memory on CPU".to_string()
                ));
            }
            ptr
        };
        
        Ok(ptr)
    }
    
    fn free(&self, ptr: *mut u8, _device_index: usize) -> Result<()> {
        if ptr.is_null() {
            return Ok(());
        }
        
        // In a real implementation, we would keep track of allocations and their layouts
        // For now, we'll use a fixed layout with a large alignment to be safe
        // This is not ideal but works for demonstration purposes
        unsafe {
            // Assume 64-byte alignment for all allocations
            let layout = Layout::from_size_align(1, 64)
                .map_err(|e| PhynexusError::MemoryError(e.to_string()))?;
            
            alloc::dealloc(ptr, layout);
        }
        
        Ok(())
    }
    
    fn copy_host_to_device(&self, host_ptr: *const u8, device_ptr: *mut u8, size: usize, _device_index: usize) -> Result<()> {
        if host_ptr.is_null() || device_ptr.is_null() {
            return Err(PhynexusError::InvalidArgument(
                "Null pointer passed to copy_host_to_device".to_string()
            ));
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(host_ptr, device_ptr, size);
        }
        
        Ok(())
    }
    
    fn copy_device_to_host(&self, device_ptr: *const u8, host_ptr: *mut u8, size: usize, _device_index: usize) -> Result<()> {
        if device_ptr.is_null() || host_ptr.is_null() {
            return Err(PhynexusError::InvalidArgument(
                "Null pointer passed to copy_device_to_host".to_string()
            ));
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(device_ptr, host_ptr, size);
        }
        
        Ok(())
    }
    
    fn copy_device_to_device(&self, src_ptr: *const u8, dst_ptr: *mut u8, size: usize, _src_device_index: usize, _dst_device_index: usize) -> Result<()> {
        if src_ptr.is_null() || dst_ptr.is_null() {
            return Err(PhynexusError::InvalidArgument(
                "Null pointer passed to copy_device_to_device".to_string()
            ));
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
        }
        
        Ok(())
    }
    
    fn synchronize(&self, _device_index: usize) -> Result<()> {
        // No need to synchronize on CPU
        Ok(())
    }
}
