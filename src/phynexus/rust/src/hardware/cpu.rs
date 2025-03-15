//! CPU backend implementation for the Phynexus engine

use crate::error::Result;
use crate::hardware::Backend;
use std::alloc::{alloc, dealloc, Layout};

/// CPU backend for hardware-specific operations
pub struct CpuBackend;

impl CpuBackend {
    /// Create a new CPU backend
    pub fn new() -> Self {
        Self
    }
}

impl Backend for CpuBackend {
    fn get_device_count(&self) -> Result<usize> {
        Ok(1)
    }
    
    fn allocate(&self, size: usize, _device_index: usize) -> Result<*mut u8> {
        unsafe {
            let layout = Layout::from_size_align(size, 64).unwrap();
            let ptr = alloc(layout);
            Ok(ptr)
        }
    }
    
    fn free(&self, ptr: *mut u8, _device_index: usize) -> Result<()> {
        unsafe {
            if !ptr.is_null() {
                let layout = Layout::from_size_align(1, 1).unwrap();
                dealloc(ptr, layout);
            }
        }
        Ok(())
    }
    
    fn copy_host_to_device(&self, host_ptr: *const u8, device_ptr: *mut u8, size: usize, _device_index: usize) -> Result<()> {
        unsafe {
            std::ptr::copy_nonoverlapping(host_ptr, device_ptr, size);
        }
        Ok(())
    }
    
    fn copy_device_to_host(&self, device_ptr: *const u8, host_ptr: *mut u8, size: usize, _device_index: usize) -> Result<()> {
        unsafe {
            std::ptr::copy_nonoverlapping(device_ptr, host_ptr, size);
        }
        Ok(())
    }
    
    fn copy_device_to_device(&self, src_ptr: *const u8, dst_ptr: *mut u8, size: usize, _src_device_index: usize, _dst_device_index: usize) -> Result<()> {
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
        }
        Ok(())
    }
    
    fn synchronize(&self, _device_index: usize) -> Result<()> {
        // CPU operations are synchronous, so nothing to do
        Ok(())
    }
}
