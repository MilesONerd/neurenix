//! Memory management for the Phynexus engine

use crate::error::{PhynexusError, Result};
use std::alloc::{self, Layout};
use std::ptr::NonNull;

/// Trait for memory buffers on different devices
pub trait Memory: Send + Sync {
    /// Copy data from host (CPU) to the memory buffer
    fn copy_from_host(&self, data: &[u8]) -> Result<()>;
    
    /// Copy data from the memory buffer to host (CPU)
    fn copy_to_host(&self, data: &mut [u8]) -> Result<()>;
    
    /// Get the size of the memory buffer in bytes
    fn size(&self) -> usize;
}

/// CPU memory implementation
pub struct CpuMemory {
    /// Pointer to the allocated memory
    ptr: NonNull<u8>,
    
    /// Size of the allocated memory in bytes
    size: usize,
    
    /// Layout of the allocated memory
    layout: Layout,
}

impl CpuMemory {
    /// Create a new CPU memory buffer with the given size
    pub fn new(size: usize) -> Result<Self> {
        let layout = Layout::array::<u8>(size)
            .map_err(|e| PhynexusError::MemoryError(e.to_string()))?;
        
        let ptr = unsafe {
            let ptr = alloc::alloc(layout);
            if ptr.is_null() {
                return Err(PhynexusError::MemoryError(
                    "Failed to allocate memory".to_string()
                ));
            }
            NonNull::new_unchecked(ptr)
        };
        
        Ok(Self {
            ptr,
            size,
            layout,
        })
    }
}

impl Memory for CpuMemory {
    fn copy_from_host(&self, data: &[u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(PhynexusError::MemoryError(
                format!("Data size {} exceeds buffer size {}", data.len(), self.size)
            ));
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                self.ptr.as_ptr(),
                data.len(),
            );
        }
        
        Ok(())
    }
    
    fn copy_to_host(&self, data: &mut [u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(PhynexusError::MemoryError(
                format!("Buffer size {} is smaller than requested size {}", self.size, data.len())
            ));
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.ptr.as_ptr(),
                data.as_mut_ptr(),
                data.len(),
            );
        }
        
        Ok(())
    }
    
    fn size(&self) -> usize {
        self.size
    }
}

impl Drop for CpuMemory {
    fn drop(&mut self) {
        unsafe {
            alloc::dealloc(self.ptr.as_ptr(), self.layout);
        }
    }
}

// Implement Send and Sync for CpuMemory
unsafe impl Send for CpuMemory {}
unsafe impl Sync for CpuMemory {}
