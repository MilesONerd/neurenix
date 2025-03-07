//! WebGPU backend implementation for the Phynexus engine
//! 
//! This backend provides WebGPU support for the Phynexus engine, with a focus on
//! WebAssembly integration for client-side AI execution in browsers.

use crate::error::{PhynexusError, Result};
use crate::hardware::Backend;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use web_sys::{self, GpuDevice};
#[cfg(target_arch = "wasm32")]
use js_sys::{ArrayBuffer, Float32Array};

/// WebGPU backend implementation
pub struct WebGpuBackend;

impl WebGpuBackend {
    /// Create a new WebGPU backend
    pub fn new() -> Self {
        Self
    }
}

impl Backend for WebGpuBackend {
    fn name(&self) -> &str {
        "WebGPU"
    }
    
    fn is_available(&self) -> bool {
        // Check if WebGPU is available in the current environment
        #[cfg(target_arch = "wasm32")]
        {
            // In WebAssembly context, check if WebGPU API is available
            if let Some(window) = web_sys::window() {
                if let Ok(gpu) = js_sys::Reflect::get(&window, &JsValue::from_str("gpu")) {
                    return !gpu.is_undefined();
                }
            }
            false
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            false // WebGPU not available outside WebAssembly context
        }
    }
    
    fn device_count(&self) -> usize {
        if self.is_available() { 1 } else { 0 }
    }
    
    fn allocate(&self, size: usize, _device_index: usize) -> Result<*mut u8> {
        #[cfg(target_arch = "wasm32")]
        {
            if !self.is_available() {
                return Err(PhynexusError::UnsupportedOperation(
                    "WebGPU not available in this environment".to_string()
                ));
            }
            
            // Allocate memory using WebAssembly memory
            let buffer = ArrayBuffer::new(size as u32);
            let ptr = buffer.as_ptr() as *mut u8;
            Ok(ptr)
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Err(PhynexusError::UnsupportedOperation(
                "WebGPU backend only available in WebAssembly context".to_string()
            ))
        }
    }
    
    fn free(&self, ptr: *mut u8, _device_index: usize) -> Result<()> {
        #[cfg(target_arch = "wasm32")]
        {
            if !self.is_available() {
                return Err(PhynexusError::UnsupportedOperation(
                    "WebGPU not available in this environment".to_string()
                ));
            }
            
            // In WebAssembly, memory is garbage collected
            // No explicit free is needed
            Ok(())
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Err(PhynexusError::UnsupportedOperation(
                "WebGPU backend only available in WebAssembly context".to_string()
            ))
        }
    }
    
    fn copy_host_to_device(&self, host_ptr: *const u8, device_ptr: *mut u8, size: usize, _device_index: usize) -> Result<()> {
        #[cfg(target_arch = "wasm32")]
        {
            if !self.is_available() {
                return Err(PhynexusError::UnsupportedOperation(
                    "WebGPU not available in this environment".to_string()
                ));
            }
            
            // In WebAssembly, host and device memory are the same
            unsafe {
                std::ptr::copy_nonoverlapping(host_ptr, device_ptr, size);
            }
            Ok(())
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Err(PhynexusError::UnsupportedOperation(
                "WebGPU backend only available in WebAssembly context".to_string()
            ))
        }
    }
    
    fn copy_device_to_host(&self, device_ptr: *const u8, host_ptr: *mut u8, size: usize, _device_index: usize) -> Result<()> {
        #[cfg(target_arch = "wasm32")]
        {
            if !self.is_available() {
                return Err(PhynexusError::UnsupportedOperation(
                    "WebGPU not available in this environment".to_string()
                ));
            }
            
            // In WebAssembly, host and device memory are the same
            unsafe {
                std::ptr::copy_nonoverlapping(device_ptr, host_ptr, size);
            }
            Ok(())
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Err(PhynexusError::UnsupportedOperation(
                "WebGPU backend only available in WebAssembly context".to_string()
            ))
        }
    }
    
    fn copy_device_to_device(&self, src_ptr: *const u8, dst_ptr: *mut u8, size: usize, _src_device_index: usize, _dst_device_index: usize) -> Result<()> {
        #[cfg(target_arch = "wasm32")]
        {
            if !self.is_available() {
                return Err(PhynexusError::UnsupportedOperation(
                    "WebGPU not available in this environment".to_string()
                ));
            }
            
            // In WebAssembly, all device memory is the same
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
            }
            Ok(())
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Err(PhynexusError::UnsupportedOperation(
                "WebGPU backend only available in WebAssembly context".to_string()
            ))
        }
    }
    
    fn synchronize(&self, _device_index: usize) -> Result<()> {
        #[cfg(target_arch = "wasm32")]
        {
            if !self.is_available() {
                return Err(PhynexusError::UnsupportedOperation(
                    "WebGPU not available in this environment".to_string()
                ));
            }
            
            // In WebAssembly, synchronization is handled by the browser
            Ok(())
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            Err(PhynexusError::UnsupportedOperation(
                "WebGPU backend only available in WebAssembly context".to_string()
            ))
        }
    }
}
