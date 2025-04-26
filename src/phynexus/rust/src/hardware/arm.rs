
use crate::error::{PhynexusError, Result};
use crate::hardware::Backend;

pub struct ArmBackend;

impl ArmBackend {
    pub fn new() -> Result<Self> {
        #[cfg(target_arch = "arm")]
        {
            #[cfg(target_feature = "neon")]
            let has_neon = true;
            #[cfg(not(target_feature = "neon"))]
            let has_neon = false;
            
            #[cfg(target_feature = "sve")]
            let has_sve = true;
            #[cfg(not(target_feature = "sve"))]
            let has_sve = false;
            
            if !has_neon && !has_sve {
                return Err(PhynexusError::DeviceNotAvailable(
                    "No ARM SIMD features available".to_string()
                ));
            }
        }
        
        #[cfg(not(target_arch = "arm"))]
        {
            return Err(PhynexusError::UnsupportedOperation(
                "ARM support not available on this architecture".to_string()
            ));
        }
        
        Ok(Self)
    }
}

impl Backend for ArmBackend {
    fn get_device_count(&self) -> Result<usize> {
        Ok(1)
    }
    
    fn allocate(&self, size: usize, device_index: usize) -> Result<*mut u8> {
        if device_index != 0 {
            return Err(PhynexusError::InvalidDevice(
                format!("Invalid ARM device index: {}", device_index)
            ));
        }
        
        unsafe {
            let layout = std::alloc::Layout::from_size_align(size, 64)
                .map_err(|e| PhynexusError::AllocationError(e.to_string()))?;
            
            let ptr = std::alloc::alloc(layout);
            if ptr.is_null() {
                Err(PhynexusError::AllocationError(
                    format!("Failed to allocate {} bytes on ARM device", size)
                ))
            } else {
                Ok(ptr)
            }
        }
    }
    
    fn free(&self, ptr: *mut u8, device_index: usize) -> Result<()> {
        if device_index != 0 {
            return Err(PhynexusError::InvalidDevice(
                format!("Invalid ARM device index: {}", device_index)
            ));
        }
        
        if ptr.is_null() {
            return Ok(());
        }
        
        unsafe {
            Ok(())
        }
    }
    
    fn copy_host_to_device(&self, host_ptr: *const u8, device_ptr: *mut u8, size: usize, device_index: usize) -> Result<()> {
        if device_index != 0 {
            return Err(PhynexusError::InvalidDevice(
                format!("Invalid ARM device index: {}", device_index)
            ));
        }
        
        if host_ptr.is_null() || device_ptr.is_null() {
            return Err(PhynexusError::InvalidArgument(
                "Host or device pointer is null".to_string()
            ));
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(host_ptr, device_ptr, size);
        }
        
        Ok(())
    }
    
    fn copy_device_to_host(&self, device_ptr: *const u8, host_ptr: *mut u8, size: usize, device_index: usize) -> Result<()> {
        if device_index != 0 {
            return Err(PhynexusError::InvalidDevice(
                format!("Invalid ARM device index: {}", device_index)
            ));
        }
        
        if device_ptr.is_null() || host_ptr.is_null() {
            return Err(PhynexusError::InvalidArgument(
                "Device or host pointer is null".to_string()
            ));
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(device_ptr, host_ptr, size);
        }
        
        Ok(())
    }
    
    fn copy_device_to_device(&self, src_ptr: *const u8, dst_ptr: *mut u8, size: usize, src_device_index: usize, dst_device_index: usize) -> Result<()> {
        if src_device_index != 0 || dst_device_index != 0 {
            return Err(PhynexusError::InvalidDevice(
                format!("Invalid ARM device indices: {} -> {}", src_device_index, dst_device_index)
            ));
        }
        
        if src_ptr.is_null() || dst_ptr.is_null() {
            return Err(PhynexusError::InvalidArgument(
                "Source or destination pointer is null".to_string()
            ));
        }
        
        unsafe {
            std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, size);
        }
        
        Ok(())
    }
    
    fn synchronize(&self, device_index: usize) -> Result<()> {
        if device_index != 0 {
            return Err(PhynexusError::InvalidDevice(
                format!("Invalid ARM device index: {}", device_index)
            ));
        }
        
        Ok(())
    }
}

#[cfg(target_arch = "arm")]
pub mod neon {
    use crate::error::Result;
    
    #[cfg(target_feature = "neon")]
    pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        use std::arch::arm::*;
        
        if a.len() != b.len() || a.len() != out.len() {
            return Err(crate::error::PhynexusError::DimensionMismatch(
                "Input and output dimensions must match".to_string()
            ));
        }
        
        let len = a.len();
        let mut i = 0;
        
        unsafe {
            while i + 4 <= len {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                let vc = vaddq_f32(va, vb);
                vst1q_f32(out.as_mut_ptr().add(i), vc);
                i += 4;
            }
        }
        
        while i < len {
            out[i] = a[i] + b[i];
            i += 1;
        }
        
        Ok(())
    }
    
    #[cfg(target_feature = "neon")]
    pub fn multiply_f32(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        use std::arch::arm::*;
        
        if a.len() != b.len() || a.len() != out.len() {
            return Err(crate::error::PhynexusError::DimensionMismatch(
                "Input and output dimensions must match".to_string()
            ));
        }
        
        let len = a.len();
        let mut i = 0;
        
        unsafe {
            while i + 4 <= len {
                let va = vld1q_f32(a.as_ptr().add(i));
                let vb = vld1q_f32(b.as_ptr().add(i));
                let vc = vmulq_f32(va, vb);
                vst1q_f32(out.as_mut_ptr().add(i), vc);
                i += 4;
            }
        }
        
        while i < len {
            out[i] = a[i] * b[i];
            i += 1;
        }
        
        Ok(())
    }
}

#[cfg(target_arch = "arm")]
pub mod sve {
    use crate::error::Result;
    
    #[cfg(target_feature = "sve")]
    pub fn add_f32(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        use std::arch::aarch64::*;
        
        if a.len() != b.len() || a.len() != out.len() {
            return Err(crate::error::PhynexusError::DimensionMismatch(
                "Input and output dimensions must match".to_string()
            ));
        }
        
        let len = a.len();
        let mut i = 0;
        
        unsafe {
            let pg = svptrue_b32();
            let vl = svcntw();
            
            while i + vl <= len {
                let va = svld1_f32(pg, a.as_ptr().add(i));
                let vb = svld1_f32(pg, b.as_ptr().add(i));
                let vc = svadd_f32_z(pg, va, vb);
                svst1_f32(pg, out.as_mut_ptr().add(i), vc);
                i += vl;
            }
            
            if i < len {
                let pg_rem = svwhilelt_b32(i as u64, len as u64);
                let va = svld1_f32(pg_rem, a.as_ptr().add(i));
                let vb = svld1_f32(pg_rem, b.as_ptr().add(i));
                let vc = svadd_f32_z(pg_rem, va, vb);
                svst1_f32(pg_rem, out.as_mut_ptr().add(i), vc);
            }
        }
        
        Ok(())
    }
    
    #[cfg(target_feature = "sve")]
    pub fn multiply_f32(a: &[f32], b: &[f32], out: &mut [f32]) -> Result<()> {
        use std::arch::aarch64::*;
        
        if a.len() != b.len() || a.len() != out.len() {
            return Err(crate::error::PhynexusError::DimensionMismatch(
                "Input and output dimensions must match".to_string()
            ));
        }
        
        let len = a.len();
        let mut i = 0;
        
        unsafe {
            let pg = svptrue_b32();
            let vl = svcntw();
            
            while i + vl <= len {
                let va = svld1_f32(pg, a.as_ptr().add(i));
                let vb = svld1_f32(pg, b.as_ptr().add(i));
                let vc = svmul_f32_z(pg, va, vb);
                svst1_f32(pg, out.as_mut_ptr().add(i), vc);
                i += vl;
            }
            
            if i < len {
                let pg_rem = svwhilelt_b32(i as u64, len as u64);
                let va = svld1_f32(pg_rem, a.as_ptr().add(i));
                let vb = svld1_f32(pg_rem, b.as_ptr().add(i));
                let vc = svmul_f32_z(pg_rem, va, vb);
                svst1_f32(pg_rem, out.as_mut_ptr().add(i), vc);
            }
        }
        
        Ok(())
    }
}
