use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;

pub fn arm_add(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    if a.shape() != b.shape() || a.shape() != out.shape() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for ARM addition: a={:?}, b={:?}, out={:?}", 
                    a.shape(), b.shape(), out.shape())
        ));
    }
    
    if !a.device().is_arm() || !b.device().is_arm() || !out.device().is_arm() {
        return Err(PhynexusError::InvalidDevice(
            "All tensors must be on ARM device".to_string()
        ));
    }
    
    let a_data = a.data();
    let b_data = b.data();
    let out_data = out.data_mut();
    
    let size = a.numel();
    
    #[cfg(all(target_arch = "arm", target_feature = "neon"))]
    {
        unsafe {
            if crate::hardware::arm::neon::add_f32(
                std::slice::from_raw_parts(a_data as *const f32, size),
                std::slice::from_raw_parts(b_data as *const f32, size),
                std::slice::from_raw_parts_mut(out_data as *mut f32, size),
            ).is_ok() {
                return Ok(());
            }
        }
    }
    
    #[cfg(all(target_arch = "arm", target_feature = "sve"))]
    {
        unsafe {
            if crate::hardware::arm::sve::add_f32(
                std::slice::from_raw_parts(a_data as *const f32, size),
                std::slice::from_raw_parts(b_data as *const f32, size),
                std::slice::from_raw_parts_mut(out_data as *mut f32, size),
            ).is_ok() {
                return Ok(());
            }
        }
    }
    
    unsafe {
        let a_slice = std::slice::from_raw_parts(a_data as *const f32, size);
        let b_slice = std::slice::from_raw_parts(b_data as *const f32, size);
        let out_slice = std::slice::from_raw_parts_mut(out_data as *mut f32, size);
        
        for i in 0..size {
            out_slice[i] = a_slice[i] + b_slice[i];
        }
    }
    
    Ok(())
}

pub fn arm_subtract(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    if a.shape() != b.shape() || a.shape() != out.shape() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for ARM subtraction: a={:?}, b={:?}, out={:?}", 
                    a.shape(), b.shape(), out.shape())
        ));
    }
    
    if !a.device().is_arm() || !b.device().is_arm() || !out.device().is_arm() {
        return Err(PhynexusError::InvalidDevice(
            "All tensors must be on ARM device".to_string()
        ));
    }
    
    let a_data = a.data();
    let b_data = b.data();
    let out_data = out.data_mut();
    
    let size = a.numel();
    
    unsafe {
        let a_slice = std::slice::from_raw_parts(a_data as *const f32, size);
        let b_slice = std::slice::from_raw_parts(b_data as *const f32, size);
        let out_slice = std::slice::from_raw_parts_mut(out_data as *mut f32, size);
        
        for i in 0..size {
            out_slice[i] = a_slice[i] - b_slice[i];
        }
    }
    
    Ok(())
}

pub fn arm_multiply(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    if a.shape() != b.shape() || a.shape() != out.shape() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for ARM multiplication: a={:?}, b={:?}, out={:?}", 
                    a.shape(), b.shape(), out.shape())
        ));
    }
    
    if !a.device().is_arm() || !b.device().is_arm() || !out.device().is_arm() {
        return Err(PhynexusError::InvalidDevice(
            "All tensors must be on ARM device".to_string()
        ));
    }
    
    let a_data = a.data();
    let b_data = b.data();
    let out_data = out.data_mut();
    
    let size = a.numel();
    
    #[cfg(all(target_arch = "arm", target_feature = "neon"))]
    {
        unsafe {
            if crate::hardware::arm::neon::multiply_f32(
                std::slice::from_raw_parts(a_data as *const f32, size),
                std::slice::from_raw_parts(b_data as *const f32, size),
                std::slice::from_raw_parts_mut(out_data as *mut f32, size),
            ).is_ok() {
                return Ok(());
            }
        }
    }
    
    #[cfg(all(target_arch = "arm", target_feature = "sve"))]
    {
        unsafe {
            if crate::hardware::arm::sve::multiply_f32(
                std::slice::from_raw_parts(a_data as *const f32, size),
                std::slice::from_raw_parts(b_data as *const f32, size),
                std::slice::from_raw_parts_mut(out_data as *mut f32, size),
            ).is_ok() {
                return Ok(());
            }
        }
    }
    
    unsafe {
        let a_slice = std::slice::from_raw_parts(a_data as *const f32, size);
        let b_slice = std::slice::from_raw_parts(b_data as *const f32, size);
        let out_slice = std::slice::from_raw_parts_mut(out_data as *mut f32, size);
        
        for i in 0..size {
            out_slice[i] = a_slice[i] * b_slice[i];
        }
    }
    
    Ok(())
}

pub fn arm_divide(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    if a.shape() != b.shape() || a.shape() != out.shape() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for ARM division: a={:?}, b={:?}, out={:?}", 
                    a.shape(), b.shape(), out.shape())
        ));
    }
    
    if !a.device().is_arm() || !b.device().is_arm() || !out.device().is_arm() {
        return Err(PhynexusError::InvalidDevice(
            "All tensors must be on ARM device".to_string()
        ));
    }
    
    let a_data = a.data();
    let b_data = b.data();
    let out_data = out.data_mut();
    
    let size = a.numel();
    
    unsafe {
        let a_slice = std::slice::from_raw_parts(a_data as *const f32, size);
        let b_slice = std::slice::from_raw_parts(b_data as *const f32, size);
        let out_slice = std::slice::from_raw_parts_mut(out_data as *mut f32, size);
        
        for i in 0..size {
            out_slice[i] = a_slice[i] / b_slice[i];
        }
    }
    
    Ok(())
}
