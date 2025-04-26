use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;

pub fn npu_add(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let npu_device = crate::hardware::npu::get_npu_device()?;
    let npu_context = crate::hardware::npu::get_npu_context()?;
    
    let a_data = a.data_npu()?;
    let b_data = b.data_npu()?;
    let mut out_data = out.data_npu_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for NPU addition: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    unsafe {
        crate::hardware::npu::execute_npu_op(
            npu_device,
            npu_context,
            "npu_elementwise_add",
            &[a_data, b_data],
            &mut [out_data],
            &[out_data.len() as i32],
        )?;
    }
    
    crate::hardware::npu::wait_for_completion(npu_context)?;
    
    Ok(())
}

pub fn npu_subtract(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let npu_device = crate::hardware::npu::get_npu_device()?;
    let npu_context = crate::hardware::npu::get_npu_context()?;
    
    let a_data = a.data_npu()?;
    let b_data = b.data_npu()?;
    let mut out_data = out.data_npu_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for NPU subtraction: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    unsafe {
        crate::hardware::npu::execute_npu_op(
            npu_device,
            npu_context,
            "npu_elementwise_subtract",
            &[a_data, b_data],
            &mut [out_data],
            &[out_data.len() as i32],
        )?;
    }
    
    crate::hardware::npu::wait_for_completion(npu_context)?;
    
    Ok(())
}

pub fn npu_multiply(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let npu_device = crate::hardware::npu::get_npu_device()?;
    let npu_context = crate::hardware::npu::get_npu_context()?;
    
    let a_data = a.data_npu()?;
    let b_data = b.data_npu()?;
    let mut out_data = out.data_npu_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for NPU multiplication: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    unsafe {
        crate::hardware::npu::execute_npu_op(
            npu_device,
            npu_context,
            "npu_elementwise_multiply",
            &[a_data, b_data],
            &mut [out_data],
            &[out_data.len() as i32],
        )?;
    }
    
    crate::hardware::npu::wait_for_completion(npu_context)?;
    
    Ok(())
}

pub fn npu_divide(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let npu_device = crate::hardware::npu::get_npu_device()?;
    let npu_context = crate::hardware::npu::get_npu_context()?;
    
    let a_data = a.data_npu()?;
    let b_data = b.data_npu()?;
    let mut out_data = out.data_npu_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for NPU division: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    unsafe {
        crate::hardware::npu::execute_npu_op(
            npu_device,
            npu_context,
            "npu_elementwise_divide",
            &[a_data, b_data],
            &mut [out_data],
            &[out_data.len() as i32],
        )?;
    }
    
    crate::hardware::npu::wait_for_completion(npu_context)?;
    
    Ok(())
}
