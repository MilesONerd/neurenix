
use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;

pub fn tpu_add(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let tpu_device = crate::hardware::tpu::get_tpu_device()?;
    let tpu_context = crate::hardware::tpu::get_tpu_context()?;
    
    let a_data = a.data_tpu()?;
    let b_data = b.data_tpu()?;
    let mut out_data = out.data_tpu_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for TPU addition: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    unsafe {
        crate::hardware::tpu::execute_tpu_op(
            tpu_device,
            tpu_context,
            "tpu_elementwise_add",
            &[a_data, b_data],
            &mut [out_data],
            &[out_data.len() as i32],
        )?;
    }
    
    crate::hardware::tpu::wait_for_completion(tpu_context)?;
    
    Ok(())
}

pub fn tpu_subtract(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let tpu_device = crate::hardware::tpu::get_tpu_device()?;
    let tpu_context = crate::hardware::tpu::get_tpu_context()?;
    
    let a_data = a.data_tpu()?;
    let b_data = b.data_tpu()?;
    let mut out_data = out.data_tpu_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for TPU subtraction: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    unsafe {
        crate::hardware::tpu::execute_tpu_op(
            tpu_device,
            tpu_context,
            "tpu_elementwise_subtract",
            &[a_data, b_data],
            &mut [out_data],
            &[out_data.len() as i32],
        )?;
    }
    
    crate::hardware::tpu::wait_for_completion(tpu_context)?;
    
    Ok(())
}

pub fn tpu_multiply(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let tpu_device = crate::hardware::tpu::get_tpu_device()?;
    let tpu_context = crate::hardware::tpu::get_tpu_context()?;
    
    let a_data = a.data_tpu()?;
    let b_data = b.data_tpu()?;
    let mut out_data = out.data_tpu_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for TPU multiplication: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    unsafe {
        crate::hardware::tpu::execute_tpu_op(
            tpu_device,
            tpu_context,
            "tpu_elementwise_multiply",
            &[a_data, b_data],
            &mut [out_data],
            &[out_data.len() as i32],
        )?;
    }
    
    crate::hardware::tpu::wait_for_completion(tpu_context)?;
    
    Ok(())
}

pub fn tpu_divide(a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    let tpu_device = crate::hardware::tpu::get_tpu_device()?;
    let tpu_context = crate::hardware::tpu::get_tpu_context()?;
    
    let a_data = a.data_tpu()?;
    let b_data = b.data_tpu()?;
    let mut out_data = out.data_tpu_mut()?;
    
    if a_data.len() != out_data.len() || b_data.len() != out_data.len() {
        return Err(PhynexusError::DimensionMismatch(
            format!("Tensor dimensions do not match for TPU division: a={}, b={}, out={}", 
                    a_data.len(), b_data.len(), out_data.len())
        ));
    }
    
    unsafe {
        crate::hardware::tpu::execute_tpu_op(
            tpu_device,
            tpu_context,
            "tpu_elementwise_divide",
            &[a_data, b_data],
            &mut [out_data],
            &[out_data.len() as i32],
        )?;
    }
    
    crate::hardware::tpu::wait_for_completion(tpu_context)?;
    
    Ok(())
}
