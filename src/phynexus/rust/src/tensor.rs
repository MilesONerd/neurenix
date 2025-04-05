//! Tensor implementation for the Phynexus engine

use crate::device::Device;
use crate::dtype::DataType;
use crate::error::Result;
use std::fmt;

/// A multi-dimensional array with a specific data type and device
pub struct Tensor {
    /// Shape of the tensor
    shape: Vec<usize>,
    
    /// Data type of the tensor
    dtype: DataType,
    
    /// Device where the tensor is stored
    device: Device,
    
    /// Raw pointer to the tensor data
    data: *mut u8,
    
    /// Size of the tensor data in bytes
    size: usize,
    
    /// Whether the tensor requires gradients
    requires_grad: bool,
    
    /// Gradient of the tensor
    grad: Option<Box<Tensor>>,
}

impl Tensor {
    /// Create a new tensor with the given shape, data type, and device
    pub fn new(shape: Vec<usize>, dtype: DataType, device: Device) -> Result<Self> {
        let size = shape.iter().product::<usize>() * dtype.size();
        
        // Allocate memory on the device
        let data = device.allocate(size)?;
        
        Ok(Self {
            shape,
            dtype,
            device,
            data,
            size,
            requires_grad: false,
            grad: None,
        })
    }
    
    /// Create a new tensor with the given shape and data type on the CPU
    pub fn new_cpu<T: 'static + Copy>(shape: Vec<usize>) -> Result<Self> {
        let dtype = DataType::from_type::<T>()?;
        let device = Device::cpu();
        
        Self::new(shape, dtype, device)
    }
    
    /// Create a new tensor with the given shape and data type on the specified device
    pub fn new_with_device<T: 'static + Copy>(shape: Vec<usize>, device: Device) -> Result<Self> {
        let dtype = DataType::from_type::<T>()?;
        
        Self::new(shape, dtype, device)
    }
    
    /// Get the shape of the tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    
    /// Get the data type of the tensor
    pub fn dtype(&self) -> DataType {
        self.dtype
    }
    
    /// Get the device where the tensor is stored
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Get the raw pointer to the tensor data
    pub fn data(&self) -> *const u8 {
        self.data
    }
    
    /// Get the mutable raw pointer to the tensor data
    pub fn data_mut(&mut self) -> *mut u8 {
        self.data
    }
    
    /// Get the size of the tensor data in bytes
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Get the number of elements in the tensor
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    
    /// Set whether the tensor requires gradients
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }
    
    /// Get whether the tensor requires gradients
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    
    /// Get the gradient of the tensor
    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref().map(|g| g.as_ref())
    }
    
    /// Set the gradient of the tensor
    pub fn set_grad(&mut self, grad: Option<Box<Tensor>>) {
        self.grad = grad;
    }
    
    /// Move the tensor to a different device
    pub fn to_device(&self, device: Device) -> Result<Self> {
        if self.device == device {
            return Ok(self.clone());
        }
        
        let device_clone = device.clone();
        let mut result = Self::new(self.shape.clone(), self.dtype, device_clone.clone())?;
        
        // Copy data from this tensor to the new tensor
        device_clone.copy_from_device(self.data, self.device.clone(), result.data_mut())?;
        
        // Copy requires_grad and grad
        result.requires_grad = self.requires_grad;
        if let Some(grad) = &self.grad {
            result.grad = Some(Box::new((**grad).to_device(device_clone)?));
        }
        
        Ok(result)
    }
    
    /// Move the tensor to the CPU
    pub fn to_cpu(&self) -> Result<Self> {
        self.to_device(Device::cpu())
    }
    
    /// Move the tensor to the CUDA device
    pub fn to_cuda(&self) -> Result<Self> {
        self.to_device(Device::cuda(0))
    }
    
    /// Move the tensor to the ROCm device
    pub fn to_rocm(&self) -> Result<Self> {
        self.to_device(Device::rocm(0))
    }
    
    /// Move the tensor to the WebGPU device
    pub fn to_webgpu(&self) -> Result<Self> {
        self.to_device(Device::webgpu())
    }
    
    /// Move the tensor to the TPU device
    pub fn to_tpu(&self) -> Result<Self> {
        self.to_device(Device::tpu(0))
    }
    
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self> {
        if dim0 >= self.shape.len() || dim1 >= self.shape.len() {
            return Err(crate::error::PhynexusError::InvalidArgument(format!(
                "Dimension out of range: dim0={}, dim1={}, ndim={}",
                dim0, dim1, self.shape.len()
            )));
        }
        
        let mut new_shape = self.shape.clone();
        new_shape.swap(dim0, dim1);
        
        let mut result = Self::new(new_shape, self.dtype, self.device.clone())?;
        
        
        let ndim = self.shape.len();
        
        let mut strides = vec![1; ndim];
        for i in (0..ndim-1).rev() {
            strides[i] = strides[i+1] * self.shape[i+1];
        }
        
        let mut new_strides = strides.clone();
        new_strides.swap(dim0, dim1);
        
        let mut indices = vec![0; ndim];
        let numel = self.numel();
        
        for i in 0..numel {
            let mut src_idx = 0;
            for d in 0..ndim {
                src_idx += indices[d] * strides[d];
            }
            
            let mut dst_idx = 0;
            for d in 0..ndim {
                dst_idx += indices[d] * new_strides[d];
            }
            
            unsafe {
                let src_ptr = self.data.add(src_idx * self.dtype.size());
                let dst_ptr = result.data_mut().add(dst_idx * self.dtype.size());
                std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, self.dtype.size());
            }
            
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < self.shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }
        
        Ok(result)
    }
    
    pub fn transpose_2d(&self) -> Result<Self> {
        if self.shape.len() != 2 {
            return Err(crate::error::PhynexusError::InvalidArgument(format!(
                "transpose_2d requires a 2D tensor, got shape {:?}",
                self.shape
            )));
        }
        
        self.transpose(0, 1)
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        let mut result = Self::new(
            self.shape.clone(),
            self.dtype,
            self.device.clone(),
        ).unwrap();
        
        // Copy data from this tensor to the new tensor
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.data,
                result.data_mut(),
                self.size,
            );
        }
        
        // Copy requires_grad and grad
        result.requires_grad = self.requires_grad;
        if let Some(grad) = &self.grad {
            result.grad = Some(Box::new((**grad).clone()));
        }
        
        result
    }
}

impl Drop for Tensor {
    fn drop(&mut self) {
        // Free memory on the device
        let _ = self.device.free(self.data);
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .field("requires_grad", &self.requires_grad)
            .finish()
    }
}
