//! Tensor implementation for the Phynexus engine

use crate::error::{PhynexusError, Result};
use crate::device::{Device, DeviceType};
use crate::memory::Memory;
use ndarray::{Array, ArrayD, IxDyn};
use std::fmt;
use std::sync::Arc;

/// Represents a multi-dimensional array with support for various hardware devices
pub struct Tensor {
    /// The shape of the tensor
    shape: Vec<usize>,
    
    /// The data type of the tensor elements
    dtype: DataType,
    
    /// The device where the tensor is stored
    device: Device,
    
    /// The memory buffer containing the tensor data
    memory: Arc<dyn Memory>,
}

/// Supported data types for tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    Float32,
    Float64,
    Int32,
    Int64,
    Bool,
}

impl Tensor {
    /// Create a new tensor with the given shape and data type on the specified device
    pub fn new(shape: Vec<usize>, dtype: DataType, device: Device) -> Result<Self> {
        let size = shape.iter().product::<usize>() * dtype.size_in_bytes();
        let memory = device.allocate_memory(size)?;
        
        Ok(Self {
            shape,
            dtype,
            device,
            memory,
        })
    }
    
    /// Create a tensor from CPU data
    pub fn from_cpu_data<T>(data: &[T], shape: Vec<usize>, device: Device) -> Result<Self> 
    where
        T: Copy,
    {
        let dtype = DataType::from_type::<T>()?;
        let mut tensor = Self::new(shape, dtype, device)?;
        tensor.copy_from_cpu(data)?;
        Ok(tensor)
    }
    
    /// Copy data from CPU to the tensor
    pub fn copy_from_cpu<T>(&mut self, data: &[T]) -> Result<()> 
    where
        T: Copy,
    {
        let expected_len = self.shape.iter().product::<usize>();
        if data.len() != expected_len {
            return Err(PhynexusError::ShapeMismatch(format!(
                "Data length {} does not match tensor size {}", 
                data.len(), expected_len
            )));
        }
        
        let bytes = std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * std::mem::size_of::<T>(),
        );
        
        self.memory.copy_from_host(bytes)?;
        Ok(())
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
}

impl DataType {
    /// Get the size of the data type in bytes
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::Bool => 1,
        }
    }
    
    /// Create a DataType from a Rust type
    pub fn from_type<T>() -> Result<Self> {
        let type_id = std::any::TypeId::of::<T>();
        
        if type_id == std::any::TypeId::of::<f32>() {
            Ok(DataType::Float32)
        } else if type_id == std::any::TypeId::of::<f64>() {
            Ok(DataType::Float64)
        } else if type_id == std::any::TypeId::of::<i32>() {
            Ok(DataType::Int32)
        } else if type_id == std::any::TypeId::of::<i64>() {
            Ok(DataType::Int64)
        } else if type_id == std::any::TypeId::of::<bool>() {
            Ok(DataType::Bool)
        } else {
            Err(PhynexusError::UnsupportedOperation(format!(
                "Unsupported data type"
            )))
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("device", &self.device)
            .finish()
    }
}
