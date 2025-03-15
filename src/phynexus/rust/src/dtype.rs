//! Data type definitions for the Phynexus engine

use crate::error::{PhynexusError, Result};
use std::fmt;

/// Data type enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DataType {
    /// 32-bit floating point
    Float32,
    
    /// 64-bit floating point
    Float64,
    
    /// 8-bit signed integer
    Int8,
    
    /// 16-bit signed integer
    Int16,
    
    /// 32-bit signed integer
    Int32,
    
    /// 64-bit signed integer
    Int64,
    
    /// 8-bit unsigned integer
    Uint8,
    
    /// 16-bit unsigned integer
    Uint16,
    
    /// 32-bit unsigned integer
    Uint32,
    
    /// 64-bit unsigned integer
    Uint64,
    
    /// Boolean
    Bool,
}

impl DataType {
    /// Get the size of the data type in bytes
    pub fn size(&self) -> usize {
        match self {
            DataType::Float32 => 4,
            DataType::Float64 => 8,
            DataType::Int8 => 1,
            DataType::Int16 => 2,
            DataType::Int32 => 4,
            DataType::Int64 => 8,
            DataType::Uint8 => 1,
            DataType::Uint16 => 2,
            DataType::Uint32 => 4,
            DataType::Uint64 => 8,
            DataType::Bool => 1,
        }
    }
    
    /// Get the data type from a Rust type
    pub fn from_type<T: 'static>() -> Result<Self> {
        let type_id = std::any::TypeId::of::<T>();
        
        if type_id == std::any::TypeId::of::<f32>() {
            Ok(DataType::Float32)
        } else if type_id == std::any::TypeId::of::<f64>() {
            Ok(DataType::Float64)
        } else if type_id == std::any::TypeId::of::<i8>() {
            Ok(DataType::Int8)
        } else if type_id == std::any::TypeId::of::<i16>() {
            Ok(DataType::Int16)
        } else if type_id == std::any::TypeId::of::<i32>() {
            Ok(DataType::Int32)
        } else if type_id == std::any::TypeId::of::<i64>() {
            Ok(DataType::Int64)
        } else if type_id == std::any::TypeId::of::<u8>() {
            Ok(DataType::Uint8)
        } else if type_id == std::any::TypeId::of::<u16>() {
            Ok(DataType::Uint16)
        } else if type_id == std::any::TypeId::of::<u32>() {
            Ok(DataType::Uint32)
        } else if type_id == std::any::TypeId::of::<u64>() {
            Ok(DataType::Uint64)
        } else if type_id == std::any::TypeId::of::<bool>() {
            Ok(DataType::Bool)
        } else {
            Err(PhynexusError::UnsupportedDataType(
                format!("Unsupported data type: {:?}", std::any::type_name::<T>())
            ))
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Float32 => write!(f, "float32"),
            DataType::Float64 => write!(f, "float64"),
            DataType::Int8 => write!(f, "int8"),
            DataType::Int16 => write!(f, "int16"),
            DataType::Int32 => write!(f, "int32"),
            DataType::Int64 => write!(f, "int64"),
            DataType::Uint8 => write!(f, "uint8"),
            DataType::Uint16 => write!(f, "uint16"),
            DataType::Uint32 => write!(f, "uint32"),
            DataType::Uint64 => write!(f, "uint64"),
            DataType::Bool => write!(f, "bool"),
        }
    }
}

/// Trait for data types that can be used in tensors
pub trait DType: 'static + Copy + Send + Sync {
    /// Get the data type enum for this type
    fn dtype() -> DataType;
}

impl DType for f32 {
    fn dtype() -> DataType {
        DataType::Float32
    }
}

impl DType for f64 {
    fn dtype() -> DataType {
        DataType::Float64
    }
}

impl DType for i8 {
    fn dtype() -> DataType {
        DataType::Int8
    }
}

impl DType for i16 {
    fn dtype() -> DataType {
        DataType::Int16
    }
}

impl DType for i32 {
    fn dtype() -> DataType {
        DataType::Int32
    }
}

impl DType for i64 {
    fn dtype() -> DataType {
        DataType::Int64
    }
}

impl DType for u8 {
    fn dtype() -> DataType {
        DataType::Uint8
    }
}

impl DType for u16 {
    fn dtype() -> DataType {
        DataType::Uint16
    }
}

impl DType for u32 {
    fn dtype() -> DataType {
        DataType::Uint32
    }
}

impl DType for u64 {
    fn dtype() -> DataType {
        DataType::Uint64
    }
}

impl DType for bool {
    fn dtype() -> DataType {
        DataType::Bool
    }
}
