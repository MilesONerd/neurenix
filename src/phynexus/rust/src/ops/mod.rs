//! Operation implementations for the Phynexus engine

pub mod activation;
pub mod blas;
pub mod conv;
pub mod elementwise;
pub mod matmul;
pub mod reduction;
pub mod tpu;

// Export specific functions to avoid name conflicts
pub use activation::{ActivationType, relu, sigmoid, tanh, softmax};
pub use elementwise::{add, subtract, multiply, divide, pow, exp, log};
pub use matmul::{matmul, batch_matmul};
pub use reduction::{ReductionOp, sum, mean, max, min};
pub use tpu::*;
