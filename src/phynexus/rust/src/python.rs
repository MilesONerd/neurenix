//! Python bindings for the Phynexus engine

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use numpy::{PyArray, PyArrayDyn, IntoPyArray, PyReadonlyArrayDyn};
use ndarray::{Array, ArrayD, IxDyn};

use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;
use crate::device::{Device, DeviceType};
use crate::tensor_ops;

/// Python module for the Phynexus engine
#[pymodule]
fn phynexus(_py: Python, m: &PyModule) -> PyResult<()> {
    // Initialize the Phynexus engine
    crate::init().map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    
    // Add version information
    m.add("__version__", crate::version())?;
    
    // Add device types
    m.add_class::<PyDeviceType>()?;
    
    // Add tensor class
    m.add_class::<PyTensor>()?;
    
    // Add functions
    m.add_function(wrap_pyfunction!(py_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(py_add, m)?)?;
    m.add_function(wrap_pyfunction!(py_subtract, m)?)?;
    m.add_function(wrap_pyfunction!(py_multiply, m)?)?;
    m.add_function(wrap_pyfunction!(py_divide, m)?)?;
    m.add_function(wrap_pyfunction!(py_reshape, m)?)?;
    m.add_function(wrap_pyfunction!(py_transpose, m)?)?;
    
    // Add submodules
    let nn = PyModule::new(_py, "nn")?;
    nn.add_class::<PyLinear>()?;
    nn.add_class::<PyConv2d>()?;
    nn.add_class::<PyLSTM>()?;
    m.add_submodule(nn)?;
    
    let optim = PyModule::new(_py, "optim")?;
    optim.add_class::<PySGD>()?;
    optim.add_class::<PyAdam>()?;
    m.add_submodule(optim)?;
    
    Ok(())
}

/// Python wrapper for DeviceType
#[pyclass]
#[derive(Clone)]
struct PyDeviceType {
    device_type: DeviceType,
}

#[pymethods]
impl PyDeviceType {
    #[classattr]
    const CPU: &'static str = "cpu";
    
    #[classattr]
    const CUDA: &'static str = "cuda";
    
    #[classattr]
    const ROCM: &'static str = "rocm";
    
    #[classattr]
    const WEBGPU: &'static str = "webgpu";
    
    #[new]
    fn new(device_str: &str) -> PyResult<Self> {
        let device_type = match device_str.to_lowercase().as_str() {
            "cpu" => DeviceType::CPU,
            "cuda" => DeviceType::CUDA,
            "rocm" => DeviceType::ROCm,
            "webgpu" => DeviceType::WebGPU,
            _ => return Err(PyValueError::new_err(format!("Unknown device type: {}", device_str))),
        };
        
        Ok(Self { device_type })
    }
    
    fn __repr__(&self) -> String {
        format!("DeviceType({})", self.device_type)
    }
}

/// Python wrapper for Tensor
#[pyclass]
struct PyTensor {
    tensor: Tensor,
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: &PyAny, device: Option<&PyDeviceType>) -> PyResult<Self> {
        // Convert PyAny to ndarray
        let array: ArrayD<f32> = if let Ok(array) = data.extract::<PyReadonlyArrayDyn<f32>>() {
            array.as_array().to_owned()
        } else {
            return Err(PyValueError::new_err("Input data must be a numpy array"));
        };
        
        // Create device
        let device_type = if let Some(device) = device {
            device.device_type.clone()
        } else {
            DeviceType::CPU
        };
        
        let device = Device::new(device_type, 0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        // Create tensor
        let tensor = Tensor::from_cpu_data(array.as_slice().unwrap(), array.shape().to_vec(), device)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(Self { tensor })
    }
    
    fn shape(&self) -> Vec<usize> {
        self.tensor.shape().to_vec()
    }
    
    fn to_numpy<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArrayDyn<f32>> {
        // TODO: Implement conversion from Tensor to numpy array
        // For now, just return a dummy array
        let shape = self.tensor.shape();
        let array = Array::zeros(IxDyn(shape));
        Ok(array.into_pyarray(py))
    }
    
    fn __repr__(&self) -> String {
        format!("Tensor(shape={:?}, device={})", self.tensor.shape(), self.tensor.device())
    }
}

/// Python wrapper for matrix multiplication
#[pyfunction]
fn py_matmul(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    let result = tensor_ops::matmul(&a.tensor, &b.tensor)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    
    Ok(PyTensor { tensor: result })
}

/// Python wrapper for addition
#[pyfunction]
fn py_add(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    let result = tensor_ops::add(&a.tensor, &b.tensor)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    
    Ok(PyTensor { tensor: result })
}

/// Python wrapper for subtraction
#[pyfunction]
fn py_subtract(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    let result = tensor_ops::subtract(&a.tensor, &b.tensor)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    
    Ok(PyTensor { tensor: result })
}

/// Python wrapper for multiplication
#[pyfunction]
fn py_multiply(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    let result = tensor_ops::multiply(&a.tensor, &b.tensor)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    
    Ok(PyTensor { tensor: result })
}

/// Python wrapper for division
#[pyfunction]
fn py_divide(a: &PyTensor, b: &PyTensor) -> PyResult<PyTensor> {
    let result = tensor_ops::divide(&a.tensor, &b.tensor)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    
    Ok(PyTensor { tensor: result })
}

/// Python wrapper for reshape
#[pyfunction]
fn py_reshape(tensor: &PyTensor, shape: Vec<usize>) -> PyResult<PyTensor> {
    let result = tensor_ops::reshape(&tensor.tensor, &shape)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    
    Ok(PyTensor { tensor: result })
}

/// Python wrapper for transpose
#[pyfunction]
fn py_transpose(tensor: &PyTensor, dim0: usize, dim1: usize) -> PyResult<PyTensor> {
    let result = tensor_ops::transpose(&tensor.tensor, dim0, dim1)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    
    Ok(PyTensor { tensor: result })
}

/// Python wrapper for Linear layer
#[pyclass]
struct PyLinear {
    in_features: usize,
    out_features: usize,
    bias: bool,
    weight: PyTensor,
    bias_tensor: Option<PyTensor>,
}

#[pymethods]
impl PyLinear {
    #[new]
    fn new(in_features: usize, out_features: usize, bias: Option<bool>) -> PyResult<Self> {
        let bias = bias.unwrap_or(true);
        
        // Create weight tensor
        let weight_data = Array::zeros(IxDyn(&[out_features, in_features]));
        let device = Device::new(DeviceType::CPU, 0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let weight_tensor = Tensor::from_cpu_data(
            weight_data.as_slice().unwrap(),
            weight_data.shape().to_vec(),
            device.clone()
        ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        // Create bias tensor if needed
        let bias_tensor = if bias {
            let bias_data = Array::zeros(IxDyn(&[out_features]));
            let bias_tensor = Tensor::from_cpu_data(
                bias_data.as_slice().unwrap(),
                bias_data.shape().to_vec(),
                device
            ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Some(PyTensor { tensor: bias_tensor })
        } else {
            None
        };
        
        Ok(Self {
            in_features,
            out_features,
            bias,
            weight: PyTensor { tensor: weight_tensor },
            bias_tensor,
        })
    }
    
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Perform matrix multiplication: input @ weight.T
        let weight_t = tensor_ops::transpose(&self.weight.tensor, 0, 1)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        let output = tensor_ops::matmul(&input.tensor, &weight_t)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        // Add bias if present
        if let Some(bias) = &self.bias_tensor {
            let result = tensor_ops::add(&output, &bias.tensor)
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Ok(PyTensor { tensor: result })
        } else {
            Ok(PyTensor { tensor: output })
        }
    }
    
    fn __repr__(&self) -> String {
        format!("Linear(in_features={}, out_features={}, bias={})",
                self.in_features, self.out_features, self.bias)
    }
}

/// Python wrapper for Conv2d layer
#[pyclass]
struct PyConv2d {
    in_channels: usize,
    out_channels: usize,
    kernel_size: Vec<usize>,
    stride: Vec<usize>,
    padding: Vec<usize>,
    dilation: Vec<usize>,
    groups: usize,
    bias: bool,
    weight: PyTensor,
    bias_tensor: Option<PyTensor>,
}

#[pymethods]
impl PyConv2d {
    #[new]
    #[pyo3(signature = (in_channels, out_channels, kernel_size, stride=None, padding=None, dilation=None, groups=1, bias=None))]
    fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: &PyAny,
        stride: Option<&PyAny>,
        padding: Option<&PyAny>,
        dilation: Option<&PyAny>,
        groups: Option<usize>,
        bias: Option<bool>,
    ) -> PyResult<Self> {
        // Parse kernel_size
        let kernel_size = if let Ok(size) = kernel_size.extract::<usize>() {
            vec![size, size]
        } else if let Ok(sizes) = kernel_size.extract::<Vec<usize>>() {
            sizes
        } else {
            return Err(PyValueError::new_err("kernel_size must be an int or tuple of ints"));
        };
        
        // Parse stride
        let stride = if let Some(s) = stride {
            if let Ok(size) = s.extract::<usize>() {
                vec![size, size]
            } else if let Ok(sizes) = s.extract::<Vec<usize>>() {
                sizes
            } else {
                return Err(PyValueError::new_err("stride must be an int or tuple of ints"));
            }
        } else {
            vec![1, 1]
        };
        
        // Parse padding
        let padding = if let Some(p) = padding {
            if let Ok(size) = p.extract::<usize>() {
                vec![size, size]
            } else if let Ok(sizes) = p.extract::<Vec<usize>>() {
                sizes
            } else {
                return Err(PyValueError::new_err("padding must be an int or tuple of ints"));
            }
        } else {
            vec![0, 0]
        };
        
        // Parse dilation
        let dilation = if let Some(d) = dilation {
            if let Ok(size) = d.extract::<usize>() {
                vec![size, size]
            } else if let Ok(sizes) = d.extract::<Vec<usize>>() {
                sizes
            } else {
                return Err(PyValueError::new_err("dilation must be an int or tuple of ints"));
            }
        } else {
            vec![1, 1]
        };
        
        let groups = groups.unwrap_or(1);
        let bias = bias.unwrap_or(true);
        
        // Create weight tensor
        let weight_shape = vec![out_channels, in_channels / groups, kernel_size[0], kernel_size[1]];
        let weight_data = Array::zeros(IxDyn(&weight_shape));
        let device = Device::new(DeviceType::CPU, 0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let weight_tensor = Tensor::from_cpu_data(
            weight_data.as_slice().unwrap(),
            weight_data.shape().to_vec(),
            device.clone()
        ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        // Create bias tensor if needed
        let bias_tensor = if bias {
            let bias_data = Array::zeros(IxDyn(&[out_channels]));
            let bias_tensor = Tensor::from_cpu_data(
                bias_data.as_slice().unwrap(),
                bias_data.shape().to_vec(),
                device
            ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
            Some(PyTensor { tensor: bias_tensor })
        } else {
            None
        };
        
        Ok(Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            weight: PyTensor { tensor: weight_tensor },
            bias_tensor,
        })
    }
    
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Perform convolution
        let bias_ref = self.bias_tensor.as_ref().map(|b| &b.tensor);
        
        let output = tensor_ops::conv(
            &input.tensor,
            &self.weight.tensor,
            bias_ref,
            &self.stride,
            &self.padding,
            &self.dilation,
            self.groups
        ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        Ok(PyTensor { tensor: output })
    }
    
    fn __repr__(&self) -> String {
        format!(
            "Conv2d(in_channels={}, out_channels={}, kernel_size={:?}, stride={:?}, padding={:?}, dilation={:?}, groups={}, bias={})",
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.dilation, self.groups, self.bias
        )
    }
}

/// Python wrapper for LSTM layer
#[pyclass]
struct PyLSTM {
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    bias: bool,
    batch_first: bool,
    bidirectional: bool,
    dropout: f32,
}

#[pymethods]
impl PyLSTM {
    #[new]
    #[pyo3(signature = (input_size, hidden_size, num_layers=1, bias=true, batch_first=false, dropout=0.0, bidirectional=false))]
    fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: Option<usize>,
        bias: Option<bool>,
        batch_first: Option<bool>,
        dropout: Option<f32>,
        bidirectional: Option<bool>,
    ) -> PyResult<Self> {
        let num_layers = num_layers.unwrap_or(1);
        let bias = bias.unwrap_or(true);
        let batch_first = batch_first.unwrap_or(false);
        let dropout = dropout.unwrap_or(0.0);
        let bidirectional = bidirectional.unwrap_or(false);
        
        Ok(Self {
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            bidirectional,
            dropout,
        })
    }
    
    fn forward(&self, input: &PyTensor, hidden: Option<&PyTuple>) -> PyResult<(PyTensor, PyTuple)> {
        // TODO: Implement LSTM forward pass
        // For now, just return dummy outputs
        
        let py = input.py();
        
        // Create dummy output tensor
        let batch_size = input.shape()[0];
        let seq_len = if self.batch_first { input.shape()[1] } else { input.shape()[0] };
        let num_directions = if self.bidirectional { 2 } else { 1 };
        
        let output_shape = if self.batch_first {
            vec![batch_size, seq_len, self.hidden_size * num_directions]
        } else {
            vec![seq_len, batch_size, self.hidden_size * num_directions]
        };
        
        let output_data = Array::zeros(IxDyn(&output_shape));
        let device = Device::new(DeviceType::CPU, 0)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let output_tensor = Tensor::from_cpu_data(
            output_data.as_slice().unwrap(),
            output_data.shape().to_vec(),
            device.clone()
        ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        // Create dummy hidden state
        let h_shape = vec![self.num_layers * num_directions, batch_size, self.hidden_size];
        let h_data = Array::zeros(IxDyn(&h_shape));
        let h_tensor = Tensor::from_cpu_data(
            h_data.as_slice().unwrap(),
            h_data.shape().to_vec(),
            device.clone()
        ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        let c_shape = vec![self.num_layers * num_directions, batch_size, self.hidden_size];
        let c_data = Array::zeros(IxDyn(&c_shape));
        let c_tensor = Tensor::from_cpu_data(
            c_data.as_slice().unwrap(),
            c_data.shape().to_vec(),
            device
        ).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        
        let h_state = PyTensor { tensor: h_tensor };
        let c_state = PyTensor { tensor: c_tensor };
        
        let hidden_tuple = PyTuple::new(py, &[h_state.into_py(py), c_state.into_py(py)]);
        
        Ok((PyTensor { tensor: output_tensor }, hidden_tuple))
    }
    
    fn __repr__(&self) -> String {
        format!(
            "LSTM(input_size={}, hidden_size={}, num_layers={}, bias={}, batch_first={}, dropout={}, bidirectional={})",
            self.input_size, self.hidden_size, self.num_layers, self.bias, self.batch_first, self.dropout, self.bidirectional
        )
    }
}

/// Python wrapper for SGD optimizer
#[pyclass]
struct PySGD {
    lr: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
    nesterov: bool,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (lr=0.01, momentum=0.0, dampening=0.0, weight_decay=0.0, nesterov=false))]
    fn new(
        lr: Option<f32>,
        momentum: Option<f32>,
        dampening: Option<f32>,
        weight_decay: Option<f32>,
        nesterov: Option<bool>,
    ) -> PyResult<Self> {
        let lr = lr.unwrap_or(0.01);
        let momentum = momentum.unwrap_or(0.0);
        let dampening = dampening.unwrap_or(0.0);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let nesterov = nesterov.unwrap_or(false);
        
        Ok(Self {
            lr,
            momentum,
            dampening,
            weight_decay,
            nesterov,
        })
    }
    
    fn step(&self) -> PyResult<()> {
        // TODO: Implement SGD step
        Ok(())
    }
    
    fn zero_grad(&self) -> PyResult<()> {
        // TODO: Implement zero_grad
        Ok(())
    }
    
    fn __repr__(&self) -> String {
        format!(
            "SGD(lr={}, momentum={}, dampening={}, weight_decay={}, nesterov={})",
            self.lr, self.momentum, self.dampening, self.weight_decay, self.nesterov
        )
    }
}

/// Python wrapper for Adam optimizer
#[pyclass]
struct PyAdam {
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (lr=0.001, betas=None, eps=1e-8, weight_decay=0.0, amsgrad=false))]
    fn new(
        lr: Option<f32>,
        betas: Option<(f32, f32)>,
        eps: Option<f32>,
        weight_decay: Option<f32>,
        amsgrad: Option<bool>,
    ) -> PyResult<Self> {
        let lr = lr.unwrap_or(0.001);
        let betas = betas.unwrap_or((0.9, 0.999));
        let eps = eps.unwrap_or(1e-8);
        let weight_decay = weight_decay.unwrap_or(0.0);
        let amsgrad = amsgrad.unwrap_or(false);
        
        Ok(Self {
            lr,
            betas,
            eps,
            weight_decay,
            amsgrad,
        })
    }
    
    fn step(&self) -> PyResult<()> {
        // TODO: Implement Adam step
        Ok(())
    }
    
    fn zero_grad(&self) -> PyResult<()> {
        // TODO: Implement zero_grad
        Ok(())
    }
    
    fn __repr__(&self) -> String {
        format!(
            "Adam(lr={}, betas={:?}, eps={}, weight_decay={}, amsgrad={})",
            self.lr, self.betas, self.eps, self.weight_decay, self.amsgrad
        )
    }
}
