//! Neural network module for the Phynexus engine

use crate::device::Device;
use crate::error::Result;
use crate::tensor::Tensor;

/// Module trait for neural network layers
pub trait Module {
    /// Forward pass
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
}

/// Linear layer
pub struct Linear {
    /// Weight tensor
    #[allow(dead_code)]
    weight: Tensor,
    
    /// Bias tensor
    #[allow(dead_code)]
    bias: Option<Tensor>,
}

impl Linear {
    /// Create a new linear layer
    pub fn new(in_features: usize, out_features: usize, bias: bool, device: Device) -> Result<Self> {
        use crate::ops::init::{kaiming_uniform, zeros};
        
        let weight = kaiming_uniform(&[out_features, in_features], device)?;
        
        let bias = if bias {
            Some(zeros(&[out_features], device)?)
        } else {
            None
        };
        
        Ok(Self { weight, bias })
    }
}

impl Module for Linear {
    /// Forward pass
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        use crate::ops::matmul::matmul;
        
        if input.shape().len() < 2 {
            return Err(crate::error::PhynexusError::InvalidShape(
                format!("Expected input with at least 2 dimensions, got {:?}", input.shape())
            ));
        }
        
        let in_features = input.shape()[input.shape().len() - 1];
        if in_features != self.weight.shape()[1] {
            return Err(crate::error::PhynexusError::InvalidShape(
                format!("Input features {} doesn't match weight shape {:?}", in_features, self.weight.shape())
            ));
        }
        
        let mut output = matmul(input, &self.weight.transpose()?)?;
        
        if let Some(bias) = &self.bias {
            output = output.add(bias)?;
        }
        
        Ok(output)
    }
}

/// Convolutional layer
pub struct Conv2d {
    /// Weight tensor
    #[allow(dead_code)]
    weight: Tensor,
    
    /// Bias tensor
    #[allow(dead_code)]
    bias: Option<Tensor>,
    
    /// Stride
    #[allow(dead_code)]
    stride: Vec<usize>,
    
    /// Padding
    #[allow(dead_code)]
    padding: Vec<usize>,
    
    /// Dilation
    #[allow(dead_code)]
    dilation: Vec<usize>,
    
    /// Groups
    #[allow(dead_code)]
    groups: usize,
}

impl Conv2d {
    /// Create a new convolutional layer
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
        bias: bool,
        device: Device,
    ) -> Result<Self> {
        use crate::ops::init::{kaiming_uniform, zeros};
        
        if kernel_size.len() != 2 {
            return Err(crate::error::PhynexusError::InvalidShape(
                format!("Expected kernel_size with 2 dimensions, got {:?}", kernel_size)
            ));
        }
        
        if stride.len() != 2 {
            return Err(crate::error::PhynexusError::InvalidShape(
                format!("Expected stride with 2 dimensions, got {:?}", stride)
            ));
        }
        
        if padding.len() != 2 {
            return Err(crate::error::PhynexusError::InvalidShape(
                format!("Expected padding with 2 dimensions, got {:?}", padding)
            ));
        }
        
        if dilation.len() != 2 {
            return Err(crate::error::PhynexusError::InvalidShape(
                format!("Expected dilation with 2 dimensions, got {:?}", dilation)
            ));
        }
        
        if in_channels % groups != 0 || out_channels % groups != 0 {
            return Err(crate::error::PhynexusError::InvalidValue(
                format!("in_channels ({}) and out_channels ({}) must be divisible by groups ({})", 
                        in_channels, out_channels, groups)
            ));
        }
        
        let weight_shape = vec![out_channels, in_channels / groups, kernel_size[0], kernel_size[1]];
        let weight = kaiming_uniform(&weight_shape, device)?;
        
        let bias = if bias {
            Some(zeros(&[out_channels], device)?)
        } else {
            None
        };
        
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            dilation,
            groups,
        })
    }
}

impl Module for Conv2d {
    /// Forward pass
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        use crate::ops::conv::{conv2d, Conv2dParams};
        
        if input.shape().len() != 4 {
            return Err(crate::error::PhynexusError::InvalidShape(
                format!("Expected 4D input (batch_size, channels, height, width), got {:?}", input.shape())
            ));
        }
        
        let in_channels = input.shape()[1];
        let expected_in_channels = self.weight.shape()[1] * self.groups;
        
        if in_channels != expected_in_channels {
            return Err(crate::error::PhynexusError::InvalidShape(
                format!("Input channels {} doesn't match weight shape {:?} with groups {}", 
                        in_channels, self.weight.shape(), self.groups)
            ));
        }
        
        let params = Conv2dParams {
            stride: self.stride.clone(),
            padding: self.padding.clone(),
            dilation: self.dilation.clone(),
            groups: self.groups,
        };
        
        let mut output = conv2d(input, &self.weight, &params)?;
        
        if let Some(bias) = &self.bias {
            let bias_shape = vec![1, bias.shape()[0], 1, 1];
            let bias_reshaped = bias.reshape(&bias_shape)?;
            
            output = output.add(&bias_reshaped)?;
        }
        
        Ok(output)
    }
}

/// LSTM layer
pub struct LSTM {
    /// Input size
    #[allow(dead_code)]
    input_size: usize,
    
    /// Hidden size
    #[allow(dead_code)]
    hidden_size: usize,
    
    /// Number of layers
    #[allow(dead_code)]
    num_layers: usize,
    
    /// Whether to use bias
    #[allow(dead_code)]
    bias: bool,
    
    /// Whether batch dimension is first
    #[allow(dead_code)]
    batch_first: bool,
    
    /// Dropout probability
    #[allow(dead_code)]
    dropout: f32,
    
    /// Whether to use bidirectional LSTM
    #[allow(dead_code)]
    bidirectional: bool,
}

impl LSTM {
    /// Create a new LSTM layer
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bias: bool,
        batch_first: bool,
        dropout: f32,
        bidirectional: bool,
        device: Device,
    ) -> Result<Self> {
        if num_layers == 0 {
            return Err(crate::error::PhynexusError::InvalidValue(
                "Number of layers must be at least 1".to_string()
            ));
        }
        
        if dropout < 0.0 || dropout >= 1.0 {
            return Err(crate::error::PhynexusError::InvalidValue(
                format!("Dropout probability must be between 0 and 1, got {}", dropout)
            ));
        }
        
        Ok(Self {
            input_size,
            hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        })
    }
}

impl Module for LSTM {
    /// Forward pass
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        use crate::ops::rnn::{lstm_forward, LSTMState};
        
        if input.shape().len() != 3 {
            return Err(crate::error::PhynexusError::InvalidShape(
                format!("Expected 3D input (seq_len, batch, input_size) or (batch, seq_len, input_size), got {:?}", input.shape())
            ));
        }
        
        let (seq_len, batch_size, input_dim) = if self.batch_first {
            (input.shape()[1], input.shape()[0], input.shape()[2])
        } else {
            (input.shape()[0], input.shape()[1], input.shape()[2])
        };
        
        if input_dim != self.input_size {
            return Err(crate::error::PhynexusError::InvalidShape(
                format!("Input dimension {} doesn't match expected input size {}", input_dim, self.input_size)
            ));
        }
        
        let hidden_size = self.hidden_size * (if self.bidirectional { 2 } else { 1 });
        let device = input.device();
        
        let h0 = Tensor::zeros(&[self.num_layers, batch_size, hidden_size], device)?;
        let c0 = Tensor::zeros(&[self.num_layers, batch_size, hidden_size], device)?;
        
        let initial_state = LSTMState { h: h0, c: c0 };
        
        let (output, _) = lstm_forward(
            input,
            &initial_state,
            self.input_size,
            self.hidden_size,
            self.num_layers,
            self.dropout,
            self.bidirectional,
            self.batch_first,
        )?;
        
        Ok(output)
    }
}
