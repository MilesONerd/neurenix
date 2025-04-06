//! Computational graph for the Phynexus engine

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::fmt;

use crate::error::{PhynexusError, Result};
use crate::tensor::Tensor;
use crate::device::Device;

/// Represents a node in the computational graph
pub struct Node {
    /// Unique identifier for the node
    id: usize,
    
    /// Name of the node
    name: String,
    
    /// Operation performed by the node
    op: Op,
    
    /// Input nodes
    inputs: Vec<Arc<Node>>,
    
    /// Output tensor
    output: Option<Tensor>,
    
    /// Whether the node requires gradient computation
    requires_grad: bool,
    
    /// Gradient of the output tensor
    grad: Option<Tensor>,
}

/// Represents an operation in the computational graph
#[derive(Clone)]
pub enum Op {
    /// Input tensor
    Input,
    
    /// Constant tensor
    Constant,
    
    /// Variable tensor
    Variable,
    
    /// Matrix multiplication
    MatMul,
    
    /// Element-wise addition
    Add,
    
    /// Element-wise subtraction
    Sub,
    
    /// Element-wise multiplication
    Mul,
    
    /// Element-wise division
    Div,
    
    /// ReLU activation
    ReLU,
    
    /// Sigmoid activation
    Sigmoid,
    
    /// Tanh activation
    Tanh,
    
    /// Softmax activation
    Softmax { dim: i64 },
    
    /// Convolution
    Conv2d {
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
    },
    
    /// Transposed convolution
    ConvTranspose2d {
        stride: Vec<usize>,
        padding: Vec<usize>,
        output_padding: Vec<usize>,
        dilation: Vec<usize>,
        groups: usize,
    },
    
    /// Max pooling
    MaxPool2d {
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
        dilation: Vec<usize>,
    },
    
    /// Average pooling
    AvgPool2d {
        kernel_size: Vec<usize>,
        stride: Vec<usize>,
        padding: Vec<usize>,
    },
    
    /// Reshape
    Reshape { shape: Vec<usize> },
    
    /// Transpose
    Transpose { dims: Vec<usize> },
    
    /// Concatenate tensors along a dimension
    Concat { dim: usize },
    
    /// Split a tensor into chunks along a dimension
    Split { dim: usize, chunks: Vec<usize> },
    
    /// Reduce sum along dimensions
    Sum { dims: Vec<usize>, keep_dims: bool },
    
    /// Reduce mean along dimensions
    Mean { dims: Vec<usize>, keep_dims: bool },
    
    /// Reduce max along dimensions
    Max { dims: Vec<usize>, keep_dims: bool },
    
    /// Reduce min along dimensions
    Min { dims: Vec<usize>, keep_dims: bool },
    
    /// Custom operation
    Custom {
        name: String,
        forward: Arc<dyn Fn(&[&Tensor]) -> Result<Tensor> + Send + Sync>,
        backward: Option<Arc<dyn Fn(&[&Tensor], &Tensor) -> Result<Vec<Tensor>> + Send + Sync>>,
    },
}

impl Node {
    /// Create a new node
    pub fn new(id: usize, name: String, op: Op, inputs: Vec<Arc<Node>>, requires_grad: bool) -> Self {
        Self {
            id,
            name,
            op,
            inputs,
            output: None,
            requires_grad,
            grad: None,
        }
    }
    
    /// Get the ID of the node
    pub fn id(&self) -> usize {
        self.id
    }
    
    /// Get the name of the node
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the operation of the node
    pub fn op(&self) -> &Op {
        &self.op
    }
    
    /// Get the input nodes
    pub fn inputs(&self) -> &[Arc<Node>] {
        &self.inputs
    }
    
    /// Get the output tensor
    pub fn output(&self) -> Option<&Tensor> {
        self.output.as_ref()
    }
    
    /// Set the output tensor
    pub fn set_output(&mut self, output: Tensor) {
        self.output = Some(output);
    }
    
    /// Check if the node requires gradient computation
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    
    /// Get the gradient of the output tensor
    pub fn grad(&self) -> Option<&Tensor> {
        self.grad.as_ref()
    }
    
    /// Set the gradient of the output tensor
    pub fn set_grad(&mut self, grad: Tensor) {
        self.grad = Some(grad);
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Node")
            .field("id", &self.id)
            .field("name", &self.name)
            .field("op", &format!("{:?}", self.op))
            .field("inputs", &self.inputs.iter().map(|n| n.id).collect::<Vec<_>>())
            .field("requires_grad", &self.requires_grad)
            .finish()
    }
}

impl fmt::Debug for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Op::Input => write!(f, "Input"),
            Op::Constant => write!(f, "Constant"),
            Op::Variable => write!(f, "Variable"),
            Op::MatMul => write!(f, "MatMul"),
            Op::Add => write!(f, "Add"),
            Op::Sub => write!(f, "Sub"),
            Op::Mul => write!(f, "Mul"),
            Op::Div => write!(f, "Div"),
            Op::ReLU => write!(f, "ReLU"),
            Op::Sigmoid => write!(f, "Sigmoid"),
            Op::Tanh => write!(f, "Tanh"),
            Op::Softmax { dim } => write!(f, "Softmax(dim={})", dim),
            Op::Conv2d { stride, padding, dilation, groups } => {
                write!(f, "Conv2d(stride={:?}, padding={:?}, dilation={:?}, groups={})",
                       stride, padding, dilation, groups)
            },
            Op::ConvTranspose2d { stride, padding, output_padding, dilation, groups } => {
                write!(f, "ConvTranspose2d(stride={:?}, padding={:?}, output_padding={:?}, dilation={:?}, groups={})",
                       stride, padding, output_padding, dilation, groups)
            },
            Op::MaxPool2d { kernel_size, stride, padding, dilation } => {
                write!(f, "MaxPool2d(kernel_size={:?}, stride={:?}, padding={:?}, dilation={:?})",
                       kernel_size, stride, padding, dilation)
            },
            Op::AvgPool2d { kernel_size, stride, padding } => {
                write!(f, "AvgPool2d(kernel_size={:?}, stride={:?}, padding={:?})",
                       kernel_size, stride, padding)
            },
            Op::Reshape { shape } => write!(f, "Reshape(shape={:?})", shape),
            Op::Transpose { dims } => write!(f, "Transpose(dims={:?})", dims),
            Op::Concat { dim } => write!(f, "Concat(dim={})", dim),
            Op::Split { dim, chunks } => write!(f, "Split(dim={}, chunks={:?})", dim, chunks),
            Op::Sum { dims, keep_dims } => write!(f, "Sum(dims={:?}, keep_dims={})", dims, keep_dims),
            Op::Mean { dims, keep_dims } => write!(f, "Mean(dims={:?}, keep_dims={})", dims, keep_dims),
            Op::Max { dims, keep_dims } => write!(f, "Max(dims={:?}, keep_dims={})", dims, keep_dims),
            Op::Min { dims, keep_dims } => write!(f, "Min(dims={:?}, keep_dims={})", dims, keep_dims),
            Op::Custom { name, .. } => write!(f, "Custom(name={})", name),
        }
    }
}

/// Computational graph for the Phynexus engine
pub struct Graph {
    /// Nodes in the graph
    nodes: HashMap<usize, Arc<Mutex<Node>>>,
    
    /// Next node ID
    next_id: usize,
    
    /// Input nodes
    inputs: Vec<Arc<Mutex<Node>>>,
    
    /// Output nodes
    outputs: Vec<Arc<Mutex<Node>>>,
    
    /// Default device for the graph
    device: Device,
}

impl Graph {
    /// Create a new graph
    pub fn new(device: Device) -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            inputs: Vec::new(),
            outputs: Vec::new(),
            device,
        }
    }
    
    /// Add a node to the graph
    pub fn add_node(&mut self, name: &str, op: Op, inputs: Vec<Arc<Mutex<Node>>>, requires_grad: bool) -> Result<Arc<Mutex<Node>>> {
        let id = self.next_id;
        self.next_id += 1;
        
        let input_nodes = inputs.iter()
            .map(|n| {
                let node = n.lock().unwrap();
                Arc::new(Node::new(
                    node.id,
                    node.name.clone(),
                    node.op.clone(),
                    node.inputs.clone(),
                    node.requires_grad,
                ))
            })
            .collect();
        
        let node = Node::new(id, name.to_string(), op, input_nodes, requires_grad);
        let node = Arc::new(Mutex::new(node));
        
        self.nodes.insert(id, node.clone());
        
        Ok(node)
    }
    
    /// Add an input node to the graph
    pub fn add_input(&mut self, name: &str, tensor: Tensor, requires_grad: bool) -> Result<Arc<Mutex<Node>>> {
        let node = self.add_node(name, Op::Input, Vec::new(), requires_grad)?;
        node.lock().unwrap().set_output(tensor);
        
        self.inputs.push(node.clone());
        
        Ok(node)
    }
    
    /// Add a constant node to the graph
    pub fn add_constant(&mut self, name: &str, tensor: Tensor) -> Result<Arc<Mutex<Node>>> {
        let node = self.add_node(name, Op::Constant, Vec::new(), false)?;
        node.lock().unwrap().set_output(tensor);
        
        Ok(node)
    }
    
    /// Add a variable node to the graph
    pub fn add_variable(&mut self, name: &str, tensor: Tensor, requires_grad: bool) -> Result<Arc<Mutex<Node>>> {
        let node = self.add_node(name, Op::Variable, Vec::new(), requires_grad)?;
        node.lock().unwrap().set_output(tensor);
        
        Ok(node)
    }
    
    /// Add an output node to the graph
    pub fn add_output(&mut self, node: Arc<Mutex<Node>>) {
        self.outputs.push(node);
    }
    
    /// Topological sort of the nodes in the graph
    fn topological_sort(&self) -> Result<Vec<Arc<Mutex<Node>>>> {
        let mut visited = HashSet::new();
        let mut sorted = Vec::new();
        
        // Helper function for depth-first search
        fn dfs(
            node_id: usize,
            nodes: &HashMap<usize, Arc<Mutex<Node>>>,
            visited: &mut HashSet<usize>,
            sorted: &mut Vec<Arc<Mutex<Node>>>,
        ) -> Result<()> {
            if visited.contains(&node_id) {
                return Ok(());
            }
            
            visited.insert(node_id);
            
            let node = nodes.get(&node_id)
                .ok_or_else(|| PhynexusError::InvalidArgument(
                    format!("Node with ID {} not found in the graph", node_id)
                ))?;
            
            let input_ids: Vec<usize> = {
                let node = node.lock().unwrap();
                node.inputs.iter().map(|n| n.id()).collect()
            };
            
            for input_id in input_ids {
                dfs(input_id, nodes, visited, sorted)?;
            }
            
            sorted.push(node.clone());
            
            Ok(())
        }
        
        // Perform DFS from each output node
        for output in &self.outputs {
            let node_id = output.lock().unwrap().id;
            dfs(node_id, &self.nodes, &mut visited, &mut sorted)?;
        }
        
        Ok(sorted)
    }
    
    /// Forward pass through the graph
    pub fn forward(&self) -> Result<Vec<Tensor>> {
        let sorted_nodes = self.topological_sort()?;
        
        for node in &sorted_nodes {
            let mut node = node.lock().unwrap();
            
            if node.output.is_some() {
                continue;
            }
            
            let input_tensors: Result<Vec<Tensor>> = node.inputs
                .iter()
                .map(|input| {
                    let input_node = self.nodes.get(&input.id())
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} not found in the graph", input.id())
                        ))?;
                    
                    let input_node = input_node.lock().unwrap();
                    input_node.output()
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} has no output tensor", input.id())
                        ))
                        .map(|t| t.clone())
                })
                .collect();
            
            let input_tensors = input_tensors?;
            
            let output = match &node.op {
                Op::Input | Op::Constant | Op::Variable => {
                    continue;
                },
                Op::MatMul => {
                    if input_tensors.len() != 2 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("MatMul operation requires 2 inputs, got {}", input_tensors.len())
                        ));
                    }
                    
                    crate::ops::matmul(&input_tensors[0], &input_tensors[1])?
                },
                Op::Add => {
                    if input_tensors.len() != 2 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Add operation requires 2 inputs, got {}", input_tensors.len())
                        ));
                    }
                    
                    &input_tensors[0] + &input_tensors[1]
                },
                Op::Sub => {
                    if input_tensors.len() != 2 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Sub operation requires 2 inputs, got {}", input_tensors.len())
                        ));
                    }
                    
                    &input_tensors[0] - &input_tensors[1]
                },
                Op::Mul => {
                    if input_tensors.len() != 2 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Mul operation requires 2 inputs, got {}", input_tensors.len())
                        ));
                    }
                    
                    &input_tensors[0] * &input_tensors[1]
                },
                Op::Div => {
                    if input_tensors.len() != 2 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Div operation requires 2 inputs, got {}", input_tensors.len())
                        ));
                    }
                    
                    &input_tensors[0] / &input_tensors[1]
                },
                Op::ReLU => {
                    if input_tensors.len() != 1 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("ReLU operation requires 1 input, got {}", input_tensors.len())
                        ));
                    }
                    
                    crate::ops::relu(&input_tensors[0])?
                },
                Op::Sigmoid => {
                    if input_tensors.len() != 1 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Sigmoid operation requires 1 input, got {}", input_tensors.len())
                        ));
                    }
                    
                    crate::ops::sigmoid(&input_tensors[0])?
                },
                Op::Tanh => {
                    if input_tensors.len() != 1 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Tanh operation requires 1 input, got {}", input_tensors.len())
                        ));
                    }
                    
                    crate::ops::tanh(&input_tensors[0])?
                },
                Op::Softmax { dim } => {
                    if input_tensors.len() != 1 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Softmax operation requires 1 input, got {}", input_tensors.len())
                        ));
                    }
                    
                    crate::ops::softmax(&input_tensors[0], *dim)?
                },
                Op::Conv2d { stride, padding, dilation, groups } => {
                    if input_tensors.len() != 2 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Conv2d operation requires 2 inputs, got {}", input_tensors.len())
                        ));
                    }
                    
                    let params = crate::ops::conv::Conv2dParams {
                        stride: stride.clone(),
                        padding: padding.clone(),
                        dilation: dilation.clone(),
                        groups: *groups,
                    };
                    
                    crate::ops::conv::conv2d(&input_tensors[0], &input_tensors[1], &params)?
                },
                Op::ConvTranspose2d { stride, padding, output_padding, dilation, groups } => {
                    if input_tensors.len() != 2 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("ConvTranspose2d operation requires 2 inputs, got {}", input_tensors.len())
                        ));
                    }
                    
                    return Err(PhynexusError::UnsupportedOperation(
                        "ConvTranspose2d operation not yet implemented".to_string()
                    ));
                },
                Op::MaxPool2d { kernel_size, stride, padding, dilation } => {
                    if input_tensors.len() != 1 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("MaxPool2d operation requires 1 input, got {}", input_tensors.len())
                        ));
                    }
                    
                    return Err(PhynexusError::UnsupportedOperation(
                        "MaxPool2d operation not yet implemented".to_string()
                    ));
                },
                Op::AvgPool2d { kernel_size, stride, padding } => {
                    if input_tensors.len() != 1 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("AvgPool2d operation requires 1 input, got {}", input_tensors.len())
                        ));
                    }
                    
                    return Err(PhynexusError::UnsupportedOperation(
                        "AvgPool2d operation not yet implemented".to_string()
                    ));
                },
                Op::Reshape { shape } => {
                    if input_tensors.len() != 1 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Reshape operation requires 1 input, got {}", input_tensors.len())
                        ));
                    }
                    
                    input_tensors[0].reshape(shape)?
                },
                Op::Transpose { dims } => {
                    if input_tensors.len() != 1 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Transpose operation requires 1 input, got {}", input_tensors.len())
                        ));
                    }
                    
                    input_tensors[0].transpose(dims)?
                },
                Op::Concat { dim } => {
                    if input_tensors.is_empty() {
                        return Err(PhynexusError::InvalidArgument(
                            "Concat operation requires at least 1 input".to_string()
                        ));
                    }
                    
                    crate::ops::concat(&input_tensors, *dim)?
                },
                Op::Split { dim, chunks } => {
                    if input_tensors.len() != 1 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Split operation requires 1 input, got {}", input_tensors.len())
                        ));
                    }
                    
                    return Err(PhynexusError::UnsupportedOperation(
                        "Split operation not yet implemented".to_string()
                    ));
                },
                Op::Sum { dims, keep_dims } => {
                    if input_tensors.len() != 1 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Sum operation requires 1 input, got {}", input_tensors.len())
                        ));
                    }
                    
                    input_tensors[0].sum(dims, *keep_dims)?
                },
                Op::Mean { dims, keep_dims } => {
                    if input_tensors.len() != 1 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Mean operation requires 1 input, got {}", input_tensors.len())
                        ));
                    }
                    
                    input_tensors[0].mean(dims, *keep_dims)?
                },
                Op::Max { dims, keep_dims } => {
                    if input_tensors.len() != 1 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Max operation requires 1 input, got {}", input_tensors.len())
                        ));
                    }
                    
                    input_tensors[0].max(dims, *keep_dims)?
                },
                Op::Min { dims, keep_dims } => {
                    if input_tensors.len() != 1 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Min operation requires 1 input, got {}", input_tensors.len())
                        ));
                    }
                    
                    input_tensors[0].min(dims, *keep_dims)?
                },
                Op::Custom { forward, .. } => {
                    let input_refs: Vec<&Tensor> = input_tensors.iter().collect();
                    forward(&input_refs)?
                },
            };
            
            node.set_output(output);
        }
        
        let mut outputs = Vec::new();
        for output_node in &self.outputs {
            let output_node = output_node.lock().unwrap();
            let output = output_node.output()
                .ok_or_else(|| PhynexusError::InvalidArgument(
                    format!("Output node with ID {} has no output tensor", output_node.id)
                ))?
                .clone();
            
            outputs.push(output);
        }
        
        Ok(outputs)
    }
    
    /// Backward pass through the graph
    pub fn backward(&self) -> Result<()> {
        let mut sorted_nodes = self.topological_sort()?;
        sorted_nodes.reverse();
        
        for output_node in &self.outputs {
            let mut output_node = output_node.lock().unwrap();
            
            if !output_node.requires_grad() {
                continue;
            }
            
            if output_node.grad().is_none() {
                let output = output_node.output()
                    .ok_or_else(|| PhynexusError::InvalidArgument(
                        format!("Output node with ID {} has no output tensor", output_node.id)
                    ))?;
                
                let grad = output.ones_like()?;
                output_node.set_grad(grad);
            }
        }
        
        for node in &sorted_nodes {
            let node_lock = node.lock().unwrap();
            
            if !node_lock.requires_grad() {
                continue;
            }
            
            let grad = match node_lock.grad() {
                Some(g) => g.clone(),
                None => continue, // Skip nodes with no gradient
            };
            
            let op = node_lock.op().clone();
            let inputs = node_lock.inputs().to_vec();
            
            drop(node_lock);
            
            match op {
                Op::Input | Op::Constant => {
                    continue;
                },
                Op::Variable => {
                    continue;
                },
                Op::MatMul => {
                    if inputs.len() != 2 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("MatMul operation requires 2 inputs, got {}", inputs.len())
                        ));
                    }
                    
                    let input_a = self.nodes.get(&inputs[0].id())
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} not found in the graph", inputs[0].id())
                        ))?;
                    
                    let input_b = self.nodes.get(&inputs[1].id())
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} not found in the graph", inputs[1].id())
                        ))?;
                    
                    let a = input_a.lock().unwrap().output()
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} has no output tensor", inputs[0].id())
                        ))?
                        .clone();
                    
                    let b = input_b.lock().unwrap().output()
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} has no output tensor", inputs[1].id())
                        ))?
                        .clone();
                    
                    let grad_a = crate::ops::matmul(&grad, &b.transpose(&[1, 0])?)?;
                    let grad_b = crate::ops::matmul(&a.transpose(&[1, 0])?, &grad)?;
                    
                    if input_a.lock().unwrap().requires_grad() {
                        let mut input_a = input_a.lock().unwrap();
                        if let Some(existing_grad) = input_a.grad() {
                            input_a.set_grad(&existing_grad + &grad_a);
                        } else {
                            input_a.set_grad(grad_a);
                        }
                    }
                    
                    if input_b.lock().unwrap().requires_grad() {
                        let mut input_b = input_b.lock().unwrap();
                        if let Some(existing_grad) = input_b.grad() {
                            input_b.set_grad(&existing_grad + &grad_b);
                        } else {
                            input_b.set_grad(grad_b);
                        }
                    }
                },
                Op::Add => {
                    if inputs.len() != 2 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Add operation requires 2 inputs, got {}", inputs.len())
                        ));
                    }
                    
                    let input_a = self.nodes.get(&inputs[0].id())
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} not found in the graph", inputs[0].id())
                        ))?;
                    
                    let input_b = self.nodes.get(&inputs[1].id())
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} not found in the graph", inputs[1].id())
                        ))?;
                    
                    if input_a.lock().unwrap().requires_grad() {
                        let mut input_a = input_a.lock().unwrap();
                        if let Some(existing_grad) = input_a.grad() {
                            input_a.set_grad(&existing_grad + &grad);
                        } else {
                            input_a.set_grad(grad.clone());
                        }
                    }
                    
                    if input_b.lock().unwrap().requires_grad() {
                        let mut input_b = input_b.lock().unwrap();
                        if let Some(existing_grad) = input_b.grad() {
                            input_b.set_grad(&existing_grad + &grad);
                        } else {
                            input_b.set_grad(grad);
                        }
                    }
                },
                Op::Sub => {
                    if inputs.len() != 2 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Sub operation requires 2 inputs, got {}", inputs.len())
                        ));
                    }
                    
                    let input_a = self.nodes.get(&inputs[0].id())
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} not found in the graph", inputs[0].id())
                        ))?;
                    
                    let input_b = self.nodes.get(&inputs[1].id())
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} not found in the graph", inputs[1].id())
                        ))?;
                    
                    if input_a.lock().unwrap().requires_grad() {
                        let mut input_a = input_a.lock().unwrap();
                        if let Some(existing_grad) = input_a.grad() {
                            input_a.set_grad(&existing_grad + &grad);
                        } else {
                            input_a.set_grad(grad.clone());
                        }
                    }
                    
                    if input_b.lock().unwrap().requires_grad() {
                        let mut input_b = input_b.lock().unwrap();
                        let neg_grad = -&grad;
                        if let Some(existing_grad) = input_b.grad() {
                            input_b.set_grad(&existing_grad + &neg_grad);
                        } else {
                            input_b.set_grad(neg_grad);
                        }
                    }
                },
                Op::Mul => {
                    if inputs.len() != 2 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Mul operation requires 2 inputs, got {}", inputs.len())
                        ));
                    }
                    
                    let input_a = self.nodes.get(&inputs[0].id())
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} not found in the graph", inputs[0].id())
                        ))?;
                    
                    let input_b = self.nodes.get(&inputs[1].id())
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} not found in the graph", inputs[1].id())
                        ))?;
                    
                    let a = input_a.lock().unwrap().output()
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} has no output tensor", inputs[0].id())
                        ))?
                        .clone();
                    
                    let b = input_b.lock().unwrap().output()
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} has no output tensor", inputs[1].id())
                        ))?
                        .clone();
                    
                    if input_a.lock().unwrap().requires_grad() {
                        let mut input_a = input_a.lock().unwrap();
                        let grad_a = &grad * &b;
                        if let Some(existing_grad) = input_a.grad() {
                            input_a.set_grad(&existing_grad + &grad_a);
                        } else {
                            input_a.set_grad(grad_a);
                        }
                    }
                    
                    if input_b.lock().unwrap().requires_grad() {
                        let mut input_b = input_b.lock().unwrap();
                        let grad_b = &grad * &a;
                        if let Some(existing_grad) = input_b.grad() {
                            input_b.set_grad(&existing_grad + &grad_b);
                        } else {
                            input_b.set_grad(grad_b);
                        }
                    }
                },
                Op::Div => {
                    if inputs.len() != 2 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("Div operation requires 2 inputs, got {}", inputs.len())
                        ));
                    }
                    
                    let input_a = self.nodes.get(&inputs[0].id())
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} not found in the graph", inputs[0].id())
                        ))?;
                    
                    let input_b = self.nodes.get(&inputs[1].id())
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} not found in the graph", inputs[1].id())
                        ))?;
                    
                    let a = input_a.lock().unwrap().output()
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} has no output tensor", inputs[0].id())
                        ))?
                        .clone();
                    
                    let b = input_b.lock().unwrap().output()
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} has no output tensor", inputs[1].id())
                        ))?
                        .clone();
                    
                    if input_a.lock().unwrap().requires_grad() {
                        let mut input_a = input_a.lock().unwrap();
                        let grad_a = &grad / &b;
                        if let Some(existing_grad) = input_a.grad() {
                            input_a.set_grad(&existing_grad + &grad_a);
                        } else {
                            input_a.set_grad(grad_a);
                        }
                    }
                    
                    if input_b.lock().unwrap().requires_grad() {
                        let mut input_b = input_b.lock().unwrap();
                        let b_squared = &b * &b;
                        let grad_b = -(&grad * &a) / &b_squared;
                        if let Some(existing_grad) = input_b.grad() {
                            input_b.set_grad(&existing_grad + &grad_b);
                        } else {
                            input_b.set_grad(grad_b);
                        }
                    }
                },
                Op::ReLU => {
                    if inputs.len() != 1 {
                        return Err(PhynexusError::InvalidArgument(
                            format!("ReLU operation requires 1 input, got {}", inputs.len())
                        ));
                    }
                    
                    let input = self.nodes.get(&inputs[0].id())
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} not found in the graph", inputs[0].id())
                        ))?;
                    
                    let x = input.lock().unwrap().output()
                        .ok_or_else(|| PhynexusError::InvalidArgument(
                            format!("Input node with ID {} has no output tensor", inputs[0].id())
                        ))?
                        .clone();
                    
                    if input.lock().unwrap().requires_grad() {
                        let mut input = input.lock().unwrap();
                        let mask = x.gt(&x.zeros_like()?)?;
                        let grad_input = &grad * &mask;
                        if let Some(existing_grad) = input.grad() {
                            input.set_grad(&existing_grad + &grad_input);
                        } else {
                            input.set_grad(grad_input);
                        }
                    }
                },
                _ => {
                    return Err(PhynexusError::UnsupportedOperation(
                        format!("Backward pass not yet implemented for operation {:?}", op)
                    ));
                }
            }
        }
        
        Ok(())
    }
}
