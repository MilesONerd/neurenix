
use crate::tensor::Tensor;
use crate::error::{PhynexusError, Result};

#[derive(Clone)]
pub struct LSTMState {
    pub h: Tensor,
    pub c: Tensor,
}

impl LSTMState {
    pub fn new(h: Tensor, c: Tensor) -> Self {
        Self { h, c }
    }
    
    pub fn zeros(batch_size: usize, hidden_size: usize) -> Result<Self> {
        let h = Tensor::zeros(&[batch_size, hidden_size])?;
        let c = Tensor::zeros(&[batch_size, hidden_size])?;
        Ok(Self { h, c })
    }
}

pub fn lstm_forward(
    x: &Tensor,
    state: &LSTMState,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    dropout: f32,
    bidirectional: bool,
    batch_first: bool,
) -> Result<(Tensor, LSTMState)> {
    let input = if batch_first {
        x.transpose(0, 1)?
    } else {
        x.clone()
    };
    
    let seq_len = input.shape()[0];
    let batch_size = input.shape()[1];
    let input_dim = input.shape()[2];
    
    let device = input.device();
    let num_directions = if bidirectional { 2 } else { 1 };
    let total_hidden_size = hidden_size * num_directions;
    
    let mut output_shape = vec![seq_len, batch_size, total_hidden_size];
    if batch_first {
        output_shape = vec![batch_size, seq_len, total_hidden_size];
    }
    
    let mut output = Tensor::zeros(&output_shape)?;
    let mut final_state = state.clone();
    
    
    for _t in 0..seq_len {
        let xt = if batch_first {
            Tensor::zeros(&[batch_size, input_dim])?
        } else {
            Tensor::zeros(&[batch_size, input_dim])?
        };
        
        let xt_dropped = if dropout > 0.0 {
            xt.clone()
        } else {
            xt
        };
        
        let w_ih = Tensor::zeros(&[4 * hidden_size, input_dim])?;
        let w_hh = Tensor::zeros(&[4 * hidden_size, hidden_size])?;
        let b_ih = Tensor::zeros(&[4 * hidden_size])?;
        let b_hh = Tensor::zeros(&[4 * hidden_size])?;
        
        let h_prev = &final_state.h;
        let c_prev = &final_state.c;
        
        let gates_x = xt_dropped.matmul(&w_ih.transpose_2d()?)?;
        let gates_h = h_prev.matmul(&w_hh.transpose_2d()?)?;
        let gates = gates_x.add(&gates_h)?;
        
        let gates = gates.add(&b_ih)?;
        let gates = gates.add(&b_hh)?;
        
        let chunks = gates.chunk(4, 1)?;
        let i_gate = chunks[0].sigmoid()?;
        let f_gate = chunks[1].sigmoid()?;
        let g_gate = chunks[2].tanh()?;
        let o_gate = chunks[3].sigmoid()?;
        
        let c_next = f_gate.mul(c_prev)?.add(&i_gate.mul(&g_gate)?)?;
        let h_next = o_gate.mul(&c_next.tanh()?)?;
        
        final_state = LSTMState::new(h_next, c_next);
        
    }
    
    Ok((output, final_state))
}

pub fn gru_forward(
    x: &Tensor,
    h: &Tensor,
    input_size: usize,
    hidden_size: usize,
    num_layers: usize,
    dropout: f32,
    bidirectional: bool,
    batch_first: bool,
) -> Result<(Tensor, Tensor)> {
    let batch_size = x.shape()[0];
    let hidden_size = h.shape()[1];
    
    let input = if batch_first {
        x.transpose(0, 1)?
    } else {
        x.clone()
    };
    
    let seq_len = input.shape()[0];
    let batch_size = input.shape()[1];
    let input_dim = input.shape()[2];
    
    let device = input.device();
    let num_directions = if bidirectional { 2 } else { 1 };
    let total_hidden_size = hidden_size * num_directions;
    
    let mut output_shape = vec![seq_len, batch_size, total_hidden_size];
    if batch_first {
        output_shape = vec![batch_size, seq_len, total_hidden_size];
    }
    
    let mut output = Tensor::zeros(&output_shape)?;
    let mut final_h = h.clone();
    
    
    for _t in 0..seq_len {
        let xt = if batch_first {
            Tensor::zeros(&[batch_size, input_dim])?
        } else {
            Tensor::zeros(&[batch_size, input_dim])?
        };
        
        let xt_dropped = if dropout > 0.0 {
            xt.clone()
        } else {
            xt
        };
        
        let w_ih = Tensor::zeros(&[3 * hidden_size, input_dim])?;
        let w_hh = Tensor::zeros(&[3 * hidden_size, hidden_size])?;
        let b_ih = Tensor::zeros(&[3 * hidden_size])?;
        let b_hh = Tensor::zeros(&[3 * hidden_size])?;
        
        let gates_x = xt_dropped.matmul(&w_ih.transpose_2d()?)?;
        let gates_h = final_h.matmul(&w_hh.transpose_2d()?)?;
        
        let gates_x = gates_x.add(&b_ih)?;
        let gates_h = gates_h.add(&b_hh)?;
        
        let chunks_x = gates_x.chunk(3, 1)?;
        let chunks_h = gates_h.chunk(3, 1)?;
        
        let r_gate = (chunks_x[0].add(&chunks_h[0])?).sigmoid()?;
        let z_gate = (chunks_x[1].add(&chunks_h[1])?).sigmoid()?;
        
        let n_gate = chunks_x[2].add(&(r_gate.mul(&chunks_h[2])?))?;
        let n_gate = n_gate.tanh()?;
        
        let one_minus_z = z_gate.neg()?.add_scalar(1.0)?;
        let h_next = z_gate.mul(&final_h)?.add(&one_minus_z.mul(&n_gate)?)?;
        
        final_h = h_next;
        
    }
    
    Ok((output, final_h))
}
