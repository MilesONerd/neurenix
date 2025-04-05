
use crate::tensor::Tensor;
use crate::error::{PhynexusError, Result};

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
    w_ih: &Tensor,
    w_hh: &Tensor,
    b_ih: &Tensor,
    b_hh: &Tensor,
) -> Result<LSTMState> {
    let batch_size = x.shape()[0];
    let hidden_size = state.h.shape()[1];
    
    let h_prev = &state.h;
    let c_prev = &state.c;
    
    let gates_x = x.matmul(w_ih.transpose(0, 1)?)?;
    let gates_h = h_prev.matmul(w_hh.transpose(0, 1)?)?;
    let gates = gates_x.add(&gates_h)?;
    
    let gates = gates.add(b_ih)?;
    let gates = gates.add(b_hh)?;
    
    let chunks = gates.chunk(4, 1)?;
    let i_gate = chunks[0].sigmoid()?;
    let f_gate = chunks[1].sigmoid()?;
    let g_gate = chunks[2].tanh()?;
    let o_gate = chunks[3].sigmoid()?;
    
    let c_next = f_gate.mul(c_prev)?.add(&i_gate.mul(&g_gate)?)?;
    
    let h_next = o_gate.mul(&c_next.tanh()?)?;
    
    Ok(LSTMState::new(h_next, c_next))
}

pub fn gru_forward(
    x: &Tensor,
    h: &Tensor,
    w_ih: &Tensor,
    w_hh: &Tensor,
    b_ih: &Tensor,
    b_hh: &Tensor,
) -> Result<Tensor> {
    let batch_size = x.shape()[0];
    let hidden_size = h.shape()[1];
    
    let gates_x = x.matmul(w_ih.transpose(0, 1)?)?;
    let gates_h = h.matmul(w_hh.transpose(0, 1)?)?;
    
    let gates_x = gates_x.add(b_ih)?;
    let gates_h = gates_h.add(b_hh)?;
    
    let chunks_x = gates_x.chunk(3, 1)?;
    let chunks_h = gates_h.chunk(3, 1)?;
    
    let r_gate = (chunks_x[0].add(&chunks_h[0])?).sigmoid()?;
    let z_gate = (chunks_x[1].add(&chunks_h[1])?).sigmoid()?;
    
    let n_gate = chunks_x[2].add(&(r_gate.mul(&chunks_h[2])?))?;
    let n_gate = n_gate.tanh()?;
    
    let one_minus_z = z_gate.neg()?.add_scalar(1.0)?;
    let h_next = z_gate.mul(h)?.add(&one_minus_z.mul(&n_gate)?)?;
    
    Ok(h_next)
}
