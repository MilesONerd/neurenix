"""
Recurrent neural network layers for the Neurenix framework.

This module provides recurrent neural network layers such as RNN, LSTM, and GRU.
"""

from typing import Tuple, Optional, Union, List

import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor

class RNNCell(Module):
    """
    An Elman RNN cell with tanh or ReLU non-linearity.
    
    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be 'tanh' or 'relu'
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        nonlinearity: str = "tanh",
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        if nonlinearity not in ["tanh", "relu"]:
            raise ValueError(f"Unknown nonlinearity: {nonlinearity}")
        self.nonlinearity = nonlinearity
        
        self.weight_ih = Tensor.randn((hidden_size, input_size), requires_grad=True)
        self.weight_hh = Tensor.randn((hidden_size, hidden_size), requires_grad=True)
        
        if bias:
            self.bias_ih = Tensor.zeros((hidden_size,), requires_grad=True)
            self.bias_hh = Tensor.zeros((hidden_size,), requires_grad=True)
        else:
            self.bias_ih = None
            self.bias_hh = None
    
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the RNN cell.
        
        Args:
            input: Input tensor of shape (batch, input_size)
            hx: Hidden state tensor of shape (batch, hidden_size)
            
        Returns:
            New hidden state tensor of shape (batch, hidden_size)
        """
        if hx is None:
            hx = Tensor.zeros((input.shape[0], self.hidden_size), device=input.device)
        
        # Linear transformations
        ih = input.matmul(self.weight_ih.t())
        if self.bias_ih is not None:
            ih = ih + self.bias_ih
        
        hh = hx.matmul(self.weight_hh.t())
        if self.bias_hh is not None:
            hh = hh + self.bias_hh
        
        # Apply nonlinearity
        if self.nonlinearity == "tanh":
            return (ih + hh).tanh()
        else:  # relu
            return (ih + hh).relu()
    
    def __repr__(self) -> str:
        return (
            f"RNNCell({self.input_size}, {self.hidden_size}, "
            f"bias={self.bias}, nonlinearity='{self.nonlinearity}')"
        )


class LSTMCell(Module):
    """
    Long Short-Term Memory (LSTM) cell.
    
    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        bias: If False, then the layer does not use bias weights
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Input-to-hidden weights (4 gates: input, forget, cell, output)
        self.weight_ih = Tensor.randn((4 * hidden_size, input_size), requires_grad=True)
        
        # Hidden-to-hidden weights (4 gates: input, forget, cell, output)
        self.weight_hh = Tensor.randn((4 * hidden_size, hidden_size), requires_grad=True)
        
        if bias:
            self.bias_ih = Tensor.zeros((4 * hidden_size,), requires_grad=True)
            self.bias_hh = Tensor.zeros((4 * hidden_size,), requires_grad=True)
        else:
            self.bias_ih = None
            self.bias_hh = None
    
    def forward(self, input: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the LSTM cell.
        
        Args:
            input: Input tensor of shape (batch, input_size)
            state: Tuple of (h_0, c_0) where h_0 is the hidden state and c_0 is the cell state,
                   both of shape (batch, hidden_size)
            
        Returns:
            Tuple of (h_1, c_1) where h_1 is the new hidden state and c_1 is the new cell state,
            both of shape (batch, hidden_size)
        """
        batch_size = input.shape[0]
        
        if state is None:
            h_0 = Tensor.zeros((batch_size, self.hidden_size), device=input.device)
            c_0 = Tensor.zeros((batch_size, self.hidden_size), device=input.device)
        else:
            h_0, c_0 = state
        
        # Linear transformations
        gates_ih = input.matmul(self.weight_ih.t())
        if self.bias_ih is not None:
            gates_ih = gates_ih + self.bias_ih
        
        gates_hh = h_0.matmul(self.weight_hh.t())
        if self.bias_hh is not None:
            gates_hh = gates_hh + self.bias_hh
        
        gates = gates_ih + gates_hh
        
        # Split gates
        i, f, g, o = gates.chunk(4, dim=1)
        
        # Apply gate nonlinearities
        i = i.sigmoid()  # input gate
        f = f.sigmoid()  # forget gate
        g = g.tanh()     # cell gate
        o = o.sigmoid()  # output gate
        
        # Update cell state
        c_1 = f * c_0 + i * g
        
        # Update hidden state
        h_1 = o * c_1.tanh()
        
        return h_1, c_1
    
    def __repr__(self) -> str:
        return f"LSTMCell({self.input_size}, {self.hidden_size}, bias={self.bias})"


class GRUCell(Module):
    """
    Gated Recurrent Unit (GRU) cell.
    
    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        bias: If False, then the layer does not use bias weights
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Input-to-hidden weights (3 gates: reset, update, new)
        self.weight_ih = Tensor.randn((3 * hidden_size, input_size), requires_grad=True)
        
        # Hidden-to-hidden weights (3 gates: reset, update, new)
        self.weight_hh = Tensor.randn((3 * hidden_size, hidden_size), requires_grad=True)
        
        if bias:
            self.bias_ih = Tensor.zeros((3 * hidden_size,), requires_grad=True)
            self.bias_hh = Tensor.zeros((3 * hidden_size,), requires_grad=True)
        else:
            self.bias_ih = None
            self.bias_hh = None
    
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of the GRU cell.
        
        Args:
            input: Input tensor of shape (batch, input_size)
            hx: Hidden state tensor of shape (batch, hidden_size)
            
        Returns:
            New hidden state tensor of shape (batch, hidden_size)
        """
        if hx is None:
            hx = Tensor.zeros((input.shape[0], self.hidden_size), device=input.device)
        
        # Linear transformations for input
        gates_ih = input.matmul(self.weight_ih.t())
        if self.bias_ih is not None:
            gates_ih = gates_ih + self.bias_ih
        
        # Linear transformations for hidden state
        gates_hh = hx.matmul(self.weight_hh.t())
        if self.bias_hh is not None:
            gates_hh = gates_hh + self.bias_hh
        
        # Split input gates
        r_i, z_i, n_i = gates_ih.chunk(3, dim=1)
        
        # Split hidden gates
        r_h, z_h, n_h = gates_hh.chunk(3, dim=1)
        
        # Reset and update gates
        r = (r_i + r_h).sigmoid()
        z = (z_i + z_h).sigmoid()
        
        # New gate
        n = (n_i + r * n_h).tanh()
        
        # Update hidden state
        h_1 = (1 - z) * n + z * hx
        
        return h_1
    
    def __repr__(self) -> str:
        return f"GRUCell({self.input_size}, {self.hidden_size}, bias={self.bias})"


class RNN(Module):
    """
    Applies a multi-layer Elman RNN with tanh or ReLU non-linearity to an input sequence.
    
    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        num_layers: Number of recurrent layers
        nonlinearity: The non-linearity to use. Can be 'tanh' or 'relu'
        bias: If False, then the layer does not use bias weights
        batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        bidirectional: If True, becomes a bidirectional RNN
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        nonlinearity: str = "tanh",
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.num_directions = 2 if bidirectional else 1
        
        # Create RNN cells for each layer and direction
        self.cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            
            # Forward direction
            self.cells.append(
                RNNCell(layer_input_size, hidden_size, bias, nonlinearity)
            )
            
            # Backward direction (if bidirectional)
            if bidirectional:
                self.cells.append(
                    RNNCell(layer_input_size, hidden_size, bias, nonlinearity)
                )
    
    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the RNN.
        
        Args:
            input: Input tensor of shape (seq_len, batch, input_size) or (batch, seq_len, input_size)
            hx: Hidden state tensor of shape (num_layers * num_directions, batch, hidden_size)
            
        Returns:
            Tuple of (output, h_n) where:
                output: Tensor of shape (seq_len, batch, num_directions * hidden_size) or
                        (batch, seq_len, num_directions * hidden_size)
                h_n: Tensor of shape (num_layers * num_directions, batch, hidden_size)
        """
        # Ensure input is (seq_len, batch, input_size)
        if self.batch_first:
            input = input.transpose(0, 1)
        
        seq_len, batch_size, _ = input.shape
        
        # Initialize hidden state if not provided
        if hx is None:
            hx = Tensor.zeros(
                (self.num_layers * self.num_directions, batch_size, self.hidden_size),
                device=input.device
            )
        
        # Process each layer
        output = input
        h_n = []
        
        for layer in range(self.num_layers):
            # Get hidden state for this layer
            layer_hx = hx[layer * self.num_directions:(layer + 1) * self.num_directions]
            
            # Process sequence
            layer_output = []
            
            if self.bidirectional:
                # Forward direction
                forward_hx = layer_hx[0]
                forward_outputs = []
                
                for t in range(seq_len):
                    forward_hx = self.cells[layer * 2](output[t], forward_hx)
                    forward_outputs.append(forward_hx)
                
                # Backward direction
                backward_hx = layer_hx[1] if layer_hx.shape[0] > 1 else layer_hx[0]
                backward_outputs = []
                
                for t in range(seq_len - 1, -1, -1):
                    backward_hx = self.cells[layer * 2 + 1](output[t], backward_hx)
                    backward_outputs.append(backward_hx)
                
                # Combine directions
                for t in range(seq_len):
                    layer_output.append(
                        Tensor.cat([forward_outputs[t], backward_outputs[seq_len - t - 1]], dim=1)
                    )
                
                # Update hidden state
                h_n.append(forward_hx)
                h_n.append(backward_hx)
            else:
                # Forward direction only
                forward_hx = layer_hx[0]
                
                for t in range(seq_len):
                    forward_hx = self.cells[layer](output[t], forward_hx)
                    layer_output.append(forward_hx)
                
                # Update hidden state
                h_n.append(forward_hx)
            
            # Stack outputs
            output = Tensor.stack(layer_output, dim=0)
            
            # Apply dropout (except last layer)
            if layer < self.num_layers - 1 and self.dropout > 0:
                output = output.dropout(self.dropout)
        
        # Stack hidden states
        h_n = Tensor.stack(h_n, dim=0)
        
        # Convert output to batch_first if needed
        if self.batch_first:
            output = output.transpose(0, 1)
        
        return output, h_n
    
    def __repr__(self) -> str:
        return (
            f"RNN({self.input_size}, {self.hidden_size}, "
            f"num_layers={self.num_layers}, nonlinearity='{self.nonlinearity}', "
            f"bias={self.bias}, batch_first={self.batch_first}, "
            f"dropout={self.dropout}, bidirectional={self.bidirectional})"
        )


class LSTM(Module):
    """
    Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.
    
    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        num_layers: Number of recurrent layers
        bias: If False, then the layer does not use bias weights
        batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each LSTM layer except the last layer
        bidirectional: If True, becomes a bidirectional LSTM
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.num_directions = 2 if bidirectional else 1
        
        # Create LSTM cells for each layer and direction
        self.cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            
            # Forward direction
            self.cells.append(
                LSTMCell(layer_input_size, hidden_size, bias)
            )
            
            # Backward direction (if bidirectional)
            if bidirectional:
                self.cells.append(
                    LSTMCell(layer_input_size, hidden_size, bias)
                )
    
    def forward(
        self,
        input: Tensor,
        hx: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass of the LSTM.
        
        Args:
            input: Input tensor of shape (seq_len, batch, input_size) or (batch, seq_len, input_size)
            hx: Tuple of (h_0, c_0) where:
                h_0: Hidden state tensor of shape (num_layers * num_directions, batch, hidden_size)
                c_0: Cell state tensor of shape (num_layers * num_directions, batch, hidden_size)
            
        Returns:
            Tuple of (output, (h_n, c_n)) where:
                output: Tensor of shape (seq_len, batch, num_directions * hidden_size) or
                        (batch, seq_len, num_directions * hidden_size)
                h_n: Tensor of shape (num_layers * num_directions, batch, hidden_size)
                c_n: Tensor of shape (num_layers * num_directions, batch, hidden_size)
        """
        # Ensure input is (seq_len, batch, input_size)
        if self.batch_first:
            input = input.transpose(0, 1)
        
        seq_len, batch_size, _ = input.shape
        
        # Initialize hidden and cell states if not provided
        if hx is None:
            h_0 = Tensor.zeros(
                (self.num_layers * self.num_directions, batch_size, self.hidden_size),
                device=input.device
            )
            c_0 = Tensor.zeros(
                (self.num_layers * self.num_directions, batch_size, self.hidden_size),
                device=input.device
            )
        else:
            h_0, c_0 = hx
        
        # Process each layer
        output = input
        h_n = []
        c_n = []
        
        for layer in range(self.num_layers):
            # Get hidden and cell states for this layer
            layer_h = h_0[layer * self.num_directions:(layer + 1) * self.num_directions]
            layer_c = c_0[layer * self.num_directions:(layer + 1) * self.num_directions]
            
            # Process sequence
            layer_output = []
            
            if self.bidirectional:
                # Forward direction
                forward_h = layer_h[0]
                forward_c = layer_c[0]
                forward_outputs = []
                
                for t in range(seq_len):
                    forward_h, forward_c = self.cells[layer * 2](
                        output[t], (forward_h, forward_c)
                    )
                    forward_outputs.append(forward_h)
                
                # Backward direction
                backward_h = layer_h[1] if layer_h.shape[0] > 1 else layer_h[0]
                backward_c = layer_c[1] if layer_c.shape[0] > 1 else layer_c[0]
                backward_outputs = []
                
                for t in range(seq_len - 1, -1, -1):
                    backward_h, backward_c = self.cells[layer * 2 + 1](
                        output[t], (backward_h, backward_c)
                    )
                    backward_outputs.append(backward_h)
                
                # Combine directions
                for t in range(seq_len):
                    layer_output.append(
                        Tensor.cat([forward_outputs[t], backward_outputs[seq_len - t - 1]], dim=1)
                    )
                
                # Update hidden and cell states
                h_n.append(forward_h)
                h_n.append(backward_h)
                c_n.append(forward_c)
                c_n.append(backward_c)
            else:
                # Forward direction only
                forward_h = layer_h[0]
                forward_c = layer_c[0]
                
                for t in range(seq_len):
                    forward_h, forward_c = self.cells[layer](
                        output[t], (forward_h, forward_c)
                    )
                    layer_output.append(forward_h)
                
                # Update hidden and cell states
                h_n.append(forward_h)
                c_n.append(forward_c)
            
            # Stack outputs
            output = Tensor.stack(layer_output, dim=0)
            
            # Apply dropout (except last layer)
            if layer < self.num_layers - 1 and self.dropout > 0:
                output = output.dropout(self.dropout)
        
        # Stack hidden and cell states
        h_n = Tensor.stack(h_n, dim=0)
        c_n = Tensor.stack(c_n, dim=0)
        
        # Convert output to batch_first if needed
        if self.batch_first:
            output = output.transpose(0, 1)
        
        return output, (h_n, c_n)
    
    def __repr__(self) -> str:
        return (
            f"LSTM({self.input_size}, {self.hidden_size}, "
            f"num_layers={self.num_layers}, bias={self.bias}, "
            f"batch_first={self.batch_first}, dropout={self.dropout}, "
            f"bidirectional={self.bidirectional})"
        )


class GRU(Module):
    """
    Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.
    
    Args:
        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        num_layers: Number of recurrent layers
        bias: If False, then the layer does not use bias weights
        batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
        dropout: If non-zero, introduces a dropout layer on the outputs of each GRU layer except the last layer
        bidirectional: If True, becomes a bidirectional GRU
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.num_directions = 2 if bidirectional else 1
        
        # Create GRU cells for each layer and direction
        self.cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size * self.num_directions
            
            # Forward direction
            self.cells.append(
                GRUCell(layer_input_size, hidden_size, bias)
            )
            
            # Backward direction (if bidirectional)
            if bidirectional:
                self.cells.append(
                    GRUCell(layer_input_size, hidden_size, bias)
                )
    
    def forward(
        self,
        input: Tensor,
        hx: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the GRU.
        
        Args:
            input: Input tensor of shape (seq_len, batch, input_size) or (batch, seq_len, input_size)
            hx: Hidden state tensor of shape (num_layers * num_directions, batch, hidden_size)
            
        Returns:
            Tuple of (output, h_n) where:
                output: Tensor of shape (seq_len, batch, num_directions * hidden_size) or
                        (batch, seq_len, num_directions * hidden_size)
                h_n: Tensor of shape (num_layers * num_directions, batch, hidden_size)
        """
        # Ensure input is (seq_len, batch, input_size)
        if self.batch_first:
            input = input.transpose(0, 1)
        
        seq_len, batch_size, _ = input.shape
        
        # Initialize hidden state if not provided
        if hx is None:
            hx = Tensor.zeros(
                (self.num_layers * self.num_directions, batch_size, self.hidden_size),
                device=input.device
            )
        
        # Process each layer
        output = input
        h_n = []
        
        for layer in range(self.num_layers):
            # Get hidden state for this layer
            layer_hx = hx[layer * self.num_directions:(layer + 1) * self.num_directions]
            
            # Process sequence
            layer_output = []
            
            if self.bidirectional:
                # Forward direction
                forward_hx = layer_hx[0]
                forward_outputs = []
                
                for t in range(seq_len):
                    forward_hx = self.cells[layer * 2](output[t], forward_hx)
                    forward_outputs.append(forward_hx)
                
                # Backward direction
                backward_hx = layer_hx[1] if layer_hx.shape[0] > 1 else layer_hx[0]
                backward_outputs = []
                
                for t in range(seq_len - 1, -1, -1):
                    backward_hx = self.cells[layer * 2 + 1](output[t], backward_hx)
                    backward_outputs.append(backward_hx)
                
                # Combine directions
                for t in range(seq_len):
                    layer_output.append(
                        Tensor.cat([forward_outputs[t], backward_outputs[seq_len - t - 1]], dim=1)
                    )
                
                # Update hidden state
                h_n.append(forward_hx)
                h_n.append(backward_hx)
            else:
                # Forward direction only
                forward_hx = layer_hx[0]
                
                for t in range(seq_len):
                    forward_hx = self.cells[layer](output[t], forward_hx)
                    layer_output.append(forward_hx)
                
                # Update hidden state
                h_n.append(forward_hx)
            
            # Stack outputs
            output = Tensor.stack(layer_output, dim=0)
            
            # Apply dropout (except last layer)
            if layer < self.num_layers - 1 and self.dropout > 0:
                output = output.dropout(self.dropout)
        
        # Stack hidden states
        h_n = Tensor.stack(h_n, dim=0)
        
        # Convert output to batch_first if needed
        if self.batch_first:
            output = output.transpose(0, 1)
        
        return output, h_n
    
    def __repr__(self) -> str:
        return (
            f"GRU({self.input_size}, {self.hidden_size}, "
            f"num_layers={self.num_layers}, bias={self.bias}, "
            f"batch_first={self.batch_first}, dropout={self.dropout}, "
            f"bidirectional={self.bidirectional})"
        )
