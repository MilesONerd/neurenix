"""
Utility functions for quantum computing in Neurenix.

This module provides utility functions for working with quantum states,
measurements, and other quantum computing operations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Callable

from neurenix.tensor import Tensor

def state_to_tensor(state: np.ndarray) -> Tensor:
    """
    Convert a quantum state vector to a Neurenix tensor.
    
    Args:
        state: Quantum state vector
        
    Returns:
        Tensor representation of the state
    """
    return Tensor(state)

def tensor_to_state(tensor: Tensor) -> np.ndarray:
    """
    Convert a Neurenix tensor to a quantum state vector.
    
    Args:
        tensor: Tensor representation of the state
        
    Returns:
        Quantum state vector
    """
    return tensor.to_numpy()

def measure_expectation(state: Tensor, observable: Tensor) -> Tensor:
    """
    Measure the expectation value of an observable with respect to a state.
    
    Args:
        state: Quantum state tensor
        observable: Observable tensor
        
    Returns:
        Expectation value
    """
    state_conj = Tensor.conj(state)
    expectation = Tensor.matmul(state_conj, Tensor.matmul(observable, state))
    
    return expectation

def quantum_gradient(circuit_func: Callable, params: Tensor, epsilon: float = 1e-4) -> Tensor:
    """
    Compute the gradient of a quantum circuit with respect to its parameters.
    
    Args:
        circuit_func: Function that takes parameters and returns a quantum state
        params: Circuit parameters
        epsilon: Small value for finite difference approximation
        
    Returns:
        Gradient tensor
    """
    num_params = params.shape[0]
    gradients = []
    
    f0 = circuit_func(params)
    
    for i in range(num_params):
        params_plus = params.clone()
        params_plus[i] += epsilon
        
        f_plus = circuit_func(params_plus)
        
        grad_i = (f_plus - f0) / epsilon
        gradients.append(grad_i)
        
    return Tensor.stack(gradients)

def quantum_fisher_information(circuit_func: Callable, params: Tensor, epsilon: float = 1e-4) -> Tensor:
    """
    Compute the quantum Fisher information matrix.
    
    Args:
        circuit_func: Function that takes parameters and returns a quantum state
        params: Circuit parameters
        epsilon: Small value for finite difference approximation
        
    Returns:
        Fisher information matrix
    """
    num_params = params.shape[0]
    fisher = Tensor.zeros((num_params, num_params))
    
    gradients = quantum_gradient(circuit_func, params, epsilon)
    
    for i in range(num_params):
        for j in range(num_params):
            fisher[i, j] = Tensor.dot(gradients[i], gradients[j])
            
    return fisher

def quantum_natural_gradient(circuit_func: Callable, params: Tensor, epsilon: float = 1e-4, lambda_reg: float = 1e-4) -> Tensor:
    """
    Compute the quantum natural gradient.
    
    Args:
        circuit_func: Function that takes parameters and returns a quantum state
        params: Circuit parameters
        epsilon: Small value for finite difference approximation
        lambda_reg: Regularization parameter for matrix inversion
        
    Returns:
        Natural gradient tensor
    """
    gradient = quantum_gradient(circuit_func, params, epsilon)
    
    fisher = quantum_fisher_information(circuit_func, params, epsilon)
    
    fisher_reg = fisher + lambda_reg * Tensor.eye(fisher.shape[0])
    
    natural_gradient = Tensor.matmul(Tensor.inverse(fisher_reg), gradient)
    
    return natural_gradient

def quantum_state_fidelity(state1: Tensor, state2: Tensor) -> Tensor:
    """
    Compute the fidelity between two quantum states.
    
    Args:
        state1: First quantum state
        state2: Second quantum state
        
    Returns:
        Fidelity between the states
    """
    overlap = Tensor.abs(Tensor.dot(Tensor.conj(state1), state2)) ** 2
    
    return overlap

def quantum_state_trace_distance(state1: Tensor, state2: Tensor) -> Tensor:
    """
    Compute the trace distance between two quantum states.
    
    Args:
        state1: First quantum state
        state2: Second quantum state
        
    Returns:
        Trace distance between the states
    """
    diff = Tensor.outer(state1, Tensor.conj(state1)) - Tensor.outer(state2, Tensor.conj(state2))
    trace_dist = 0.5 * Tensor.trace(Tensor.abs(diff))
    
    return trace_dist

def quantum_state_entropy(state: Tensor) -> Tensor:
    """
    Compute the von Neumann entropy of a quantum state.
    
    Args:
        state: Quantum state
        
    Returns:
        Entropy of the state
    """
    rho = Tensor.outer(state, Tensor.conj(state))
    
    eigenvalues = Tensor.eigvals(rho)
    
    entropy = -Tensor.sum(eigenvalues * Tensor.log(eigenvalues + 1e-10))
    
    return entropy

def quantum_state_purity(state: Tensor) -> Tensor:
    """
    Compute the purity of a quantum state.
    
    Args:
        state: Quantum state
        
    Returns:
        Purity of the state
    """
    rho = Tensor.outer(state, Tensor.conj(state))
    
    purity = Tensor.trace(Tensor.matmul(rho, rho))
    
    return purity

def quantum_state_concurrence(state: Tensor) -> Tensor:
    """
    Compute the concurrence of a two-qubit state.
    
    Args:
        state: Two-qubit state
        
    Returns:
        Concurrence of the state
    """
    rho = Tensor.outer(state, Tensor.conj(state))
    
    sigma_y = Tensor([[0, -1j], [1j, 0]])
    sigma_y_tensor = Tensor.kron(sigma_y, sigma_y)
    rho_tilde = Tensor.matmul(Tensor.matmul(sigma_y_tensor, Tensor.conj(rho)), sigma_y_tensor)
    
    R = Tensor.matmul(rho, rho_tilde)
    
    eigenvalues = Tensor.eigvals(R)
    eigenvalues = Tensor.sqrt(eigenvalues)
    
    eigenvalues = Tensor.sort(eigenvalues, descending=True)
    
    concurrence = Tensor.maximum(0, eigenvalues[0] - eigenvalues[1] - eigenvalues[2] - eigenvalues[3])
    
    return concurrence

def quantum_state_negativity(state: Tensor, dims: Tuple[int, int]) -> Tensor:
    """
    Compute the negativity of a bipartite quantum state.
    
    Args:
        state: Bipartite quantum state
        dims: Dimensions of the subsystems
        
    Returns:
        Negativity of the state
    """
    rho = Tensor.outer(state, Tensor.conj(state))
    
    rho_pt = Tensor.partial_transpose(rho, dims, 1)
    
    eigenvalues = Tensor.eigvals(rho_pt)
    
    negativity = 0.5 * (Tensor.sum(Tensor.abs(eigenvalues)) - 1)
    
    return negativity
