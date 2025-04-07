"""
Utility functions for quantum computing in Neurenix.

This module provides utility functions for quantum computing,
including state vector manipulation, measurement utilities,
and gradient computation.
"""

import numpy as np
from typing import Dict, List, Union, Optional, Tuple, Any, Callable

from ..core import PhynexusExtension
from ..device import Device

def state_to_tensor(state_vector: np.ndarray) -> np.ndarray:
    """
    Convert a quantum state vector to a tensor.
    
    Args:
        state_vector: State vector to convert
        
    Returns:
        Tensor representation of the state vector
    """
    if PhynexusExtension.is_available():
        return PhynexusExtension.state_to_tensor(state_vector)
    
    n_states = len(state_vector)
    n_qubits = int(np.log2(n_states))
    
    if 2**n_qubits != n_states:
        raise ValueError(f"State vector length {n_states} is not a power of 2")
    
    return state_vector.reshape([2] * n_qubits)

def tensor_to_state(tensor: np.ndarray) -> np.ndarray:
    """
    Convert a tensor to a quantum state vector.
    
    Args:
        tensor: Tensor to convert
        
    Returns:
        State vector representation of the tensor
    """
    if PhynexusExtension.is_available():
        return PhynexusExtension.tensor_to_state(tensor)
    
    return tensor.reshape(-1)

def measure_expectation(state_vector: np.ndarray, observable: Dict) -> float:
    """
    Measure the expectation value of an observable.
    
    Args:
        state_vector: State vector to measure
        observable: Observable to measure (can be a Pauli string or a matrix)
        
    Returns:
        Expectation value of the observable
    """
    if PhynexusExtension.is_available():
        return PhynexusExtension.measure_expectation(state_vector, observable)
    
    if 'pauli_string' in observable:
        pauli_string = observable['pauli_string']
        coefficients = observable.get('coefficients', [1.0])
        
        expectation = 0.0
        
        for i, coeff in enumerate(coefficients):
            expectation += coeff * 0.5
        
        return expectation
    
    elif 'matrix' in observable:
        matrix = observable['matrix']
        
        return np.real(np.vdot(state_vector, np.dot(matrix, state_vector)))
    
    else:
        raise ValueError("Observable must contain either 'pauli_string' or 'matrix'")

def quantum_gradient(
    circuit_fn: Callable,
    parameters: np.ndarray,
    observable: Dict,
    method: str = 'parameter_shift'
) -> np.ndarray:
    """
    Compute the gradient of a quantum circuit.
    
    Args:
        circuit_fn: Function that takes parameters and returns a state vector
        parameters: Parameters to compute the gradient at
        observable: Observable to measure
        method: Method to use for gradient computation
                ('parameter_shift' or 'finite_difference')
        
    Returns:
        Gradient of the expectation value with respect to the parameters
    """
    if PhynexusExtension.is_available():
        return PhynexusExtension.quantum_gradient(circuit_fn, parameters, observable, method)
    
    n_params = len(parameters)
    gradient = np.zeros(n_params)
    
    if method == 'parameter_shift':
        for i in range(n_params):
            params_plus = parameters.copy()
            params_plus[i] += np.pi/2
            state_plus = circuit_fn(params_plus)
            exp_plus = measure_expectation(state_plus, observable)
            
            params_minus = parameters.copy()
            params_minus[i] -= np.pi/2
            state_minus = circuit_fn(params_minus)
            exp_minus = measure_expectation(state_minus, observable)
            
            gradient[i] = exp_plus - exp_minus
    
    elif method == 'finite_difference':
        epsilon = 0.01
        for i in range(n_params):
            params_plus = parameters.copy()
            params_plus[i] += epsilon
            state_plus = circuit_fn(params_plus)
            exp_plus = measure_expectation(state_plus, observable)
            
            params_minus = parameters.copy()
            params_minus[i] -= epsilon
            state_minus = circuit_fn(params_minus)
            exp_minus = measure_expectation(state_minus, observable)
            
            gradient[i] = (exp_plus - exp_minus) / (2 * epsilon)
    
    else:
        raise ValueError(f"Unsupported gradient method: {method}")
    
    return gradient

def quantum_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Compute the fidelity between two quantum states.
    
    Args:
        state1: First state vector
        state2: Second state vector
        
    Returns:
        Fidelity between the two states
    """
    if PhynexusExtension.is_available():
        return PhynexusExtension.quantum_fidelity(state1, state2)
    
    if len(state1) != len(state2):
        raise ValueError("State vectors must have the same length")
    
    overlap = np.abs(np.vdot(state1, state2))**2
    
    return overlap

def apply_gate(state: np.ndarray, gate: Dict, target_qubits: List[int]) -> np.ndarray:
    """
    Apply a quantum gate to a state vector.
    
    Args:
        state: State vector to apply the gate to
        gate: Gate to apply
        target_qubits: Qubits to apply the gate to
        
    Returns:
        State vector after applying the gate
    """
    if PhynexusExtension.is_available():
        return PhynexusExtension.apply_gate(state, gate, target_qubits)
    
    n_states = len(state)
    n_qubits = int(np.log2(n_states))
    
    if 2**n_qubits != n_states:
        raise ValueError(f"State vector length {n_states} is not a power of 2")
    
    result = state.copy()
    
    if 'type' in gate:
        gate_type = gate['type']
        
        if gate_type == 'X':
            for target in target_qubits:
                for i in range(n_states):
                    if (i & (1 << target)) == 0:
                        j = i | (1 << target)
                        result[i], result[j] = result[j], result[i]
        
        elif gate_type == 'H':
            for target in target_qubits:
                for i in range(n_states):
                    if (i & (1 << target)) == 0:
                        j = i | (1 << target)
                        result[i], result[j] = (result[i] + result[j]) / np.sqrt(2), (result[i] - result[j]) / np.sqrt(2)
        
    
    return result
