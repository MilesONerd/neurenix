"""
Utility functions for quantum computing in Neurenix.
"""

from typing import List, Dict, Optional, Union, Any, Tuple
import numpy as np

from neurenix.device import Device, DeviceType
from neurenix.tensor import Tensor
from neurenix.quantum.circuit import QuantumCircuit, ParameterizedCircuit

def state_fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Calculate the fidelity between two quantum states.
    
    Args:
        state1: First quantum state
        state2: Second quantum state
        
    Returns:
        Fidelity between the states
    """
    return np.abs(np.vdot(state1, state2))**2

def process_tomography(circuit: QuantumCircuit, shots: int = 1024) -> np.ndarray:
    """
    Perform quantum process tomography on a circuit.
    
    Args:
        circuit: Quantum circuit
        shots: Number of measurement shots
        
    Returns:
        Process matrix
    """
    
    n = circuit.num_qubits
    dim = 2**n
    
    basis_states = []
    for i in range(dim):
        state = np.zeros(dim, dtype=np.complex128)
        state[i] = 1.0
        basis_states.append(state)
        
    output_states = []
    for state in basis_states:
        output_state = circuit.to_matrix() @ state
        output_states.append(output_state)
        
    process_matrix = np.zeros((dim, dim, dim, dim), dtype=np.complex128)
    
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for l in range(dim):
                    process_matrix[i, j, k, l] = np.vdot(basis_states[i], output_states[j]) * np.vdot(output_states[k], basis_states[l])
                    
    return process_matrix

def density_matrix(state: np.ndarray) -> np.ndarray:
    """
    Calculate the density matrix of a quantum state.
    
    Args:
        state: Quantum state vector
        
    Returns:
        Density matrix
    """
    return np.outer(state, np.conj(state))

def circuit_to_tensor(circuit: QuantumCircuit) -> Tensor:
    """
    Convert a quantum circuit to a Neurenix tensor.
    
    Args:
        circuit: Quantum circuit
        
    Returns:
        Tensor representation of the circuit
    """
    state_vector = circuit.to_matrix() @ np.array([1] + [0] * (2**circuit.num_qubits - 1))
    
    return Tensor(state_vector, device=circuit.device)

def tensor_to_circuit(tensor: Tensor, num_qubits: Optional[int] = None) -> QuantumCircuit:
    """
    Convert a Neurenix tensor to a quantum circuit.
    
    Args:
        tensor: Tensor to convert
        num_qubits: Number of qubits (if None, inferred from tensor)
        
    Returns:
        Quantum circuit
    """
    state_vector = tensor.numpy()
    
    if num_qubits is None:
        num_qubits = int(np.log2(len(state_vector)))
        
    circuit = QuantumCircuit(num_qubits, device=tensor.device)
    
    
    return circuit

def quantum_gradient(circuit: ParameterizedCircuit, 
                     parameter: str, 
                     observable: Union[np.ndarray, List[Tuple[str, float]]],
                     epsilon: float = 1e-5) -> float:
    """
    Calculate the gradient of an observable with respect to a circuit parameter.
    
    Args:
        circuit: Parameterized quantum circuit
        parameter: Parameter to calculate gradient for
        observable: Observable to measure
        epsilon: Small value for finite difference
        
    Returns:
        Gradient value
    """
    current_value = circuit.parameter_values.get(parameter, 0.0)
    
    forward_params = circuit.parameter_values.copy()
    forward_params[parameter] = current_value + epsilon
    forward_circuit = circuit.bind_parameters(forward_params)
    
    backward_params = circuit.parameter_values.copy()
    backward_params[parameter] = current_value - epsilon
    backward_circuit = circuit.bind_parameters(backward_params)
    
    if isinstance(observable, np.ndarray):
        forward_state = forward_circuit.to_matrix() @ np.array([1] + [0] * (2**circuit.num_qubits - 1))
        forward_expectation = np.real(np.vdot(forward_state, observable @ forward_state))
        
        backward_state = backward_circuit.to_matrix() @ np.array([1] + [0] * (2**circuit.num_qubits - 1))
        backward_expectation = np.real(np.vdot(backward_state, observable @ backward_state))
    else:
        forward_expectation = 0.0
        backward_expectation = 0.0
        
        for pauli_string, coefficient in observable:
            forward_counts = forward_circuit.run(shots=1024)
            backward_counts = backward_circuit.run(shots=1024)
            
            for bitstring, count in forward_counts.items():
                parity = 1 if bitstring.count('1') % 2 == 0 else -1
                forward_expectation += coefficient * parity * count / 1024
                
            for bitstring, count in backward_counts.items():
                parity = 1 if bitstring.count('1') % 2 == 0 else -1
                backward_expectation += coefficient * parity * count / 1024
                
    gradient = (forward_expectation - backward_expectation) / (2 * epsilon)
    
    return gradient
