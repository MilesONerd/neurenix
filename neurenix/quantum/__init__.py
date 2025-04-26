"""
Quantum Computing module for Neurenix.

This module provides implementations of quantum computing algorithms
and integrations with popular quantum frameworks like Qiskit and Cirq.
"""

from neurenix.quantum.circuit import (
    QuantumCircuit,
    ParameterizedCircuit,
    CircuitTemplate
)

from neurenix.quantum.backends import (
    QuantumBackend,
    QiskitBackend,
    CirqBackend
)

from neurenix.quantum.operations import (
    HGate,
    XGate,
    YGate,
    ZGate,
    CXGate,
    CZGate,
    SwapGate,
    TGate,
    SGate,
    RXGate,
    RYGate,
    RZGate,
    U1Gate,
    U2Gate,
    U3Gate,
    MeasurementGate
)

from neurenix.quantum.algorithms import (
    VQE,
    QAOA,
    Grover,
    Shor,
    QuantumPhaseEstimation
)

from neurenix.quantum.hybrid import (
    HybridModel,
    QuantumLayer,
    QuantumTensor
)

from neurenix.quantum.utils import (
    state_fidelity,
    process_tomography,
    density_matrix,
    circuit_to_tensor,
    tensor_to_circuit,
    quantum_gradient
)

__all__ = [
    'QuantumCircuit',
    'ParameterizedCircuit',
    'CircuitTemplate',
    
    'QuantumBackend',
    'QiskitBackend',
    'CirqBackend',
    
    'HGate',
    'XGate',
    'YGate',
    'ZGate',
    'CXGate',
    'CZGate',
    'SwapGate',
    'TGate',
    'SGate',
    'RXGate',
    'RYGate',
    'RZGate',
    'U1Gate',
    'U2Gate',
    'U3Gate',
    'MeasurementGate',
    
    'VQE',
    'QAOA',
    'Grover',
    'Shor',
    'QuantumPhaseEstimation',
    
    'HybridModel',
    'QuantumLayer',
    'QuantumTensor',
    
    'state_fidelity',
    'process_tomography',
    'density_matrix',
    'circuit_to_tensor',
    'tensor_to_circuit',
    'quantum_gradient'
]
