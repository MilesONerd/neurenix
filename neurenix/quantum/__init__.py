"""
Quantum Computing module for Neurenix.

This module provides integration with quantum computing frameworks
like Qiskit and Cirq, allowing for hybrid classical-quantum models
and quantum machine learning algorithms.
"""

from .qiskit_interface import (
    QiskitBackend, QiskitCircuit, QiskitQuantumRegister,
    QiskitClassicalRegister, QiskitQuantumGate
)
from .cirq_interface import (
    CirqBackend, CirqCircuit, CirqQubit, CirqGate,
    CirqSimulator
)
from .hybrid import (
    QuantumLayer, QuantumNeuralNetwork, HybridOptimizer,
    ParameterizedQuantumCircuit
)
from .algorithms import (
    QAOA, VQE, QuantumKernelTrainer, QuantumSVM,
    QuantumFeatureMap
)
from .utils import (
    state_to_tensor, tensor_to_state, measure_expectation,
    quantum_gradient
)

__all__ = [
    'QiskitBackend', 'QiskitCircuit', 'QiskitQuantumRegister',
    'QiskitClassicalRegister', 'QiskitQuantumGate',
    
    'CirqBackend', 'CirqCircuit', 'CirqQubit', 'CirqGate',
    'CirqSimulator',
    
    'QuantumLayer', 'QuantumNeuralNetwork', 'HybridOptimizer',
    'ParameterizedQuantumCircuit',
    
    'QAOA', 'VQE', 'QuantumKernelTrainer', 'QuantumSVM',
    'QuantumFeatureMap',
    
    'state_to_tensor', 'tensor_to_state', 'measure_expectation',
    'quantum_gradient',
]
