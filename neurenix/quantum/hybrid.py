"""
Hybrid quantum-classical models for Neurenix quantum computing module.

This module provides tools for creating hybrid quantum-classical neural networks,
where classical neural networks are combined with quantum circuits to leverage
the strengths of both paradigms.
"""

import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any, Callable, Sequence

from ..core import PhynexusExtension
from ..device import Device
from ..nn.module import Module
from ..tensor import Tensor

class QuantumLayer(Module):
    """
    A quantum layer that can be integrated into classical neural networks.
    
    This layer encapsulates a quantum circuit that can be used as part of
    a hybrid quantum-classical neural network.
    """
    
    def __init__(self, 
                 n_qubits: int,
                 backend_type: str = 'qiskit',
                 backend_name: str = 'qasm_simulator',
                 shots: int = 1024,
                 device: Optional[Device] = None):
        """
        Initialize a quantum layer.
        
        Args:
            n_qubits: Number of qubits in the quantum circuit
            backend_type: Type of quantum backend ('qiskit' or 'cirq')
            backend_name: Name of the specific backend to use
            shots: Number of shots for quantum circuit execution
            device: Optional Neurenix device for accelerated simulation
        """
        super().__init__()
        self.n_qubits = n_qubits
        self.backend_type = backend_type
        self.backend_name = backend_name
        self.shots = shots
        self.device = device
        self._circuit = None
        self._backend = None
        
        if backend_type == 'qiskit':
            from .qiskit_interface import QiskitBackend, QiskitCircuit, QiskitQuantumRegister, QiskitClassicalRegister
            self.qreg = QiskitQuantumRegister(n_qubits)
            self.creg = QiskitClassicalRegister(n_qubits)
            self._circuit = QiskitCircuit([self.qreg], [self.creg])
            self._backend = QiskitBackend(backend_name, shots=shots, device=device)
        elif backend_type == 'cirq':
            from .cirq_interface import CirqBackend, CirqCircuit, CirqQubit
            self._circuit = CirqCircuit()
            self.qubits = [CirqQubit(id=i) for i in range(n_qubits)]
            self._backend = CirqBackend(backend_name, shots=shots, device=device)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
        
        self.register_parameter("theta", Tensor.zeros((n_qubits, 3)))
    
    def build_circuit(self):
        """
        Build the quantum circuit based on the current parameters.
        
        This method should be overridden by subclasses to define the
        specific quantum circuit architecture.
        """
        if self.backend_type == 'qiskit':
            for i in range(self.n_qubits):
                self._circuit.h(self.qreg.register[i])
                
            for i in range(self.n_qubits):
                self._circuit.rx(self.theta[i, 0].item(), self.qreg.register[i])
                self._circuit.ry(self.theta[i, 1].item(), self.qreg.register[i])
                self._circuit.rz(self.theta[i, 2].item(), self.qreg.register[i])
            
            for i in range(self.n_qubits - 1):
                self._circuit.cx(self.qreg.register[i], self.qreg.register[i + 1])
            
            self._circuit.measure_all()
        
        elif self.backend_type == 'cirq':
            for i in range(self.n_qubits):
                self._circuit.h(self.qubits[i])
                
            for i in range(self.n_qubits):
                self._circuit.rx(self.qubits[i], self.theta[i, 0].item())
                self._circuit.ry(self.qubits[i], self.theta[i, 1].item())
                self._circuit.rz(self.qubits[i], self.theta[i, 2].item())
            
            for i in range(self.n_qubits - 1):
                self._circuit.cnot(self.qubits[i], self.qubits[i + 1])
            
            self._circuit.measure_all(self.qubits)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the quantum layer.
        
        Args:
            x: Input tensor with shape (batch_size, n_features)
            
        Returns:
            Output tensor with shape (batch_size, n_qubits)
        """
        batch_size = x.shape[0]
        results = []
        
        for i in range(batch_size):
            input_sample = x[i]
            n_features = min(input_sample.shape[0], self.n_qubits * 3)
            
            reshaped_input = input_sample[:n_features].reshape(-1, 3)
            if reshaped_input.shape[0] < self.n_qubits:
                padding = Tensor.zeros((self.n_qubits - reshaped_input.shape[0], 3))
                reshaped_input = Tensor.cat([reshaped_input, padding], dim=0)
            
            self.theta.data = reshaped_input
            
            self.build_circuit()
            result = self._backend.run_circuit(self._circuit.circuit)
            counts = self._backend.get_counts(result)
            
            output = np.zeros(self.n_qubits)
            for bitstring, count in counts.items():
                if isinstance(bitstring, str):
                    for i, bit in enumerate(reversed(bitstring[-self.n_qubits:])):
                        output[i] += int(bit) * count / self.shots
            
            results.append(output)
        
        return Tensor(np.array(results))
    
    def extra_repr(self) -> str:
        """Return extra information about the layer."""
        return f'n_qubits={self.n_qubits}, backend={self.backend_type}:{self.backend_name}'


class QuantumNeuralNetwork(Module):
    """
    A hybrid quantum-classical neural network.
    
    This module combines classical neural network layers with quantum layers
    to create a hybrid architecture.
    """
    
    def __init__(self, 
                 classical_layers: List[Module],
                 quantum_layer: QuantumLayer,
                 output_layers: List[Module]):
        """
        Initialize a hybrid quantum-classical neural network.
        
        Args:
            classical_layers: List of classical neural network layers
            quantum_layer: Quantum layer to use in the hybrid model
            output_layers: List of classical output layers
        """
        super().__init__()
        self.classical_layers = Module()
        for i, layer in enumerate(classical_layers):
            setattr(self.classical_layers, f'layer{i}', layer)
        
        self.quantum_layer = quantum_layer
        
        self.output_layers = Module()
        for i, layer in enumerate(output_layers):
            setattr(self.output_layers, f'layer{i}', layer)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the hybrid quantum-classical neural network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        x = self.classical_layers(x)
        
        x = self.quantum_layer(x)
        
        x = self.output_layers(x)
        
        return x


class ParameterizedQuantumCircuit:
    """
    A parameterized quantum circuit that can be optimized.
    
    This class provides a way to define and optimize parameterized
    quantum circuits for various quantum algorithms.
    """
    
    def __init__(self, 
                 n_qubits: int,
                 n_parameters: int,
                 backend_type: str = 'qiskit',
                 backend_name: str = 'qasm_simulator',
                 shots: int = 1024,
                 device: Optional[Device] = None):
        """
        Initialize a parameterized quantum circuit.
        
        Args:
            n_qubits: Number of qubits in the quantum circuit
            n_parameters: Number of parameters in the quantum circuit
            backend_type: Type of quantum backend ('qiskit' or 'cirq')
            backend_name: Name of the specific backend to use
            shots: Number of shots for quantum circuit execution
            device: Optional Neurenix device for accelerated simulation
        """
        self.n_qubits = n_qubits
        self.n_parameters = n_parameters
        self.backend_type = backend_type
        self.backend_name = backend_name
        self.shots = shots
        self.device = device
        self._circuit = None
        self._backend = None
        self.parameters = np.random.randn(n_parameters)
        
        if backend_type == 'qiskit':
            from .qiskit_interface import QiskitBackend, QiskitCircuit, QiskitQuantumRegister, QiskitClassicalRegister
            self.qreg = QiskitQuantumRegister(n_qubits)
            self.creg = QiskitClassicalRegister(n_qubits)
            self._circuit = QiskitCircuit([self.qreg], [self.creg])
            self._backend = QiskitBackend(backend_name, shots=shots, device=device)
        elif backend_type == 'cirq':
            from .cirq_interface import CirqBackend, CirqCircuit, CirqQubit
            self._circuit = CirqCircuit()
            self.qubits = [CirqQubit(id=i) for i in range(n_qubits)]
            self._backend = CirqBackend(backend_name, shots=shots, device=device)
        else:
            raise ValueError(f"Unsupported backend type: {backend_type}")
    
    def build_circuit(self, parameters: Optional[np.ndarray] = None):
        """
        Build the quantum circuit based on the provided parameters.
        
        Args:
            parameters: Optional parameters to use for building the circuit
        """
        if parameters is not None:
            self.parameters = parameters
        
        if self.backend_type == 'qiskit':
            self._circuit = self._circuit.__class__([self.qreg], [self.creg])
            
            param_idx = 0
            for i in range(self.n_qubits):
                self._circuit.h(self.qreg.register[i])
                
                if param_idx < self.n_parameters:
                    self._circuit.rx(self.parameters[param_idx], self.qreg.register[i])
                    param_idx += 1
                
                if param_idx < self.n_parameters:
                    self._circuit.ry(self.parameters[param_idx], self.qreg.register[i])
                    param_idx += 1
                
                if param_idx < self.n_parameters:
                    self._circuit.rz(self.parameters[param_idx], self.qreg.register[i])
                    param_idx += 1
            
            for i in range(self.n_qubits - 1):
                self._circuit.cx(self.qreg.register[i], self.qreg.register[i + 1])
                
                if param_idx < self.n_parameters:
                    self._circuit.rz(self.parameters[param_idx], self.qreg.register[i + 1])
                    param_idx += 1
            
            self._circuit.measure_all()
        
        elif self.backend_type == 'cirq':
            self._circuit = self._circuit.__class__()
            
            param_idx = 0
            for i in range(self.n_qubits):
                self._circuit.h(self.qubits[i])
                
                if param_idx < self.n_parameters:
                    self._circuit.rx(self.qubits[i], self.parameters[param_idx])
                    param_idx += 1
                
                if param_idx < self.n_parameters:
                    self._circuit.ry(self.qubits[i], self.parameters[param_idx])
                    param_idx += 1
                
                if param_idx < self.n_parameters:
                    self._circuit.rz(self.qubits[i], self.parameters[param_idx])
                    param_idx += 1
            
            for i in range(self.n_qubits - 1):
                self._circuit.cnot(self.qubits[i], self.qubits[i + 1])
                
                if param_idx < self.n_parameters:
                    self._circuit.rz(self.qubits[i + 1], self.parameters[param_idx])
                    param_idx += 1
            
            self._circuit.measure_all(self.qubits)
    
    def run(self, parameters: Optional[np.ndarray] = None) -> Dict:
        """
        Run the quantum circuit with the provided parameters.
        
        Args:
            parameters: Optional parameters to use for running the circuit
            
        Returns:
            Dictionary of measurement results and counts
        """
        self.build_circuit(parameters)
        result = self._backend.run_circuit(self._circuit.circuit)
        return self._backend.get_counts(result)
    
    def expectation(self, parameters: Optional[np.ndarray] = None, observable: Optional[str] = 'Z') -> float:
        """
        Calculate the expectation value of an observable.
        
        Args:
            parameters: Optional parameters to use for running the circuit
            observable: Observable to measure ('Z', 'X', 'Y')
            
        Returns:
            Expectation value of the observable
        """
        counts = self.run(parameters)
        
        expectation = 0.0
        total_shots = sum(counts.values())
        
        if observable == 'Z':
            for bitstring, count in counts.items():
                if isinstance(bitstring, str):
                    parity = sum(int(bit) for bit in bitstring) % 2
                    expectation += (-1) ** parity * count / total_shots
        
        return expectation


class HybridOptimizer:
    """
    Optimizer for hybrid quantum-classical models.
    
    This class provides methods for optimizing parameters of hybrid
    quantum-classical models using classical optimization algorithms.
    """
    
    def __init__(self, 
                 quantum_circuit: ParameterizedQuantumCircuit,
                 optimizer_type: str = 'adam',
                 learning_rate: float = 0.01):
        """
        Initialize a hybrid optimizer.
        
        Args:
            quantum_circuit: Parameterized quantum circuit to optimize
            optimizer_type: Type of classical optimizer to use
            learning_rate: Learning rate for the optimizer
        """
        self.quantum_circuit = quantum_circuit
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.parameters = quantum_circuit.parameters.copy()
        
        self.step_count = 0
        if optimizer_type == 'adam':
            self.m = np.zeros_like(self.parameters)
            self.v = np.zeros_like(self.parameters)
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
    
    def compute_gradient(self, cost_function: Callable[[np.ndarray], float], epsilon: float = 0.01) -> np.ndarray:
        """
        Compute the gradient of a cost function with respect to the parameters.
        
        Args:
            cost_function: Cost function to optimize
            epsilon: Small value for finite difference approximation
            
        Returns:
            Gradient of the cost function
        """
        gradient = np.zeros_like(self.parameters)
        cost = cost_function(self.parameters)
        
        for i in range(len(self.parameters)):
            params_plus = self.parameters.copy()
            params_plus[i] += epsilon
            cost_plus = cost_function(params_plus)
            
            gradient[i] = (cost_plus - cost) / epsilon
        
        return gradient
    
    def step(self, cost_function: Callable[[np.ndarray], float]) -> float:
        """
        Perform one optimization step.
        
        Args:
            cost_function: Cost function to optimize
            
        Returns:
            Current value of the cost function
        """
        self.step_count += 1
        
        gradient = self.compute_gradient(cost_function)
        
        if self.optimizer_type == 'sgd':
            self.parameters -= self.learning_rate * gradient
        
        elif self.optimizer_type == 'adam':
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            
            self.v = self.beta2 * self.v + (1 - self.beta2) * gradient**2
            
            m_hat = self.m / (1 - self.beta1**self.step_count)
            
            v_hat = self.v / (1 - self.beta2**self.step_count)
            
            self.parameters -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        self.quantum_circuit.parameters = self.parameters.copy()
        
        return cost_function(self.parameters)
    
    def optimize(self, 
                cost_function: Callable[[np.ndarray], float], 
                n_steps: int = 100, 
                tolerance: float = 1e-5) -> Tuple[np.ndarray, List[float]]:
        """
        Optimize the parameters of the quantum circuit.
        
        Args:
            cost_function: Cost function to optimize
            n_steps: Maximum number of optimization steps
            tolerance: Convergence tolerance
            
        Returns:
            Tuple of optimized parameters and cost history
        """
        cost_history = []
        prev_cost = float('inf')
        
        for _ in range(n_steps):
            cost = self.step(cost_function)
            cost_history.append(cost)
            
            if abs(prev_cost - cost) < tolerance:
                break
            
            prev_cost = cost
        
        return self.parameters, cost_history
