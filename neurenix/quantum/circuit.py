"""
Implementation of quantum circuits for Neurenix.
"""

from typing import List, Dict, Optional, Union, Any, Tuple, Callable
import numpy as np

from neurenix.device import Device, DeviceType
from neurenix.tensor import Tensor

class QuantumCircuit:
    """
    Base class for quantum circuits in Neurenix.
    """
    
    def __init__(self, num_qubits: int, name: str = None, device: Optional[Device] = None):
        """
        Initialize a quantum circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
            name: Optional name for the circuit
            device: Device to run the circuit on
        """
        self.num_qubits = num_qubits
        self.name = name or f"circuit_{num_qubits}q"
        self.device = device or Device(DeviceType.QUANTUM)
        self.operations = []
        self._backend = None
        
    def h(self, qubit: int) -> 'QuantumCircuit':
        """Add Hadamard gate."""
        self.operations.append(('h', qubit))
        return self
        
    def x(self, qubit: int) -> 'QuantumCircuit':
        """Add Pauli-X gate."""
        self.operations.append(('x', qubit))
        return self
        
    def y(self, qubit: int) -> 'QuantumCircuit':
        """Add Pauli-Y gate."""
        self.operations.append(('y', qubit))
        return self
        
    def z(self, qubit: int) -> 'QuantumCircuit':
        """Add Pauli-Z gate."""
        self.operations.append(('z', qubit))
        return self
        
    def cx(self, control: int, target: int) -> 'QuantumCircuit':
        """Add CNOT gate."""
        self.operations.append(('cx', control, target))
        return self
        
    def cz(self, control: int, target: int) -> 'QuantumCircuit':
        """Add CZ gate."""
        self.operations.append(('cz', control, target))
        return self
        
    def rx(self, qubit: int, theta: float) -> 'QuantumCircuit':
        """Add RX rotation gate."""
        self.operations.append(('rx', qubit, theta))
        return self
        
    def ry(self, qubit: int, theta: float) -> 'QuantumCircuit':
        """Add RY rotation gate."""
        self.operations.append(('ry', qubit, theta))
        return self
        
    def rz(self, qubit: int, theta: float) -> 'QuantumCircuit':
        """Add RZ rotation gate."""
        self.operations.append(('rz', qubit, theta))
        return self
        
    def measure(self, qubit: int, classical_bit: Optional[int] = None) -> 'QuantumCircuit':
        """Add measurement operation."""
        if classical_bit is None:
            classical_bit = qubit
        self.operations.append(('measure', qubit, classical_bit))
        return self
        
    def to_matrix(self) -> np.ndarray:
        """
        Convert the circuit to a unitary matrix.
        
        Returns:
            Unitary matrix representation of the circuit
        """
        self._ensure_backend()
        return self._backend.to_matrix(self)
        
    def to_tensor(self) -> Tensor:
        """
        Convert the circuit to a Neurenix tensor.
        
        Returns:
            Tensor representation of the circuit
        """
        matrix = self.to_matrix()
        return Tensor(matrix, device=self.device)
        
    def run(self, shots: int = 1024) -> Dict[str, int]:
        """
        Run the circuit and return measurement results.
        
        Args:
            shots: Number of shots to run
            
        Returns:
            Dictionary of measurement results and their counts
        """
        self._ensure_backend()
        return self._backend.run(self, shots)
        
    def _ensure_backend(self):
        """Ensure a backend is available for the circuit."""
        if self._backend is None:
            from neurenix.quantum.backends import get_default_backend
            self._backend = get_default_backend(self.device)
            
    def __repr__(self) -> str:
        return f"QuantumCircuit(num_qubits={self.num_qubits}, name='{self.name}', operations={len(self.operations)})"


class ParameterizedCircuit(QuantumCircuit):
    """
    Parameterized quantum circuit for variational algorithms.
    """
    
    def __init__(self, num_qubits: int, parameters: Optional[List[str]] = None, name: str = None, device: Optional[Device] = None):
        """
        Initialize a parameterized quantum circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit
            parameters: List of parameter names
            name: Optional name for the circuit
            device: Device to run the circuit on
        """
        super().__init__(num_qubits, name, device)
        self.parameters = parameters or []
        self.parameter_values = {param: 0.0 for param in self.parameters}
        
    def rx_param(self, qubit: int, param_name: str) -> 'ParameterizedCircuit':
        """Add parameterized RX rotation gate."""
        if param_name not in self.parameters:
            self.parameters.append(param_name)
            self.parameter_values[param_name] = 0.0
        self.operations.append(('rx_param', qubit, param_name))
        return self
        
    def ry_param(self, qubit: int, param_name: str) -> 'ParameterizedCircuit':
        """Add parameterized RY rotation gate."""
        if param_name not in self.parameters:
            self.parameters.append(param_name)
            self.parameter_values[param_name] = 0.0
        self.operations.append(('ry_param', qubit, param_name))
        return self
        
    def rz_param(self, qubit: int, param_name: str) -> 'ParameterizedCircuit':
        """Add parameterized RZ rotation gate."""
        if param_name not in self.parameters:
            self.parameters.append(param_name)
            self.parameter_values[param_name] = 0.0
        self.operations.append(('rz_param', qubit, param_name))
        return self
        
    def bind_parameters(self, parameter_values: Dict[str, float]) -> 'QuantumCircuit':
        """
        Bind parameters to values and return a concrete circuit.
        
        Args:
            parameter_values: Dictionary mapping parameter names to values
            
        Returns:
            Concrete quantum circuit with parameters bound to values
        """
        circuit = QuantumCircuit(self.num_qubits, self.name, self.device)
        
        for op in self.operations:
            if op[0].endswith('_param'):
                base_op = op[0].split('_')[0]
                qubit = op[1]
                param_name = op[2]
                param_value = parameter_values.get(param_name, self.parameter_values.get(param_name, 0.0))
                getattr(circuit, base_op)(qubit, param_value)
            else:
                if len(op) == 2:
                    getattr(circuit, op[0])(op[1])
                elif len(op) == 3:
                    getattr(circuit, op[0])(op[1], op[2])
        
        return circuit
        
    def __repr__(self) -> str:
        return f"ParameterizedCircuit(num_qubits={self.num_qubits}, name='{self.name}', parameters={len(self.parameters)})"


class CircuitTemplate:
    """
    Predefined quantum circuit templates for common operations.
    """
    
    @staticmethod
    def bell_pair(device: Optional[Device] = None) -> QuantumCircuit:
        """Create a Bell pair circuit."""
        circuit = QuantumCircuit(2, "bell_pair", device)
        circuit.h(0).cx(0, 1)
        return circuit
        
    @staticmethod
    def ghz_state(num_qubits: int, device: Optional[Device] = None) -> QuantumCircuit:
        """Create a GHZ state circuit."""
        circuit = QuantumCircuit(num_qubits, f"ghz_{num_qubits}", device)
        circuit.h(0)
        for i in range(num_qubits - 1):
            circuit.cx(i, i+1)
        return circuit
        
    @staticmethod
    def w_state(num_qubits: int, device: Optional[Device] = None) -> QuantumCircuit:
        """Create a W state circuit."""
        circuit = QuantumCircuit(num_qubits, f"w_{num_qubits}", device)
        circuit.x(0)
        for i in range(num_qubits - 1):
            theta = np.arccos(np.sqrt(1.0 / (num_qubits - i)))
            circuit.ry(i+1, theta)
            circuit.cx(i, i+1)
        return circuit
        
    @staticmethod
    def qft(num_qubits: int, device: Optional[Device] = None) -> QuantumCircuit:
        """Create a Quantum Fourier Transform circuit."""
        circuit = QuantumCircuit(num_qubits, f"qft_{num_qubits}", device)
        for i in range(num_qubits):
            circuit.h(i)
            for j in range(i + 1, num_qubits):
                angle = np.pi / (2 ** (j - i))
                circuit.cz(i, j)
        for i in range(num_qubits // 2):
            circuit.cx(i, num_qubits-i-1)
            circuit.cx(num_qubits-i-1, i)
            circuit.cx(i, num_qubits-i-1)
        return circuit
