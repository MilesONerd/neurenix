"""
Implementation of quantum backends for Neurenix.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import logging

from neurenix.device import Device, DeviceType

logger = logging.getLogger("neurenix")

class QuantumBackend(ABC):
    """
    Abstract base class for quantum backends.
    """
    
    @abstractmethod
    def run(self, circuit, shots: int = 1024) -> Dict[str, int]:
        """Run a quantum circuit and return measurement results."""
        pass
        
    @abstractmethod
    def to_matrix(self, circuit) -> np.ndarray:
        """Convert a quantum circuit to a unitary matrix."""
        pass


class QiskitBackend(QuantumBackend):
    """
    Quantum backend using Qiskit.
    """
    
    def __init__(self, device: Optional[Device] = None):
        """
        Initialize a Qiskit backend.
        
        Args:
            device: Quantum device for the backend
        """
        self.device = device or Device(DeviceType.QUANTUM)
        self._backend = None
        
        try:
            import qiskit
            self.qiskit = qiskit
            self.available = True
        except ImportError:
            logger.warning("Qiskit not found. QiskitBackend will not be available.")
            self.qiskit = None
            self.available = False
            
    def run(self, circuit, shots: int = 1024) -> Dict[str, int]:
        """
        Run a quantum circuit using Qiskit and return measurement results.
        
        Args:
            circuit: Quantum circuit to run
            shots: Number of shots to run
            
        Returns:
            Dictionary of measurement results and their counts
        """
        if not self.available:
            raise RuntimeError("Qiskit is not available. Please install qiskit.")
            
        qiskit_circuit = self._convert_to_qiskit(circuit)
        
        if self._backend is None:
            self._backend = self.qiskit.Aer.get_backend('qasm_simulator')
            
        result = self.qiskit.execute(
            qiskit_circuit,
            backend=self._backend,
            shots=shots
        ).result()
        
        counts = result.get_counts(qiskit_circuit)
        return counts
        
    def to_matrix(self, circuit) -> np.ndarray:
        """
        Convert a quantum circuit to a unitary matrix using Qiskit.
        
        Args:
            circuit: Quantum circuit to convert
            
        Returns:
            Unitary matrix representation of the circuit
        """
        if not self.available:
            raise RuntimeError("Qiskit is not available. Please install qiskit.")
            
        qiskit_circuit = self._convert_to_qiskit(circuit)
        
        simulator = self.qiskit.Aer.get_backend('unitary_simulator')
        result = self.qiskit.execute(qiskit_circuit, backend=simulator).result()
        unitary = result.get_unitary(qiskit_circuit)
        
        return unitary
        
    def _convert_to_qiskit(self, circuit):
        """
        Convert a Neurenix quantum circuit to a Qiskit circuit.
        
        Args:
            circuit: Neurenix quantum circuit
            
        Returns:
            Qiskit quantum circuit
        """
        from qiskit import QuantumCircuit as QiskitQuantumCircuit
        
        qiskit_circuit = QiskitQuantumCircuit(circuit.num_qubits, circuit.num_qubits)
        
        for op in circuit.operations:
            if op[0] == 'h':
                qiskit_circuit.h(op[1])
            elif op[0] == 'x':
                qiskit_circuit.x(op[1])
            elif op[0] == 'y':
                qiskit_circuit.y(op[1])
            elif op[0] == 'z':
                qiskit_circuit.z(op[1])
            elif op[0] == 'cx':
                qiskit_circuit.cx(op[1], op[2])
            elif op[0] == 'cz':
                qiskit_circuit.cz(op[1], op[2])
            elif op[0] == 'rx':
                qiskit_circuit.rx(op[2], op[1])
            elif op[0] == 'ry':
                qiskit_circuit.ry(op[2], op[1])
            elif op[0] == 'rz':
                qiskit_circuit.rz(op[2], op[1])
            elif op[0] == 'measure':
                qiskit_circuit.measure(op[1], op[2])
        
        return qiskit_circuit


class CirqBackend(QuantumBackend):
    """
    Quantum backend using Cirq.
    """
    
    def __init__(self, device: Optional[Device] = None):
        """
        Initialize a Cirq backend.
        
        Args:
            device: Quantum device for the backend
        """
        self.device = device or Device(DeviceType.QUANTUM)
        
        try:
            import cirq
            self.cirq = cirq
            self.available = True
        except ImportError:
            logger.warning("Cirq not found. CirqBackend will not be available.")
            self.cirq = None
            self.available = False
            
    def run(self, circuit, shots: int = 1024) -> Dict[str, int]:
        """
        Run a quantum circuit using Cirq and return measurement results.
        
        Args:
            circuit: Quantum circuit to run
            shots: Number of shots to run
            
        Returns:
            Dictionary of measurement results and their counts
        """
        if not self.available:
            raise RuntimeError("Cirq is not available. Please install cirq.")
            
        cirq_circuit, qubit_map = self._convert_to_cirq(circuit)
        
        simulator = self.cirq.Simulator()
        result = simulator.run(cirq_circuit, repetitions=shots)
        
        counts = {}
        for key, val in result.histogram(key='all').items():
            key_str = ''.join([str(int((key >> i) & 1)) for i in range(circuit.num_qubits)])
            counts[key_str] = val
            
        return counts
        
    def to_matrix(self, circuit) -> np.ndarray:
        """
        Convert a quantum circuit to a unitary matrix using Cirq.
        
        Args:
            circuit: Quantum circuit to convert
            
        Returns:
            Unitary matrix representation of the circuit
        """
        if not self.available:
            raise RuntimeError("Cirq is not available. Please install cirq.")
            
        cirq_circuit, qubit_map = self._convert_to_cirq(circuit)
        
        unitary = self.cirq.unitary(cirq_circuit)
        
        return unitary
        
    def _convert_to_cirq(self, circuit):
        """
        Convert a Neurenix quantum circuit to a Cirq circuit.
        
        Args:
            circuit: Neurenix quantum circuit
            
        Returns:
            Tuple of (Cirq circuit, qubit mapping)
        """
        cirq_circuit = self.cirq.Circuit()
        
        qubits = [self.cirq.LineQubit(i) for i in range(circuit.num_qubits)]
        qubit_map = {i: qubit for i, qubit in enumerate(qubits)}
        
        for op in circuit.operations:
            if op[0] == 'h':
                cirq_circuit.append(self.cirq.H(qubits[op[1]]))
            elif op[0] == 'x':
                cirq_circuit.append(self.cirq.X(qubits[op[1]]))
            elif op[0] == 'y':
                cirq_circuit.append(self.cirq.Y(qubits[op[1]]))
            elif op[0] == 'z':
                cirq_circuit.append(self.cirq.Z(qubits[op[1]]))
            elif op[0] == 'cx':
                cirq_circuit.append(self.cirq.CNOT(qubits[op[1]], qubits[op[2]]))
            elif op[0] == 'cz':
                cirq_circuit.append(self.cirq.CZ(qubits[op[1]], qubits[op[2]]))
            elif op[0] == 'rx':
                cirq_circuit.append(self.cirq.rx(op[2])(qubits[op[1]]))
            elif op[0] == 'ry':
                cirq_circuit.append(self.cirq.ry(op[2])(qubits[op[1]]))
            elif op[0] == 'rz':
                cirq_circuit.append(self.cirq.rz(op[2])(qubits[op[1]]))
            elif op[0] == 'measure':
                cirq_circuit.append(self.cirq.measure(qubits[op[1]], key=str(op[2])))
        
        return cirq_circuit, qubit_map


def get_default_backend(device: Optional[Device] = None) -> QuantumBackend:
    """
    Get the default quantum backend based on availability.
    
    Args:
        device: Device to use for the backend
        
    Returns:
        Default quantum backend
    """
    qiskit_backend = QiskitBackend(device)
    if qiskit_backend.available:
        return qiskit_backend
        
    cirq_backend = CirqBackend(device)
    if cirq_backend.available:
        return cirq_backend
        
    return FallbackBackend(device)




class FallbackBackend(QuantumBackend):
    """
    Fallback quantum backend using NumPy for basic operations.
    """
    
    def __init__(self, device: Optional[Device] = None):
        """
        Initialize a fallback backend.
        
        Args:
            device: Quantum device for the backend
        """
        self.device = device or Device(DeviceType.CPU)
        self.available = True
        
    def run(self, circuit, shots: int = 1024) -> Dict[str, int]:
        """
        Run a quantum circuit using NumPy and return measurement results.
        
        Args:
            circuit: Quantum circuit to run
            shots: Number of shots to run
            
        Returns:
            Dictionary of measurement results and their counts
        """
        unitary = self.to_matrix(circuit)
        
        state = np.zeros(2**circuit.num_qubits, dtype=np.complex128)
        state[0] = 1.0
        
        final_state = unitary @ state
        
        probabilities = np.abs(final_state)**2
        
        outcomes = np.random.choice(2**circuit.num_qubits, size=shots, p=probabilities)
        
        counts = {}
        for outcome in outcomes:
            binary = format(outcome, f'0{circuit.num_qubits}b')
            counts[binary] = counts.get(binary, 0) + 1
            
        return counts
        
    def to_matrix(self, circuit) -> np.ndarray:
        """
        Convert a quantum circuit to a unitary matrix using NumPy.
        
        Args:
            circuit: Quantum circuit to convert
            
        Returns:
            Unitary matrix representation of the circuit
        """
        n = circuit.num_qubits
        dim = 2**n
        
        unitary = np.eye(dim, dtype=np.complex128)
        
        h_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        x_gate = np.array([[0, 1], [1, 0]])
        y_gate = np.array([[0, -1j], [1j, 0]])
        z_gate = np.array([[1, 0], [0, -1]])
        
        for op in circuit.operations:
            if op[0] == 'h':
                qubit = op[1]
                unitary = self._apply_single_qubit_gate(unitary, h_gate, qubit, n)
            elif op[0] == 'x':
                qubit = op[1]
                unitary = self._apply_single_qubit_gate(unitary, x_gate, qubit, n)
            elif op[0] == 'y':
                qubit = op[1]
                unitary = self._apply_single_qubit_gate(unitary, y_gate, qubit, n)
            elif op[0] == 'z':
                qubit = op[1]
                unitary = self._apply_single_qubit_gate(unitary, z_gate, qubit, n)
            
        return unitary
        
    def _apply_single_qubit_gate(self, unitary: np.ndarray, gate: np.ndarray, qubit: int, n: int) -> np.ndarray:
        """Apply a single-qubit gate to the unitary matrix."""
        if n == 1:
            return gate @ unitary
            
        result = np.zeros_like(unitary)
        
        for i in range(2**n):
            if (i & (1 << qubit)) == 0:
                j = i
                k = i | (1 << qubit)
                result[j] += gate[0, 0] * unitary[j] + gate[0, 1] * unitary[k]
                result[k] += gate[1, 0] * unitary[j] + gate[1, 1] * unitary[k]
                
        return result
