"""
Implementation of quantum operations for Neurenix.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np

class QuantumGate(ABC):
    """
    Abstract base class for quantum gates.
    """
    
    @abstractmethod
    def matrix(self) -> np.ndarray:
        """Return the matrix representation of the gate."""
        pass
        
    @abstractmethod
    def apply(self, circuit, *args, **kwargs):
        """Apply the gate to a quantum circuit."""
        pass


class HGate(QuantumGate):
    """Hadamard gate."""
    
    def matrix(self) -> np.ndarray:
        return np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        
    def apply(self, circuit, qubit: int):
        return circuit.h(qubit)


class XGate(QuantumGate):
    """Pauli-X gate."""
    
    def matrix(self) -> np.ndarray:
        return np.array([[0, 1], [1, 0]])
        
    def apply(self, circuit, qubit: int):
        return circuit.x(qubit)


class YGate(QuantumGate):
    """Pauli-Y gate."""
    
    def matrix(self) -> np.ndarray:
        return np.array([[0, -1j], [1j, 0]])
        
    def apply(self, circuit, qubit: int):
        return circuit.y(qubit)


class ZGate(QuantumGate):
    """Pauli-Z gate."""
    
    def matrix(self) -> np.ndarray:
        return np.array([[1, 0], [0, -1]])
        
    def apply(self, circuit, qubit: int):
        return circuit.z(qubit)


class CXGate(QuantumGate):
    """CNOT gate."""
    
    def matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        
    def apply(self, circuit, control: int, target: int):
        return circuit.cx(control, target)


class CZGate(QuantumGate):
    """CZ gate."""
    
    def matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ])
        
    def apply(self, circuit, control: int, target: int):
        return circuit.cz(control, target)


class SwapGate(QuantumGate):
    """SWAP gate."""
    
    def matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
        
    def apply(self, circuit, qubit1: int, qubit2: int):
        circuit.cx(qubit1, qubit2)
        circuit.cx(qubit2, qubit1)
        circuit.cx(qubit1, qubit2)
        return circuit


class TGate(QuantumGate):
    """T gate."""
    
    def matrix(self) -> np.ndarray:
        return np.array([
            [1, 0],
            [0, np.exp(1j * np.pi / 4)]
        ])
        
    def apply(self, circuit, qubit: int):
        return circuit.rz(qubit, np.pi / 4)


class SGate(QuantumGate):
    """S gate."""
    
    def matrix(self) -> np.ndarray:
        return np.array([
            [1, 0],
            [0, 1j]
        ])
        
    def apply(self, circuit, qubit: int):
        return circuit.rz(qubit, np.pi / 2)


class RXGate(QuantumGate):
    """RX rotation gate."""
    
    def __init__(self, theta: float):
        self.theta = theta
        
    def matrix(self) -> np.ndarray:
        cos = np.cos(self.theta / 2)
        sin = np.sin(self.theta / 2)
        return np.array([
            [cos, -1j * sin],
            [-1j * sin, cos]
        ])
        
    def apply(self, circuit, qubit: int):
        return circuit.rx(qubit, self.theta)


class RYGate(QuantumGate):
    """RY rotation gate."""
    
    def __init__(self, theta: float):
        self.theta = theta
        
    def matrix(self) -> np.ndarray:
        cos = np.cos(self.theta / 2)
        sin = np.sin(self.theta / 2)
        return np.array([
            [cos, -sin],
            [sin, cos]
        ])
        
    def apply(self, circuit, qubit: int):
        return circuit.ry(qubit, self.theta)


class RZGate(QuantumGate):
    """RZ rotation gate."""
    
    def __init__(self, theta: float):
        self.theta = theta
        
    def matrix(self) -> np.ndarray:
        return np.array([
            [np.exp(-1j * self.theta / 2), 0],
            [0, np.exp(1j * self.theta / 2)]
        ])
        
    def apply(self, circuit, qubit: int):
        return circuit.rz(qubit, self.theta)


class U1Gate(QuantumGate):
    """U1 gate (phase gate)."""
    
    def __init__(self, theta: float):
        self.theta = theta
        
    def matrix(self) -> np.ndarray:
        return np.array([
            [1, 0],
            [0, np.exp(1j * self.theta)]
        ])
        
    def apply(self, circuit, qubit: int):
        return circuit.rz(qubit, self.theta)


class U2Gate(QuantumGate):
    """U2 gate."""
    
    def __init__(self, phi: float, lam: float):
        self.phi = phi
        self.lam = lam
        
    def matrix(self) -> np.ndarray:
        return np.array([
            [1, -np.exp(1j * self.lam)],
            [np.exp(1j * self.phi), np.exp(1j * (self.phi + self.lam))]
        ]) / np.sqrt(2)
        
    def apply(self, circuit, qubit: int):
        circuit.rz(qubit, self.phi)
        circuit.ry(qubit, np.pi / 2)
        circuit.rz(qubit, self.lam)
        return circuit


class U3Gate(QuantumGate):
    """U3 gate (general single-qubit rotation)."""
    
    def __init__(self, theta: float, phi: float, lam: float):
        self.theta = theta
        self.phi = phi
        self.lam = lam
        
    def matrix(self) -> np.ndarray:
        return np.array([
            [np.cos(self.theta/2), -np.exp(1j*self.lam) * np.sin(self.theta/2)],
            [np.exp(1j*self.phi) * np.sin(self.theta/2), np.exp(1j*(self.phi+self.lam)) * np.cos(self.theta/2)]
        ])
        
    def apply(self, circuit, qubit: int):
        circuit.rz(qubit, self.phi)
        circuit.ry(qubit, self.theta)
        circuit.rz(qubit, self.lam)
        return circuit


class MeasurementGate(QuantumGate):
    """Measurement gate."""
    
    def matrix(self) -> np.ndarray:
        return None
        
    def apply(self, circuit, qubit: int, classical_bit: Optional[int] = None):
        return circuit.measure(qubit, classical_bit)
