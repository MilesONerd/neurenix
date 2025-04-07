"""
Qiskit interface for Neurenix quantum computing module.

This module provides integration with IBM's Qiskit framework,
allowing for quantum circuit creation, execution, and integration
with classical neural networks.
"""

import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any, Callable

from ..core import PhynexusExtension
from ..device import Device

class QiskitBackend:
    """
    Interface to Qiskit quantum computing backends.
    
    This class provides a unified interface to both simulators and
    real quantum hardware through IBM's Qiskit framework.
    """
    
    def __init__(self, 
                 backend_name: str = 'qasm_simulator', 
                 provider: Optional[str] = None,
                 shots: int = 1024,
                 device: Optional[Device] = None):
        """
        Initialize a Qiskit backend.
        
        Args:
            backend_name: Name of the Qiskit backend to use
            provider: Optional provider for the backend (e.g., 'ibm-q')
            shots: Number of shots for quantum circuit execution
            device: Optional Neurenix device for accelerated simulation
        """
        self.backend_name = backend_name
        self.provider_name = provider
        self.shots = shots
        self.device = device
        self._backend = None
        self._provider = None
        
        try:
            import qiskit
            self.qiskit = qiskit
            self._initialize_backend()
        except ImportError:
            self.qiskit = None
            print("Warning: Qiskit not installed. Please install with 'pip install qiskit'")
    
    def _initialize_backend(self):
        """Initialize the Qiskit backend."""
        if self.qiskit is None:
            return
        
        if self.provider_name == 'ibm-q':
            try:
                from qiskit_ibm_provider import IBMProvider
                self._provider = IBMProvider()
                self._backend = self._provider.get_backend(self.backend_name)
            except ImportError:
                print("Warning: qiskit-ibm-provider not installed. Please install with 'pip install qiskit-ibm-provider'")
                self._backend = self.qiskit.Aer.get_backend('qasm_simulator')
        else:
            self._backend = self.qiskit.Aer.get_backend(self.backend_name)
    
    def run_circuit(self, circuit, **kwargs):
        """
        Run a quantum circuit on the backend.
        
        Args:
            circuit: Qiskit circuit to run
            **kwargs: Additional arguments to pass to the backend
            
        Returns:
            Result object from Qiskit
        """
        if self.qiskit is None:
            raise ImportError("Qiskit not installed. Please install with 'pip install qiskit'")
        
        if self._backend is None:
            self._initialize_backend()
        
        shots = kwargs.pop('shots', self.shots)
        
        if self.device and PhynexusExtension.is_available():
            return PhynexusExtension.run_qiskit_circuit(
                circuit, self.backend_name, shots, self.device.device_id
            )
        
        job = self.qiskit.execute(circuit, self._backend, shots=shots, **kwargs)
        return job.result()
    
    def get_counts(self, result, circuit=None):
        """
        Get measurement counts from a result.
        
        Args:
            result: Result object from run_circuit
            circuit: Optional circuit to get counts for (if result contains multiple)
            
        Returns:
            Dictionary of measurement results and counts
        """
        if circuit is None:
            return result.get_counts()
        return result.get_counts(circuit)
    
    def get_statevector(self, result, circuit=None):
        """
        Get the statevector from a result.
        
        Args:
            result: Result object from run_circuit
            circuit: Optional circuit to get statevector for (if result contains multiple)
            
        Returns:
            Numpy array containing the statevector
        """
        if self.backend_name != 'statevector_simulator':
            raise ValueError("Statevector is only available when using the 'statevector_simulator' backend")
        
        if circuit is None:
            return result.get_statevector()
        return result.get_statevector(circuit)


class QiskitQuantumRegister:
    """Wrapper for Qiskit's QuantumRegister."""
    
    def __init__(self, size: int, name: str = 'q'):
        """
        Initialize a quantum register.
        
        Args:
            size: Number of qubits in the register
            name: Name of the register
        """
        self.size = size
        self.name = name
        self._register = None
        
        try:
            import qiskit
            self.qiskit = qiskit
            self._register = qiskit.QuantumRegister(size, name)
        except ImportError:
            self.qiskit = None
    
    @property
    def register(self):
        """Get the underlying Qiskit register."""
        if self.qiskit is None:
            raise ImportError("Qiskit not installed. Please install with 'pip install qiskit'")
        
        if self._register is None:
            self._register = self.qiskit.QuantumRegister(self.size, self.name)
        
        return self._register


class QiskitClassicalRegister:
    """Wrapper for Qiskit's ClassicalRegister."""
    
    def __init__(self, size: int, name: str = 'c'):
        """
        Initialize a classical register.
        
        Args:
            size: Number of bits in the register
            name: Name of the register
        """
        self.size = size
        self.name = name
        self._register = None
        
        try:
            import qiskit
            self.qiskit = qiskit
            self._register = qiskit.ClassicalRegister(size, name)
        except ImportError:
            self.qiskit = None
    
    @property
    def register(self):
        """Get the underlying Qiskit register."""
        if self.qiskit is None:
            raise ImportError("Qiskit not installed. Please install with 'pip install qiskit'")
        
        if self._register is None:
            self._register = self.qiskit.ClassicalRegister(self.size, self.name)
        
        return self._register


class QiskitCircuit:
    """Wrapper for Qiskit's QuantumCircuit."""
    
    def __init__(self, 
                 qregs: Optional[List[QiskitQuantumRegister]] = None,
                 cregs: Optional[List[QiskitClassicalRegister]] = None,
                 name: str = 'circuit'):
        """
        Initialize a quantum circuit.
        
        Args:
            qregs: List of quantum registers
            cregs: List of classical registers
            name: Name of the circuit
        """
        self.name = name
        self.qregs = qregs or []
        self.cregs = cregs or []
        self._circuit = None
        
        try:
            import qiskit
            self.qiskit = qiskit
            self._initialize_circuit()
        except ImportError:
            self.qiskit = None
    
    def _initialize_circuit(self):
        """Initialize the Qiskit circuit."""
        if self.qiskit is None:
            return
        
        qregs = [qreg.register for qreg in self.qregs]
        cregs = [creg.register for creg in self.cregs]
        
        self._circuit = self.qiskit.QuantumCircuit(*qregs, *cregs, name=self.name)
    
    @property
    def circuit(self):
        """Get the underlying Qiskit circuit."""
        if self.qiskit is None:
            raise ImportError("Qiskit not installed. Please install with 'pip install qiskit'")
        
        if self._circuit is None:
            self._initialize_circuit()
        
        return self._circuit
    
    def h(self, qubit):
        """Apply Hadamard gate to qubit."""
        self.circuit.h(qubit)
        return self
    
    def x(self, qubit):
        """Apply Pauli-X gate to qubit."""
        self.circuit.x(qubit)
        return self
    
    def y(self, qubit):
        """Apply Pauli-Y gate to qubit."""
        self.circuit.y(qubit)
        return self
    
    def z(self, qubit):
        """Apply Pauli-Z gate to qubit."""
        self.circuit.z(qubit)
        return self
    
    def cx(self, control, target):
        """Apply CNOT gate with control and target qubits."""
        self.circuit.cx(control, target)
        return self
    
    def cz(self, control, target):
        """Apply CZ gate with control and target qubits."""
        self.circuit.cz(control, target)
        return self
    
    def measure(self, qubit, cbit):
        """Measure qubit and store result in classical bit."""
        self.circuit.measure(qubit, cbit)
        return self
    
    def measure_all(self):
        """Measure all qubits."""
        self.circuit.measure_all()
        return self
    
    def barrier(self):
        """Add a barrier to the circuit."""
        self.circuit.barrier()
        return self
    
    def draw(self, **kwargs):
        """Draw the circuit."""
        return self.circuit.draw(**kwargs)


class QiskitQuantumGate:
    """Factory for Qiskit quantum gates."""
    
    @staticmethod
    def rx(theta):
        """Create a rotation around X-axis gate."""
        def apply(circuit, qubit):
            circuit.circuit.rx(theta, qubit)
            return circuit
        return apply
    
    @staticmethod
    def ry(theta):
        """Create a rotation around Y-axis gate."""
        def apply(circuit, qubit):
            circuit.circuit.ry(theta, qubit)
            return circuit
        return apply
    
    @staticmethod
    def rz(theta):
        """Create a rotation around Z-axis gate."""
        def apply(circuit, qubit):
            circuit.circuit.rz(theta, qubit)
            return circuit
        return apply
    
    @staticmethod
    def u(theta, phi, lam):
        """Create a U gate with parameters theta, phi, and lambda."""
        def apply(circuit, qubit):
            circuit.circuit.u(theta, phi, lam, qubit)
            return circuit
        return apply
    
    @staticmethod
    def crx(theta):
        """Create a controlled rotation around X-axis gate."""
        def apply(circuit, control, target):
            circuit.circuit.crx(theta, control, target)
            return circuit
        return apply
    
    @staticmethod
    def cry(theta):
        """Create a controlled rotation around Y-axis gate."""
        def apply(circuit, control, target):
            circuit.circuit.cry(theta, control, target)
            return circuit
        return apply
    
    @staticmethod
    def crz(theta):
        """Create a controlled rotation around Z-axis gate."""
        def apply(circuit, control, target):
            circuit.circuit.crz(theta, control, target)
            return circuit
        return apply
