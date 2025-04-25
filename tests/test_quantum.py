"""
Tests for the quantum computing module.
"""

import unittest
import numpy as np

from neurenix.device import Device, DeviceType
from neurenix.quantum import (
    QuantumCircuit,
    ParameterizedCircuit,
    CircuitTemplate,
    QiskitBackend,
    CirqBackend,
    HGate,
    XGate,
    YGate,
    ZGate,
    CXGate,
    CZGate,
    RXGate,
    RYGate,
    RZGate
)
from neurenix.quantum.utils import (
    state_fidelity,
    density_matrix,
    circuit_to_tensor,
    tensor_to_circuit
)

class TestQuantumCircuit(unittest.TestCase):
    """Test the QuantumCircuit class."""
    
    def test_circuit_creation(self):
        """Test creating a quantum circuit."""
        circuit = QuantumCircuit(2, "test_circuit")
        self.assertEqual(circuit.num_qubits, 2)
        self.assertEqual(circuit.name, "test_circuit")
        self.assertEqual(len(circuit.operations), 0)
        
    def test_gate_operations(self):
        """Test adding gate operations to a circuit."""
        circuit = QuantumCircuit(2)
        
        circuit.h(0).x(1).cx(0, 1).measure(0, 0)
        
        self.assertEqual(len(circuit.operations), 4)
        
    def test_bell_pair(self):
        """Test creating a Bell pair circuit."""
        circuit = CircuitTemplate.bell_pair()
        
        self.assertEqual(circuit.num_qubits, 2)
        self.assertEqual(len(circuit.operations), 2)
        
        counts = circuit.run(shots=1024)
        
        self.assertIn('00', counts)
        self.assertIn('11', counts)
        
        total = counts.get('00', 0) + counts.get('11', 0)
        self.assertGreater(counts.get('00', 0) / total, 0.4)
        self.assertGreater(counts.get('11', 0) / total, 0.4)
        
class TestParameterizedCircuit(unittest.TestCase):
    """Test the ParameterizedCircuit class."""
    
    def test_parameterized_circuit(self):
        """Test creating a parameterized circuit."""
        circuit = ParameterizedCircuit(2, ["theta", "phi"])
        
        circuit.rx_param(0, "theta").ry_param(1, "phi")
        
        self.assertEqual(len(circuit.operations), 2)
        self.assertEqual(len(circuit.parameters), 2)
        
    def test_bind_parameters(self):
        """Test binding parameters to a circuit."""
        circuit = ParameterizedCircuit(2, ["theta", "phi"])
        
        circuit.rx_param(0, "theta").ry_param(1, "phi")
        
        bound_circuit = circuit.bind_parameters({"theta": np.pi, "phi": np.pi/2})
        
        self.assertEqual(len(bound_circuit.operations), 2)
        
class TestQuantumBackends(unittest.TestCase):
    """Test the quantum backends."""
    
    def test_qiskit_backend(self):
        """Test the Qiskit backend."""
        try:
            import qiskit
            
            backend = QiskitBackend()
            circuit = CircuitTemplate.bell_pair()
            
            counts = backend.run(circuit, shots=1024)
            
            self.assertIn('00', counts)
            self.assertIn('11', counts)
            
            total = counts.get('00', 0) + counts.get('11', 0)
            self.assertGreater(counts.get('00', 0) / total, 0.4)
            self.assertGreater(counts.get('11', 0) / total, 0.4)
        except ImportError:
            self.skipTest("Qiskit not installed")
            
    def test_cirq_backend(self):
        """Test the Cirq backend."""
        try:
            import cirq
            
            backend = CirqBackend()
            circuit = CircuitTemplate.bell_pair()
            
            counts = backend.run(circuit, shots=1024)
            
            self.assertIn('00', counts)
            self.assertIn('11', counts)
            
            total = counts.get('00', 0) + counts.get('11', 0)
            self.assertGreater(counts.get('00', 0) / total, 0.4)
            self.assertGreater(counts.get('11', 0) / total, 0.4)
        except ImportError:
            self.skipTest("Cirq not installed")
            
class TestQuantumUtils(unittest.TestCase):
    """Test the quantum utility functions."""
    
    def test_state_fidelity(self):
        """Test calculating state fidelity."""
        state1 = np.array([1.0, 0.0], dtype=np.complex128) / np.sqrt(2)
        state2 = np.array([0.0, 1.0], dtype=np.complex128) / np.sqrt(2)
        
        fidelity = state_fidelity(state1, state2)
        
        self.assertAlmostEqual(fidelity, 0.0)
        
    def test_density_matrix(self):
        """Test calculating density matrix."""
        state = np.array([1.0, 0.0], dtype=np.complex128) / np.sqrt(2)
        
        rho = density_matrix(state)
        
        self.assertEqual(rho.shape, (2, 2))
        self.assertAlmostEqual(rho[0, 0], 0.5)
        
    def test_circuit_to_tensor(self):
        """Test converting a circuit to a tensor."""
        circuit = CircuitTemplate.bell_pair()
        
        tensor = circuit_to_tensor(circuit)
        
        self.assertEqual(tensor.shape, (4,))
        
    def test_tensor_to_circuit(self):
        """Test converting a tensor to a circuit."""
        circuit = CircuitTemplate.bell_pair()
        tensor = circuit_to_tensor(circuit)
        
        
        new_circuit = tensor_to_circuit(tensor)
        
        self.assertEqual(new_circuit.num_qubits, 2)

if __name__ == '__main__':
    unittest.main()
