# Quantum Computing API Documentation

## Overview

The Quantum Computing module provides functionality for quantum machine learning and quantum algorithm implementation within the Neurenix framework. This module enables developers to create, simulate, and execute quantum circuits, implement quantum algorithms, and build hybrid quantum-classical models that leverage both quantum and classical computing resources.

Quantum computing offers potential advantages for certain computational tasks, including optimization problems, simulation of quantum systems, and specific machine learning applications. The Neurenix Quantum Computing module provides a unified interface to various quantum computing backends, including simulators and real quantum hardware, while maintaining the same programming model used throughout the framework.

## Key Concepts

### Quantum Circuits

Quantum circuits are the fundamental building blocks of quantum algorithms:

- **Qubits**: Quantum bits, the basic units of quantum information
- **Quantum Gates**: Operations that manipulate qubits
- **Measurements**: Operations that extract classical information from qubits
- **Circuit Execution**: Running quantum circuits on simulators or quantum hardware

### Quantum Backends

The module supports multiple quantum computing backends:

- **Simulators**: Software that simulates quantum circuits
- **Real Quantum Hardware**: Access to actual quantum computers
- **Backend Management**: Selection and configuration of quantum backends

### Quantum Gates

Quantum gates are the building blocks of quantum circuits:

- **Single-Qubit Gates**: Gates that operate on a single qubit (X, Y, Z, H, etc.)
- **Multi-Qubit Gates**: Gates that operate on multiple qubits (CNOT, SWAP, etc.)
- **Parameterized Gates**: Gates with adjustable parameters for variational algorithms
- **Custom Gates**: User-defined quantum gates

### Quantum Algorithms

The module implements various quantum algorithms:

- **VQE**: Variational Quantum Eigensolver for chemistry and optimization
- **QAOA**: Quantum Approximate Optimization Algorithm
- **Grover's Algorithm**: Quantum search algorithm
- **Shor's Algorithm**: Quantum factoring algorithm
- **Quantum Machine Learning**: Quantum algorithms for machine learning tasks

### Hybrid Quantum-Classical Models

Hybrid models combine quantum and classical components:

- **Variational Circuits**: Quantum circuits with trainable parameters
- **Quantum Layers**: Quantum components that can be integrated into classical neural networks
- **Quantum Feature Maps**: Mappings from classical data to quantum states
- **Quantum Kernels**: Kernel methods that leverage quantum computations

## API Reference

### Quantum Circuit

```python
neurenix.quantum.QuantumCircuit(
    num_qubits: int,
    num_cbits: int = 0,
    name: str = None
)
```

Creates a quantum circuit with the specified number of qubits and classical bits.

**Parameters:**
- `num_qubits`: Number of qubits in the circuit
- `num_cbits`: Number of classical bits in the circuit
- `name`: Optional name for the circuit

**Methods:**
- `h(qubit)`: Apply Hadamard gate to a qubit
- `x(qubit)`: Apply Pauli-X gate to a qubit
- `y(qubit)`: Apply Pauli-Y gate to a qubit
- `z(qubit)`: Apply Pauli-Z gate to a qubit
- `rx(theta, qubit)`: Apply parameterized rotation around X-axis
- `ry(theta, qubit)`: Apply parameterized rotation around Y-axis
- `rz(theta, qubit)`: Apply parameterized rotation around Z-axis
- `cnot(control, target)`: Apply CNOT gate
- `measure(qubit, cbit)`: Measure a qubit and store result in a classical bit
- `barrier()`: Add a barrier to the circuit
- `to_matrix()`: Convert the circuit to a matrix representation
- `draw()`: Draw the circuit

**Example:**
```python
from neurenix.quantum import QuantumCircuit

# Create a quantum circuit with 2 qubits and 2 classical bits
circuit = QuantumCircuit(2, 2)

# Apply gates
circuit.h(0)
circuit.cnot(0, 1)

# Measure qubits
circuit.measure(0, 0)
circuit.measure(1, 1)

# Draw the circuit
circuit.draw()
```

### Quantum Backend

```python
neurenix.quantum.QuantumBackend(
    backend_type: str = "simulator",
    backend_name: str = "default",
    config: Dict[str, Any] = None
)
```

Creates a quantum backend for executing quantum circuits.

**Parameters:**
- `backend_type`: Type of backend ("simulator", "hardware")
- `backend_name`: Name of the specific backend
- `config`: Configuration options for the backend

**Methods:**
- `run(circuit, shots)`: Execute a quantum circuit
- `get_result(job_id)`: Get the result of a quantum job
- `get_status(job_id)`: Get the status of a quantum job
- `get_available_backends()`: Get a list of available backends
- `get_backend_properties()`: Get properties of the current backend

**Example:**
```python
from neurenix.quantum import QuantumCircuit, QuantumBackend

# Create a quantum circuit
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cnot(0, 1)
circuit.measure(0, 0)
circuit.measure(1, 1)

# Create a quantum backend
backend = QuantumBackend(backend_type="simulator")

# Run the circuit
result = backend.run(circuit, shots=1024)

# Print the result
print(f"Result: {result.counts}")
```

### Quantum Gate

```python
neurenix.quantum.QuantumGate(
    name: str,
    matrix: Union[np.ndarray, List[List[complex]]],
    qubits: List[int],
    parameters: List[float] = None
)
```

Creates a custom quantum gate.

**Parameters:**
- `name`: Name of the gate
- `matrix`: Matrix representation of the gate
- `qubits`: Qubits the gate acts on
- `parameters`: Optional parameters for parameterized gates

**Methods:**
- `apply(circuit)`: Apply the gate to a circuit
- `adjoint()`: Get the adjoint (conjugate transpose) of the gate
- `tensor(other)`: Tensor product with another gate
- `to_matrix()`: Get the matrix representation of the gate

**Example:**
```python
import numpy as np
from neurenix.quantum import QuantumGate, QuantumCircuit

# Create a custom gate (Hadamard)
h_matrix = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
h_gate = QuantumGate("H_custom", h_matrix, [0])

# Create a quantum circuit
circuit = QuantumCircuit(1, 1)

# Apply the custom gate
h_gate.apply(circuit)

# Measure the qubit
circuit.measure(0, 0)

# Draw the circuit
circuit.draw()
```

### Quantum Algorithm

```python
neurenix.quantum.VQE(
    ansatz: QuantumCircuit,
    observable: Union[np.ndarray, List[List[complex]]],
    optimizer: str = "COBYLA",
    initial_params: List[float] = None,
    backend: QuantumBackend = None
)
```

Creates a Variational Quantum Eigensolver (VQE) algorithm.

**Parameters:**
- `ansatz`: Parameterized quantum circuit
- `observable`: Observable to minimize
- `optimizer`: Classical optimization algorithm
- `initial_params`: Initial parameters for the ansatz
- `backend`: Quantum backend to use

**Methods:**
- `run(max_iterations)`: Run the VQE algorithm
- `get_optimal_parameters()`: Get the optimal parameters
- `get_optimal_value()`: Get the optimal value of the observable
- `get_optimal_circuit()`: Get the optimal circuit

**Example:**
```python
import numpy as np
from neurenix.quantum import QuantumCircuit, QuantumBackend, VQE

# Create a parameterized circuit (ansatz)
ansatz = QuantumCircuit(2)
ansatz.rx("theta1", 0)
ansatz.ry("theta2", 1)
ansatz.cnot(0, 1)

# Define an observable (Pauli-Z tensor product)
observable = np.kron(np.array([[1, 0], [0, -1]]), np.array([[1, 0], [0, -1]]))

# Create a VQE instance
vqe = VQE(
    ansatz=ansatz,
    observable=observable,
    optimizer="COBYLA",
    initial_params=[0.0, 0.0],
    backend=QuantumBackend()
)

# Run the VQE algorithm
result = vqe.run(max_iterations=100)

# Print the results
print(f"Optimal parameters: {vqe.get_optimal_parameters()}")
print(f"Optimal value: {vqe.get_optimal_value()}")
```

```python
neurenix.quantum.QAOA(
    problem: Dict[str, Any],
    p: int = 1,
    optimizer: str = "COBYLA",
    initial_params: List[float] = None,
    backend: QuantumBackend = None
)
```

Creates a Quantum Approximate Optimization Algorithm (QAOA) instance.

**Parameters:**
- `problem`: Problem definition (e.g., MaxCut problem)
- `p`: Number of QAOA layers
- `optimizer`: Classical optimization algorithm
- `initial_params`: Initial parameters for the algorithm
- `backend`: Quantum backend to use

**Methods:**
- `run(max_iterations)`: Run the QAOA algorithm
- `get_optimal_parameters()`: Get the optimal parameters
- `get_optimal_value()`: Get the optimal value of the cost function
- `get_optimal_solution()`: Get the optimal solution

```python
neurenix.quantum.GroverSearch(
    oracle: QuantumCircuit,
    num_qubits: int,
    backend: QuantumBackend = None
)
```

Creates a Grover's search algorithm instance.

**Parameters:**
- `oracle`: Oracle circuit that marks the solution
- `num_qubits`: Number of qubits in the search space
- `backend`: Quantum backend to use

**Methods:**
- `run(shots)`: Run the Grover's search algorithm
- `get_result()`: Get the search result
- `get_probability_distribution()`: Get the probability distribution of outcomes

```python
neurenix.quantum.ShorFactoring(
    number: int,
    backend: QuantumBackend = None
)
```

Creates a Shor's factoring algorithm instance.

**Parameters:**
- `number`: Number to factor
- `backend`: Quantum backend to use

**Methods:**
- `run()`: Run the Shor's factoring algorithm
- `get_factors()`: Get the factors of the number
- `get_circuit()`: Get the quantum circuit used for factoring

### Quantum Machine Learning

```python
neurenix.quantum.QuantumLayer(
    circuit: QuantumCircuit,
    input_dim: int,
    output_dim: int,
    input_encoding: str = "angle",
    output_decoding: str = "expectation"
)
```

Creates a quantum layer that can be integrated into classical neural networks.

**Parameters:**
- `circuit`: Parameterized quantum circuit
- `input_dim`: Dimension of the input data
- `output_dim`: Dimension of the output data
- `input_encoding`: Method for encoding classical data into quantum states
- `output_decoding`: Method for decoding quantum states into classical data

**Methods:**
- `forward(x)`: Process input through the quantum layer
- `parameters()`: Get the trainable parameters of the layer
- `to_circuit(x)`: Convert input to a quantum circuit

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.quantum import QuantumCircuit, QuantumLayer

# Create a parameterized quantum circuit
circuit = QuantumCircuit(4)
for i in range(4):
    circuit.rx("theta" + str(i), i)
    circuit.ry("phi" + str(i), i)
for i in range(3):
    circuit.cnot(i, i+1)
for i in range(4):
    circuit.rz("gamma" + str(i), i)

# Create a quantum layer
quantum_layer = QuantumLayer(
    circuit=circuit,
    input_dim=8,
    output_dim=4,
    input_encoding="angle",
    output_decoding="expectation"
)

# Create a hybrid quantum-classical model
model = Sequential(
    Linear(10, 8),
    ReLU(),
    quantum_layer,
    Linear(4, 2)
)

# Process input
input_tensor = nx.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
output = model(input_tensor)
```

```python
neurenix.quantum.QuantumKernel(
    feature_map: QuantumCircuit,
    backend: QuantumBackend = None
)
```

Creates a quantum kernel for kernel-based machine learning methods.

**Parameters:**
- `feature_map`: Quantum circuit for mapping classical data to quantum states
- `backend`: Quantum backend to use

**Methods:**
- `compute(x1, x2)`: Compute the kernel matrix between two sets of data points
- `evaluate(x1, x2)`: Evaluate the kernel function for two data points
- `fit(X, y)`: Fit a kernel-based model to data
- `predict(X)`: Make predictions using the fitted model

**Example:**
```python
import numpy as np
from neurenix.quantum import QuantumCircuit, QuantumKernel

# Create a feature map circuit
feature_map = QuantumCircuit(2)
for i in range(2):
    feature_map.h(i)
    feature_map.rz("x" + str(i), i)
feature_map.cnot(0, 1)
for i in range(2):
    feature_map.rz("x" + str(i+2), i)
    feature_map.h(i)

# Create a quantum kernel
kernel = QuantumKernel(feature_map=feature_map)

# Generate some data
X_train = np.random.rand(10, 4)
y_train = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])
X_test = np.random.rand(5, 4)

# Fit the kernel-based model
kernel.fit(X_train, y_train)

# Make predictions
predictions = kernel.predict(X_test)
```

### Quantum Simulators

```python
neurenix.quantum.StateVectorSimulator(
    max_qubits: int = 24,
    gpu_enabled: bool = False
)
```

Creates a state vector simulator for quantum circuits.

**Parameters:**
- `max_qubits`: Maximum number of qubits the simulator can handle
- `gpu_enabled`: Whether to use GPU acceleration

**Methods:**
- `run(circuit)`: Run a quantum circuit
- `get_statevector(circuit)`: Get the state vector of a circuit
- `get_probabilities(circuit)`: Get the measurement probabilities
- `get_expectation(circuit, observable)`: Get the expectation value of an observable

**Example:**
```python
from neurenix.quantum import QuantumCircuit, StateVectorSimulator

# Create a quantum circuit
circuit = QuantumCircuit(3)
circuit.h(0)
circuit.cnot(0, 1)
circuit.cnot(1, 2)

# Create a simulator
simulator = StateVectorSimulator(max_qubits=10)

# Get the state vector
statevector = simulator.get_statevector(circuit)
print(f"State vector: {statevector}")

# Get the probabilities
probabilities = simulator.get_probabilities(circuit)
print(f"Probabilities: {probabilities}")
```

```python
neurenix.quantum.DensityMatrixSimulator(
    max_qubits: int = 12,
    gpu_enabled: bool = False
)
```

Creates a density matrix simulator for quantum circuits with noise.

**Parameters:**
- `max_qubits`: Maximum number of qubits the simulator can handle
- `gpu_enabled`: Whether to use GPU acceleration

**Methods:**
- `run(circuit, noise_model)`: Run a quantum circuit with noise
- `get_density_matrix(circuit, noise_model)`: Get the density matrix of a circuit
- `get_probabilities(circuit, noise_model)`: Get the measurement probabilities
- `get_expectation(circuit, observable, noise_model)`: Get the expectation value of an observable

### Quantum Integration

```python
neurenix.quantum.QiskitBackend(
    backend_name: str = "qasm_simulator",
    provider: str = "Aer",
    config: Dict[str, Any] = None
)
```

Creates a backend that interfaces with Qiskit.

**Parameters:**
- `backend_name`: Name of the Qiskit backend
- `provider`: Provider of the backend ("Aer", "IBMQ", etc.)
- `config`: Configuration options for the backend

**Methods:**
- `run(circuit, shots)`: Execute a quantum circuit
- `from_qiskit_circuit(qiskit_circuit)`: Convert a Qiskit circuit to a Neurenix circuit
- `to_qiskit_circuit(circuit)`: Convert a Neurenix circuit to a Qiskit circuit

**Example:**
```python
from neurenix.quantum import QuantumCircuit, QiskitBackend

# Create a quantum circuit
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cnot(0, 1)
circuit.measure(0, 0)
circuit.measure(1, 1)

# Create a Qiskit backend
backend = QiskitBackend(backend_name="qasm_simulator")

# Run the circuit
result = backend.run(circuit, shots=1024)

# Print the result
print(f"Result: {result.counts}")
```

```python
neurenix.quantum.CirqBackend(
    backend_name: str = "simulator",
    config: Dict[str, Any] = None
)
```

Creates a backend that interfaces with Cirq.

**Parameters:**
- `backend_name`: Name of the Cirq backend
- `config`: Configuration options for the backend

**Methods:**
- `run(circuit, shots)`: Execute a quantum circuit
- `from_cirq_circuit(cirq_circuit)`: Convert a Cirq circuit to a Neurenix circuit
- `to_cirq_circuit(circuit)`: Convert a Neurenix circuit to a Cirq circuit

## Framework Comparison

### Neurenix vs. TensorFlow Quantum

| Feature | Neurenix | TensorFlow Quantum |
|---------|----------|-------------------|
| **Integration with Classical ML** | Seamless integration with Neurenix | Seamless integration with TensorFlow |
| **Supported Backends** | Multiple (Qiskit, Cirq, custom) | Primarily Cirq |
| **Quantum Algorithms** | Comprehensive set (VQE, QAOA, Grover, Shor) | Focus on variational algorithms |
| **API Consistency** | Consistent with rest of Neurenix | TensorFlow-specific API |
| **Hardware Support** | Multiple vendors | Limited to Google hardware |
| **Hybrid Models** | First-class support | First-class support |

Neurenix provides a more backend-agnostic approach compared to TensorFlow Quantum, which is primarily built on Cirq and focused on Google's quantum hardware. Neurenix's quantum module offers a wider range of quantum algorithms and better integration with multiple quantum hardware providers, while maintaining the same programming model used throughout the framework.

### Neurenix vs. PyTorch Quantum

| Feature | Neurenix | PyTorch Quantum |
|---------|----------|----------------|
| **Integration with Classical ML** | Seamless integration with Neurenix | Seamless integration with PyTorch |
| **Supported Backends** | Multiple (Qiskit, Cirq, custom) | Limited backends |
| **Quantum Algorithms** | Comprehensive set (VQE, QAOA, Grover, Shor) | Focus on variational algorithms |
| **API Consistency** | Consistent with rest of Neurenix | PyTorch-specific API |
| **Hardware Support** | Multiple vendors | Limited hardware support |
| **Hybrid Models** | First-class support | First-class support |

While PyTorch Quantum provides good integration with PyTorch for hybrid quantum-classical models, Neurenix offers a more comprehensive set of quantum algorithms and better support for multiple quantum hardware providers. Neurenix's quantum module is designed to work seamlessly with the rest of the framework, providing a consistent API for both quantum and classical components.

### Neurenix vs. Qiskit

| Feature | Neurenix | Qiskit |
|---------|----------|--------|
| **Integration with Classical ML** | Seamless integration with Neurenix | Requires additional libraries |
| **Supported Backends** | Multiple (Qiskit, Cirq, custom) | IBM Quantum and Aer simulators |
| **Quantum Algorithms** | Comprehensive set (VQE, QAOA, Grover, Shor) | Comprehensive set |
| **API Consistency** | Consistent with rest of Neurenix | Quantum-specific API |
| **Hardware Support** | Multiple vendors | Primarily IBM hardware |
| **Hybrid Models** | First-class support | Limited support |

Qiskit is a dedicated quantum computing framework with comprehensive support for quantum algorithms and IBM's quantum hardware. Neurenix's quantum module provides better integration with classical machine learning and supports multiple quantum hardware providers, making it more suitable for hybrid quantum-classical applications. Neurenix also offers a more consistent API across both quantum and classical components.

## Best Practices

### Circuit Design

Design efficient quantum circuits:

```python
from neurenix.quantum import QuantumCircuit

# Inefficient circuit
inefficient = QuantumCircuit(2)
inefficient.h(0)
inefficient.h(0)  # Applying H twice is equivalent to identity
inefficient.x(1)
inefficient.z(1)
inefficient.x(1)  # This sequence can be simplified

# Efficient circuit
efficient = QuantumCircuit(2)
efficient.h(0)
# H*H = I, so we omit the second H
efficient.y(1)  # X*Z*X = Y
```

### Backend Selection

Choose appropriate backends based on your needs:

```python
from neurenix.quantum import QuantumBackend

# For rapid prototyping, use a fast simulator
fast_backend = QuantumBackend(backend_type="simulator", backend_name="statevector")

# For testing with noise, use a noise simulator
noise_backend = QuantumBackend(
    backend_type="simulator",
    backend_name="qasm_simulator",
    config={"noise_model": noise_model}
)

# For final results, use real hardware
hardware_backend = QuantumBackend(backend_type="hardware", backend_name="ibmq_manila")
```

### Parameterized Circuits

Use parameterized circuits for variational algorithms:

```python
from neurenix.quantum import QuantumCircuit

# Create a parameterized circuit
circuit = QuantumCircuit(2)

# Add parameterized gates
circuit.rx("theta1", 0)
circuit.ry("theta2", 1)
circuit.rz("theta3", 0)
circuit.rz("theta4", 1)
circuit.cnot(0, 1)

# Bind parameters for execution
bound_circuit = circuit.bind_parameters({"theta1": 0.1, "theta2": 0.2, "theta3": 0.3, "theta4": 0.4})
```

### Error Mitigation

Apply error mitigation techniques:

```python
from neurenix.quantum import QuantumCircuit, QuantumBackend, error_mitigation

# Create a circuit
circuit = QuantumCircuit(2, 2)
circuit.h(0)
circuit.cnot(0, 1)
circuit.measure(0, 0)
circuit.measure(1, 1)

# Create a backend with noise
noisy_backend = QuantumBackend(
    backend_type="simulator",
    backend_name="qasm_simulator",
    config={"noise_model": noise_model}
)

# Apply error mitigation
mitigated_result = error_mitigation.readout_mitigation(
    circuit=circuit,
    backend=noisy_backend,
    shots=1024
)

print(f"Mitigated result: {mitigated_result.counts}")
```

## Tutorials

### Quantum Teleportation

```python
from neurenix.quantum import QuantumCircuit, QuantumBackend

# Create a quantum teleportation circuit
def create_teleportation_circuit():
    # Create a circuit with 3 qubits and 2 classical bits
    circuit = QuantumCircuit(3, 2)
    
    # Prepare the state to teleport (qubit 0)
    circuit.rx(0.5, 0)  # Arbitrary state
    
    # Create entanglement between qubits 1 and 2
    circuit.h(1)
    circuit.cnot(1, 2)
    
    # Bell measurement on qubits 0 and 1
    circuit.cnot(0, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.measure(1, 1)
    
    # Apply corrections to qubit 2 based on measurements
    circuit.x(2).controlled_by(1)
    circuit.z(2).controlled_by(0)
    
    return circuit

# Create the teleportation circuit
teleportation_circuit = create_teleportation_circuit()

# Draw the circuit
teleportation_circuit.draw()

# Create a quantum backend
backend = QuantumBackend(backend_type="simulator")

# Run the circuit
result = backend.run(teleportation_circuit, shots=1024)

# Print the result
print(f"Result: {result.counts}")
```

### Variational Quantum Eigensolver (VQE)

```python
import numpy as np
from neurenix.quantum import QuantumCircuit, QuantumBackend, VQE

# Define a Hamiltonian (e.g., for H2 molecule)
hamiltonian = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.5, 0.5, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

# Create a parameterized circuit (ansatz)
def create_ansatz():
    circuit = QuantumCircuit(2)
    
    # Initial state
    circuit.ry("theta1", 0)
    circuit.ry("theta2", 1)
    
    # Entanglement
    circuit.cnot(0, 1)
    
    # More rotations
    circuit.ry("theta3", 0)
    circuit.ry("theta4", 1)
    
    return circuit

# Create the ansatz
ansatz = create_ansatz()

# Create a quantum backend
backend = QuantumBackend(backend_type="simulator")

# Create a VQE instance
vqe = VQE(
    ansatz=ansatz,
    observable=hamiltonian,
    optimizer="COBYLA",
    initial_params=[0.0, 0.0, 0.0, 0.0],
    backend=backend
)

# Run the VQE algorithm
result = vqe.run(max_iterations=100)

# Print the results
print(f"Optimal parameters: {vqe.get_optimal_parameters()}")
print(f"Ground state energy: {vqe.get_optimal_value()}")

# Get the optimal circuit
optimal_circuit = vqe.get_optimal_circuit()

# Draw the optimal circuit
optimal_circuit.draw()
```

### Quantum Machine Learning

```python
import numpy as np
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.optim import Adam
from neurenix.quantum import QuantumCircuit, QuantumLayer

# Generate some data
np.random.seed(42)
X_train = np.random.rand(100, 8)
y_train = np.sin(X_train.sum(axis=1)) + 0.1 * np.random.randn(100)

# Convert to tensors
X_tensor = nx.Tensor(X_train)
y_tensor = nx.Tensor(y_train).reshape(-1, 1)

# Create a parameterized quantum circuit
def create_quantum_circuit():
    circuit = QuantumCircuit(4)
    
    # Encode classical data
    for i in range(4):
        circuit.rx("x" + str(i), i)
        circuit.rz("z" + str(i), i)
    
    # Entanglement
    for i in range(3):
        circuit.cnot(i, i+1)
    
    # More rotations
    for i in range(4):
        circuit.ry("theta" + str(i), i)
    
    return circuit

# Create a quantum layer
quantum_layer = QuantumLayer(
    circuit=create_quantum_circuit(),
    input_dim=8,
    output_dim=4,
    input_encoding="angle",
    output_decoding="expectation"
)

# Create a hybrid quantum-classical model
model = Sequential(
    Linear(8, 8),
    ReLU(),
    quantum_layer,
    Linear(4, 1)
)

# Define loss function and optimizer
loss_fn = nx.nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
batch_size = 10
num_batches = len(X_train) // batch_size

for epoch in range(num_epochs):
    total_loss = 0
    
    # Shuffle data
    indices = np.random.permutation(len(X_train))
    X_shuffled = X_tensor[indices]
    y_shuffled = y_tensor[indices]
    
    for batch in range(num_batches):
        # Get batch
        start = batch * batch_size
        end = start + batch_size
        X_batch = X_shuffled[start:end]
        y_batch = y_shuffled[start:end]
        
        # Forward pass
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/num_batches:.4f}")

# Evaluate the model
with nx.no_grad():
    y_pred = model(X_tensor)
    mse = ((y_pred - y_tensor) ** 2).mean().item()
    print(f"Final MSE: {mse:.4f}")
```

This documentation provides a comprehensive overview of the Quantum Computing module in Neurenix, including key concepts, API reference, framework comparisons, best practices, and tutorials for quantum computing and quantum machine learning.
