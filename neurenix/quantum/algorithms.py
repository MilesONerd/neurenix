"""
Implementation of quantum algorithms for Neurenix.
"""

from typing import List, Dict, Optional, Union, Any, Tuple, Callable
import numpy as np

from neurenix.device import Device, DeviceType
from neurenix.tensor import Tensor
from neurenix.quantum.circuit import QuantumCircuit, ParameterizedCircuit

class QuantumOptimizer:
    """
    Simple optimizer for quantum algorithms.
    """
    
    def __init__(self, lr: float = 0.01):
        """
        Initialize the optimizer.
        
        Args:
            lr: Learning rate
        """
        self.lr = lr
        
    def minimize(self, objective_fn: Callable, initial_params: np.ndarray, max_iterations: int = 100, tolerance: float = 1e-5) -> np.ndarray:
        """
        Minimize an objective function.
        
        Args:
            objective_fn: Objective function to minimize
            initial_params: Initial parameters
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Optimized parameters
        """
        params = initial_params.copy()
        
        for i in range(max_iterations):
            value = objective_fn(params)
            
            grad = np.zeros_like(params)
            for j in range(len(params)):
                params_plus = params.copy()
                params_plus[j] += tolerance
                value_plus = objective_fn(params_plus)
                grad[j] = (value_plus - value) / tolerance
                
            # Update parameters
            params -= self.lr * grad
            
            if np.max(np.abs(grad)) < tolerance:
                break
                
        return params

class QuantumAlgorithm:
    """
    Base class for quantum algorithms.
    """
    
    def __init__(self, device: Optional[Device] = None):
        """
        Initialize a quantum algorithm.
        
        Args:
            device: Device to run the algorithm on
        """
        self.device = device or Device(DeviceType.QUANTUM)


class VQE(QuantumAlgorithm):
    """
    Variational Quantum Eigensolver algorithm.
    """
    
    def __init__(self, 
                 ansatz: ParameterizedCircuit,
                 observable: Union[np.ndarray, List[Tuple[str, float]]],
                 optimizer: Optional[Any] = None,
                 device: Optional[Device] = None):
        """
        Initialize the VQE algorithm.
        
        Args:
            ansatz: Parameterized quantum circuit for the variational form
            observable: Hamiltonian as matrix or list of Pauli strings with coefficients
            optimizer: Classical optimizer (defaults to gradient descent)
            device: Device to run the algorithm on
        """
        super().__init__(device)
        self.ansatz = ansatz
        self.observable = observable
        
        if optimizer is None:
            self.optimizer = QuantumOptimizer(lr=0.01)
        else:
            self.optimizer = optimizer
            
        self.parameters = {param: 0.0 for param in ansatz.parameters}
        self.optimal_parameters = None
        self.optimal_value = None
        self.history = []
        
    def compute_expectation(self, parameters: Dict[str, float]) -> float:
        """
        Compute the expectation value of the observable with the given parameters.
        
        Args:
            parameters: Dictionary of parameter values
            
        Returns:
            Expectation value
        """
        circuit = self.ansatz.bind_parameters(parameters)
        
        if isinstance(self.observable, np.ndarray):
            state_vector = circuit.to_matrix() @ np.array([1] + [0] * (2**circuit.num_qubits - 1))
            expectation = np.real(np.vdot(state_vector, self.observable @ state_vector))
            return expectation
            
        expectation = 0.0
        for pauli_string, coefficient in self.observable:
            measurement_circuit = self._prepare_measurement_circuit(circuit, pauli_string)
            counts = measurement_circuit.run(shots=1024)
            
            expectation_term = 0.0
            for bitstring, count in counts.items():
                parity = self._compute_parity(bitstring, pauli_string)
                expectation_term += parity * count / 1024
                
            expectation += coefficient * expectation_term
            
        return expectation
        
    def _prepare_measurement_circuit(self, circuit, pauli_string: str) -> QuantumCircuit:
        """Prepare a circuit that measures in the appropriate basis for a Pauli string."""
        measurement_circuit = QuantumCircuit(circuit.num_qubits, device=self.device)
        
        for op in circuit.operations:
            if op[0] == 'h':
                measurement_circuit.h(op[1])
            elif op[0] == 'x':
                measurement_circuit.x(op[1])
            elif op[0] == 'y':
                measurement_circuit.y(op[1])
            elif op[0] == 'z':
                measurement_circuit.z(op[1])
            elif op[0] == 'cx':
                measurement_circuit.cx(op[1], op[2])
            elif op[0] == 'cz':
                measurement_circuit.cz(op[1], op[2])
            elif op[0] == 'rx':
                measurement_circuit.rx(op[1], op[2])
            elif op[0] == 'ry':
                measurement_circuit.ry(op[1], op[2])
            elif op[0] == 'rz':
                measurement_circuit.rz(op[1], op[2])
        
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                measurement_circuit.h(i)
            elif pauli == 'Y':
                measurement_circuit.rx(i, -np.pi/2)
        
        for i in range(circuit.num_qubits):
            measurement_circuit.measure(i, i)
            
        return measurement_circuit
        
    def _compute_parity(self, bitstring: str, pauli_string: str) -> int:
        """Compute the parity of a measurement result for a Pauli string."""
        parity = 0
        for i, pauli in enumerate(pauli_string):
            if pauli in ['X', 'Y', 'Z'] and bitstring[i] == '1':
                parity += 1
                
        return 1 if parity % 2 == 0 else -1
        
    def optimize(self, max_iterations: int = 100, tolerance: float = 1e-5) -> Dict[str, float]:
        """
        Run the VQE optimization.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Optimal parameters
        """
        params = np.array(list(self.parameters.values()))
        param_keys = list(self.parameters.keys())
        
        def objective(params):
            param_dict = {param_keys[i]: params[i] for i in range(len(params))}
            energy = self.compute_expectation(param_dict)
            self.history.append(energy)
            return energy
            
        result = self.optimizer.minimize(objective, params, max_iterations, tolerance)
        
        self.optimal_parameters = {param_keys[i]: result[i] for i in range(len(result))}
        self.optimal_value = objective(result)
        
        return self.optimal_parameters
        
    def get_optimal_value(self) -> float:
        """Get the optimal value found by the algorithm."""
        if self.optimal_value is None:
            raise ValueError("Optimization has not been run yet.")
        return self.optimal_value
        
    def get_optimal_state(self) -> np.ndarray:
        """Get the optimal quantum state found by the algorithm."""
        if self.optimal_parameters is None:
            raise ValueError("Optimization has not been run yet.")
            
        circuit = self.ansatz.bind_parameters(self.optimal_parameters)
        state_vector = circuit.to_matrix() @ np.array([1] + [0] * (2**circuit.num_qubits - 1))
        
        return state_vector


class QAOA(QuantumAlgorithm):
    """
    Quantum Approximate Optimization Algorithm.
    """
    
    def __init__(self, 
                 cost_hamiltonian: Union[np.ndarray, List[Tuple[str, float]]],
                 p: int = 1,
                 optimizer: Optional[Any] = None,
                 device: Optional[Device] = None):
        """
        Initialize the QAOA algorithm.
        
        Args:
            cost_hamiltonian: Problem Hamiltonian
            p: Number of QAOA layers
            optimizer: Classical optimizer
            device: Device to run the algorithm on
        """
        super().__init__(device)
        self.cost_hamiltonian = cost_hamiltonian
        self.p = p
        
        if optimizer is None:
            self.optimizer = QuantumOptimizer(lr=0.01)
        else:
            self.optimizer = optimizer
            
        if isinstance(cost_hamiltonian, np.ndarray):
            n = int(np.log2(cost_hamiltonian.shape[0]))
        else:
            n = max(len(pauli_string) for pauli_string, _ in cost_hamiltonian)
            
        self.num_qubits = n
        
        self.ansatz = self._create_qaoa_circuit(n, p)
        
        self.parameters = {param: 0.0 for param in self.ansatz.parameters}
        self.optimal_parameters = None
        self.optimal_value = None
        self.history = []
        
    def _create_qaoa_circuit(self, n: int, p: int) -> ParameterizedCircuit:
        """Create a QAOA circuit with p layers."""
        circuit = ParameterizedCircuit(n, name=f"qaoa_{n}q_{p}p", device=self.device)
        
        for i in range(n):
            circuit.h(i)
            
        for layer in range(p):
            gamma_param = f"gamma_{layer}"
            
            if isinstance(self.cost_hamiltonian, np.ndarray):
                for i in range(n):
                    circuit.rz_param(i, gamma_param)
                for i in range(n-1):
                    circuit.cx(i, i+1)
                    circuit.rz_param(i+1, gamma_param)
                    circuit.cx(i, i+1)
            else:
                for pauli_string, coefficient in self.cost_hamiltonian:
                    self._apply_pauli_string_exp(circuit, pauli_string, gamma_param, coefficient)
            
            beta_param = f"beta_{layer}"
            for i in range(n):
                circuit.rx_param(i, beta_param)
                
        for i in range(n):
            circuit.measure(i, i)
            
        return circuit
        
    def _apply_pauli_string_exp(self, circuit, pauli_string: str, param_name: str, coefficient: float):
        """Apply the exponential of a Pauli string operator to the circuit."""
        
        for i, pauli in enumerate(pauli_string):
            if pauli == 'Z':
                circuit.rz_param(i, param_name)
                
        for i, pauli in enumerate(pauli_string):
            if pauli == 'X':
                circuit.h(i)
                circuit.rz_param(i, param_name)
                circuit.h(i)
                
        for i, pauli in enumerate(pauli_string):
            if pauli == 'Y':
                circuit.rx(i, np.pi/2)
                circuit.rz_param(i, param_name)
                circuit.rx(i, -np.pi/2)
                
        
    def optimize(self, max_iterations: int = 100, tolerance: float = 1e-5) -> Dict[str, float]:
        """
        Run the QAOA optimization.
        
        Args:
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Optimal parameters
        """
        params = np.array(list(self.parameters.values()))
        param_keys = list(self.parameters.keys())
        
        def objective(params):
            param_dict = {param_keys[i]: params[i] for i in range(len(params))}
            circuit = self.ansatz.bind_parameters(param_dict)
            
            counts = circuit.run(shots=1024)
            
            expectation = 0.0
            for bitstring, count in counts.items():
                cost = self._compute_cost(bitstring)
                expectation += cost * count / 1024
                
            self.history.append(expectation)
            return expectation
            
        result = self.optimizer.minimize(objective, params, max_iterations, tolerance)
        
        self.optimal_parameters = {param_keys[i]: result[i] for i in range(len(result))}
        self.optimal_value = objective(result)
        
        return self.optimal_parameters
        
    def _compute_cost(self, bitstring: str) -> float:
        """Compute the cost function value for a given bitstring."""
        x = np.array([int(bit) for bit in bitstring])
        
        if isinstance(self.cost_hamiltonian, np.ndarray):
            idx = int(bitstring, 2)
            return self.cost_hamiltonian[idx, idx]
            
        cost = 0.0
        for pauli_string, coefficient in self.cost_hamiltonian:
            term_value = 1.0
            for i, pauli in enumerate(pauli_string):
                if pauli == 'Z':
                    term_value *= 1 - 2 * x[i]  # Convert 0->1, 1->-1
                elif pauli == 'X':
                    term_value = 0  # Simplified
                elif pauli == 'Y':
                    term_value = 0  # Simplified
                    
            cost += coefficient * term_value
            
        return cost


class Grover(QuantumAlgorithm):
    """
    Grover's search algorithm.
    """
    
    def __init__(self, 
                 num_qubits: int,
                 oracle: Union[Callable[[QuantumCircuit, List[int]], None], List[str]],
                 device: Optional[Device] = None):
        """
        Initialize Grover's algorithm.
        
        Args:
            num_qubits: Number of qubits
            oracle: Oracle function or list of marked states
            device: Device to run the algorithm on
        """
        super().__init__(device)
        self.num_qubits = num_qubits
        self.oracle = oracle
        
    def run(self, num_iterations: Optional[int] = None) -> Dict[str, int]:
        """
        Run Grover's algorithm.
        
        Args:
            num_iterations: Number of Grover iterations (if None, uses optimal)
            
        Returns:
            Measurement results
        """
        if num_iterations is None:
            n = self.num_qubits
            num_iterations = int(np.pi/4 * np.sqrt(2**n))
            
        circuit = QuantumCircuit(self.num_qubits + 1, name=f"grover_{self.num_qubits}q", device=self.device)
        
        for i in range(self.num_qubits):
            circuit.h(i)
            
        circuit.x(self.num_qubits)
        circuit.h(self.num_qubits)
        
        for _ in range(num_iterations):
            self._apply_oracle(circuit)
            
            for i in range(self.num_qubits):
                circuit.h(i)
                circuit.x(i)
                
            for i in range(self.num_qubits - 1):
                circuit.cz(i, i+1)
                
            for i in range(self.num_qubits):
                circuit.x(i)
                circuit.h(i)
                
        for i in range(self.num_qubits):
            circuit.measure(i, i)
            
        return circuit.run(shots=1024)
        
    def _apply_oracle(self, circuit: QuantumCircuit):
        """Apply the oracle to the circuit."""
        if callable(self.oracle):
            self.oracle(circuit, list(range(self.num_qubits)))
        else:
            for marked_state in self.oracle:
                for i, bit in enumerate(marked_state):
                    if bit == '0':
                        circuit.x(i)
                        
                for i in range(self.num_qubits):
                    circuit.cz(i, self.num_qubits)
                    
                for i, bit in enumerate(marked_state):
                    if bit == '0':
                        circuit.x(i)


class Shor(QuantumAlgorithm):
    """
    Shor's factoring algorithm.
    """
    
    def __init__(self, 
                 number_to_factor: int,
                 device: Optional[Device] = None):
        """
        Initialize Shor's algorithm.
        
        Args:
            number_to_factor: Number to factor
            device: Device to run the algorithm on
        """
        super().__init__(device)
        self.number_to_factor = number_to_factor
        
        self.n = number_to_factor.bit_length()
        self.num_qubits = 2 * self.n
        
    def run(self) -> Tuple[int, int]:
        """
        Run Shor's algorithm to factor the number.
        
        Returns:
            Tuple of factors (p, q) such that p * q = number_to_factor
        """
        
        import math
        from fractions import Fraction
        
        N = self.number_to_factor
        
        if N % 2 == 0:
            return 2, N // 2
            
        for i in range(2, int(math.log2(N)) + 1):
            root = round(N ** (1/i))
            if root ** i == N:
                return root, root
                
        import random
        a = random.randint(2, N-1)
        
        g = math.gcd(a, N)
        if g > 1:
            return g, N // g
            
        
        r = self._find_period_classically(a, N)
        
        if r % 2 == 1:
            return self.run()
            
        if pow(a, r//2, N) == N - 1:
            return self.run()
            
        p = math.gcd(pow(a, r//2) - 1, N)
        q = math.gcd(pow(a, r//2) + 1, N)
        
        return p, q
        
    def _find_period_classically(self, a: int, N: int) -> int:
        """Find the period of a^r mod N classically."""
        
        values = {}
        for r in range(1, N):
            val = pow(a, r, N)
            if val in values:
                return r - values[val]
            values[val] = r
            
        return 0


class QuantumPhaseEstimation(QuantumAlgorithm):
    """
    Quantum Phase Estimation algorithm.
    """
    
    def __init__(self, 
                 unitary: Union[np.ndarray, Callable[[QuantumCircuit, int, int], None]],
                 num_qubits: int,
                 precision_qubits: int,
                 device: Optional[Device] = None):
        """
        Initialize Quantum Phase Estimation.
        
        Args:
            unitary: Unitary operator or function to apply controlled-U
            num_qubits: Number of qubits for the eigenstate
            precision_qubits: Number of qubits for phase estimation
            device: Device to run the algorithm on
        """
        super().__init__(device)
        self.unitary = unitary
        self.num_qubits = num_qubits
        self.precision_qubits = precision_qubits
        self.total_qubits = num_qubits + precision_qubits
        
    def run(self, initial_state: Optional[np.ndarray] = None) -> float:
        """
        Run Quantum Phase Estimation.
        
        Args:
            initial_state: Initial state (eigenstate of the unitary)
            
        Returns:
            Estimated phase
        """
        circuit = QuantumCircuit(self.total_qubits, name="qpe", device=self.device)
        
        if initial_state is not None:
            pass
        else:
            pass
            
        for i in range(self.precision_qubits):
            circuit.h(i)
            
        for j in range(self.precision_qubits):
            power = 2 ** j
            self._apply_controlled_unitary(circuit, j, power)
            
        self._apply_inverse_qft(circuit)
        
        for i in range(self.precision_qubits):
            circuit.measure(i, i)
            
        counts = circuit.run(shots=1024)
        
        phase = 0.0
        total_shots = 0
        
        for bitstring, count in counts.items():
            precision_bits = bitstring[:self.precision_qubits]
            
            phase_value = int(precision_bits, 2) / (2 ** self.precision_qubits)
            
            phase += phase_value * count
            total_shots += count
            
        return phase / total_shots
        
    def _apply_controlled_unitary(self, circuit: QuantumCircuit, control: int, power: int):
        """Apply the controlled-U^power operation."""
        if callable(self.unitary):
            for _ in range(power):
                for target in range(self.precision_qubits, self.total_qubits):
                    self.unitary(circuit, control, target)
        else:
            pass
            
    def _apply_inverse_qft(self, circuit: QuantumCircuit):
        """Apply the inverse Quantum Fourier Transform."""
        for i in range(self.precision_qubits // 2):
            circuit.cx(i, self.precision_qubits - i - 1)
            circuit.cx(self.precision_qubits - i - 1, i)
            circuit.cx(i, self.precision_qubits - i - 1)
            
        for i in range(self.precision_qubits):
            circuit.h(i)
            
            for j in range(i):
                angle = -np.pi / (2 ** (i - j))
                circuit.cz(j, i)
