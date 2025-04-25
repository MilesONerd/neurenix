/**
 * @file quantum.h
 * @brief Header file for Quantum Computing module in Neurenix C++ backend.
 */

#ifndef PHYNEXUS_QUANTUM_H
#define PHYNEXUS_QUANTUM_H

#include <vector>
#include <complex>
#include <string>
#include <unordered_map>
#include <memory>
#include <functional>
#include <cmath>

namespace phynexus {
namespace quantum {

/**
 * @brief Complex number type for quantum operations.
 */
using Complex = std::complex<double>;

/**
 * @brief Quantum state vector.
 */
using StateVector = std::vector<Complex>;

/**
 * @brief Quantum unitary matrix.
 */
using UnitaryMatrix = std::vector<std::vector<Complex>>;

/**
 * @brief Enum for quantum gate types.
 */
enum class GateType {
    H,      // Hadamard
    X,      // Pauli-X
    Y,      // Pauli-Y
    Z,      // Pauli-Z
    CX,     // CNOT
    CZ,     // Controlled-Z
    SWAP,   // SWAP
    T,      // T gate
    S,      // S gate
    RX,     // RX rotation
    RY,     // RY rotation
    RZ,     // RZ rotation
    U1,     // U1 gate
    U2,     // U2 gate
    U3,     // U3 gate
    MEASURE // Measurement
};

/**
 * @brief Structure representing a quantum gate operation.
 */
struct GateOperation {
    GateType type;
    std::vector<int> qubits;
    std::vector<double> parameters;
};

/**
 * @brief Class representing a quantum circuit.
 */
class QuantumCircuit {
public:
    /**
     * @brief Constructor.
     * @param num_qubits Number of qubits in the circuit.
     * @param name Optional name for the circuit.
     */
    QuantumCircuit(int num_qubits, const std::string& name = "");
    
    /**
     * @brief Add a Hadamard gate.
     * @param qubit Qubit to apply the gate to.
     * @return Reference to this circuit.
     */
    QuantumCircuit& h(int qubit);
    
    /**
     * @brief Add a Pauli-X gate.
     * @param qubit Qubit to apply the gate to.
     * @return Reference to this circuit.
     */
    QuantumCircuit& x(int qubit);
    
    /**
     * @brief Add a Pauli-Y gate.
     * @param qubit Qubit to apply the gate to.
     * @return Reference to this circuit.
     */
    QuantumCircuit& y(int qubit);
    
    /**
     * @brief Add a Pauli-Z gate.
     * @param qubit Qubit to apply the gate to.
     * @return Reference to this circuit.
     */
    QuantumCircuit& z(int qubit);
    
    /**
     * @brief Add a CNOT gate.
     * @param control Control qubit.
     * @param target Target qubit.
     * @return Reference to this circuit.
     */
    QuantumCircuit& cx(int control, int target);
    
    /**
     * @brief Add a Controlled-Z gate.
     * @param control Control qubit.
     * @param target Target qubit.
     * @return Reference to this circuit.
     */
    QuantumCircuit& cz(int control, int target);
    
    /**
     * @brief Add an RX rotation gate.
     * @param qubit Qubit to apply the gate to.
     * @param theta Rotation angle.
     * @return Reference to this circuit.
     */
    QuantumCircuit& rx(int qubit, double theta);
    
    /**
     * @brief Add an RY rotation gate.
     * @param qubit Qubit to apply the gate to.
     * @param theta Rotation angle.
     * @return Reference to this circuit.
     */
    QuantumCircuit& ry(int qubit, double theta);
    
    /**
     * @brief Add an RZ rotation gate.
     * @param qubit Qubit to apply the gate to.
     * @param theta Rotation angle.
     * @return Reference to this circuit.
     */
    QuantumCircuit& rz(int qubit, double theta);
    
    /**
     * @brief Add a measurement operation.
     * @param qubit Qubit to measure.
     * @param classical_bit Classical bit to store the result.
     * @return Reference to this circuit.
     */
    QuantumCircuit& measure(int qubit, int classical_bit);
    
    /**
     * @brief Get the unitary matrix representation of the circuit.
     * @return Unitary matrix.
     */
    UnitaryMatrix to_matrix() const;
    
    /**
     * @brief Run the circuit and return measurement results.
     * @param shots Number of shots to run.
     * @return Map of measurement results and their counts.
     */
    std::unordered_map<std::string, int> run(int shots = 1024) const;
    
    /**
     * @brief Get the number of qubits in the circuit.
     * @return Number of qubits.
     */
    int num_qubits() const { return num_qubits_; }
    
    /**
     * @brief Get the name of the circuit.
     * @return Circuit name.
     */
    const std::string& name() const { return name_; }
    
    /**
     * @brief Get the operations in the circuit.
     * @return Vector of gate operations.
     */
    const std::vector<GateOperation>& operations() const { return operations_; }
    
private:
    int num_qubits_;
    std::string name_;
    std::vector<GateOperation> operations_;
};

/**
 * @brief Class representing a parameterized quantum circuit.
 */
class ParameterizedCircuit : public QuantumCircuit {
public:
    /**
     * @brief Constructor.
     * @param num_qubits Number of qubits in the circuit.
     * @param parameters List of parameter names.
     * @param name Optional name for the circuit.
     */
    ParameterizedCircuit(int num_qubits, const std::vector<std::string>& parameters = {}, const std::string& name = "");
    
    /**
     * @brief Add a parameterized RX rotation gate.
     * @param qubit Qubit to apply the gate to.
     * @param param_name Parameter name.
     * @return Reference to this circuit.
     */
    ParameterizedCircuit& rx_param(int qubit, const std::string& param_name);
    
    /**
     * @brief Add a parameterized RY rotation gate.
     * @param qubit Qubit to apply the gate to.
     * @param param_name Parameter name.
     * @return Reference to this circuit.
     */
    ParameterizedCircuit& ry_param(int qubit, const std::string& param_name);
    
    /**
     * @brief Add a parameterized RZ rotation gate.
     * @param qubit Qubit to apply the gate to.
     * @param param_name Parameter name.
     * @return Reference to this circuit.
     */
    ParameterizedCircuit& rz_param(int qubit, const std::string& param_name);
    
    /**
     * @brief Bind parameters to values and return a concrete circuit.
     * @param parameter_values Map of parameter names to values.
     * @return Concrete quantum circuit.
     */
    QuantumCircuit bind_parameters(const std::unordered_map<std::string, double>& parameter_values) const;
    
    /**
     * @brief Get the parameters of the circuit.
     * @return Vector of parameter names.
     */
    const std::vector<std::string>& parameters() const { return parameters_; }
    
    /**
     * @brief Get the parameter values of the circuit.
     * @return Map of parameter names to values.
     */
    const std::unordered_map<std::string, double>& parameter_values() const { return parameter_values_; }
    
    /**
     * @brief Set a parameter value.
     * @param param_name Parameter name.
     * @param value Parameter value.
     */
    void set_parameter(const std::string& param_name, double value);
    
private:
    std::vector<std::string> parameters_;
    std::unordered_map<std::string, double> parameter_values_;
};

/**
 * @brief Abstract base class for quantum backends.
 */
class QuantumBackend {
public:
    /**
     * @brief Virtual destructor.
     */
    virtual ~QuantumBackend() = default;
    
    /**
     * @brief Run a quantum circuit and return measurement results.
     * @param circuit Quantum circuit to run.
     * @param shots Number of shots to run.
     * @return Map of measurement results and their counts.
     */
    virtual std::unordered_map<std::string, int> run(const QuantumCircuit& circuit, int shots = 1024) const = 0;
    
    /**
     * @brief Convert a quantum circuit to a unitary matrix.
     * @param circuit Quantum circuit to convert.
     * @return Unitary matrix representation of the circuit.
     */
    virtual UnitaryMatrix to_matrix(const QuantumCircuit& circuit) const = 0;
};

/**
 * @brief Quantum backend using a simulator.
 */
class SimulatorBackend : public QuantumBackend {
public:
    /**
     * @brief Constructor.
     */
    SimulatorBackend() = default;
    
    /**
     * @brief Run a quantum circuit and return measurement results.
     * @param circuit Quantum circuit to run.
     * @param shots Number of shots to run.
     * @return Map of measurement results and their counts.
     */
    std::unordered_map<std::string, int> run(const QuantumCircuit& circuit, int shots = 1024) const override;
    
    /**
     * @brief Convert a quantum circuit to a unitary matrix.
     * @param circuit Quantum circuit to convert.
     * @return Unitary matrix representation of the circuit.
     */
    UnitaryMatrix to_matrix(const QuantumCircuit& circuit) const override;
    
private:
    /**
     * @brief Apply a gate to a state vector.
     * @param gate Gate operation to apply.
     * @param state State vector to modify.
     */
    void apply_gate(const GateOperation& gate, StateVector& state) const;
    
    /**
     * @brief Get the matrix representation of a gate.
     * @param gate Gate operation.
     * @return Matrix representation of the gate.
     */
    UnitaryMatrix get_gate_matrix(const GateOperation& gate) const;
};

/**
 * @brief Utility functions for quantum computing.
 */
namespace utils {

/**
 * @brief Calculate the fidelity between two quantum states.
 * @param state1 First quantum state.
 * @param state2 Second quantum state.
 * @return Fidelity between the states.
 */
double state_fidelity(const StateVector& state1, const StateVector& state2);

/**
 * @brief Calculate the density matrix of a quantum state.
 * @param state Quantum state vector.
 * @return Density matrix.
 */
std::vector<std::vector<Complex>> density_matrix(const StateVector& state);

/**
 * @brief Create a Bell pair circuit.
 * @return Bell pair circuit.
 */
QuantumCircuit bell_pair();

/**
 * @brief Create a GHZ state circuit.
 * @param num_qubits Number of qubits.
 * @return GHZ state circuit.
 */
QuantumCircuit ghz_state(int num_qubits);

/**
 * @brief Create a W state circuit.
 * @param num_qubits Number of qubits.
 * @return W state circuit.
 */
QuantumCircuit w_state(int num_qubits);

/**
 * @brief Create a Quantum Fourier Transform circuit.
 * @param num_qubits Number of qubits.
 * @return QFT circuit.
 */
QuantumCircuit qft(int num_qubits);

} // namespace utils

} // namespace quantum
} // namespace phynexus

#endif // PHYNEXUS_QUANTUM_H
