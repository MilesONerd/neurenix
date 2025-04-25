/**
 * @file quantum.cpp
 * @brief Implementation of Quantum Computing module for Neurenix C++ backend.
 */

#include "quantum/quantum.h"
#include <random>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <bitset>

namespace phynexus {
namespace quantum {

QuantumCircuit::QuantumCircuit(int num_qubits, const std::string& name)
    : num_qubits_(num_qubits), name_(name.empty() ? "circuit_" + std::to_string(num_qubits) + "q" : name) {
}

QuantumCircuit& QuantumCircuit::h(int qubit) {
    operations_.push_back({GateType::H, {qubit}, {}});
    return *this;
}

QuantumCircuit& QuantumCircuit::x(int qubit) {
    operations_.push_back({GateType::X, {qubit}, {}});
    return *this;
}

QuantumCircuit& QuantumCircuit::y(int qubit) {
    operations_.push_back({GateType::Y, {qubit}, {}});
    return *this;
}

QuantumCircuit& QuantumCircuit::z(int qubit) {
    operations_.push_back({GateType::Z, {qubit}, {}});
    return *this;
}

QuantumCircuit& QuantumCircuit::cx(int control, int target) {
    operations_.push_back({GateType::CX, {control, target}, {}});
    return *this;
}

QuantumCircuit& QuantumCircuit::cz(int control, int target) {
    operations_.push_back({GateType::CZ, {control, target}, {}});
    return *this;
}

QuantumCircuit& QuantumCircuit::rx(int qubit, double theta) {
    operations_.push_back({GateType::RX, {qubit}, {theta}});
    return *this;
}

QuantumCircuit& QuantumCircuit::ry(int qubit, double theta) {
    operations_.push_back({GateType::RY, {qubit}, {theta}});
    return *this;
}

QuantumCircuit& QuantumCircuit::rz(int qubit, double theta) {
    operations_.push_back({GateType::RZ, {qubit}, {theta}});
    return *this;
}

QuantumCircuit& QuantumCircuit::measure(int qubit, int classical_bit) {
    operations_.push_back({GateType::MEASURE, {qubit, classical_bit}, {}});
    return *this;
}

UnitaryMatrix QuantumCircuit::to_matrix() const {
    SimulatorBackend backend;
    return backend.to_matrix(*this);
}

std::unordered_map<std::string, int> QuantumCircuit::run(int shots) const {
    SimulatorBackend backend;
    return backend.run(*this, shots);
}

ParameterizedCircuit::ParameterizedCircuit(int num_qubits, const std::vector<std::string>& parameters, const std::string& name)
    : QuantumCircuit(num_qubits, name), parameters_(parameters) {
    
    for (const auto& param : parameters) {
        parameter_values_[param] = 0.0;
    }
}

ParameterizedCircuit& ParameterizedCircuit::rx_param(int qubit, const std::string& param_name) {
    if (std::find(parameters_.begin(), parameters_.end(), param_name) == parameters_.end()) {
        parameters_.push_back(param_name);
        parameter_values_[param_name] = 0.0;
    }
    
    operations_.push_back({GateType::RX, {qubit}, {0.0}});
    return *this;
}

ParameterizedCircuit& ParameterizedCircuit::ry_param(int qubit, const std::string& param_name) {
    if (std::find(parameters_.begin(), parameters_.end(), param_name) == parameters_.end()) {
        parameters_.push_back(param_name);
        parameter_values_[param_name] = 0.0;
    }
    
    operations_.push_back({GateType::RY, {qubit}, {0.0}});
    return *this;
}

ParameterizedCircuit& ParameterizedCircuit::rz_param(int qubit, const std::string& param_name) {
    if (std::find(parameters_.begin(), parameters_.end(), param_name) == parameters_.end()) {
        parameters_.push_back(param_name);
        parameter_values_[param_name] = 0.0;
    }
    
    operations_.push_back({GateType::RZ, {qubit}, {0.0}});
    return *this;
}

QuantumCircuit ParameterizedCircuit::bind_parameters(const std::unordered_map<std::string, double>& parameter_values) const {
    QuantumCircuit circuit(num_qubits(), name());
    
    for (const auto& op : operations()) {
        if (op.type == GateType::RX && op.parameters.empty()) {
            const std::string& param_name = parameters_[op.parameters[0]];
            double param_value = parameter_values.at(param_name);
            circuit.rx(op.qubits[0], param_value);
        } else if (op.type == GateType::RY && op.parameters.empty()) {
            const std::string& param_name = parameters_[op.parameters[0]];
            double param_value = parameter_values.at(param_name);
            circuit.ry(op.qubits[0], param_value);
        } else if (op.type == GateType::RZ && op.parameters.empty()) {
            const std::string& param_name = parameters_[op.parameters[0]];
            double param_value = parameter_values.at(param_name);
            circuit.rz(op.qubits[0], param_value);
        } else {
            switch (op.type) {
                case GateType::H:
                    circuit.h(op.qubits[0]);
                    break;
                case GateType::X:
                    circuit.x(op.qubits[0]);
                    break;
                case GateType::Y:
                    circuit.y(op.qubits[0]);
                    break;
                case GateType::Z:
                    circuit.z(op.qubits[0]);
                    break;
                case GateType::CX:
                    circuit.cx(op.qubits[0], op.qubits[1]);
                    break;
                case GateType::CZ:
                    circuit.cz(op.qubits[0], op.qubits[1]);
                    break;
                case GateType::RX:
                    circuit.rx(op.qubits[0], op.parameters[0]);
                    break;
                case GateType::RY:
                    circuit.ry(op.qubits[0], op.parameters[0]);
                    break;
                case GateType::RZ:
                    circuit.rz(op.qubits[0], op.parameters[0]);
                    break;
                case GateType::MEASURE:
                    circuit.measure(op.qubits[0], op.qubits[1]);
                    break;
                default:
                    break;
            }
        }
    }
    
    return circuit;
}

void ParameterizedCircuit::set_parameter(const std::string& param_name, double value) {
    if (parameter_values_.find(param_name) == parameter_values_.end()) {
        parameters_.push_back(param_name);
    }
    
    parameter_values_[param_name] = value;
}

std::unordered_map<std::string, int> SimulatorBackend::run(const QuantumCircuit& circuit, int shots) const {
    int n = circuit.num_qubits();
    int dim = 1 << n;
    StateVector state(dim, 0.0);
    state[0] = 1.0;
    
    for (const auto& op : circuit.operations()) {
        if (op.type != GateType::MEASURE) {
            apply_gate(op, state);
        }
    }
    
    std::unordered_map<std::string, int> counts;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::vector<double> probabilities;
    for (const auto& amplitude : state) {
        probabilities.push_back(std::norm(amplitude));
    }
    
    std::discrete_distribution<> dist(probabilities.begin(), probabilities.end());
    
    for (int i = 0; i < shots; ++i) {
        int outcome = dist(gen);
        std::string bitstring = std::bitset<64>(outcome).to_string().substr(64 - n);
        counts[bitstring]++;
    }
    
    return counts;
}

UnitaryMatrix SimulatorBackend::to_matrix(const QuantumCircuit& circuit) const {
    int n = circuit.num_qubits();
    int dim = 1 << n;
    
    UnitaryMatrix unitary(dim, std::vector<Complex>(dim, 0.0));
    for (int i = 0; i < dim; ++i) {
        unitary[i][i] = 1.0;
    }
    
    for (const auto& op : circuit.operations()) {
        if (op.type != GateType::MEASURE) {
            UnitaryMatrix gate_matrix = get_gate_matrix(op);
            
            UnitaryMatrix result(dim, std::vector<Complex>(dim, 0.0));
            for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < dim; ++j) {
                    for (int k = 0; k < dim; ++k) {
                        result[i][j] += gate_matrix[i][k] * unitary[k][j];
                    }
                }
            }
            
            unitary = result;
        }
    }
    
    return unitary;
}

void SimulatorBackend::apply_gate(const GateOperation& gate, StateVector& state) const {
    UnitaryMatrix gate_matrix = get_gate_matrix(gate);
    int dim = state.size();
    
    StateVector result(dim, 0.0);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            result[i] += gate_matrix[i][j] * state[j];
        }
    }
    
    state = result;
}

UnitaryMatrix SimulatorBackend::get_gate_matrix(const GateOperation& gate) const {
    int n = 0;
    for (int qubit : gate.qubits) {
        n = std::max(n, qubit + 1);
    }
    
    int dim = 1 << n;
    UnitaryMatrix matrix(dim, std::vector<Complex>(dim, 0.0));
    
    if (gate.type == GateType::H) {
        int qubit = gate.qubits[0];
        int mask = 1 << qubit;
        
        for (int i = 0; i < dim; ++i) {
            int j = i ^ mask;
            if (i & mask) {
                matrix[i][i] = -1.0 / std::sqrt(2.0);
                matrix[i][j] = 1.0 / std::sqrt(2.0);
            } else {
                matrix[i][i] = 1.0 / std::sqrt(2.0);
                matrix[i][j] = 1.0 / std::sqrt(2.0);
            }
        }
    } else if (gate.type == GateType::X) {
        int qubit = gate.qubits[0];
        int mask = 1 << qubit;
        
        for (int i = 0; i < dim; ++i) {
            int j = i ^ mask;
            matrix[i][j] = 1.0;
        }
    } else if (gate.type == GateType::Y) {
        int qubit = gate.qubits[0];
        int mask = 1 << qubit;
        
        for (int i = 0; i < dim; ++i) {
            int j = i ^ mask;
            if (i & mask) {
                matrix[i][j] = Complex(0.0, -1.0);
            } else {
                matrix[i][j] = Complex(0.0, 1.0);
            }
        }
    } else if (gate.type == GateType::Z) {
        int qubit = gate.qubits[0];
        int mask = 1 << qubit;
        
        for (int i = 0; i < dim; ++i) {
            if (i & mask) {
                matrix[i][i] = -1.0;
            } else {
                matrix[i][i] = 1.0;
            }
        }
    } else if (gate.type == GateType::RX) {
        int qubit = gate.qubits[0];
        double theta = gate.parameters[0];
        int mask = 1 << qubit;
        
        double cos = std::cos(theta / 2.0);
        double sin = std::sin(theta / 2.0);
        
        for (int i = 0; i < dim; ++i) {
            int j = i ^ mask;
            if (i & mask) {
                matrix[i][i] = cos;
                matrix[i][j] = Complex(0.0, -sin);
            } else {
                matrix[i][i] = cos;
                matrix[i][j] = Complex(0.0, -sin);
            }
        }
    } else if (gate.type == GateType::RY) {
        int qubit = gate.qubits[0];
        double theta = gate.parameters[0];
        int mask = 1 << qubit;
        
        double cos = std::cos(theta / 2.0);
        double sin = std::sin(theta / 2.0);
        
        for (int i = 0; i < dim; ++i) {
            int j = i ^ mask;
            if (i & mask) {
                matrix[i][i] = cos;
                matrix[i][j] = -sin;
            } else {
                matrix[i][i] = cos;
                matrix[i][j] = sin;
            }
        }
    } else if (gate.type == GateType::RZ) {
        int qubit = gate.qubits[0];
        double theta = gate.parameters[0];
        int mask = 1 << qubit;
        
        Complex phase_pos = std::exp(Complex(0.0, -theta / 2.0));
        Complex phase_neg = std::exp(Complex(0.0, theta / 2.0));
        
        for (int i = 0; i < dim; ++i) {
            if (i & mask) {
                matrix[i][i] = phase_neg;
            } else {
                matrix[i][i] = phase_pos;
            }
        }
    }
    
    else if (gate.type == GateType::CX) {
        int control = gate.qubits[0];
        int target = gate.qubits[1];
        int control_mask = 1 << control;
        int target_mask = 1 << target;
        
        for (int i = 0; i < dim; ++i) {
            if (i & control_mask) {
                int j = i ^ target_mask;
                matrix[i][j] = 1.0;
            } else {
                matrix[i][i] = 1.0;
            }
        }
    } else if (gate.type == GateType::CZ) {
        int control = gate.qubits[0];
        int target = gate.qubits[1];
        int control_mask = 1 << control;
        int target_mask = 1 << target;
        
        for (int i = 0; i < dim; ++i) {
            if ((i & control_mask) && (i & target_mask)) {
                matrix[i][i] = -1.0;
            } else {
                matrix[i][i] = 1.0;
            }
        }
    }
    
    return matrix;
}

namespace utils {

double state_fidelity(const StateVector& state1, const StateVector& state2) {
    Complex inner_product = 0.0;
    for (size_t i = 0; i < state1.size(); ++i) {
        inner_product += std::conj(state1[i]) * state2[i];
    }
    
    return std::abs(inner_product) * std::abs(inner_product);
}

std::vector<std::vector<Complex>> density_matrix(const StateVector& state) {
    int dim = state.size();
    std::vector<std::vector<Complex>> rho(dim, std::vector<Complex>(dim, 0.0));
    
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            rho[i][j] = state[i] * std::conj(state[j]);
        }
    }
    
    return rho;
}

QuantumCircuit bell_pair() {
    QuantumCircuit circuit(2, "bell_pair");
    circuit.h(0).cx(0, 1);
    return circuit;
}

QuantumCircuit ghz_state(int num_qubits) {
    QuantumCircuit circuit(num_qubits, "ghz_" + std::to_string(num_qubits));
    circuit.h(0);
    for (int i = 0; i < num_qubits - 1; ++i) {
        circuit.cx(i, i+1);
    }
    return circuit;
}

QuantumCircuit w_state(int num_qubits) {
    QuantumCircuit circuit(num_qubits, "w_" + std::to_string(num_qubits));
    circuit.x(0);
    for (int i = 0; i < num_qubits - 1; ++i) {
        double theta = std::acos(std::sqrt(1.0 / (num_qubits - i)));
        circuit.ry(i+1, theta);
        circuit.cx(i, i+1);
    }
    return circuit;
}

QuantumCircuit qft(int num_qubits) {
    QuantumCircuit circuit(num_qubits, "qft_" + std::to_string(num_qubits));
    for (int i = 0; i < num_qubits; ++i) {
        circuit.h(i);
        for (int j = i + 1; j < num_qubits; ++j) {
            double angle = M_PI / (1 << (j - i));
            circuit.cz(i, j);
        }
    }
    
    for (int i = 0; i < num_qubits / 2; ++i) {
        int j = num_qubits - i - 1;
        circuit.cx(i, j);
        circuit.cx(j, i);
        circuit.cx(i, j);
    }
    
    return circuit;
}

} // namespace utils

} // namespace quantum
} // namespace phynexus
