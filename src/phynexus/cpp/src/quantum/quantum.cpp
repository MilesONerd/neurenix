/**
 * @file quantum.cpp
 * @brief Quantum Computing implementation for Neurenix
 */

#include "quantum/quantum.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <complex>
#include <iostream>
#include <sstream>
#include <unordered_map>
#include <functional>

namespace phynexus {
namespace quantum {

const double PI = 3.14159265358979323846;

QuantumState::QuantumState(size_t num_qubits) : num_qubits_(num_qubits) {
    state_vector_.resize(1ULL << num_qubits, Complex(0.0, 0.0));
    initialize_zero();
}

QuantumCircuit::QuantumCircuit(size_t num_qubits) : num_qubits_(num_qubits) {}

size_t QuantumCircuit::num_qubits() const {
    return num_qubits_;
}

void QuantumCircuit::add_gate(std::shared_ptr<QuantumGate> gate) {
    gates_.push_back(gate);
}

void QuantumCircuit::h(size_t target_qubit) {
    add_gate(std::make_shared<HadamardGate>(target_qubit));
}

void QuantumCircuit::x(size_t target_qubit) {
    add_gate(std::make_shared<PauliXGate>(target_qubit));
}

void QuantumCircuit::y(size_t target_qubit) {
    add_gate(std::make_shared<PauliYGate>(target_qubit));
}

void QuantumCircuit::z(size_t target_qubit) {
    add_gate(std::make_shared<PauliZGate>(target_qubit));
}

void QuantumCircuit::phase(size_t target_qubit, double phase) {
    add_gate(std::make_shared<PhaseGate>(target_qubit, phase));
}

void QuantumCircuit::rx(size_t target_qubit, double theta) {
    add_gate(std::make_shared<RXGate>(target_qubit, theta));
}

void QuantumCircuit::ry(size_t target_qubit, double theta) {
    add_gate(std::make_shared<RYGate>(target_qubit, theta));
}

void QuantumCircuit::rz(size_t target_qubit, double theta) {
    add_gate(std::make_shared<RZGate>(target_qubit, theta));
}

void QuantumCircuit::cnot(size_t control_qubit, size_t target_qubit) {
    add_gate(std::make_shared<CNOTGate>(control_qubit, target_qubit));
}

void QuantumCircuit::cz(size_t control_qubit, size_t target_qubit) {
    add_gate(std::make_shared<CZGate>(control_qubit, target_qubit));
}

void QuantumCircuit::swap(size_t qubit1, size_t qubit2) {
    add_gate(std::make_shared<SwapGate>(qubit1, qubit2));
}

void QuantumCircuit::toffoli(size_t control_qubit1, size_t control_qubit2, size_t target_qubit) {
    add_gate(std::make_shared<ToffoliGate>(control_qubit1, control_qubit2, target_qubit));
}

void QuantumCircuit::execute(QuantumState& state) const {
    if (state.num_qubits() != num_qubits_) {
        throw std::invalid_argument("State and circuit qubit counts do not match");
    }
    
    for (const auto& gate : gates_) {
        gate->apply(state);
    }
}

QuantumState QuantumCircuit::execute() const {
    QuantumState state(num_qubits_);
    execute(state);
    return state;
}

std::unordered_map<std::string, int> QuantumCircuit::execute_shots(int num_shots) const {
    std::unordered_map<std::string, int> results;
    
    for (int i = 0; i < num_shots; ++i) {
        QuantumState state(num_qubits_);
        execute(state);
        std::string measurement = state.measure();
        results[measurement]++;
    }
    
    return results;
}

size_t QuantumState::num_qubits() const {
    return num_qubits_;
}

size_t QuantumState::dimension() const {
    return 1ULL << num_qubits_;
}

const std::vector<Complex>& QuantumState::state_vector() const {
    return state_vector_;
}

void QuantumState::set_state_vector(const std::vector<Complex>& state_vector) {
    if (state_vector.size() != dimension()) {
        throw std::invalid_argument("State vector dimension mismatch");
    }
    state_vector_ = state_vector;
}

void QuantumState::initialize_zero() {
    std::fill(state_vector_.begin(), state_vector_.end(), Complex(0.0, 0.0));
    state_vector_[0] = Complex(1.0, 0.0);
}

void QuantumState::initialize_random() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    
    double norm = 0.0;
    for (size_t i = 0; i < dimension(); ++i) {
        double real = dist(gen);
        double imag = dist(gen);
        state_vector_[i] = Complex(real, imag);
        norm += std::norm(state_vector_[i]);
    }
    
    norm = std::sqrt(norm);
    for (size_t i = 0; i < dimension(); ++i) {
        state_vector_[i] /= norm;
    }
}

double QuantumState::probability(const std::string& bit_string) const {
    if (bit_string.length() != num_qubits_) {
        throw std::invalid_argument("Bit string length mismatch");
    }
    
    size_t index = 0;
    for (size_t i = 0; i < num_qubits_; ++i) {
        if (bit_string[i] == '1') {
            index |= (1ULL << (num_qubits_ - i - 1));
        } else if (bit_string[i] != '0') {
            throw std::invalid_argument("Bit string must contain only '0' and '1'");
        }
    }
    
    return std::norm(state_vector_[index]);
}

std::string QuantumState::measure() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    double rand_val = dist(gen);
    double cumulative_prob = 0.0;
    
    for (size_t i = 0; i < dimension(); ++i) {
        cumulative_prob += std::norm(state_vector_[i]);
        if (rand_val <= cumulative_prob) {
            std::fill(state_vector_.begin(), state_vector_.end(), Complex(0.0, 0.0));
            state_vector_[i] = Complex(1.0, 0.0);
            
            std::string result(num_qubits_, '0');
            for (size_t j = 0; j < num_qubits_; ++j) {
                if ((i >> (num_qubits_ - j - 1)) & 1) {
                    result[j] = '1';
                }
            }
            
            return result;
        }
    }
    
    return std::string(num_qubits_, '0');
}

int QuantumState::measure_qubit(size_t qubit_index) {
    if (qubit_index >= num_qubits_) {
        throw std::out_of_range("Qubit index out of range");
    }
    
    double prob_one = 0.0;
    for (size_t i = 0; i < dimension(); ++i) {
        if ((i >> (num_qubits_ - qubit_index - 1)) & 1) {
            prob_one += std::norm(state_vector_[i]);
        }
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    int result = (dist(gen) < prob_one) ? 1 : 0;
    
    double norm_factor = 0.0;
    for (size_t i = 0; i < dimension(); ++i) {
        bool bit_value = (i >> (num_qubits_ - qubit_index - 1)) & 1;
        if (bit_value != static_cast<bool>(result)) {
            state_vector_[i] = Complex(0.0, 0.0);
        } else {
            norm_factor += std::norm(state_vector_[i]);
        }
    }
    
    norm_factor = std::sqrt(norm_factor);
    for (size_t i = 0; i < dimension(); ++i) {
        state_vector_[i] /= norm_factor;
    }
    
    return result;
}

} // namespace quantum
} // namespace phynexus
