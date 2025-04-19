/**
 * @file quantum_part2.cpp
 * @brief Implementation of additional quantum gates for Neurenix
 */

#include "quantum/quantum.h"
#include <algorithm>
#include <cmath>
#include <complex>

namespace phynexus {
namespace quantum {

CZGate::CZGate(size_t control_qubit, size_t target_qubit) 
    : control_qubit_(control_qubit), target_qubit_(target_qubit) {}

void CZGate::apply(QuantumState& state) const {
    if (control_qubit_ >= state.num_qubits() || target_qubit_ >= state.num_qubits()) {
        throw std::out_of_range("Qubit index out of range");
    }
    
    if (control_qubit_ == target_qubit_) {
        throw std::invalid_argument("Control and target qubits must be different");
    }
    
    size_t n = state.num_qubits();
    size_t dim = state.dimension();
    std::vector<Complex> new_state = state.state_vector();
    
    for (size_t i = 0; i < dim; ++i) {
        bool control_bit = (i >> (n - control_qubit_ - 1)) & 1;
        bool target_bit = (i >> (n - target_qubit_ - 1)) & 1;
        
        if (control_bit && target_bit) {
            new_state[i] = -new_state[i];
        }
    }
    
    state.set_state_vector(new_state);
}

std::vector<std::vector<Complex>> CZGate::matrix() const {
    return {
        {Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0)},
        {Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0)},
        {Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0)},
        {Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(-1.0, 0.0)}
    };
}

std::string CZGate::name() const {
    return "CZ";
}

SwapGate::SwapGate(size_t qubit1, size_t qubit2) : qubit1_(qubit1), qubit2_(qubit2) {}

void SwapGate::apply(QuantumState& state) const {
    if (qubit1_ >= state.num_qubits() || qubit2_ >= state.num_qubits()) {
        throw std::out_of_range("Qubit index out of range");
    }
    
    if (qubit1_ == qubit2_) {
        return; // Swapping a qubit with itself does nothing
    }
    
    size_t n = state.num_qubits();
    size_t dim = state.dimension();
    std::vector<Complex> new_state = state.state_vector();
    
    for (size_t i = 0; i < dim; ++i) {
        bool bit1 = (i >> (n - qubit1_ - 1)) & 1;
        bool bit2 = (i >> (n - qubit2_ - 1)) & 1;
        
        if (bit1 != bit2) {
            size_t swapped = i ^ (1ULL << (n - qubit1_ - 1)) ^ (1ULL << (n - qubit2_ - 1));
            std::swap(new_state[i], new_state[swapped]);
        }
    }
    
    state.set_state_vector(new_state);
}

std::vector<std::vector<Complex>> SwapGate::matrix() const {
    return {
        {Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0)},
        {Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0)},
        {Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0)},
        {Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)}
    };
}

std::string SwapGate::name() const {
    return "SWAP";
}

CNOTGate::CNOTGate(size_t control_qubit, size_t target_qubit)
    : control_qubit_(control_qubit), target_qubit_(target_qubit) {}

void CNOTGate::apply(QuantumState& state) const {
    if (control_qubit_ >= state.num_qubits() || target_qubit_ >= state.num_qubits()) {
        throw std::out_of_range("Qubit index out of range");
    }
    
    if (control_qubit_ == target_qubit_) {
        throw std::invalid_argument("Control and target qubits must be different");
    }
    
    size_t n = state.num_qubits();
    size_t dim = state.dimension();
    std::vector<Complex> new_state = state.state_vector();
    
    for (size_t i = 0; i < dim; ++i) {
        bool control_bit = (i >> (n - control_qubit_ - 1)) & 1;
        
        if (control_bit) {
            size_t flipped = i ^ (1ULL << (n - target_qubit_ - 1));
            std::swap(new_state[i], new_state[flipped]);
        }
    }
    
    state.set_state_vector(new_state);
}

std::vector<std::vector<Complex>> CNOTGate::matrix() const {
    return {
        {Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0)},
        {Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0)},
        {Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)},
        {Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0)}
    };
}

std::string CNOTGate::name() const {
    return "CNOT";
}

} // namespace quantum
} // namespace phynexus
