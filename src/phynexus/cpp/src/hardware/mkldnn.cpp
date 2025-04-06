/**
 * MKL-DNN hardware backend implementation for Phynexus
 */

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <unordered_map>

#include "hardware/mkldnn.h"
#include "error.h"
#include "tensor.h"
#include "memory.h"

#ifdef PHYNEXUS_WITH_MKLDNN
#include <mkldnn.hpp>
#endif

namespace phynexus {
namespace hardware {

bool MKLDNNBackend::is_available() {
#ifdef PHYNEXUS_WITH_MKLDNN
    try {
        mkldnn::engine cpu_engine(mkldnn::engine::kind::cpu, 0);
        return true;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

MKLDNNBackend::MKLDNNBackend() : initialized_(false), device_count_(0), engine_(nullptr) {
    try {
        initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize MKL-DNN backend: " << e.what() << std::endl;
    }
}

MKLDNNBackend::~MKLDNNBackend() {
    if (initialized_) {
        try {
            cleanup();
        } catch (const std::exception& e) {
            std::cerr << "Error during MKL-DNN backend cleanup: " << e.what() << std::endl;
        }
    }
}


} // namespace hardware
} // namespace phynexus
