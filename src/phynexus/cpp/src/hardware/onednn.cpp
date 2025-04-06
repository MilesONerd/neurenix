/**
 * oneDNN hardware backend implementation for Phynexus
 */

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <unordered_map>

#include "hardware/onednn.h"
#include "error.h"
#include "tensor.h"
#include "memory.h"

#ifdef PHYNEXUS_WITH_ONEDNN
#include <dnnl.hpp>
#endif

namespace phynexus {
namespace hardware {

bool OneDNNBackend::is_available() {
#ifdef PHYNEXUS_WITH_ONEDNN
    try {
        dnnl::engine cpu_engine(dnnl::engine::kind::cpu, 0);
        return true;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

OneDNNBackend::OneDNNBackend() : initialized_(false), device_count_(0), engine_(nullptr) {
    try {
        initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize oneDNN backend: " << e.what() << std::endl;
    }
}

OneDNNBackend::~OneDNNBackend() {
    if (initialized_) {
        try {
            cleanup();
        } catch (const std::exception& e) {
            std::cerr << "Error during oneDNN backend cleanup: " << e.what() << std::endl;
        }
    }
}


} // namespace hardware
} // namespace phynexus
