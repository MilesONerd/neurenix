/**
 * TensorRT hardware backend implementation for Phynexus
 */

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <unordered_map>

#include "hardware/tensorrt.h"
#include "error.h"
#include "tensor.h"
#include "memory.h"

#ifdef PHYNEXUS_WITH_TENSORRT
#include <NvInfer.h>
#include <cuda_runtime_api.h>

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << msg << std::endl;
        }
    }
} gLogger;
#endif

namespace phynexus {
namespace hardware {

bool TensorRTBackend::is_available() {
#ifdef PHYNEXUS_WITH_TENSORRT
    try {
        auto builder = nvinfer1::createInferBuilder(gLogger);
        if (!builder) {
            return false;
        }
        
        int device_count = 0;
        cudaError_t error = cudaGetDeviceCount(&device_count);
        if (error != cudaSuccess || device_count == 0) {
            return false;
        }
        
        return true;
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

TensorRTBackend::TensorRTBackend() : initialized_(false), device_count_(0), runtime_(nullptr) {
    try {
        initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize TensorRT backend: " << e.what() << std::endl;
    }
}

TensorRTBackend::~TensorRTBackend() {
    if (initialized_) {
        try {
            cleanup();
        } catch (const std::exception& e) {
            std::cerr << "Error during TensorRT backend cleanup: " << e.what() << std::endl;
        }
    }
}


} // namespace hardware
} // namespace phynexus
