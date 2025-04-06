/**
 * DirectML hardware backend implementation for Phynexus
 */

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <unordered_map>

#include "hardware/directml.h"
#include "error.h"
#include "tensor.h"
#include "memory.h"

#ifdef PHYNEXUS_WITH_DIRECTML
#include <DirectML.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
using Microsoft::WRL::ComPtr;
#endif

namespace phynexus {
namespace hardware {

bool DirectMLBackend::is_available() {
#ifdef PHYNEXUS_WITH_DIRECTML
    try {
        ComPtr<IDXGIFactory6> factory;
        HRESULT hr = CreateDXGIFactory2(0, IID_PPV_ARGS(&factory));
        
        if (FAILED(hr)) {
            return false;
        }
        
        ComPtr<IDXGIAdapter1> adapter;
        if (FAILED(factory->EnumAdapters1(0, &adapter))) {
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

DirectMLBackend::DirectMLBackend() : initialized_(false), device_count_(0), device_factory_(nullptr) {
    try {
        initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize DirectML backend: " << e.what() << std::endl;
    }
}

DirectMLBackend::~DirectMLBackend() {
    if (initialized_) {
        try {
            cleanup();
        } catch (const std::exception& e) {
            std::cerr << "Error during DirectML backend cleanup: " << e.what() << std::endl;
        }
    }
}


} // namespace hardware
} // namespace phynexus
