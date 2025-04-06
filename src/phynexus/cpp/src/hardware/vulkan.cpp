/**
 * Vulkan hardware backend implementation for Phynexus
 */

#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <unordered_map>

#include "hardware/vulkan.h"
#include "error.h"
#include "tensor.h"
#include "memory.h"

namespace phynexus {
namespace hardware {

bool VulkanBackend::is_available() {
    try {
        #ifdef _WIN32
        void* vulkan_lib = LoadLibrary("vulkan-1.dll");
        if (!vulkan_lib) return false;
        FreeLibrary((HMODULE)vulkan_lib);
        #else
        void* vulkan_lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!vulkan_lib) return false;
        dlclose(vulkan_lib);
        #endif
        
        return true;
    } catch (...) {
        return false;
    }
}

VulkanBackend::VulkanBackend() : initialized_(false), device_count_(0) {
    try {
        initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Vulkan backend: " << e.what() << std::endl;
    }
}

VulkanBackend::~VulkanBackend() {
    if (initialized_) {
        try {
            cleanup();
        } catch (const std::exception& e) {
            std::cerr << "Error during Vulkan backend cleanup: " << e.what() << std::endl;
        }
    }
}


} // namespace hardware
} // namespace phynexus
