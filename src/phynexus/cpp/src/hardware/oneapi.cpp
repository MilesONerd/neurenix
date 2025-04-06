/**
 * oneAPI hardware backend implementation for Phynexus
 */

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <unordered_map>

#include "hardware/oneapi.h"
#include "error.h"
#include "tensor.h"
#include "memory.h"

#ifdef PHYNEXUS_WITH_ONEAPI
#include <CL/sycl.hpp>
#endif

namespace phynexus {
namespace hardware {

bool OneAPIBackend::is_available() {
#ifdef PHYNEXUS_WITH_ONEAPI
    try {
        auto platforms = sycl::platform::get_platforms();
        return !platforms.empty();
    } catch (...) {
        return false;
    }
#else
    return false;
#endif
}

OneAPIBackend::OneAPIBackend() : initialized_(false), device_count_(0), platform_(nullptr) {
    try {
        initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize oneAPI backend: " << e.what() << std::endl;
    }
}

OneAPIBackend::~OneAPIBackend() {
    if (initialized_) {
        try {
            cleanup();
        } catch (const std::exception& e) {
            std::cerr << "Error during oneAPI backend cleanup: " << e.what() << std::endl;
        }
    }
}

void OneAPIBackend::initialize() {
#ifdef PHYNEXUS_WITH_ONEAPI
    if (initialized_) return;
    
    if (!is_available()) {
        throw PhynexusError("oneAPI is not available on this system");
    }
    
    try {
        auto platforms = sycl::platform::get_platforms();
        
        if (platforms.empty()) {
            throw PhynexusError("No oneAPI platforms found");
        }
        
        auto platform = platforms[0];
        platform_ = new sycl::platform(platform);
        
        auto devices = platform.get_devices();
        
        if (devices.empty()) {
            throw PhynexusError("No oneAPI devices found");
        }
        
        for (const auto& device : devices) {
            if (device.is_gpu() || device.is_accelerator()) {
                sycl::device* dev_ptr = new sycl::device(device);
                devices_.push_back(dev_ptr);
                
                sycl::context* ctx_ptr = new sycl::context(device);
                contexts_.push_back(ctx_ptr);
                
                sycl::queue* queue_ptr = new sycl::queue(device);
                queues_.push_back(queue_ptr);
            }
        }
        
        device_count_ = devices_.size();
        
        if (device_count_ == 0) {
            throw PhynexusError("No suitable oneAPI devices found");
        }
        
        initialized_ = true;
    } catch (const sycl::exception& e) {
        throw PhynexusError(std::string("oneAPI error: ") + e.what());
    }
#else
    throw PhynexusError("oneAPI support not compiled in");
#endif
}

void OneAPIBackend::cleanup() {
#ifdef PHYNEXUS_WITH_ONEAPI
    if (!initialized_) return;
    
    for (auto& allocation : allocations_) {
        free_buffer(allocation.first);
    }
    allocations_.clear();
    
    for (auto& queue : queues_) {
        delete static_cast<sycl::queue*>(queue);
    }
    queues_.clear();
    
    for (auto& context : contexts_) {
        delete static_cast<sycl::context*>(context);
    }
    contexts_.clear();
    
    for (auto& device : devices_) {
        delete static_cast<sycl::device*>(device);
    }
    devices_.clear();
    
    delete static_cast<sycl::platform*>(platform_);
    platform_ = nullptr;
    
    initialized_ = false;
    device_count_ = 0;
#endif
}

size_t OneAPIBackend::get_device_count() const {
    return device_count_;
}

std::string OneAPIBackend::get_device_name(size_t device_index) const {
#ifdef PHYNEXUS_WITH_ONEAPI
    if (device_index >= device_count_) {
        throw PhynexusError("Invalid device index");
    }
    
    auto device = static_cast<sycl::device*>(devices_[device_index]);
    return device->get_info<sycl::info::device::name>();
#else
    throw PhynexusError("oneAPI support not compiled in");
#endif
}

size_t OneAPIBackend::get_device_memory(size_t device_index) const {
#ifdef PHYNEXUS_WITH_ONEAPI
    if (device_index >= device_count_) {
        throw PhynexusError("Invalid device index");
    }
    
    auto device = static_cast<sycl::device*>(devices_[device_index]);
    return device->get_info<sycl::info::device::global_mem_size>();
#else
    throw PhynexusError("oneAPI support not compiled in");
#endif
}

void* OneAPIBackend::allocate(size_t size, size_t device_index) {
#ifdef PHYNEXUS_WITH_ONEAPI
    if (!initialized_) {
        throw PhynexusError("oneAPI backend not initialized");
    }
    
    if (device_index >= device_count_) {
        throw PhynexusError("Invalid device index");
    }
    
    try {
        auto queue = static_cast<sycl::queue*>(queues_[device_index]);
        auto buffer = new sycl::buffer<uint8_t, 1>(sycl::range<1>(size));
        
        OneAPIAllocation* allocation = new OneAPIAllocation{
            buffer,
            size,
            device_index
        };
        
        allocations_[allocation] = allocation;
        
        return allocation;
    } catch (const sycl::exception& e) {
        throw PhynexusError(std::string("oneAPI allocation error: ") + e.what());
    }
#else
    throw PhynexusError("oneAPI support not compiled in");
#endif
}

void OneAPIBackend::free(void* ptr) {
    if (!initialized_) {
        throw PhynexusError("oneAPI backend not initialized");
    }
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        throw PhynexusError("Invalid oneAPI allocation pointer");
    }
    
    free_buffer(ptr);
    allocations_.erase(it);
}

void OneAPIBackend::free_buffer(void* ptr) {
#ifdef PHYNEXUS_WITH_ONEAPI
    auto allocation = static_cast<OneAPIAllocation*>(ptr);
    
    delete static_cast<sycl::buffer<uint8_t, 1>*>(allocation->buffer);
    delete allocation;
#endif
}

void OneAPIBackend::copy_host_to_device(const void* host_ptr, void* device_ptr, size_t size) {
#ifdef PHYNEXUS_WITH_ONEAPI
    if (!initialized_) {
        throw PhynexusError("oneAPI backend not initialized");
    }
    
    auto it = allocations_.find(device_ptr);
    if (it == allocations_.end()) {
        throw PhynexusError("Invalid oneAPI allocation pointer");
    }
    
    auto allocation = static_cast<OneAPIAllocation*>(device_ptr);
    size_t device_index = allocation->device_index;
    
    if (device_index >= device_count_) {
        throw PhynexusError("Invalid device index in oneAPI allocation");
    }
    
    try {
        auto queue = static_cast<sycl::queue*>(queues_[device_index]);
        auto buffer = static_cast<sycl::buffer<uint8_t, 1>*>(allocation->buffer);
        
        buffer->set_final_data(nullptr);  // Don't copy back to host
        
        buffer->get_access<sycl::access::mode::write>(sycl::range<1>(size));
        
        queue->submit([&](sycl::handler& h) {
            auto accessor = buffer->get_access<sycl::access::mode::write>(h);
            h.copy(static_cast<const uint8_t*>(host_ptr), accessor);
        });
        
        queue->wait();
    } catch (const sycl::exception& e) {
        throw PhynexusError(std::string("oneAPI copy error: ") + e.what());
    }
#else
    throw PhynexusError("oneAPI support not compiled in");
#endif
}

void OneAPIBackend::copy_device_to_host(const void* device_ptr, void* host_ptr, size_t size) {
#ifdef PHYNEXUS_WITH_ONEAPI
    if (!initialized_) {
        throw PhynexusError("oneAPI backend not initialized");
    }
    
    auto it = allocations_.find(const_cast<void*>(device_ptr));
    if (it == allocations_.end()) {
        throw PhynexusError("Invalid oneAPI allocation pointer");
    }
    
    auto allocation = static_cast<const OneAPIAllocation*>(device_ptr);
    size_t device_index = allocation->device_index;
    
    if (device_index >= device_count_) {
        throw PhynexusError("Invalid device index in oneAPI allocation");
    }
    
    try {
        auto queue = static_cast<sycl::queue*>(queues_[device_index]);
        auto buffer = static_cast<sycl::buffer<uint8_t, 1>*>(allocation->buffer);
        
        buffer->set_final_data(static_cast<uint8_t*>(host_ptr));
        
        auto accessor = buffer->get_access<sycl::access::mode::read>();
        
    } catch (const sycl::exception& e) {
        throw PhynexusError(std::string("oneAPI copy error: ") + e.what());
    }
#else
    throw PhynexusError("oneAPI support not compiled in");
#endif
}

void OneAPIBackend::copy_device_to_device(const void* src_ptr, void* dst_ptr, size_t size) {
#ifdef PHYNEXUS_WITH_ONEAPI
    if (!initialized_) {
        throw PhynexusError("oneAPI backend not initialized");
    }
    
    auto src_it = allocations_.find(const_cast<void*>(src_ptr));
    if (src_it == allocations_.end()) {
        throw PhynexusError("Invalid source oneAPI allocation pointer");
    }
    
    auto dst_it = allocations_.find(dst_ptr);
    if (dst_it == allocations_.end()) {
        throw PhynexusError("Invalid destination oneAPI allocation pointer");
    }
    
    auto src_allocation = static_cast<const OneAPIAllocation*>(src_ptr);
    auto dst_allocation = static_cast<OneAPIAllocation*>(dst_ptr);
    
    size_t src_device_index = src_allocation->device_index;
    size_t dst_device_index = dst_allocation->device_index;
    
    if (src_device_index >= device_count_ || dst_device_index >= device_count_) {
        throw PhynexusError("Invalid device index in oneAPI allocation");
    }
    
    try {
        auto src_queue = static_cast<sycl::queue*>(queues_[src_device_index]);
        auto dst_queue = static_cast<sycl::queue*>(queues_[dst_device_index]);
        
        auto src_buffer = static_cast<sycl::buffer<uint8_t, 1>*>(src_allocation->buffer);
        auto dst_buffer = static_cast<sycl::buffer<uint8_t, 1>*>(dst_allocation->buffer);
        
        if (src_device_index == dst_device_index) {
            src_queue->submit([&](sycl::handler& h) {
                auto src_accessor = src_buffer->get_access<sycl::access::mode::read>(h);
                auto dst_accessor = dst_buffer->get_access<sycl::access::mode::write>(h);
                h.copy(src_accessor, dst_accessor);
            });
            
            src_queue->wait();
        } else {
            std::vector<uint8_t> host_buffer(size);
            
            copy_device_to_host(src_ptr, host_buffer.data(), size);
            
            copy_host_to_device(host_buffer.data(), dst_ptr, size);
        }
    } catch (const sycl::exception& e) {
        throw PhynexusError(std::string("oneAPI copy error: ") + e.what());
    }
#else
    throw PhynexusError("oneAPI support not compiled in");
#endif
}

void OneAPIBackend::synchronize(size_t device_index) {
#ifdef PHYNEXUS_WITH_ONEAPI
    if (!initialized_) {
        throw PhynexusError("oneAPI backend not initialized");
    }
    
    if (device_index >= device_count_) {
        throw PhynexusError("Invalid device index");
    }
    
    try {
        auto queue = static_cast<sycl::queue*>(queues_[device_index]);
        queue->wait();
    } catch (const sycl::exception& e) {
        throw PhynexusError(std::string("oneAPI synchronize error: ") + e.what());
    }
#else
    throw PhynexusError("oneAPI support not compiled in");
#endif
}

} // namespace hardware
} // namespace phynexus
