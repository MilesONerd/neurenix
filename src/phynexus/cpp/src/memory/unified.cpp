/**
 * @file unified.cpp
 * @brief Unified Memory (UM) backend for efficient memory management
 */

#include "memory/unified.h"
#include <iostream>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <cstring>
#include <algorithm>

namespace phynexus {
namespace memory {

bool UnifiedMemoryBackend::initialized_ = false;
UnifiedMemoryConfig UnifiedMemoryBackend::config_;
size_t UnifiedMemoryBackend::total_allocated_ = 0;
size_t UnifiedMemoryBackend::allocation_count_ = 0;
std::unordered_map<void*, size_t> UnifiedMemoryBackend::allocations_;

bool UnifiedMemoryBackend::initialize(const UnifiedMemoryConfig& config) {
    if (initialized_) {
        std::cerr << "Unified Memory is already initialized" << std::endl;
        return true;
    }

    std::cout << "Initializing Unified Memory with prefetching: " << (config.enable_prefetching ? "enabled" : "disabled")
              << ", advise: " << (config.enable_advise ? "enabled" : "disabled")
              << ", async allocations: " << (config.enable_async_allocations ? "enabled" : "disabled")
              << ", allocation strategy: " << config.allocation_strategy << std::endl;
    
    
    config_ = config;
    initialized_ = true;
    return true;
}

void UnifiedMemoryBackend::finalize() {
    if (!initialized_) {
        std::cerr << "Unified Memory is not initialized" << std::endl;
        return;
    }
    
    for (const auto& allocation : allocations_) {
        free(allocation.first);
    }
    
    initialized_ = false;
    total_allocated_ = 0;
    allocation_count_ = 0;
    allocations_.clear();
}

void* UnifiedMemoryBackend::allocate(size_t size, size_t alignment) {
    if (!initialized_) {
        std::cerr << "Unified Memory is not initialized" << std::endl;
        return nullptr;
    }
    
    if (size == 0) {
        std::cerr << "Cannot allocate zero bytes" << std::endl;
        return nullptr;
    }
    
    if (alignment == 0) {
        alignment = config_.default_alignment;
    }
    
    if ((alignment & (alignment - 1)) != 0) {
        std::cerr << "Alignment must be a power of 2" << std::endl;
        return nullptr;
    }
    
    void* ptr = nullptr;
    
    
    ptr = aligned_alloc(alignment, size);
    
    if (ptr == nullptr) {
        std::cerr << "Failed to allocate " << size << " bytes" << std::endl;
        return nullptr;
    }
    
    allocations_[ptr] = size;
    total_allocated_ += size;
    allocation_count_++;
    
    return ptr;
}

void UnifiedMemoryBackend::free(void* ptr) {
    if (!initialized_) {
        std::cerr << "Unified Memory is not initialized" << std::endl;
        return;
    }
    
    if (ptr == nullptr) {
        std::cerr << "Cannot free null pointer" << std::endl;
        return;
    }
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        std::cerr << "Pointer not allocated by Unified Memory" << std::endl;
        return;
    }
    
    
    ::free(ptr);
    
    total_allocated_ -= it->second;
    allocation_count_--;
    allocations_.erase(it);
}

void UnifiedMemoryBackend::prefetch_to_device(void* ptr, size_t size, int device_id) {
    if (!initialized_) {
        std::cerr << "Unified Memory is not initialized" << std::endl;
        return;
    }
    
    if (!config_.enable_prefetching) {
        std::cerr << "Prefetching is disabled" << std::endl;
        return;
    }
    
    if (ptr == nullptr) {
        std::cerr << "Cannot prefetch null pointer" << std::endl;
        return;
    }
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        std::cerr << "Pointer not allocated by Unified Memory" << std::endl;
        return;
    }
    
    size = std::min(size, it->second);
    
    
    std::cout << "Prefetching " << size << " bytes to device " << device_id << std::endl;
}

void UnifiedMemoryBackend::prefetch_to_host(void* ptr, size_t size) {
    if (!initialized_) {
        std::cerr << "Unified Memory is not initialized" << std::endl;
        return;
    }
    
    if (!config_.enable_prefetching) {
        std::cerr << "Prefetching is disabled" << std::endl;
        return;
    }
    
    if (ptr == nullptr) {
        std::cerr << "Cannot prefetch null pointer" << std::endl;
        return;
    }
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        std::cerr << "Pointer not allocated by Unified Memory" << std::endl;
        return;
    }
    
    size = std::min(size, it->second);
    
    
    std::cout << "Prefetching " << size << " bytes to host" << std::endl;
}

void UnifiedMemoryBackend::set_access_hint(void* ptr, size_t size, const std::string& hint, int device_id) {
    if (!initialized_) {
        std::cerr << "Unified Memory is not initialized" << std::endl;
        return;
    }
    
    if (!config_.enable_advise) {
        std::cerr << "Memory advise is disabled" << std::endl;
        return;
    }
    
    if (ptr == nullptr) {
        std::cerr << "Cannot set access hint for null pointer" << std::endl;
        return;
    }
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        std::cerr << "Pointer not allocated by Unified Memory" << std::endl;
        return;
    }
    
    size = std::min(size, it->second);
    
    
    std::cout << "Setting access hint '" << hint << "' for " << size << " bytes"
              << (device_id >= 0 ? " on device " + std::to_string(device_id) : "") << std::endl;
}

std::unordered_map<std::string, size_t> UnifiedMemoryBackend::get_allocation_info(void* ptr) {
    std::unordered_map<std::string, size_t> info;
    
    if (!initialized_) {
        std::cerr << "Unified Memory is not initialized" << std::endl;
        return info;
    }
    
    if (ptr == nullptr) {
        std::cerr << "Cannot get info for null pointer" << std::endl;
        return info;
    }
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        std::cerr << "Pointer not allocated by Unified Memory" << std::endl;
        return info;
    }
    
    info["size"] = it->second;
    info["address"] = reinterpret_cast<size_t>(ptr);
    
    
    return info;
}

size_t UnifiedMemoryBackend::get_total_allocated() {
    if (!initialized_) {
        std::cerr << "Unified Memory is not initialized" << std::endl;
        return 0;
    }
    
    return total_allocated_;
}

size_t UnifiedMemoryBackend::get_allocation_count() {
    if (!initialized_) {
        std::cerr << "Unified Memory is not initialized" << std::endl;
        return 0;
    }
    
    return allocation_count_;
}

} // namespace memory
} // namespace phynexus
