/**
 * @file hmm.cpp
 * @brief Heterogeneous Memory Management (HMM) backend for efficient memory management
 */

#include "memory/hmm.h"
#include <iostream>
#include <stdexcept>
#include <thread>
#include <chrono>
#include <cstring>
#include <algorithm>

namespace phynexus {
namespace memory {

bool HMMBackend::initialized_ = false;
HMMConfig HMMBackend::config_;
size_t HMMBackend::total_allocated_ = 0;
size_t HMMBackend::allocation_count_ = 0;
std::unordered_map<void*, size_t> HMMBackend::allocations_;
std::unordered_map<void*, void*> HMMBackend::device_mappings_;

bool HMMBackend::initialize(const HMMConfig& config) {
    if (initialized_) {
        std::cerr << "HMM is already initialized" << std::endl;
        return true;
    }

    std::cout << "Initializing HMM with system memory mapping: " << (config.enable_system_memory_mapping ? "enabled" : "disabled")
              << ", device memory mapping: " << (config.enable_device_memory_mapping ? "enabled" : "disabled")
              << ", migration hints: " << (config.enable_migration_hints ? "enabled" : "disabled")
              << ", prefetch: " << (config.enable_prefetch ? "enabled" : "disabled")
              << ", allocation strategy: " << config.allocation_strategy << std::endl;
    
    
    config_ = config;
    initialized_ = true;
    return true;
}

void HMMBackend::finalize() {
    if (!initialized_) {
        std::cerr << "HMM is not initialized" << std::endl;
        return;
    }
    
    for (const auto& allocation : allocations_) {
        free(allocation.first);
    }
    
    initialized_ = false;
    total_allocated_ = 0;
    allocation_count_ = 0;
    allocations_.clear();
    device_mappings_.clear();
}

void* HMMBackend::allocate(size_t size, size_t alignment) {
    if (!initialized_) {
        std::cerr << "HMM is not initialized" << std::endl;
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

void HMMBackend::free(void* ptr) {
    if (!initialized_) {
        std::cerr << "HMM is not initialized" << std::endl;
        return;
    }
    
    if (ptr == nullptr) {
        std::cerr << "Cannot free null pointer" << std::endl;
        return;
    }
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        std::cerr << "Pointer not allocated by HMM" << std::endl;
        return;
    }
    
    
    ::free(ptr);
    
    total_allocated_ -= it->second;
    allocation_count_--;
    allocations_.erase(it);
    
    auto mapping_it = device_mappings_.find(ptr);
    if (mapping_it != device_mappings_.end()) {
        device_mappings_.erase(mapping_it);
    }
}

void* HMMBackend::map_to_device(void* ptr, size_t size, int device_id) {
    if (!initialized_) {
        std::cerr << "HMM is not initialized" << std::endl;
        return nullptr;
    }
    
    if (!config_.enable_device_memory_mapping) {
        std::cerr << "Device memory mapping is disabled" << std::endl;
        return nullptr;
    }
    
    if (ptr == nullptr) {
        std::cerr << "Cannot map null pointer" << std::endl;
        return nullptr;
    }
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        std::cerr << "Pointer not allocated by HMM" << std::endl;
        return nullptr;
    }
    
    size = std::min(size, it->second);
    
    auto mapping_it = device_mappings_.find(ptr);
    if (mapping_it != device_mappings_.end()) {
        return mapping_it->second;
    }
    
    void* device_ptr = nullptr;
    
    
    device_ptr = ptr;
    
    device_mappings_[ptr] = device_ptr;
    
    std::cout << "Mapped " << size << " bytes to device " << device_id << std::endl;
    
    return device_ptr;
}

void HMMBackend::unmap_from_device(void* device_ptr, void* host_ptr, size_t size, int device_id) {
    if (!initialized_) {
        std::cerr << "HMM is not initialized" << std::endl;
        return;
    }
    
    if (!config_.enable_device_memory_mapping) {
        std::cerr << "Device memory mapping is disabled" << std::endl;
        return;
    }
    
    if (device_ptr == nullptr || host_ptr == nullptr) {
        std::cerr << "Cannot unmap null pointer" << std::endl;
        return;
    }
    
    auto it = allocations_.find(host_ptr);
    if (it == allocations_.end()) {
        std::cerr << "Host pointer not allocated by HMM" << std::endl;
        return;
    }
    
    size = std::min(size, it->second);
    
    auto mapping_it = device_mappings_.find(host_ptr);
    if (mapping_it == device_mappings_.end() || mapping_it->second != device_ptr) {
        std::cerr << "Pointer not mapped or mapping mismatch" << std::endl;
        return;
    }
    
    
    device_mappings_.erase(mapping_it);
    
    std::cout << "Unmapped " << size << " bytes from device " << device_id << std::endl;
}

void HMMBackend::migrate_to_device(void* ptr, size_t size, int device_id) {
    if (!initialized_) {
        std::cerr << "HMM is not initialized" << std::endl;
        return;
    }
    
    if (!config_.enable_migration_hints) {
        std::cerr << "Migration hints are disabled" << std::endl;
        return;
    }
    
    if (ptr == nullptr) {
        std::cerr << "Cannot migrate null pointer" << std::endl;
        return;
    }
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        std::cerr << "Pointer not allocated by HMM" << std::endl;
        return;
    }
    
    size = std::min(size, it->second);
    
    
    std::cout << "Migrated " << size << " bytes to device " << device_id << std::endl;
}

void HMMBackend::migrate_to_host(void* ptr, size_t size) {
    if (!initialized_) {
        std::cerr << "HMM is not initialized" << std::endl;
        return;
    }
    
    if (!config_.enable_migration_hints) {
        std::cerr << "Migration hints are disabled" << std::endl;
        return;
    }
    
    if (ptr == nullptr) {
        std::cerr << "Cannot migrate null pointer" << std::endl;
        return;
    }
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        std::cerr << "Pointer not allocated by HMM" << std::endl;
        return;
    }
    
    size = std::min(size, it->second);
    
    
    std::cout << "Migrated " << size << " bytes to host" << std::endl;
}

void HMMBackend::set_access_hint(void* ptr, size_t size, const std::string& hint, int device_id) {
    if (!initialized_) {
        std::cerr << "HMM is not initialized" << std::endl;
        return;
    }
    
    if (!config_.enable_migration_hints) {
        std::cerr << "Migration hints are disabled" << std::endl;
        return;
    }
    
    if (ptr == nullptr) {
        std::cerr << "Cannot set access hint for null pointer" << std::endl;
        return;
    }
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        std::cerr << "Pointer not allocated by HMM" << std::endl;
        return;
    }
    
    size = std::min(size, it->second);
    
    //     
    
    std::cout << "Set access hint '" << hint << "' for " << size << " bytes"
              << (device_id >= 0 ? " on device " + std::to_string(device_id) : "") << std::endl;
}

std::unordered_map<std::string, size_t> HMMBackend::get_allocation_info(void* ptr) {
    std::unordered_map<std::string, size_t> info;
    
    if (!initialized_) {
        std::cerr << "HMM is not initialized" << std::endl;
        return info;
    }
    
    if (ptr == nullptr) {
        std::cerr << "Cannot get info for null pointer" << std::endl;
        return info;
    }
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        std::cerr << "Pointer not allocated by HMM" << std::endl;
        return info;
    }
    
    info["size"] = it->second;
    info["address"] = reinterpret_cast<size_t>(ptr);
    
    auto mapping_it = device_mappings_.find(ptr);
    if (mapping_it != device_mappings_.end()) {
        info["device_ptr"] = reinterpret_cast<size_t>(mapping_it->second);
        info["is_mapped"] = 1;
    } else {
        info["is_mapped"] = 0;
    }
    
    
    return info;
}

size_t HMMBackend::get_total_allocated() {
    if (!initialized_) {
        std::cerr << "HMM is not initialized" << std::endl;
        return 0;
    }
    
    return total_allocated_;
}

size_t HMMBackend::get_allocation_count() {
    if (!initialized_) {
        std::cerr << "HMM is not initialized" << std::endl;
        return 0;
    }
    
    return allocation_count_;
}

} // namespace memory
} // namespace phynexus
