/**
 * @file cpu.cpp
 * @brief CPU implementation for the Phynexus engine
 * 
 * This file contains the CPU implementation for the Phynexus engine.
 * 
 * @copyright Copyright (c) 2025 Neurenix
 * @license Apache License 2.0
 */

#include "phynexus/tensor.h"
#include <cstring>
#include <stdexcept>
#include <thread>

namespace phynexus {
namespace hardware {

/**
 * @brief Initialize CPU
 * 
 * @return True if CPU is available, false otherwise
 */
bool initialize_cpu() {
    // CPU is always available
    return true;
}

/**
 * @brief Get CPU device count
 * 
 * @return Number of CPU devices (always 1)
 */
int get_cpu_device_count() {
    // CPU is always available
    return 1;
}

/**
 * @brief Get CPU device properties
 * 
 * @return Device properties
 */
DeviceProperties get_cpu_device_properties() {
    DeviceProperties props;
    props.name = "CPU";
    props.total_memory = 0;  // System memory not reported
    props.compute_capability_major = 0;
    props.compute_capability_minor = 0;
    props.multi_processor_count = std::thread::hardware_concurrency();
    props.max_threads_per_block = 0;
    props.max_threads_per_multiprocessor = 0;
    props.warp_size = 0;
    return props;
}

/**
 * @brief Allocate memory on CPU
 * 
 * @param size Size in bytes
 * @return Pointer to allocated memory
 */
void* cpu_malloc(size_t size) {
    return malloc(size);
}

/**
 * @brief Free memory on CPU
 * 
 * @param ptr Pointer to memory
 */
void cpu_free(void* ptr) {
    free(ptr);
}

/**
 * @brief Copy memory on CPU
 * 
 * @param dst Destination pointer
 * @param src Source pointer
 * @param size Size in bytes
 */
void cpu_memcpy(void* dst, const void* src, size_t size) {
    memcpy(dst, src, size);
}

/**
 * @brief Set memory on CPU
 * 
 * @param ptr Pointer to memory
 * @param value Value to set
 * @param size Size in bytes
 */
void cpu_memset(void* ptr, int value, size_t size) {
    memset(ptr, value, size);
}

/**
 * @brief Parallel for loop on CPU
 * 
 * @param start Start index
 * @param end End index
 * @param num_threads Number of threads
 * @param func Function to execute
 */
void cpu_parallel_for(size_t start, size_t end, size_t num_threads, std::function<void(size_t, size_t)> func) {
    if (num_threads == 0) {
        num_threads = std::thread::hardware_concurrency();
    }
    
    if (num_threads == 1 || end - start <= 1) {
        func(start, end);
        return;
    }
    
    std::vector<std::thread> threads;
    size_t chunk_size = (end - start + num_threads - 1) / num_threads;
    
    for (size_t i = 0; i < num_threads; ++i) {
        size_t chunk_start = start + i * chunk_size;
        size_t chunk_end = std::min(chunk_start + chunk_size, end);
        
        if (chunk_start >= chunk_end) {
            break;
        }
        
        threads.push_back(std::thread([func, chunk_start, chunk_end]() {
            func(chunk_start, chunk_end);
        }));
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

} // namespace hardware
} // namespace phynexus
