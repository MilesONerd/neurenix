/**
 * @file unified.h
 * @brief Unified Memory (UM) backend for efficient memory management
 */

#ifndef PHYNEXUS_MEMORY_UNIFIED_H
#define PHYNEXUS_MEMORY_UNIFIED_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace phynexus {
namespace memory {

/**
 * @brief Configuration for Unified Memory initialization
 */
struct UnifiedMemoryConfig {
    bool enable_managed_memory{true};
    bool enable_prefetching{true};
    bool enable_advise{true};
    bool enable_async_allocations{false};
    size_t default_alignment{256};
    std::string allocation_strategy{"best_fit"};
};

/**
 * @brief Unified Memory backend for efficient memory management
 */
class UnifiedMemoryBackend {
public:
    /**
     * @brief Initialize Unified Memory backend
     * @param config Unified Memory configuration
     * @return true if initialization succeeded, false otherwise
     */
    static bool initialize(const UnifiedMemoryConfig& config = UnifiedMemoryConfig());

    /**
     * @brief Finalize Unified Memory backend
     */
    static void finalize();

    /**
     * @brief Allocate unified memory
     * @param size Size in bytes
     * @param alignment Alignment in bytes
     * @return Pointer to allocated memory
     */
    static void* allocate(size_t size, size_t alignment = 0);

    /**
     * @brief Free unified memory
     * @param ptr Pointer to memory
     */
    static void free(void* ptr);

    /**
     * @brief Prefetch unified memory to device
     * @param ptr Pointer to memory
     * @param size Size in bytes
     * @param device_id Device ID
     */
    static void prefetch_to_device(void* ptr, size_t size, int device_id);

    /**
     * @brief Prefetch unified memory to host
     * @param ptr Pointer to memory
     * @param size Size in bytes
     */
    static void prefetch_to_host(void* ptr, size_t size);

    /**
     * @brief Set memory access hint
     * @param ptr Pointer to memory
     * @param size Size in bytes
     * @param hint Access hint
     * @param device_id Device ID
     */
    static void set_access_hint(void* ptr, size_t size, const std::string& hint, int device_id = -1);

    /**
     * @brief Get allocation info
     * @param ptr Pointer to memory
     * @return Allocation info
     */
    static std::unordered_map<std::string, size_t> get_allocation_info(void* ptr);

    /**
     * @brief Get total allocated memory
     * @return Total allocated memory in bytes
     */
    static size_t get_total_allocated();

    /**
     * @brief Get allocation count
     * @return Number of allocations
     */
    static size_t get_allocation_count();

private:
    static bool initialized_;
    static UnifiedMemoryConfig config_;
    static size_t total_allocated_;
    static size_t allocation_count_;
    static std::unordered_map<void*, size_t> allocations_;
};

} // namespace memory
} // namespace phynexus

#endif // PHYNEXUS_MEMORY_UNIFIED_H
