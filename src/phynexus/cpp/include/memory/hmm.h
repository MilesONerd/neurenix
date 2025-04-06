/**
 * @file hmm.h
 * @brief Heterogeneous Memory Management (HMM) backend for efficient memory management
 */

#ifndef PHYNEXUS_MEMORY_HMM_H
#define PHYNEXUS_MEMORY_HMM_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace phynexus {
namespace memory {

/**
 * @brief Configuration for HMM initialization
 */
struct HMMConfig {
    bool enable_hmm{true};
    bool enable_system_memory_mapping{true};
    bool enable_device_memory_mapping{true};
    bool enable_migration_hints{true};
    bool enable_prefetch{true};
    size_t default_alignment{256};
    std::string allocation_strategy{"best_fit"};
};

/**
 * @brief Heterogeneous Memory Management backend for efficient memory management
 */
class HMMBackend {
public:
    /**
     * @brief Initialize HMM backend
     * @param config HMM configuration
     * @return true if initialization succeeded, false otherwise
     */
    static bool initialize(const HMMConfig& config = HMMConfig());

    /**
     * @brief Finalize HMM backend
     */
    static void finalize();

    /**
     * @brief Allocate HMM memory
     * @param size Size in bytes
     * @param alignment Alignment in bytes
     * @return Pointer to allocated memory
     */
    static void* allocate(size_t size, size_t alignment = 0);

    /**
     * @brief Free HMM memory
     * @param ptr Pointer to memory
     */
    static void free(void* ptr);

    /**
     * @brief Map memory to device
     * @param ptr Pointer to memory
     * @param size Size in bytes
     * @param device_id Device ID
     * @return Pointer to mapped memory on device
     */
    static void* map_to_device(void* ptr, size_t size, int device_id);

    /**
     * @brief Unmap memory from device
     * @param device_ptr Pointer to mapped memory on device
     * @param host_ptr Pointer to memory on host
     * @param size Size in bytes
     * @param device_id Device ID
     */
    static void unmap_from_device(void* device_ptr, void* host_ptr, size_t size, int device_id);

    /**
     * @brief Migrate memory to device
     * @param ptr Pointer to memory
     * @param size Size in bytes
     * @param device_id Device ID
     */
    static void migrate_to_device(void* ptr, size_t size, int device_id);

    /**
     * @brief Migrate memory to host
     * @param ptr Pointer to memory
     * @param size Size in bytes
     */
    static void migrate_to_host(void* ptr, size_t size);

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
    static HMMConfig config_;
    static size_t total_allocated_;
    static size_t allocation_count_;
    static std::unordered_map<void*, size_t> allocations_;
    static std::unordered_map<void*, void*> device_mappings_;
};

} // namespace memory
} // namespace phynexus

#endif // PHYNEXUS_MEMORY_HMM_H
