/**
 * oneAPI hardware backend header for Phynexus
 */

#ifndef PHYNEXUS_HARDWARE_ONEAPI_H
#define PHYNEXUS_HARDWARE_ONEAPI_H

#include <vector>
#include <string>
#include <unordered_map>

#include "hardware/backend.h"
#include "error.h"

namespace phynexus {
namespace hardware {

/**
 * oneAPI hardware backend implementation
 */
class OneAPIBackend : public Backend {
public:
    /**
     * Check if oneAPI is available on the system
     */
    static bool is_available();

    /**
     * Constructor
     */
    OneAPIBackend();

    /**
     * Destructor
     */
    ~OneAPIBackend();

    /**
     * Get the number of available devices
     */
    size_t get_device_count() const override;

    /**
     * Get the name of a device
     */
    std::string get_device_name(size_t device_index) const override;

    /**
     * Get the amount of memory on a device
     */
    size_t get_device_memory(size_t device_index) const override;

    /**
     * Allocate memory on a device
     */
    void* allocate(size_t size, size_t device_index) override;

    /**
     * Free memory on a device
     */
    void free(void* ptr) override;

    /**
     * Copy data from host to device
     */
    void copy_host_to_device(const void* host_ptr, void* device_ptr, size_t size) override;

    /**
     * Copy data from device to host
     */
    void copy_device_to_host(const void* device_ptr, void* host_ptr, size_t size) override;

    /**
     * Copy data from device to device
     */
    void copy_device_to_device(const void* src_ptr, void* dst_ptr, size_t size) override;

    /**
     * Synchronize device
     */
    void synchronize(size_t device_index) override;

private:
    /**
     * Initialize the backend
     */
    void initialize();

    /**
     * Clean up resources
     */
    void cleanup();

    /**
     * Free a buffer
     */
    void free_buffer(void* ptr);

    /**
     * oneAPI allocation structure
     */
    struct OneAPIAllocation {
        void* buffer;
        size_t size;
        size_t device_index;
    };

    bool initialized_;
    size_t device_count_;
    void* platform_;
    std::vector<void*> devices_;
    std::vector<void*> contexts_;
    std::vector<void*> queues_;
    std::unordered_map<void*, OneAPIAllocation*> allocations_;
};

} // namespace hardware
} // namespace phynexus

#endif // PHYNEXUS_HARDWARE_ONEAPI_H
