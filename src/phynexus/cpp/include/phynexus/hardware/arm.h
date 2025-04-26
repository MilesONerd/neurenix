/**
 * @file arm.h
 * @brief ARM hardware acceleration interface
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace phynexus {
namespace hardware {

// ARM device properties
struct ARMDeviceProperties {
    bool has_neon;
    bool has_sve;
    bool has_ethos_u;
    size_t sve_vector_length;
    char device_name[256];
};

// Convolution parameters
struct Conv2DParams {
    size_t batch_size;
    size_t input_channels;
    size_t input_height;
    size_t input_width;
    size_t output_channels;
    size_t output_height;
    size_t output_width;
    size_t kernel_height;
    size_t kernel_width;
    size_t stride_height;
    size_t stride_width;
    size_t padding_height;
    size_t padding_width;
};

// Initialize ARM hardware
bool arm_init();

// Check if ARM is available
bool arm_is_available();

// Get ARM device count
int arm_get_device_count();

// Set current ARM device
bool arm_set_device(int device_id);

// Get ARM device properties
bool arm_get_device_properties(int device_id, ARMDeviceProperties* props);

// Memory management
bool arm_malloc(void** ptr, size_t size);
bool arm_free(void* ptr);
bool arm_memcpy_host_to_device(void* dst, const void* src, size_t size);
bool arm_memcpy_device_to_host(void* dst, const void* src, size_t size);
bool arm_memcpy_device_to_device(void* dst, const void* src, size_t size);
bool arm_device_synchronize();

// NEON-optimized operations
bool arm_neon_add(const float* a, const float* b, float* c, size_t size);
bool arm_neon_multiply(const float* a, const float* b, float* c, size_t size);

// SVE-optimized operations
bool arm_sve_add(const float* a, const float* b, float* c, size_t size);
bool arm_sve_multiply(const float* a, const float* b, float* c, size_t size);

// ARM Compute Library operations
bool arm_acl_conv2d(const float* input, const float* weights, const float* bias,
                    float* output, const Conv2DParams& params);

// Ethos-U NPU operations
bool arm_ethos_u_conv2d(const float* input, const float* weights, const float* bias,
                        float* output, const Conv2DParams& params);

} // namespace hardware
} // namespace phynexus
