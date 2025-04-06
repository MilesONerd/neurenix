/**
 * @file hardware_backends.h
 * @brief Hardware backends for the Phynexus engine
 * 
 * This file contains the hardware backends for the Phynexus engine,
 * providing a unified interface for different hardware platforms.
 * 
 * @copyright Copyright (c) 2025 Neurenix
 * @license Apache License 2.0
 */

#ifndef PHYNEXUS_HARDWARE_BACKENDS_H
#define PHYNEXUS_HARDWARE_BACKENDS_H

#include "hardware.h"
#include <memory>

// Include all hardware backend headers
#include "hardware/vulkan.h"
#include "hardware/opencl.h"
#include "hardware/oneapi.h"
#include "hardware/directml.h"
#include "hardware/onednn.h"
#include "hardware/mkldnn.h"
#include "hardware/tensorrt.h"

namespace phynexus {
namespace hardware {

/**
 * @brief Initialize all hardware backends
 * 
 * @return True if at least one hardware backend is available, false otherwise
 */
bool initialize_all_backends();

/**
 * @brief Get the best available backend for a device type
 * 
 * @param device_type Device type
 * @return Backend pointer
 */
std::shared_ptr<Backend> get_backend(DeviceType device_type);

/**
 * @brief Check if a specific backend is available
 * 
 * @param backend_type Backend type
 * @return True if the backend is available, false otherwise
 */
bool is_backend_available(BackendType backend_type);

// Vulkan functions
bool initialize_vulkan();
int get_vulkan_device_count();
DeviceProperties get_vulkan_device_properties(int device_index);
void set_vulkan_device(int device_index);
int get_current_vulkan_device();
void* vulkan_malloc(size_t size);
void vulkan_free(void* ptr);
void vulkan_memcpy_host_to_device(void* dst, const void* src, size_t size);
void vulkan_memcpy_device_to_host(void* dst, const void* src, size_t size);
void vulkan_memcpy_device_to_device(void* dst, const void* src, size_t size);
void vulkan_memset(void* ptr, int value, size_t size);
void vulkan_synchronize();

// OpenCL functions
bool initialize_opencl();
int get_opencl_device_count();
DeviceProperties get_opencl_device_properties(int device_index);
void set_opencl_device(int device_index);
int get_current_opencl_device();
void* opencl_malloc(size_t size);
void opencl_free(void* ptr);
void opencl_memcpy_host_to_device(void* dst, const void* src, size_t size);
void opencl_memcpy_device_to_host(void* dst, const void* src, size_t size);
void opencl_memcpy_device_to_device(void* dst, const void* src, size_t size);
void opencl_memset(void* ptr, int value, size_t size);
void opencl_synchronize();

// oneAPI functions
bool initialize_oneapi();
int get_oneapi_device_count();
DeviceProperties get_oneapi_device_properties(int device_index);
void set_oneapi_device(int device_index);
int get_current_oneapi_device();
void* oneapi_malloc(size_t size);
void oneapi_free(void* ptr);
void oneapi_memcpy_host_to_device(void* dst, const void* src, size_t size);
void oneapi_memcpy_device_to_host(void* dst, const void* src, size_t size);
void oneapi_memcpy_device_to_device(void* dst, const void* src, size_t size);
void oneapi_memset(void* ptr, int value, size_t size);
void oneapi_synchronize();

// DirectML functions
bool initialize_directml();
int get_directml_device_count();
DeviceProperties get_directml_device_properties(int device_index);
void set_directml_device(int device_index);
int get_current_directml_device();
void* directml_malloc(size_t size);
void directml_free(void* ptr);
void directml_memcpy_host_to_device(void* dst, const void* src, size_t size);
void directml_memcpy_device_to_host(void* dst, const void* src, size_t size);
void directml_memcpy_device_to_device(void* dst, const void* src, size_t size);
void directml_memset(void* ptr, int value, size_t size);
void directml_synchronize();

// oneDNN functions
bool initialize_onednn();
int get_onednn_device_count();
DeviceProperties get_onednn_device_properties(int device_index);
void set_onednn_device(int device_index);
int get_current_onednn_device();
void* onednn_malloc(size_t size);
void onednn_free(void* ptr);
void onednn_memcpy_host_to_device(void* dst, const void* src, size_t size);
void onednn_memcpy_device_to_host(void* dst, const void* src, size_t size);
void onednn_memcpy_device_to_device(void* dst, const void* src, size_t size);
void onednn_memset(void* ptr, int value, size_t size);
void onednn_synchronize();

// MKL-DNN functions
bool initialize_mkldnn();
int get_mkldnn_device_count();
DeviceProperties get_mkldnn_device_properties(int device_index);
void set_mkldnn_device(int device_index);
int get_current_mkldnn_device();
void* mkldnn_malloc(size_t size);
void mkldnn_free(void* ptr);
void mkldnn_memcpy_host_to_device(void* dst, const void* src, size_t size);
void mkldnn_memcpy_device_to_host(void* dst, const void* src, size_t size);
void mkldnn_memcpy_device_to_device(void* dst, const void* src, size_t size);
void mkldnn_memset(void* ptr, int value, size_t size);
void mkldnn_synchronize();

// TensorRT functions
bool initialize_tensorrt();
int get_tensorrt_device_count();
DeviceProperties get_tensorrt_device_properties(int device_index);
void set_tensorrt_device(int device_index);
int get_current_tensorrt_device();
void* tensorrt_malloc(size_t size);
void tensorrt_free(void* ptr);
void tensorrt_memcpy_host_to_device(void* dst, const void* src, size_t size);
void tensorrt_memcpy_device_to_host(void* dst, const void* src, size_t size);
void tensorrt_memcpy_device_to_device(void* dst, const void* src, size_t size);
void tensorrt_memset(void* ptr, int value, size_t size);
void tensorrt_synchronize();

} // namespace hardware
} // namespace phynexus

#endif // PHYNEXUS_HARDWARE_BACKENDS_H
