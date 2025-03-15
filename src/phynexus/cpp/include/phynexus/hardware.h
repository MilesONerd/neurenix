/**
 * @file hardware.h
 * @brief Hardware abstraction layer for the Phynexus engine
 * 
 * This file contains the hardware abstraction layer for the Phynexus engine,
 * providing a unified interface for different hardware platforms.
 * 
 * @copyright Copyright (c) 2025 Neurenix
 * @license Apache License 2.0
 */

#ifndef PHYNEXUS_HARDWARE_H
#define PHYNEXUS_HARDWARE_H

#include "tensor.h"
#include <functional>
#include <vector>

namespace phynexus {
namespace hardware {

/**
 * @brief Initialize hardware
 * 
 * @param device_type Device type
 * @return True if hardware is available, false otherwise
 */
bool initialize_hardware(DeviceType device_type);

/**
 * @brief Get device count
 * 
 * @param device_type Device type
 * @return Number of devices
 */
int get_device_count(DeviceType device_type);

/**
 * @brief Get device properties
 * 
 * @param device Device
 * @return Device properties
 */
DeviceProperties get_device_properties(const Device& device);

/**
 * @brief Set device
 * 
 * @param device Device
 */
void set_device(const Device& device);

/**
 * @brief Get current device
 * 
 * @param device_type Device type
 * @return Current device
 */
Device get_current_device(DeviceType device_type);

/**
 * @brief Allocate memory
 * 
 * @param size Size in bytes
 * @param device Device
 * @return Pointer to allocated memory
 */
void* allocate_memory(size_t size, const Device& device);

/**
 * @brief Free memory
 * 
 * @param ptr Pointer to memory
 * @param device Device
 */
void free_memory(void* ptr, const Device& device);

/**
 * @brief Copy memory
 * 
 * @param dst Destination pointer
 * @param src Source pointer
 * @param size Size in bytes
 * @param dst_device Destination device
 * @param src_device Source device
 */
void copy_memory(void* dst, const void* src, size_t size, const Device& dst_device, const Device& src_device);

/**
 * @brief Set memory
 * 
 * @param ptr Pointer to memory
 * @param value Value to set
 * @param size Size in bytes
 * @param device Device
 */
void set_memory(void* ptr, int value, size_t size, const Device& device);

/**
 * @brief Synchronize device
 * 
 * @param device Device
 */
void synchronize(const Device& device);

/**
 * @brief Parallel for loop
 * 
 * @param start Start index
 * @param end End index
 * @param num_threads Number of threads
 * @param device Device
 * @param func Function to execute
 */
void parallel_for(size_t start, size_t end, size_t num_threads, const Device& device, std::function<void(size_t, size_t)> func);

// CPU functions
bool initialize_cpu();
int get_cpu_device_count();
DeviceProperties get_cpu_device_properties();
void* cpu_malloc(size_t size);
void cpu_free(void* ptr);
void cpu_memcpy(void* dst, const void* src, size_t size);
void cpu_memset(void* ptr, int value, size_t size);
void cpu_parallel_for(size_t start, size_t end, size_t num_threads, std::function<void(size_t, size_t)> func);

// CUDA functions
bool initialize_cuda();
int get_cuda_device_count();
DeviceProperties get_cuda_device_properties(int device_index);
void set_cuda_device(int device_index);
int get_current_cuda_device();
void* cuda_malloc(size_t size);
void cuda_free(void* ptr);
void cuda_memcpy_host_to_device(void* dst, const void* src, size_t size);
void cuda_memcpy_device_to_host(void* dst, const void* src, size_t size);
void cuda_memcpy_device_to_device(void* dst, const void* src, size_t size);
void cuda_memset(void* ptr, int value, size_t size);
void cuda_launch_kernel(void* kernel, dim3 grid_dim, dim3 block_dim, size_t shared_mem_size, void* stream, void** args);
void cuda_synchronize();
void* cuda_create_stream();
void cuda_destroy_stream(void* stream);
void cuda_stream_synchronize(void* stream);

// ROCm functions
bool initialize_rocm();
int get_rocm_device_count();
DeviceProperties get_rocm_device_properties(int device_index);
void set_rocm_device(int device_index);
int get_current_rocm_device();
void* rocm_malloc(size_t size);
void rocm_free(void* ptr);
void rocm_memcpy_host_to_device(void* dst, const void* src, size_t size);
void rocm_memcpy_device_to_host(void* dst, const void* src, size_t size);
void rocm_memcpy_device_to_device(void* dst, const void* src, size_t size);
void rocm_memset(void* ptr, int value, size_t size);
void rocm_launch_kernel(void* kernel, dim3 grid_dim, dim3 block_dim, size_t shared_mem_size, void* stream, void** args);
void rocm_synchronize();
void* rocm_create_stream();
void rocm_destroy_stream(void* stream);
void rocm_stream_synchronize(void* stream);

// WebGPU functions
bool initialize_webgpu();
int get_webgpu_device_count();
DeviceProperties get_webgpu_device_properties(int device_index);
void set_webgpu_device(int device_index);
int get_current_webgpu_device();
void* webgpu_malloc(size_t size);
void webgpu_free(void* ptr);
void webgpu_memcpy_host_to_device(void* dst, const void* src, size_t size);
void webgpu_memcpy_device_to_host(void* dst, const void* src, size_t size);
void webgpu_memcpy_device_to_device(void* dst, const void* src, size_t size);
void webgpu_memset(void* ptr, int value, size_t size);
void webgpu_launch_compute_shader(void* shader, const std::vector<size_t>& workgroup_count, 
                                 const std::vector<size_t>& workgroup_size, void** args);
void webgpu_synchronize();
void* webgpu_create_command_encoder();
void webgpu_destroy_command_encoder(void* encoder);
void webgpu_submit_command_encoder(void* encoder);

// TPU functions
bool initialize_tpu();
int get_tpu_device_count();
DeviceProperties get_tpu_device_properties(int device_index);
void set_tpu_device(int device_index);
int get_current_tpu_device();
void* tpu_malloc(size_t size);
void tpu_free(void* ptr);
void tpu_memcpy_host_to_device(void* dst, const void* src, size_t size);
void tpu_memcpy_device_to_host(void* dst, const void* src, size_t size);
void tpu_memcpy_device_to_device(void* dst, const void* src, size_t size);
void tpu_memset(void* ptr, int value, size_t size);
void tpu_launch_kernel(void* kernel, dim3 grid_dim, dim3 block_dim, size_t shared_mem_size, void* stream, void** args);
void tpu_synchronize();
void* tpu_create_stream();
void tpu_destroy_stream(void* stream);
void tpu_stream_synchronize(void* stream);

} // namespace hardware
} // namespace phynexus

#endif // PHYNEXUS_HARDWARE_H
