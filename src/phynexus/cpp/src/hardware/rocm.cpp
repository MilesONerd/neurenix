/**
 * @file rocm.cpp
 * @brief ROCm implementation for the Phynexus engine
 * 
 * This file contains the ROCm implementation for the Phynexus engine.
 * 
 * @copyright Copyright (c) 2025 Neurenix
 * @license Apache License 2.0
 */

#include "phynexus/tensor.h"
#include <stdexcept>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <rocblas/rocblas.h>
#include <miopen/miopen.h>

namespace phynexus {
namespace hardware {

#define HIP_CHECK(cmd) \
    do { \
        hipError_t error = cmd; \
        if (error != hipSuccess) { \
            fprintf(stderr, "HIP error: '%s'(%d) at %s:%d\n", \
                    hipGetErrorString(error), error, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define ROCBLAS_CHECK(cmd) \
    do { \
        rocblas_status status = cmd; \
        if (status != rocblas_status_success) { \
            fprintf(stderr, "rocBLAS error: %d at %s:%d\n", \
                    status, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define MIOPEN_CHECK(cmd) \
    do { \
        miopenStatus_t status = cmd; \
        if (status != miopenStatusSuccess) { \
            fprintf(stderr, "MIOpen error: %d at %s:%d\n", \
                    status, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

static rocblas_handle g_rocblas_handle = nullptr;
static miopenHandle_t g_miopen_handle = nullptr;
static bool g_rocm_initialized = false;

/**
 * @brief Initialize ROCm
 * 
 * @return True if ROCm is available, false otherwise
 */
bool initialize_rocm() {
    if (g_rocm_initialized) {
        return true;
    }

    // Check if ROCm is available
    int device_count = 0;
    hipError_t error = hipGetDeviceCount(&device_count);
    
    if (error != hipSuccess || device_count == 0) {
        return false;
    }
    
    rocblas_status rocblas_status = rocblas_create_handle(&g_rocblas_handle);
    if (rocblas_status != rocblas_status_success) {
        return false;
    }
    
    miopenStatus_t miopen_status = miopenCreate(&g_miopen_handle);
    if (miopen_status != miopenStatusSuccess) {
        rocblas_destroy_handle(g_rocblas_handle);
        return false;
    }
    
    g_rocm_initialized = true;
    return true;
}

/**
 * @brief Get ROCm device count
 * 
 * @return Number of ROCm devices
 */
int get_rocm_device_count() {
    int device_count = 0;
    hipError_t error = hipGetDeviceCount(&device_count);
    
    if (error != hipSuccess) {
        return 0;
    }
    
    return device_count;
}

/**
 * @brief Get ROCm device properties
 * 
 * @param device_index Device index
 * @return Device properties
 */
DeviceProperties get_rocm_device_properties(int device_index) {
    DeviceProperties props;
    hipDeviceProp_t hip_props;
    
    hipError_t error = hipGetDeviceProperties(&hip_props, device_index);
    if (error != hipSuccess) {
        props.name = "Unknown ROCm Device";
        return props;
    }
    
    props.name = hip_props.name;
    props.total_memory = hip_props.totalGlobalMem;
    props.compute_capability_major = hip_props.major;
    props.compute_capability_minor = hip_props.minor;
    props.multi_processor_count = hip_props.multiProcessorCount;
    props.max_threads_per_block = hip_props.maxThreadsPerBlock;
    props.max_threads_per_multiprocessor = hip_props.maxThreadsPerMultiProcessor;
    props.warp_size = hip_props.warpSize;
    
    return props;
}

/**
 * @brief Set ROCm device
 * 
 * @param device_index Device index
 */
void set_rocm_device(int device_index) {
    HIP_CHECK(hipSetDevice(device_index));
}

/**
 * @brief Get current ROCm device
 * 
 * @return Current ROCm device index
 */
int get_current_rocm_device() {
    int device;
    HIP_CHECK(hipGetDevice(&device));
    return device;
}

/**
 * @brief Allocate memory on ROCm device
 * 
 * @param size Size in bytes
 * @return Pointer to allocated memory
 */
void* rocm_malloc(size_t size) {
    void* ptr = nullptr;
    HIP_CHECK(hipMalloc(&ptr, size));
    return ptr;
}

/**
 * @brief Free memory on ROCm device
 * 
 * @param ptr Pointer to memory
 */
void rocm_free(void* ptr) {
    if (ptr != nullptr) {
        HIP_CHECK(hipFree(ptr));
    }
}

/**
 * @brief Copy memory from host to device
 * 
 * @param dst Destination pointer (device)
 * @param src Source pointer (host)
 * @param size Size in bytes
 */
void rocm_memcpy_host_to_device(void* dst, const void* src, size_t size) {
    HIP_CHECK(hipMemcpy(dst, src, size, hipMemcpyHostToDevice));
}

/**
 * @brief Copy memory from device to host
 * 
 * @param dst Destination pointer (host)
 * @param src Source pointer (device)
 * @param size Size in bytes
 */
void rocm_memcpy_device_to_host(void* dst, const void* src, size_t size) {
    HIP_CHECK(hipMemcpy(dst, src, size, hipMemcpyDeviceToHost));
}

/**
 * @brief Copy memory from device to device
 * 
 * @param dst Destination pointer (device)
 * @param src Source pointer (device)
 * @param size Size in bytes
 */
void rocm_memcpy_device_to_device(void* dst, const void* src, size_t size) {
    HIP_CHECK(hipMemcpy(dst, src, size, hipMemcpyDeviceToDevice));
}

/**
 * @brief Set memory on ROCm device
 * 
 * @param ptr Pointer to memory
 * @param value Value to set
 * @param size Size in bytes
 */
void rocm_memset(void* ptr, int value, size_t size) {
    HIP_CHECK(hipMemset(ptr, value, size));
}

/**
 * @brief Launch ROCm kernel
 * 
 * @param kernel Kernel function
 * @param grid_dim Grid dimensions
 * @param block_dim Block dimensions
 * @param shared_mem_size Shared memory size
 * @param stream Stream
 * @param args Kernel arguments
 */
void rocm_launch_kernel(void* kernel, dim3 grid_dim, dim3 block_dim, size_t shared_mem_size, void* stream, void** args) {
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    
    HIP_CHECK(hipLaunchKernel(
        kernel,
        grid_dim,
        block_dim,
        args,
        shared_mem_size,
        hip_stream
    ));
}

/**
 * @brief Synchronize ROCm device
 */
void rocm_synchronize() {
    HIP_CHECK(hipDeviceSynchronize());
}

/**
 * @brief Create ROCm stream
 * 
 * @return ROCm stream
 */
void* rocm_create_stream() {
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    return static_cast<void*>(stream);
}

/**
 * @brief Destroy ROCm stream
 * 
 * @param stream ROCm stream
 */
void rocm_destroy_stream(void* stream) {
    if (stream != nullptr) {
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        HIP_CHECK(hipStreamDestroy(hip_stream));
    }
}

/**
 * @brief Synchronize ROCm stream
 * 
 * @param stream ROCm stream
 */
void rocm_stream_synchronize(void* stream) {
    if (stream != nullptr) {
        hipStream_t hip_stream = static_cast<hipStream_t>(stream);
        HIP_CHECK(hipStreamSynchronize(hip_stream));
    }
}

void rocm_cleanup() {
    if (g_rocm_initialized) {
        if (g_rocblas_handle != nullptr) {
            rocblas_destroy_handle(g_rocblas_handle);
            g_rocblas_handle = nullptr;
        }
        
        if (g_miopen_handle != nullptr) {
            miopenDestroy(g_miopen_handle);
            g_miopen_handle = nullptr;
        }
        
        g_rocm_initialized = false;
    }
}

class ROCmCleanup {
public:
    ~ROCmCleanup() {
        rocm_cleanup();
    }
};

static ROCmCleanup g_rocm_cleanup;

} // namespace hardware
} // namespace phynexus
