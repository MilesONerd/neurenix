/**
 * @file cuda.cpp
 * @brief CUDA implementation for the Phynexus engine
 * 
 * This file contains the CUDA implementation for the Phynexus engine.
 * 
 * @copyright Copyright (c) 2025 Neurenix
 * @license Apache License 2.0
 */

#include "phynexus/tensor.h"
#include <stdexcept>

namespace phynexus {
namespace hardware {

/**
 * @brief Initialize CUDA
 * 
 * @return True if CUDA is available, false otherwise
 */
bool initialize_cuda() {
    // Check if CUDA is available
    #ifdef PHYNEXUS_WITH_CUDA
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        return false;
    }
    return deviceCount > 0;
    #else
    return false;
    #endif
}

/**
 * @brief Get CUDA device count
 * 
 * @return Number of CUDA devices
 */
int get_cuda_device_count() {
    // Get CUDA device count
    #ifdef PHYNEXUS_WITH_CUDA
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        return 0;
    }
    return deviceCount;
    #else
    return 0;
    #endif
}

/**
 * @brief Get CUDA device properties
 * 
 * @param device_index Device index
 * @return Device properties
 */
DeviceProperties get_cuda_device_properties(int device_index) {
    // Get CUDA device properties
    DeviceProperties props;
    props.name = "CUDA Device";
    props.total_memory = 0;
    props.compute_capability_major = 0;
    props.compute_capability_minor = 0;
    props.multi_processor_count = 0;
    props.max_threads_per_block = 0;
    props.max_threads_per_multiprocessor = 0;
    props.warp_size = 0;
    
    #ifdef PHYNEXUS_WITH_CUDA
    cudaDeviceProp cudaProps;
    cudaError_t error = cudaGetDeviceProperties(&cudaProps, device_index);
    if (error != cudaSuccess) {
        return props;
    }
    
    props.name = cudaProps.name;
    props.total_memory = cudaProps.totalGlobalMem;
    props.compute_capability_major = cudaProps.major;
    props.compute_capability_minor = cudaProps.minor;
    props.multi_processor_count = cudaProps.multiProcessorCount;
    props.max_threads_per_block = cudaProps.maxThreadsPerBlock;
    props.max_threads_per_multiprocessor = cudaProps.maxThreadsPerMultiProcessor;
    props.warp_size = cudaProps.warpSize;
    #endif
    
    return props;
}

/**
 * @brief Set CUDA device
 * 
 * @param device_index Device index
 */
void set_cuda_device(int device_index) {
    // Set CUDA device
    #ifdef PHYNEXUS_WITH_CUDA
    cudaError_t error = cudaSetDevice(device_index);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to set CUDA device: " + std::string(cudaGetErrorString(error)));
    }
    #endif
}

/**
 * @brief Get current CUDA device
 * 
 * @return Current CUDA device index
 */
int get_current_cuda_device() {
    // Get current CUDA device
    #ifdef PHYNEXUS_WITH_CUDA
    int device = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to get current CUDA device: " + std::string(cudaGetErrorString(error)));
    }
    return device;
    #else
    return 0;
    #endif
}

/**
 * @brief Allocate memory on CUDA device
 * 
 * @param size Size in bytes
 * @return Pointer to allocated memory
 */
void* cuda_malloc(size_t size) {
    // Allocate memory on CUDA device
    #ifdef PHYNEXUS_WITH_CUDA
    void* ptr = nullptr;
    cudaError_t error = cudaMalloc(&ptr, size);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to allocate CUDA memory: " + std::string(cudaGetErrorString(error)));
    }
    return ptr;
    #else
    return nullptr;
    #endif
}

/**
 * @brief Free memory on CUDA device
 * 
 * @param ptr Pointer to memory
 */
void cuda_free(void* ptr) {
    // Free memory on CUDA device
    #ifdef PHYNEXUS_WITH_CUDA
    if (ptr != nullptr) {
        cudaError_t error = cudaFree(ptr);
        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to free CUDA memory: " + std::string(cudaGetErrorString(error)));
        }
    }
    #endif
}

/**
 * @brief Copy memory from host to device
 * 
 * @param dst Destination pointer (device)
 * @param src Source pointer (host)
 * @param size Size in bytes
 */
void cuda_memcpy_host_to_device(void* dst, const void* src, size_t size) {
    // Copy memory from host to device
    #ifdef PHYNEXUS_WITH_CUDA
    if (dst == nullptr || src == nullptr) {
        throw std::runtime_error("Invalid pointer for CUDA host to device copy");
    }
    
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to copy memory from host to device: " + std::string(cudaGetErrorString(error)));
    }
    #endif
}

/**
 * @brief Copy memory from device to host
 * 
 * @param dst Destination pointer (host)
 * @param src Source pointer (device)
 * @param size Size in bytes
 */
void cuda_memcpy_device_to_host(void* dst, const void* src, size_t size) {
    // Copy memory from device to host
    #ifdef PHYNEXUS_WITH_CUDA
    if (dst == nullptr || src == nullptr) {
        throw std::runtime_error("Invalid pointer for CUDA device to host copy");
    }
    
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to copy memory from device to host: " + std::string(cudaGetErrorString(error)));
    }
    #endif
}

/**
 * @brief Copy memory from device to device
 * 
 * @param dst Destination pointer (device)
 * @param src Source pointer (device)
 * @param size Size in bytes
 */
void cuda_memcpy_device_to_device(void* dst, const void* src, size_t size) {
    // Copy memory from device to device
    #ifdef PHYNEXUS_WITH_CUDA
    if (dst == nullptr || src == nullptr) {
        throw std::runtime_error("Invalid pointer for CUDA device to device copy");
    }
    
    cudaError_t error = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to copy memory from device to device: " + std::string(cudaGetErrorString(error)));
    }
    #endif
}

/**
 * @brief Set memory on CUDA device
 * 
 * @param ptr Pointer to memory
 * @param value Value to set
 * @param size Size in bytes
 */
void cuda_memset(void* ptr, int value, size_t size) {
    // Set memory on CUDA device
    #ifdef PHYNEXUS_WITH_CUDA
    if (ptr == nullptr) {
        throw std::runtime_error("Invalid pointer for CUDA memset");
    }
    
    cudaError_t error = cudaMemset(ptr, value, size);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to set memory on CUDA device: " + std::string(cudaGetErrorString(error)));
    }
    #endif
}

/**
 * @brief Launch CUDA kernel
 * 
 * @param kernel Kernel function
 * @param grid_dim Grid dimensions
 * @param block_dim Block dimensions
 * @param shared_mem_size Shared memory size
 * @param stream Stream
 * @param args Kernel arguments
 */
void cuda_launch_kernel(void* kernel, dim3 grid_dim, dim3 block_dim, size_t shared_mem_size, void* stream, void** args) {
    // Launch CUDA kernel
    #ifdef PHYNEXUS_WITH_CUDA
    if (kernel == nullptr) {
        throw std::runtime_error("Invalid kernel pointer for CUDA kernel launch");
    }
    
    cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
    cudaError_t error = cudaLaunchKernel(kernel, grid_dim, block_dim, args, shared_mem_size, cudaStream);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to launch CUDA kernel: " + std::string(cudaGetErrorString(error)));
    }
    #endif
}

/**
 * @brief Synchronize CUDA device
 */
void cuda_synchronize() {
    // Synchronize CUDA device
    #ifdef PHYNEXUS_WITH_CUDA
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to synchronize CUDA device: " + std::string(cudaGetErrorString(error)));
    }
    #endif
}

/**
 * @brief Create CUDA stream
 * 
 * @return CUDA stream
 */
void* cuda_create_stream() {
    // Create CUDA stream
    #ifdef PHYNEXUS_WITH_CUDA
    cudaStream_t stream = nullptr;
    cudaError_t error = cudaStreamCreate(&stream);
    if (error != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream: " + std::string(cudaGetErrorString(error)));
    }
    return static_cast<void*>(stream);
    #else
    return nullptr;
    #endif
}

/**
 * @brief Destroy CUDA stream
 * 
 * @param stream CUDA stream
 */
void cuda_destroy_stream(void* stream) {
    // Destroy CUDA stream
    #ifdef PHYNEXUS_WITH_CUDA
    if (stream != nullptr) {
        cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
        cudaError_t error = cudaStreamDestroy(cudaStream);
        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to destroy CUDA stream: " + std::string(cudaGetErrorString(error)));
        }
    }
    #endif
}

/**
 * @brief Synchronize CUDA stream
 * 
 * @param stream CUDA stream
 */
void cuda_stream_synchronize(void* stream) {
    // Synchronize CUDA stream
    #ifdef PHYNEXUS_WITH_CUDA
    if (stream != nullptr) {
        cudaStream_t cudaStream = static_cast<cudaStream_t>(stream);
        cudaError_t error = cudaStreamSynchronize(cudaStream);
        if (error != cudaSuccess) {
            throw std::runtime_error("Failed to synchronize CUDA stream: " + std::string(cudaGetErrorString(error)));
        }
    }
    #endif
}

} // namespace hardware
} // namespace phynexus
