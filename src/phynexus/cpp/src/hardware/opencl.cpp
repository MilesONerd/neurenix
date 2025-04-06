/**
 * OpenCL hardware backend implementation for Phynexus
 */

#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <unordered_map>

#include "hardware/opencl.h"
#include "error.h"
#include "tensor.h"
#include "memory.h"

namespace phynexus {
namespace hardware {

bool OpenCLBackend::is_available() {
    try {
        cl_uint platform_count = 0;
        cl_int err = clGetPlatformIDs(0, nullptr, &platform_count);
        
        if (err != CL_SUCCESS || platform_count == 0) {
            return false;
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

OpenCLBackend::OpenCLBackend() : initialized_(false), device_count_(0), platform_(nullptr) {
    try {
        initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize OpenCL backend: " << e.what() << std::endl;
    }
}

OpenCLBackend::~OpenCLBackend() {
    if (initialized_) {
        try {
            cleanup();
        } catch (const std::exception& e) {
            std::cerr << "Error during OpenCL backend cleanup: " << e.what() << std::endl;
        }
    }
}

void OpenCLBackend::initialize() {
    if (initialized_) return;
    
    if (!is_available()) {
        throw PhynexusError("OpenCL is not available on this system");
    }
    
    cl_uint platform_count = 0;
    cl_int err = clGetPlatformIDs(0, nullptr, &platform_count);
    
    if (err != CL_SUCCESS || platform_count == 0) {
        throw PhynexusError("No OpenCL platforms found");
    }
    
    std::vector<cl_platform_id> platforms(platform_count);
    err = clGetPlatformIDs(platform_count, platforms.data(), nullptr);
    
    if (err != CL_SUCCESS) {
        throw PhynexusError("Failed to get OpenCL platforms");
    }
    
    platform_ = platforms[0];
    
    cl_uint device_count = 0;
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, 0, nullptr, &device_count);
    
    if (err != CL_SUCCESS || device_count == 0) {
        throw PhynexusError("No OpenCL devices found");
    }
    
    std::vector<cl_device_id> all_devices(device_count);
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, device_count, all_devices.data(), nullptr);
    
    if (err != CL_SUCCESS) {
        throw PhynexusError("Failed to get OpenCL devices");
    }
    
    for (cl_uint i = 0; i < device_count; i++) {
        cl_device_type device_type;
        err = clGetDeviceInfo(all_devices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, nullptr);
        
        if (err != CL_SUCCESS) {
            continue;
        }
        
        if (device_type == CL_DEVICE_TYPE_GPU || device_type == CL_DEVICE_TYPE_ACCELERATOR) {
            devices_.push_back(all_devices[i]);
        }
    }
    
    device_count_ = devices_.size();
    
    if (device_count_ == 0) {
        throw PhynexusError("No suitable OpenCL devices found");
    }
    
    for (size_t i = 0; i < device_count_; i++) {
        cl_context_properties properties[] = {
            CL_CONTEXT_PLATFORM, (cl_context_properties)platform_,
            0
        };
        
        cl_int err;
        cl_context context = clCreateContext(properties, 1, &devices_[i], nullptr, nullptr, &err);
        
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to create OpenCL context for device " << i << std::endl;
            continue;
        }
        
        contexts_.push_back(context);
        
        cl_command_queue queue = clCreateCommandQueue(context, devices_[i], 0, &err);
        
        if (err != CL_SUCCESS) {
            clReleaseContext(context);
            contexts_.pop_back();
            std::cerr << "Failed to create OpenCL command queue for device " << i << std::endl;
            continue;
        }
        
        command_queues_.push_back(queue);
    }
    
    device_count_ = contexts_.size();
    
    if (device_count_ == 0) {
        throw PhynexusError("Failed to create any OpenCL contexts or command queues");
    }
    
    initialized_ = true;
}

void OpenCLBackend::cleanup() {
    if (!initialized_) return;
    
    for (auto& allocation : allocations_) {
        free_buffer(allocation.first);
    }
    allocations_.clear();
    
    for (auto& queue : command_queues_) {
        clReleaseCommandQueue(queue);
    }
    command_queues_.clear();
    
    for (auto& context : contexts_) {
        clReleaseContext(context);
    }
    contexts_.clear();
    
    devices_.clear();
    
    initialized_ = false;
    device_count_ = 0;
    platform_ = nullptr;
}

size_t OpenCLBackend::get_device_count() const {
    return device_count_;
}

std::string OpenCLBackend::get_device_name(size_t device_index) const {
    if (device_index >= device_count_) {
        throw PhynexusError("Invalid device index");
    }
    
    char device_name[256];
    cl_int err = clGetDeviceInfo(devices_[device_index], CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    
    if (err != CL_SUCCESS) {
        throw PhynexusError("Failed to get OpenCL device name");
    }
    
    return std::string(device_name);
}

size_t OpenCLBackend::get_device_memory(size_t device_index) const {
    if (device_index >= device_count_) {
        throw PhynexusError("Invalid device index");
    }
    
    cl_ulong device_memory;
    cl_int err = clGetDeviceInfo(devices_[device_index], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &device_memory, nullptr);
    
    if (err != CL_SUCCESS) {
        throw PhynexusError("Failed to get OpenCL device memory");
    }
    
    return static_cast<size_t>(device_memory);
}

void* OpenCLBackend::allocate(size_t size, size_t device_index) {
    if (!initialized_) {
        throw PhynexusError("OpenCL backend not initialized");
    }
    
    if (device_index >= device_count_) {
        throw PhynexusError("Invalid device index");
    }
    
    cl_int err;
    cl_mem buffer = clCreateBuffer(contexts_[device_index], CL_MEM_READ_WRITE, size, nullptr, &err);
    
    if (err != CL_SUCCESS) {
        throw PhynexusError("Failed to allocate OpenCL buffer");
    }
    
    OpenCLAllocation* allocation = new OpenCLAllocation{
        buffer,
        size,
        device_index
    };
    
    allocations_[allocation] = allocation;
    
    return allocation;
}

void OpenCLBackend::free(void* ptr) {
    if (!initialized_) {
        throw PhynexusError("OpenCL backend not initialized");
    }
    
    auto it = allocations_.find(ptr);
    if (it == allocations_.end()) {
        throw PhynexusError("Invalid OpenCL allocation pointer");
    }
    
    free_buffer(ptr);
    allocations_.erase(it);
}

void OpenCLBackend::free_buffer(void* ptr) {
    auto allocation = static_cast<OpenCLAllocation*>(ptr);
    
    cl_int err = clReleaseMemObject(allocation->buffer);
    if (err != CL_SUCCESS) {
        std::cerr << "Error releasing OpenCL buffer: " << err << std::endl;
    }
    
    delete allocation;
}

void OpenCLBackend::copy_host_to_device(const void* host_ptr, void* device_ptr, size_t size) {
    if (!initialized_) {
        throw PhynexusError("OpenCL backend not initialized");
    }
    
    auto it = allocations_.find(device_ptr);
    if (it == allocations_.end()) {
        throw PhynexusError("Invalid OpenCL allocation pointer");
    }
    
    auto allocation = static_cast<OpenCLAllocation*>(device_ptr);
    size_t device_index = allocation->device_index;
    
    if (device_index >= device_count_) {
        throw PhynexusError("Invalid device index in OpenCL allocation");
    }
    
    cl_int err = clEnqueueWriteBuffer(
        command_queues_[device_index],
        allocation->buffer,
        CL_TRUE,  // Blocking write
        0,
        size,
        host_ptr,
        0,
        nullptr,
        nullptr
    );
    
    if (err != CL_SUCCESS) {
        throw PhynexusError("Failed to copy data from host to OpenCL device");
    }
}

void OpenCLBackend::copy_device_to_host(const void* device_ptr, void* host_ptr, size_t size) {
    if (!initialized_) {
        throw PhynexusError("OpenCL backend not initialized");
    }
    
    auto it = allocations_.find(const_cast<void*>(device_ptr));
    if (it == allocations_.end()) {
        throw PhynexusError("Invalid OpenCL allocation pointer");
    }
    
    auto allocation = static_cast<const OpenCLAllocation*>(device_ptr);
    size_t device_index = allocation->device_index;
    
    if (device_index >= device_count_) {
        throw PhynexusError("Invalid device index in OpenCL allocation");
    }
    
    cl_int err = clEnqueueReadBuffer(
        command_queues_[device_index],
        allocation->buffer,
        CL_TRUE,  // Blocking read
        0,
        size,
        host_ptr,
        0,
        nullptr,
        nullptr
    );
    
    if (err != CL_SUCCESS) {
        throw PhynexusError("Failed to copy data from OpenCL device to host");
    }
}

void OpenCLBackend::copy_device_to_device(const void* src_ptr, void* dst_ptr, size_t size) {
    if (!initialized_) {
        throw PhynexusError("OpenCL backend not initialized");
    }
    
    auto src_it = allocations_.find(const_cast<void*>(src_ptr));
    if (src_it == allocations_.end()) {
        throw PhynexusError("Invalid source OpenCL allocation pointer");
    }
    
    auto dst_it = allocations_.find(dst_ptr);
    if (dst_it == allocations_.end()) {
        throw PhynexusError("Invalid destination OpenCL allocation pointer");
    }
    
    auto src_allocation = static_cast<const OpenCLAllocation*>(src_ptr);
    auto dst_allocation = static_cast<OpenCLAllocation*>(dst_ptr);
    
    size_t src_device_index = src_allocation->device_index;
    size_t dst_device_index = dst_allocation->device_index;
    
    if (src_device_index >= device_count_ || dst_device_index >= device_count_) {
        throw PhynexusError("Invalid device index in OpenCL allocation");
    }
    
    if (src_device_index == dst_device_index) {
        cl_int err = clEnqueueCopyBuffer(
            command_queues_[src_device_index],
            src_allocation->buffer,
            dst_allocation->buffer,
            0,
            0,
            size,
            0,
            nullptr,
            nullptr
        );
        
        if (err != CL_SUCCESS) {
            throw PhynexusError("Failed to copy data between OpenCL buffers on the same device");
        }
    } else {
        std::vector<char> host_buffer(size);
        
        copy_device_to_host(src_ptr, host_buffer.data(), size);
        
        copy_host_to_device(host_buffer.data(), dst_ptr, size);
    }
}

void OpenCLBackend::synchronize(size_t device_index) {
    if (!initialized_) {
        throw PhynexusError("OpenCL backend not initialized");
    }
    
    if (device_index >= device_count_) {
        throw PhynexusError("Invalid device index");
    }
    
    cl_int err = clFinish(command_queues_[device_index]);
    
    if (err != CL_SUCCESS) {
        throw PhynexusError("Failed to synchronize OpenCL device");
    }
}

} // namespace hardware
} // namespace phynexus
