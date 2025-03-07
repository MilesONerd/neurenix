/**
 * @file tensor.cpp
 * @brief Implementation of the Tensor class
 * 
 * This file contains the implementation of the Tensor class for the Phynexus engine.
 * 
 * @copyright Copyright (c) 2025 Neurenix
 * @license Apache License 2.0
 */

#include "phynexus/tensor.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <sstream>
#include <cstring>

namespace phynexus {

// Constructor implementations

Tensor::Tensor() : dtype_(DataType::Float32), device_(Device::CPU), requires_grad_(false) {
    shape_ = {};
    strides_ = {};
    data_ = nullptr;
    size_ = 0;
}

Tensor::Tensor(const std::vector<size_t>& shape, DataType dtype, const Device& device)
    : shape_(shape), dtype_(dtype), device_(device), requires_grad_(false) {
    
    // Calculate size and strides
    size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    strides_ = calculate_strides(shape);
    
    // Allocate memory
    data_ = allocate_memory(size_ * get_dtype_size(dtype));
}

Tensor::Tensor(const std::vector<size_t>& shape, void* data, DataType dtype, const Device& device)
    : shape_(shape), dtype_(dtype), device_(device), requires_grad_(false) {
    
    // Calculate size and strides
    size_ = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_t>());
    strides_ = calculate_strides(shape);
    
    // Allocate memory and copy data
    data_ = allocate_memory(size_ * get_dtype_size(dtype));
    std::memcpy(data_, data, size_ * get_dtype_size(dtype));
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_), strides_(other.strides_), size_(other.size_),
      dtype_(other.dtype_), device_(other.device_), requires_grad_(other.requires_grad_) {
    
    // Allocate memory and copy data
    data_ = allocate_memory(size_ * get_dtype_size(dtype_));
    std::memcpy(data_, other.data_, size_ * get_dtype_size(dtype_));
    
    // Copy gradient if needed
    if (other.grad_) {
        grad_ = std::make_shared<Tensor>(*other.grad_);
    }
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)), strides_(std::move(other.strides_)), size_(other.size_),
      dtype_(other.dtype_), device_(other.device_), requires_grad_(other.requires_grad_),
      data_(other.data_), grad_(std::move(other.grad_)) {
    
    // Reset other
    other.data_ = nullptr;
    other.size_ = 0;
}

Tensor::~Tensor() {
    free_memory();
}

// Assignment operators

Tensor& Tensor::operator=(const Tensor& other) {
    if (this != &other) {
        // Free current memory
        free_memory();
        
        // Copy properties
        shape_ = other.shape_;
        strides_ = other.strides_;
        size_ = other.size_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        requires_grad_ = other.requires_grad_;
        
        // Allocate memory and copy data
        data_ = allocate_memory(size_ * get_dtype_size(dtype_));
        std::memcpy(data_, other.data_, size_ * get_dtype_size(dtype_));
        
        // Copy gradient if needed
        if (other.grad_) {
            grad_ = std::make_shared<Tensor>(*other.grad_);
        } else {
            grad_ = nullptr;
        }
    }
    return *this;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        // Free current memory
        free_memory();
        
        // Move properties
        shape_ = std::move(other.shape_);
        strides_ = std::move(other.strides_);
        size_ = other.size_;
        dtype_ = other.dtype_;
        device_ = other.device_;
        requires_grad_ = other.requires_grad_;
        data_ = other.data_;
        grad_ = std::move(other.grad_);
        
        // Reset other
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

// Memory management

void* Tensor::allocate_memory(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    
    // Allocate memory based on device
    if (device_.type() == DeviceType::CPU) {
        return std::malloc(size);
    } else if (device_.type() == DeviceType::CUDA) {
        // CUDA memory allocation
        void* ptr = nullptr;
        // cudaMalloc(&ptr, size);
        return ptr;
    } else if (device_.type() == DeviceType::ROCM) {
        // ROCm memory allocation
        void* ptr = nullptr;
        // hipMalloc(&ptr, size);
        return ptr;
    } else if (device_.type() == DeviceType::WEBGPU) {
        // WebGPU memory allocation
        void* ptr = nullptr;
        // webgpu_malloc(&ptr, size);
        return ptr;
    } else {
        throw std::runtime_error("Unsupported device type");
    }
}

void Tensor::free_memory() {
    if (data_ == nullptr) {
        return;
    }
    
    // Free memory based on device
    if (device_.type() == DeviceType::CPU) {
        std::free(data_);
    } else if (device_.type() == DeviceType::CUDA) {
        // CUDA memory deallocation
        // cudaFree(data_);
    } else if (device_.type() == DeviceType::ROCM) {
        // ROCm memory deallocation
        // hipFree(data_);
    } else if (device_.type() == DeviceType::WEBGPU) {
        // WebGPU memory deallocation
        // webgpu_free(data_);
    } else {
        throw std::runtime_error("Unsupported device type");
    }
    
    data_ = nullptr;
}

// Utility functions

std::vector<size_t> Tensor::calculate_strides(const std::vector<size_t>& shape) {
    std::vector<size_t> strides(shape.size());
    size_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

size_t Tensor::get_dtype_size(DataType dtype) {
    switch (dtype) {
        case DataType::Float32:
            return sizeof(float);
        case DataType::Float64:
            return sizeof(double);
        case DataType::Int32:
            return sizeof(int32_t);
        case DataType::Int64:
            return sizeof(int64_t);
        case DataType::Bool:
            return sizeof(bool);
        default:
            throw std::runtime_error("Unsupported data type");
    }
}

// Accessors

const std::vector<size_t>& Tensor::shape() const {
    return shape_;
}

const std::vector<size_t>& Tensor::strides() const {
    return strides_;
}

size_t Tensor::size() const {
    return size_;
}

size_t Tensor::ndim() const {
    return shape_.size();
}

DataType Tensor::dtype() const {
    return dtype_;
}

const Device& Tensor::device() const {
    return device_;
}

bool Tensor::requires_grad() const {
    return requires_grad_;
}

void* Tensor::data() {
    return data_;
}

const void* Tensor::data() const {
    return data_;
}

std::shared_ptr<Tensor> Tensor::grad() {
    return grad_;
}

const std::shared_ptr<Tensor> Tensor::grad() const {
    return grad_;
}

// Modifiers

Tensor& Tensor::requires_grad_(bool requires_grad) {
    requires_grad_ = requires_grad;
    if (requires_grad && !grad_) {
        grad_ = std::make_shared<Tensor>(shape_, dtype_, device_);
    }
    return *this;
}

Tensor Tensor::to(const Device& device) const {
    if (device == device_) {
        return *this;
    }
    
    // Create new tensor on target device
    Tensor result(shape_, dtype_, device);
    
    // Copy data to target device
    if (device_.type() == DeviceType::CPU && device.type() == DeviceType::CUDA) {
        // CPU to CUDA
        // cudaMemcpy(result.data_, data_, size_ * get_dtype_size(dtype_), cudaMemcpyHostToDevice);
    } else if (device_.type() == DeviceType::CUDA && device.type() == DeviceType::CPU) {
        // CUDA to CPU
        // cudaMemcpy(result.data_, data_, size_ * get_dtype_size(dtype_), cudaMemcpyDeviceToHost);
    } else if (device_.type() == DeviceType::CPU && device.type() == DeviceType::ROCM) {
        // CPU to ROCm
        // hipMemcpy(result.data_, data_, size_ * get_dtype_size(dtype_), hipMemcpyHostToDevice);
    } else if (device_.type() == DeviceType::ROCM && device.type() == DeviceType::CPU) {
        // ROCm to CPU
        // hipMemcpy(result.data_, data_, size_ * get_dtype_size(dtype_), hipMemcpyDeviceToHost);
    } else if (device_.type() == DeviceType::CPU && device.type() == DeviceType::WEBGPU) {
        // CPU to WebGPU
        // webgpu_memcpy(result.data_, data_, size_ * get_dtype_size(dtype_), WEBGPU_MEMCPY_HOST_TO_DEVICE);
    } else if (device_.type() == DeviceType::WEBGPU && device.type() == DeviceType::CPU) {
        // WebGPU to CPU
        // webgpu_memcpy(result.data_, data_, size_ * get_dtype_size(dtype_), WEBGPU_MEMCPY_DEVICE_TO_HOST);
    } else {
        // Other device combinations
        // Copy to CPU first, then to target device
        Tensor cpu_tensor = to(Device::CPU);
        return cpu_tensor.to(device);
    }
    
    return result;
}

Tensor& Tensor::to_(const Device& device) {
    if (device == device_) {
        return *this;
    }
    
    // Create temporary tensor on target device
    Tensor result = to(device);
    
    // Swap contents
    std::swap(data_, result.data_);
    device_ = device;
    
    return *this;
}

// Factory methods

Tensor Tensor::zeros(const std::vector<size_t>& shape, DataType dtype, const Device& device) {
    Tensor result(shape, dtype, device);
    
    // Fill with zeros
    size_t size = result.size_ * get_dtype_size(dtype);
    if (device.type() == DeviceType::CPU) {
        std::memset(result.data_, 0, size);
    } else if (device.type() == DeviceType::CUDA) {
        // cudaMemset(result.data_, 0, size);
    } else if (device.type() == DeviceType::ROCM) {
        // hipMemset(result.data_, 0, size);
    } else if (device.type() == DeviceType::WEBGPU) {
        // webgpu_memset(result.data_, 0, size);
    }
    
    return result;
}

Tensor Tensor::ones(const std::vector<size_t>& shape, DataType dtype, const Device& device) {
    Tensor result(shape, dtype, device);
    
    // Fill with ones
    if (device.type() == DeviceType::CPU) {
        if (dtype == DataType::Float32) {
            float* data = static_cast<float*>(result.data_);
            std::fill(data, data + result.size_, 1.0f);
        } else if (dtype == DataType::Float64) {
            double* data = static_cast<double*>(result.data_);
            std::fill(data, data + result.size_, 1.0);
        } else if (dtype == DataType::Int32) {
            int32_t* data = static_cast<int32_t*>(result.data_);
            std::fill(data, data + result.size_, 1);
        } else if (dtype == DataType::Int64) {
            int64_t* data = static_cast<int64_t*>(result.data_);
            std::fill(data, data + result.size_, 1);
        } else if (dtype == DataType::Bool) {
            bool* data = static_cast<bool*>(result.data_);
            std::fill(data, data + result.size_, true);
        }
    } else {
        // For GPU devices, create on CPU and copy
        Tensor cpu_result = ones(shape, dtype, Device::CPU);
        result = cpu_result.to(device);
    }
    
    return result;
}

Tensor Tensor::full(const std::vector<size_t>& shape, float value, DataType dtype, const Device& device) {
    Tensor result(shape, dtype, device);
    
    // Fill with value
    if (device.type() == DeviceType::CPU) {
        if (dtype == DataType::Float32) {
            float* data = static_cast<float*>(result.data_);
            std::fill(data, data + result.size_, static_cast<float>(value));
        } else if (dtype == DataType::Float64) {
            double* data = static_cast<double*>(result.data_);
            std::fill(data, data + result.size_, static_cast<double>(value));
        } else if (dtype == DataType::Int32) {
            int32_t* data = static_cast<int32_t*>(result.data_);
            std::fill(data, data + result.size_, static_cast<int32_t>(value));
        } else if (dtype == DataType::Int64) {
            int64_t* data = static_cast<int64_t*>(result.data_);
            std::fill(data, data + result.size_, static_cast<int64_t>(value));
        } else if (dtype == DataType::Bool) {
            bool* data = static_cast<bool*>(result.data_);
            std::fill(data, data + result.size_, value != 0.0f);
        }
    } else {
        // For GPU devices, create on CPU and copy
        Tensor cpu_result = full(shape, value, dtype, Device::CPU);
        result = cpu_result.to(device);
    }
    
    return result;
}

Tensor Tensor::eye(size_t n, DataType dtype, const Device& device) {
    Tensor result = zeros({n, n}, dtype, device);
    
    // Set diagonal to ones
    if (device.type() == DeviceType::CPU) {
        if (dtype == DataType::Float32) {
            float* data = static_cast<float*>(result.data_);
            for (size_t i = 0; i < n; ++i) {
                data[i * n + i] = 1.0f;
            }
        } else if (dtype == DataType::Float64) {
            double* data = static_cast<double*>(result.data_);
            for (size_t i = 0; i < n; ++i) {
                data[i * n + i] = 1.0;
            }
        } else if (dtype == DataType::Int32) {
            int32_t* data = static_cast<int32_t*>(result.data_);
            for (size_t i = 0; i < n; ++i) {
                data[i * n + i] = 1;
            }
        } else if (dtype == DataType::Int64) {
            int64_t* data = static_cast<int64_t*>(result.data_);
            for (size_t i = 0; i < n; ++i) {
                data[i * n + i] = 1;
            }
        } else if (dtype == DataType::Bool) {
            bool* data = static_cast<bool*>(result.data_);
            for (size_t i = 0; i < n; ++i) {
                data[i * n + i] = true;
            }
        }
    } else {
        // For GPU devices, create on CPU and copy
        Tensor cpu_result = eye(n, dtype, Device::CPU);
        result = cpu_result.to(device);
    }
    
    return result;
}

Tensor Tensor::randn(const std::vector<size_t>& shape, DataType dtype, const Device& device) {
    Tensor result(shape, dtype, device);
    
    // Fill with random values from normal distribution
    if (device.type() == DeviceType::CPU) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        if (dtype == DataType::Float32) {
            float* data = static_cast<float*>(result.data_);
            for (size_t i = 0; i < result.size_; ++i) {
                data[i] = dist(gen);
            }
        } else if (dtype == DataType::Float64) {
            double* data = static_cast<double*>(result.data_);
            for (size_t i = 0; i < result.size_; ++i) {
                data[i] = static_cast<double>(dist(gen));
            }
        } else {
            // For non-floating point types, create float tensor and cast
            Tensor float_result = randn(shape, DataType::Float32, device);
            result = float_result.to(dtype);
        }
    } else {
        // For GPU devices, create on CPU and copy
        Tensor cpu_result = randn(shape, dtype, Device::CPU);
        result = cpu_result.to(device);
    }
    
    return result;
}

Tensor Tensor::rand(const std::vector<size_t>& shape, DataType dtype, const Device& device) {
    Tensor result(shape, dtype, device);
    
    // Fill with random values from uniform distribution
    if (device.type() == DeviceType::CPU) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        
        if (dtype == DataType::Float32) {
            float* data = static_cast<float*>(result.data_);
            for (size_t i = 0; i < result.size_; ++i) {
                data[i] = dist(gen);
            }
        } else if (dtype == DataType::Float64) {
            double* data = static_cast<double*>(result.data_);
            for (size_t i = 0; i < result.size_; ++i) {
                data[i] = static_cast<double>(dist(gen));
            }
        } else {
            // For non-floating point types, create float tensor and cast
            Tensor float_result = rand(shape, DataType::Float32, device);
            result = float_result.to(dtype);
        }
    } else {
        // For GPU devices, create on CPU and copy
        Tensor cpu_result = rand(shape, dtype, Device::CPU);
        result = cpu_result.to(device);
    }
    
    return result;
}

// Indexing and slicing

Tensor Tensor::operator[](size_t index) const {
    if (shape_.empty()) {
        throw std::out_of_range("Cannot index into a scalar tensor");
    }
    
    if (index >= shape_[0]) {
        throw std::out_of_range("Index out of range");
    }
    
    // Calculate new shape and strides
    std::vector<size_t> new_shape(shape_.begin() + 1, shape_.end());
    std::vector<size_t> new_strides(strides_.begin() + 1, strides_.end());
    
    // Calculate new size
    size_t new_size = new_shape.empty() ? 1 : std::accumulate(new_shape.begin(), new_shape.end(), 1, std::multiplies<size_t>());
    
    // Calculate data offset
    size_t offset = index * strides_[0];
    
    // Create new tensor
    Tensor result;
    result.shape_ = new_shape;
    result.strides_ = new_strides;
    result.size_ = new_size;
    result.dtype_ = dtype_;
    result.device_ = device_;
    result.requires_grad_ = requires_grad_;
    
    // Set data pointer with offset
    if (device_.type() == DeviceType::CPU) {
        result.data_ = static_cast<char*>(data_) + offset * get_dtype_size(dtype_);
    } else {
        // For GPU devices, need to create a new tensor and copy the data
        // This is a simplified implementation
        result = to(Device::CPU)[index].to(device_);
    }
    
    return result;
}

// String representation

std::string Tensor::to_string() const {
    std::ostringstream oss;
    
    oss << "Tensor(";
    
    // Print shape
    oss << "[";
    for (size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i];
        if (i < shape_.size() - 1) {
            oss << ", ";
        }
    }
    oss << "], ";
    
    // Print dtype
    oss << "dtype=" << dtype_to_string(dtype_) << ", ";
    
    // Print device
    oss << "device=" << device_.to_string() << ", ";
    
    // Print requires_grad
    oss << "requires_grad=" << (requires_grad_ ? "true" : "false") << ")";
    
    return oss.str();
}

std::string Tensor::dtype_to_string(DataType dtype) {
    switch (dtype) {
        case DataType::Float32:
            return "float32";
        case DataType::Float64:
            return "float64";
        case DataType::Int32:
            return "int32";
        case DataType::Int64:
            return "int64";
        case DataType::Bool:
            return "bool";
        default:
            return "unknown";
    }
}

// Type conversion

Tensor Tensor::to(DataType dtype) const {
    if (dtype == dtype_) {
        return *this;
    }
    
    // Create new tensor with target dtype
    Tensor result(shape_, dtype, device_);
    
    // Convert data
    if (device_.type() == DeviceType::CPU) {
        // CPU implementation
        for (size_t i = 0; i < size_; ++i) {
            // Calculate flat index
            size_t flat_index = i;
            
            // Convert data based on source and target dtypes
            if (dtype_ == DataType::Float32 && dtype == DataType::Float64) {
                float* src = static_cast<float*>(data_);
                double* dst = static_cast<double*>(result.data_);
                dst[i] = static_cast<double>(src[i]);
            } else if (dtype_ == DataType::Float64 && dtype == DataType::Float32) {
                double* src = static_cast<double*>(data_);
                float* dst = static_cast<float*>(result.data_);
                dst[i] = static_cast<float>(src[i]);
            } else if (dtype_ == DataType::Float32 && dtype == DataType::Int32) {
                float* src = static_cast<float*>(data_);
                int32_t* dst = static_cast<int32_t*>(result.data_);
                dst[i] = static_cast<int32_t>(src[i]);
            } else if (dtype_ == DataType::Int32 && dtype == DataType::Float32) {
                int32_t* src = static_cast<int32_t*>(data_);
                float* dst = static_cast<float*>(result.data_);
                dst[i] = static_cast<float>(src[i]);
            } else if (dtype_ == DataType::Float32 && dtype == DataType::Int64) {
                float* src = static_cast<float*>(data_);
                int64_t* dst = static_cast<int64_t*>(result.data_);
                dst[i] = static_cast<int64_t>(src[i]);
            } else if (dtype_ == DataType::Int64 && dtype == DataType::Float32) {
                int64_t* src = static_cast<int64_t*>(data_);
                float* dst = static_cast<float*>(result.data_);
                dst[i] = static_cast<float>(src[i]);
            } else if (dtype_ == DataType::Float64 && dtype == DataType::Int32) {
                double* src = static_cast<double*>(data_);
                int32_t* dst = static_cast<int32_t*>(result.data_);
                dst[i] = static_cast<int32_t>(src[i]);
            } else if (dtype_ == DataType::Int32 && dtype == DataType::Float64) {
                int32_t* src = static_cast<int32_t*>(data_);
                double* dst = static_cast<double*>(result.data_);
                dst[i] = static_cast<double>(src[i]);
            } else if (dtype_ == DataType::Float64 && dtype == DataType::Int64) {
                double* src = static_cast<double*>(data_);
                int64_t* dst = static_cast<int64_t*>(result.data_);
                dst[i] = static_cast<int64_t>(src[i]);
            } else if (dtype_ == DataType::Int64 && dtype == DataType::Float64) {
                int64_t* src = static_cast<int64_t*>(data_);
                double* dst = static_cast<double*>(result.data_);
                dst[i] = static_cast<double>(src[i]);
            } else if (dtype_ == DataType::Int32 && dtype == DataType::Int64) {
                int32_t* src = static_cast<int32_t*>(data_);
                int64_t* dst = static_cast<int64_t*>(result.data_);
                dst[i] = static_cast<int64_t>(src[i]);
            } else if (dtype_ == DataType::Int64 && dtype == DataType::Int32) {
                int64_t* src = static_cast<int64_t*>(data_);
                int32_t* dst = static_cast<int32_t*>(result.data_);
                dst[i] = static_cast<int32_t>(src[i]);
            } else if (dtype_ == DataType::Float32 && dtype == DataType::Bool) {
                float* src = static_cast<float*>(data_);
                bool* dst = static_cast<bool*>(result.data_);
                dst[i] = src[i] != 0.0f;
            } else if (dtype_ == DataType::Bool && dtype == DataType::Float32) {
                bool* src = static_cast<bool*>(data_);
                float* dst = static_cast<float*>(result.data_);
                dst[i] = src[i] ? 1.0f : 0.0f;
            } else if (dtype_ == DataType::Float64 && dtype == DataType::Bool) {
                double* src = static_cast<double*>(data_);
                bool* dst = static_cast<bool*>(result.data_);
                dst[i] = src[i] != 0.0;
            } else if (dtype_ == DataType::Bool && dtype == DataType::Float64) {
                bool* src = static_cast<bool*>(data_);
                double* dst = static_cast<double*>(result.data_);
                dst[i] = src[i] ? 1.0 : 0.0;
            } else if (dtype_ == DataType::Int32 && dtype == DataType::Bool) {
                int32_t* src = static_cast<int32_t*>(data_);
                bool* dst = static_cast<bool*>(result.data_);
                dst[i] = src[i] != 0;
            } else if (dtype_ == DataType::Bool && dtype == DataType::Int32) {
                bool* src = static_cast<bool*>(data_);
                int32_t* dst = static_cast<int32_t*>(result.data_);
                dst[i] = src[i] ? 1 : 0;
            } else if (dtype_ == DataType::Int64 && dtype == DataType::Bool) {
                int64_t* src = static_cast<int64_t*>(data_);
                bool* dst = static_cast<bool*>(result.data_);
                dst[i] = src[i] != 0;
            } else if (dtype_ == DataType::Bool && dtype == DataType::Int64) {
                bool* src = static_cast<bool*>(data_);
                int64_t* dst = static_cast<int64_t*>(result.data_);
                dst[i] = src[i] ? 1 : 0;
            } else {
                throw std::runtime_error("Unsupported dtype conversion");
            }
        }
    } else {
        // For GPU devices, convert on CPU and copy back
        Tensor cpu_tensor = to(Device::CPU);
        result = cpu_tensor.to(dtype).to(device_);
    }
    
    return result;
}

} // namespace phynexus
