/**
 * @file tensor.h
 * @brief Tensor class definition for the Phynexus engine
 * 
 * This file contains the definition of the Tensor class, which is the core
 * data structure for the Phynexus engine. It provides functionality for
 * creating, manipulating, and operating on multi-dimensional arrays.
 * 
 * @copyright Copyright (c) 2025 Neurenix
 * @license Apache License 2.0
 */

#ifndef PHYNEXUS_TENSOR_H
#define PHYNEXUS_TENSOR_H

#include <vector>
#include <memory>
#include <string>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <cmath>

namespace phynexus {

/**
 * @brief Enum for supported data types
 */
enum class DataType {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64,
    UINT8,
    BOOL
};

/**
 * @brief Enum for supported device types
 */
enum class DeviceType {
    CPU,
    CUDA,
    ROCM,
    WEBGPU,
    TPU,
    NPU
};

/**
 * @brief Device class for tensor operations
 */
class Device {
public:
    /**
     * @brief Constructor
     * 
     * @param type Device type
     * @param index Device index
     */
    Device(DeviceType type = DeviceType::CPU, int index = 0)
        : type_(type), index_(index) {}

    /**
     * @brief Get device type
     * 
     * @return Device type
     */
    DeviceType type() const { return type_; }

    /**
     * @brief Get device index
     * 
     * @return Device index
     */
    int index() const { return index_; }

    /**
     * @brief Check if device is CPU
     * 
     * @return True if device is CPU, false otherwise
     */
    bool is_cpu() const { return type_ == DeviceType::CPU; }

    /**
     * @brief Check if device is CUDA
     * 
     * @return True if device is CUDA, false otherwise
     */
    bool is_cuda() const { return type_ == DeviceType::CUDA; }

    /**
     * @brief Check if device is ROCm
     * 
     * @return True if device is ROCm, false otherwise
     */
    bool is_rocm() const { return type_ == DeviceType::ROCM; }

    /**
     * @brief Check if device is WebGPU
     * 
     * @return True if device is WebGPU, false otherwise
     */
    bool is_webgpu() const { return type_ == DeviceType::WEBGPU; }
    
    /**
     * @brief Check if device is TPU
     * 
     * @return True if device is TPU, false otherwise
     */
    bool is_tpu() const { return type_ == DeviceType::TPU; }
    
    /**
     * @brief Check if device is NPU
     * 
     * @return True if device is NPU, false otherwise
     */
    bool is_npu() const { return type_ == DeviceType::NPU; }

    /**
     * @brief Get string representation of device
     * 
     * @return String representation of device
     */
    std::string to_string() const {
        std::string device_type;
        switch (type_) {
            case DeviceType::CPU:
                device_type = "CPU";
                break;
            case DeviceType::CUDA:
                device_type = "CUDA";
                break;
            case DeviceType::ROCM:
                device_type = "ROCm";
                break;
            case DeviceType::WEBGPU:
                device_type = "WebGPU";
                break;
            case DeviceType::TPU:
                device_type = "TPU";
                break;
            case DeviceType::NPU:
                device_type = "NPU";
                break;
        }
        return device_type + ":" + std::to_string(index_);
    }

private:
    DeviceType type_;
    int index_;
};

/**
 * @brief Storage class for tensor data
 */
class Storage {
public:
    /**
     * @brief Constructor
     * 
     * @param size Size of storage in bytes
     * @param device Device to allocate storage on
     */
    Storage(size_t size, const Device& device = Device())
        : size_(size), device_(device), data_(nullptr) {
        allocate();
    }

    /**
     * @brief Destructor
     */
    ~Storage() {
        deallocate();
    }

    /**
     * @brief Get size of storage
     * 
     * @return Size of storage in bytes
     */
    size_t size() const { return size_; }

    /**
     * @brief Get device
     * 
     * @return Device
     */
    const Device& device() const { return device_; }

    /**
     * @brief Get data pointer
     * 
     * @return Data pointer
     */
    void* data() const { return data_; }

    /**
     * @brief Copy data to storage
     * 
     * @param src Source data
     * @param size Size of data in bytes
     */
    void copy_from(const void* src, size_t size) {
        if (size > size_) {
            throw std::runtime_error("Copy size exceeds storage size");
        }

        if (device_.is_cpu()) {
            std::memcpy(data_, src, size);
        } else {
            // Device-specific copy implementation
            // For now, just throw an error
            throw std::runtime_error("Device copy not implemented");
        }
    }

    /**
     * @brief Copy data from storage
     * 
     * @param dst Destination data
     * @param size Size of data in bytes
     */
    void copy_to(void* dst, size_t size) const {
        if (size > size_) {
            throw std::runtime_error("Copy size exceeds storage size");
        }

        if (device_.is_cpu()) {
            std::memcpy(dst, data_, size);
        } else {
            // Device-specific copy implementation
            // For now, just throw an error
            throw std::runtime_error("Device copy not implemented");
        }
    }

private:
    size_t size_;
    Device device_;
    void* data_;

    /**
     * @brief Allocate storage
     */
    void allocate() {
        if (device_.is_cpu()) {
            data_ = std::malloc(size_);
            if (!data_) {
                throw std::bad_alloc();
            }
        } else {
            // Device-specific allocation
            // For now, just throw an error
            throw std::runtime_error("Device allocation not implemented");
        }
    }

    /**
     * @brief Deallocate storage
     */
    void deallocate() {
        if (data_) {
            if (device_.is_cpu()) {
                std::free(data_);
            } else {
                // Device-specific deallocation
                // For now, just throw an error
                throw std::runtime_error("Device deallocation not implemented");
            }
            data_ = nullptr;
        }
    }
};

/**
 * @brief Tensor class for multi-dimensional arrays
 */
class Tensor {
public:
    /**
     * @brief Constructor
     * 
     * @param shape Shape of tensor
     * @param dtype Data type of tensor
     * @param device Device to allocate tensor on
     */
    Tensor(const std::vector<size_t>& shape, DataType dtype = DataType::FLOAT32,
           const Device& device = Device())
        : shape_(shape), dtype_(dtype), device_(device) {
        
        // Calculate strides
        strides_.resize(shape.size());
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape[i];
        }
        
        // Calculate size
        size_t size = element_size() * num_elements();
        
        // Allocate storage
        storage_ = std::make_shared<Storage>(size, device);
    }

    /**
     * @brief Get shape of tensor
     * 
     * @return Shape of tensor
     */
    const std::vector<size_t>& shape() const { return shape_; }

    /**
     * @brief Get strides of tensor
     * 
     * @return Strides of tensor
     */
    const std::vector<size_t>& strides() const { return strides_; }

    /**
     * @brief Get data type of tensor
     * 
     * @return Data type of tensor
     */
    DataType dtype() const { return dtype_; }

    /**
     * @brief Get device of tensor
     * 
     * @return Device of tensor
     */
    const Device& device() const { return device_; }

    /**
     * @brief Get number of dimensions
     * 
     * @return Number of dimensions
     */
    size_t ndim() const { return shape_.size(); }

    /**
     * @brief Get number of elements
     * 
     * @return Number of elements
     */
    size_t num_elements() const {
        size_t num = 1;
        for (size_t dim : shape_) {
            num *= dim;
        }
        return num;
    }

    /**
     * @brief Get size of element in bytes
     * 
     * @return Size of element in bytes
     */
    size_t element_size() const {
        switch (dtype_) {
            case DataType::FLOAT32:
                return sizeof(float);
            case DataType::FLOAT64:
                return sizeof(double);
            case DataType::INT32:
                return sizeof(int32_t);
            case DataType::INT64:
                return sizeof(int64_t);
            case DataType::UINT8:
                return sizeof(uint8_t);
            case DataType::BOOL:
                return sizeof(bool);
            default:
                throw std::runtime_error("Unknown data type");
        }
    }

    /**
     * @brief Get data pointer
     * 
     * @return Data pointer
     */
    void* data() const { return storage_->data(); }

    /**
     * @brief Fill tensor with value
     * 
     * @tparam T Type of value
     * @param value Value to fill tensor with
     */
    template <typename T>
    void fill(T value) {
        if (device_.is_cpu()) {
            T* data_ptr = static_cast<T*>(data());
            size_t num = num_elements();
            for (size_t i = 0; i < num; ++i) {
                data_ptr[i] = value;
            }
        } else {
            // Device-specific fill implementation
            // For now, just throw an error
            throw std::runtime_error("Device fill not implemented");
        }
    }

    /**
     * @brief Copy data to tensor
     * 
     * @tparam T Type of data
     * @param data Data to copy
     * @param size Size of data in elements
     */
    template <typename T>
    void copy_from(const T* data, size_t size) {
        if (size > num_elements()) {
            throw std::runtime_error("Copy size exceeds tensor size");
        }

        storage_->copy_from(data, size * element_size());
    }

    /**
     * @brief Copy data from tensor
     * 
     * @tparam T Type of data
     * @param data Destination data
     * @param size Size of data in elements
     */
    template <typename T>
    void copy_to(T* data, size_t size) const {
        if (size > num_elements()) {
            throw std::runtime_error("Copy size exceeds tensor size");
        }

        storage_->copy_to(data, size * element_size());
    }

    /**
     * @brief Get string representation of tensor
     * 
     * @return String representation of tensor
     */
    std::string to_string() const {
        std::string result = "Tensor(";
        
        // Add shape
        result += "[";
        for (size_t i = 0; i < shape_.size(); ++i) {
            result += std::to_string(shape_[i]);
            if (i < shape_.size() - 1) {
                result += ", ";
            }
        }
        result += "], ";
        
        // Add data type
        switch (dtype_) {
            case DataType::FLOAT32:
                result += "float32";
                break;
            case DataType::FLOAT64:
                result += "float64";
                break;
            case DataType::INT32:
                result += "int32";
                break;
            case DataType::INT64:
                result += "int64";
                break;
            case DataType::UINT8:
                result += "uint8";
                break;
            case DataType::BOOL:
                result += "bool";
                break;
        }
        
        // Add device
        result += ", device=" + device_.to_string();
        
        result += ")";
        return result;
    }

private:
    std::vector<size_t> shape_;
    std::vector<size_t> strides_;
    DataType dtype_;
    Device device_;
    std::shared_ptr<Storage> storage_;
};

} // namespace phynexus

#endif // PHYNEXUS_TENSOR_H
