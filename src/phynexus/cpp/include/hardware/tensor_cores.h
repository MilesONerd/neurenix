/**
 * @file tensor_cores.h
 * @brief NVIDIA Tensor Cores hardware backend implementation.
 *
 * This header provides hardware acceleration using NVIDIA's Tensor Cores,
 * enabling high-performance matrix operations and deep learning on NVIDIA GPUs
 * with Tensor Cores capability.
 */

#ifndef PHYNEXUS_HARDWARE_TENSOR_CORES_H
#define PHYNEXUS_HARDWARE_TENSOR_CORES_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include "phynexus/tensor.h"
#include "phynexus/device.h"

namespace phynexus {
namespace hardware {

/**
 * @brief Precision modes for Tensor Cores operations
 */
enum class TensorCoresPrecision {
    FP32,  ///< 32-bit floating point precision
    FP16,  ///< 16-bit floating point precision
    Mixed  ///< Mixed precision (FP16 computation with FP32 accumulation)
};

/**
 * @brief Convert precision mode to string
 * @param precision Precision mode
 * @return String representation of precision mode
 */
std::string precision_to_string(TensorCoresPrecision precision);

/**
 * @brief Parse precision mode from string
 * @param s String representation of precision mode
 * @return Precision mode
 * @throws std::invalid_argument if the string is not a valid precision mode
 */
TensorCoresPrecision precision_from_string(const std::string& s);

/**
 * @brief Information about a device with Tensor Cores
 */
struct DeviceInfo {
    std::string name;                ///< Device name
    std::string vendor;              ///< Device vendor
    DeviceType device_type;          ///< Device type
    std::string architecture;        ///< Device architecture
    std::string compute_capability;  ///< Compute capability
    size_t compute_units;            ///< Number of compute units
    size_t memory;                   ///< Memory size in bytes
};

/**
 * @brief Check if NVIDIA Tensor Cores are available on the system
 * @return true if Tensor Cores are available, false otherwise
 */
bool is_tensor_cores_available();

/**
 * @brief Get the number of available devices with Tensor Cores
 * @return Number of available devices with Tensor Cores
 */
size_t get_tensor_cores_device_count();

/**
 * @brief Get information about a device with Tensor Cores
 * @param device_index Index of the device
 * @return Device information, or nullptr if the device is not available
 */
std::unique_ptr<DeviceInfo> get_tensor_cores_device_info(size_t device_index);

/**
 * @brief NVIDIA Tensor Cores backend for hardware acceleration
 */
class TensorCoresBackend {
public:
    /**
     * @brief Create a new Tensor Cores backend
     * @throws std::runtime_error if Tensor Cores are not available
     */
    TensorCoresBackend();
    
    /**
     * @brief Destroy the Tensor Cores backend
     */
    ~TensorCoresBackend();
    
    /**
     * @brief Initialize the Tensor Cores backend
     * @return true if initialization was successful, false otherwise
     */
    bool initialize();
    
    /**
     * @brief Clean up Tensor Cores resources
     */
    void cleanup();
    
    /**
     * @brief Set the precision mode for Tensor Cores operations
     * @param precision Precision mode
     */
    void set_precision(TensorCoresPrecision precision);
    
    /**
     * @brief Get the precision mode for Tensor Cores operations
     * @return Precision mode
     */
    TensorCoresPrecision get_precision() const;
    
    /**
     * @brief Perform matrix multiplication using Tensor Cores
     * @param a First tensor
     * @param b Second tensor
     * @return Result of matrix multiplication
     */
    Tensor matmul(const Tensor& a, const Tensor& b);
    
    /**
     * @brief Optimize a model to use Tensor Cores
     * @param model The model to optimize
     * @param precision Precision to use
     * @return Optimized model
     */
    std::shared_ptr<Model> optimize_model(const std::shared_ptr<Model>& model, TensorCoresPrecision precision = TensorCoresPrecision::Mixed);

private:
    bool initialized_;                   ///< Whether the backend is initialized
    void* handle_;                       ///< CUBLAS handle
    void* stream_;                       ///< CUDA stream
    TensorCoresPrecision precision_;     ///< Precision mode
    void* workspace_;                    ///< Workspace memory
    size_t workspace_size_;              ///< Workspace size in bytes
    
    /**
     * @brief Create CUBLAS handle with Tensor Cores enabled
     * @return CUBLAS handle, or nullptr if creation failed
     */
    void* create_cublas_handle();
    
    /**
     * @brief Destroy CUBLAS handle
     * @param handle CUBLAS handle
     */
    void destroy_cublas_handle(void* handle);
    
    /**
     * @brief Create CUDA stream
     * @return CUDA stream, or nullptr if creation failed
     */
    void* create_cuda_stream();
    
    /**
     * @brief Destroy CUDA stream
     * @param stream CUDA stream
     */
    void destroy_cuda_stream(void* stream);
    
    /**
     * @brief Allocate workspace memory
     * @param size Size in bytes
     * @return Workspace memory, or nullptr if allocation failed
     */
    void* allocate_workspace(size_t size);
    
    /**
     * @brief Free workspace memory
     * @param workspace Workspace memory
     */
    void free_workspace(void* workspace);
};

} // namespace hardware
} // namespace phynexus

#endif // PHYNEXUS_HARDWARE_TENSOR_CORES_H
