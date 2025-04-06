/**
 * Quantization module header for Phynexus
 */

#ifndef PHYNEXUS_HARDWARE_QUANTIZATION_H
#define PHYNEXUS_HARDWARE_QUANTIZATION_H

#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

#include "error.h"
#include "tensor.h"

namespace phynexus {
namespace hardware {

/**
 * Quantization type enumeration
 */
enum class QuantizationType {
    INT8,   // 8-bit integer quantization
    FP16,   // 16-bit floating point quantization
    FP8,    // 8-bit floating point quantization
};

/**
 * Quantized tensor class
 */
class QuantizedTensor {
public:
    /**
     * Constructor
     */
    QuantizedTensor(std::shared_ptr<Tensor> tensor, float scale, int32_t zero_point, QuantizationType dtype);

    /**
     * Dequantize tensor
     */
    std::shared_ptr<Tensor> dequantize() const;

    /**
     * Get scale
     */
    float get_scale() const;

    /**
     * Get zero point
     */
    int32_t get_zero_point() const;

    /**
     * Get quantization type
     */
    QuantizationType get_dtype() const;

    /**
     * Get quantized data
     */
    void* get_data() const;

    /**
     * Get tensor shape
     */
    std::vector<size_t> get_shape() const;

    /**
     * Get tensor size
     */
    size_t get_size() const;

private:
    std::shared_ptr<Tensor> tensor_;
    float scale_;
    int32_t zero_point_;
    QuantizationType dtype_;
    void* quantized_data_;
    size_t size_;
};

/**
 * Quantization configuration
 */
class QuantizationConfig {
public:
    /**
     * Constructor
     */
    QuantizationConfig(QuantizationType dtype);

    /**
     * Set per-channel quantization
     */
    QuantizationConfig& with_per_channel(bool per_channel);

    /**
     * Set symmetric quantization
     */
    QuantizationConfig& with_symmetric(bool symmetric);

    /**
     * Get quantization type
     */
    QuantizationType get_dtype() const;

    /**
     * Get per-channel flag
     */
    bool is_per_channel() const;

    /**
     * Get symmetric flag
     */
    bool is_symmetric() const;

private:
    QuantizationType dtype_;
    bool per_channel_;
    bool symmetric_;
};

/**
 * Quantization-aware training configuration
 */
class QATConfig {
public:
    /**
     * Constructor
     */
    QATConfig(QuantizationType dtype);

    /**
     * Set per-channel quantization
     */
    QATConfig& with_per_channel(bool per_channel);

    /**
     * Set symmetric quantization
     */
    QATConfig& with_symmetric(bool symmetric);

    /**
     * Set quantize weights flag
     */
    QATConfig& with_quantize_weights(bool quantize_weights);

    /**
     * Set quantize activations flag
     */
    QATConfig& with_quantize_activations(bool quantize_activations);

    /**
     * Get quantization type
     */
    QuantizationType get_dtype() const;

    /**
     * Get per-channel flag
     */
    bool is_per_channel() const;

    /**
     * Get symmetric flag
     */
    bool is_symmetric() const;

    /**
     * Get quantize weights flag
     */
    bool quantize_weights() const;

    /**
     * Get quantize activations flag
     */
    bool quantize_activations() const;

private:
    QuantizationType dtype_;
    bool per_channel_;
    bool symmetric_;
    bool quantize_weights_;
    bool quantize_activations_;
};

/**
 * Pruning configuration
 */
class PruningConfig {
public:
    /**
     * Constructor
     */
    PruningConfig(float sparsity, const std::string& method);

    /**
     * Set structured pruning flag
     */
    PruningConfig& with_structured(bool structured);

    /**
     * Get sparsity
     */
    float get_sparsity() const;

    /**
     * Get pruning method
     */
    std::string get_method() const;

    /**
     * Get structured pruning flag
     */
    bool is_structured() const;

private:
    float sparsity_;
    std::string method_;
    bool structured_;
};

/**
 * Quantize tensor
 */
std::shared_ptr<QuantizedTensor> quantize_tensor(const std::shared_ptr<Tensor>& tensor, QuantizationType dtype);

/**
 * Quantize tensor with configuration
 */
std::shared_ptr<QuantizedTensor> quantize_tensor_with_config(const std::shared_ptr<Tensor>& tensor, const QuantizationConfig& config);

/**
 * Prune tensor
 */
std::shared_ptr<Tensor> prune_tensor(const std::shared_ptr<Tensor>& tensor, float sparsity, const std::string& method);

/**
 * Prune tensor with configuration
 */
std::shared_ptr<Tensor> prune_tensor_with_config(const std::shared_ptr<Tensor>& tensor, const PruningConfig& config);

} // namespace hardware
} // namespace phynexus

#endif // PHYNEXUS_HARDWARE_QUANTIZATION_H
