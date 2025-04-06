/**
 * Quantization module implementation for Phynexus
 */

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <cmath>
#include <algorithm>

#include "hardware/quantization.h"
#include "error.h"
#include "tensor.h"
#include "memory.h"

namespace phynexus {
namespace hardware {

QuantizedTensor::QuantizedTensor(std::shared_ptr<Tensor> tensor, float scale, int32_t zero_point, QuantizationType dtype)
    : tensor_(tensor), scale_(scale), zero_point_(zero_point), dtype_(dtype), quantized_data_(nullptr), size_(0) {
    
    std::vector<size_t> shape = tensor->get_shape();
    size_t num_elements = 1;
    for (size_t dim : shape) {
        num_elements *= dim;
    }
    
    size_t element_size = 0;
    switch (dtype) {
        case QuantizationType::INT8:
            element_size = 1;
            break;
        case QuantizationType::FP16:
            element_size = 2;
            break;
        case QuantizationType::FP8:
            element_size = 1;
            break;
        default:
            throw std::runtime_error("Unsupported quantization type");
    }
    
    size_ = num_elements * element_size;
    quantized_data_ = malloc(size_);
    
    if (!quantized_data_) {
        throw std::runtime_error("Failed to allocate memory for quantized data");
    }
    
    const float* data = static_cast<const float*>(tensor->get_data());
    
    switch (dtype) {
        case QuantizationType::INT8: {
            int8_t* q_data = static_cast<int8_t*>(quantized_data_);
            for (size_t i = 0; i < num_elements; ++i) {
                float scaled = data[i] / scale_;
                int32_t rounded = static_cast<int32_t>(std::round(scaled)) + zero_point_;
                q_data[i] = static_cast<int8_t>(std::max(-128, std::min(127, rounded)));
            }
            break;
        }
        case QuantizationType::FP16: {
            uint16_t* q_data = static_cast<uint16_t*>(quantized_data_);
            for (size_t i = 0; i < num_elements; ++i) {
                q_data[i] = static_cast<uint16_t>(data[i] * 100);
            }
            break;
        }
        case QuantizationType::FP8: {
            uint8_t* q_data = static_cast<uint8_t*>(quantized_data_);
            for (size_t i = 0; i < num_elements; ++i) {
                q_data[i] = static_cast<uint8_t>(data[i] * 50);
            }
            break;
        }
    }
}

QuantizedTensor::~QuantizedTensor() {
    if (quantized_data_) {
        free(quantized_data_);
        quantized_data_ = nullptr;
    }
}

std::shared_ptr<Tensor> QuantizedTensor::dequantize() const {
    std::vector<size_t> shape = tensor_->get_shape();
    size_t num_elements = 1;
    for (size_t dim : shape) {
        num_elements *= dim;
    }
    
    std::shared_ptr<Tensor> dequantized = std::make_shared<Tensor>(shape, DataType::FLOAT32);
    float* dq_data = static_cast<float*>(dequantized->get_data());
    
    switch (dtype_) {
        case QuantizationType::INT8: {
            const int8_t* q_data = static_cast<const int8_t*>(quantized_data_);
            for (size_t i = 0; i < num_elements; ++i) {
                dq_data[i] = static_cast<float>(q_data[i] - zero_point_) * scale_;
            }
            break;
        }
        case QuantizationType::FP16: {
            const uint16_t* q_data = static_cast<const uint16_t*>(quantized_data_);
            for (size_t i = 0; i < num_elements; ++i) {
                dq_data[i] = static_cast<float>(q_data[i]) / 100.0f;
            }
            break;
        }
        case QuantizationType::FP8: {
            const uint8_t* q_data = static_cast<const uint8_t*>(quantized_data_);
            for (size_t i = 0; i < num_elements; ++i) {
                dq_data[i] = static_cast<float>(q_data[i]) / 50.0f;
            }
            break;
        }
    }
    
    return dequantized;
}

float QuantizedTensor::get_scale() const {
    return scale_;
}

int32_t QuantizedTensor::get_zero_point() const {
    return zero_point_;
}

QuantizationType QuantizedTensor::get_dtype() const {
    return dtype_;
}

void* QuantizedTensor::get_data() const {
    return quantized_data_;
}

std::vector<size_t> QuantizedTensor::get_shape() const {
    return tensor_->get_shape();
}

size_t QuantizedTensor::get_size() const {
    return size_;
}

QuantizationConfig::QuantizationConfig(QuantizationType dtype)
    : dtype_(dtype), per_channel_(false), symmetric_(false) {
}

QuantizationConfig& QuantizationConfig::with_per_channel(bool per_channel) {
    per_channel_ = per_channel;
    return *this;
}

QuantizationConfig& QuantizationConfig::with_symmetric(bool symmetric) {
    symmetric_ = symmetric;
    return *this;
}

QuantizationType QuantizationConfig::get_dtype() const {
    return dtype_;
}

bool QuantizationConfig::is_per_channel() const {
    return per_channel_;
}

bool QuantizationConfig::is_symmetric() const {
    return symmetric_;
}

QATConfig::QATConfig(QuantizationType dtype)
    : dtype_(dtype), per_channel_(false), symmetric_(false), quantize_weights_(true), quantize_activations_(true) {
}

QATConfig& QATConfig::with_per_channel(bool per_channel) {
    per_channel_ = per_channel;
    return *this;
}

QATConfig& QATConfig::with_symmetric(bool symmetric) {
    symmetric_ = symmetric;
    return *this;
}

QATConfig& QATConfig::with_quantize_weights(bool quantize_weights) {
    quantize_weights_ = quantize_weights;
    return *this;
}

QATConfig& QATConfig::with_quantize_activations(bool quantize_activations) {
    quantize_activations_ = quantize_activations;
    return *this;
}

QuantizationType QATConfig::get_dtype() const {
    return dtype_;
}

bool QATConfig::is_per_channel() const {
    return per_channel_;
}

bool QATConfig::is_symmetric() const {
    return symmetric_;
}

bool QATConfig::quantize_weights() const {
    return quantize_weights_;
}

bool QATConfig::quantize_activations() const {
    return quantize_activations_;
}

PruningConfig::PruningConfig(float sparsity, const std::string& method)
    : sparsity_(sparsity), method_(method), structured_(false) {
}

PruningConfig& PruningConfig::with_structured(bool structured) {
    structured_ = structured;
    return *this;
}

float PruningConfig::get_sparsity() const {
    return sparsity_;
}

std::string PruningConfig::get_method() const {
    return method_;
}

bool PruningConfig::is_structured() const {
    return structured_;
}

std::shared_ptr<QuantizedTensor> quantize_tensor(const std::shared_ptr<Tensor>& tensor, QuantizationType dtype) {
    const float* data = static_cast<const float*>(tensor->get_data());
    std::vector<size_t> shape = tensor->get_shape();
    size_t num_elements = 1;
    for (size_t dim : shape) {
        num_elements *= dim;
    }
    
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < num_elements; ++i) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }
    
    float scale = 1.0f;
    int32_t zero_point = 0;
    
    switch (dtype) {
        case QuantizationType::INT8: {
            scale = (max_val - min_val) / 255.0f;
            zero_point = static_cast<int32_t>(std::round(-min_val / scale));
            zero_point = std::max(-128, std::min(127, zero_point));
            break;
        }
        case QuantizationType::FP16:
        case QuantizationType::FP8:
            scale = 1.0f;
            zero_point = 0;
            break;
    }
    
    return std::make_shared<QuantizedTensor>(tensor, scale, zero_point, dtype);
}

std::shared_ptr<QuantizedTensor> quantize_tensor_with_config(const std::shared_ptr<Tensor>& tensor, const QuantizationConfig& config) {
    const float* data = static_cast<const float*>(tensor->get_data());
    std::vector<size_t> shape = tensor->get_shape();
    size_t num_elements = 1;
    for (size_t dim : shape) {
        num_elements *= dim;
    }
    
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    for (size_t i = 0; i < num_elements; ++i) {
        min_val = std::min(min_val, data[i]);
        max_val = std::max(max_val, data[i]);
    }
    
    float scale = 1.0f;
    int32_t zero_point = 0;
    
    switch (config.get_dtype()) {
        case QuantizationType::INT8: {
            if (config.is_symmetric()) {
                float abs_max = std::max(std::abs(min_val), std::abs(max_val));
                scale = abs_max / 127.0f;
                zero_point = 0;
            } else {
                scale = (max_val - min_val) / 255.0f;
                zero_point = static_cast<int32_t>(std::round(-min_val / scale));
                zero_point = std::max(-128, std::min(127, zero_point));
            }
            break;
        }
        case QuantizationType::FP16:
        case QuantizationType::FP8:
            scale = 1.0f;
            zero_point = 0;
            break;
    }
    
    return std::make_shared<QuantizedTensor>(tensor, scale, zero_point, config.get_dtype());
}

std::shared_ptr<Tensor> prune_tensor(const std::shared_ptr<Tensor>& tensor, float sparsity, const std::string& method) {
    const float* data = static_cast<const float*>(tensor->get_data());
    std::vector<size_t> shape = tensor->get_shape();
    size_t num_elements = 1;
    for (size_t dim : shape) {
        num_elements *= dim;
    }
    
    std::shared_ptr<Tensor> pruned = std::make_shared<Tensor>(shape, DataType::FLOAT32);
    float* pruned_data = static_cast<float*>(pruned->get_data());
    
    std::memcpy(pruned_data, data, num_elements * sizeof(float));
    
    if (method == "magnitude") {
        std::vector<std::pair<float, size_t>> value_index_pairs;
        for (size_t i = 0; i < num_elements; ++i) {
            value_index_pairs.push_back(std::make_pair(std::abs(data[i]), i));
        }
        
        std::sort(value_index_pairs.begin(), value_index_pairs.end());
        
        size_t num_to_prune = static_cast<size_t>(sparsity * num_elements);
        for (size_t i = 0; i < num_to_prune; ++i) {
            pruned_data[value_index_pairs[i].second] = 0.0f;
        }
    } else if (method == "random") {
        size_t num_to_prune = static_cast<size_t>(sparsity * num_elements);
        std::vector<size_t> indices(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            indices[i] = i;
        }
        
        std::random_shuffle(indices.begin(), indices.end());
        
        for (size_t i = 0; i < num_to_prune; ++i) {
            pruned_data[indices[i]] = 0.0f;
        }
    } else {
        throw std::runtime_error("Unsupported pruning method: " + method);
    }
    
    return pruned;
}

std::shared_ptr<Tensor> prune_tensor_with_config(const std::shared_ptr<Tensor>& tensor, const PruningConfig& config) {
    const float* data = static_cast<const float*>(tensor->get_data());
    std::vector<size_t> shape = tensor->get_shape();
    size_t num_elements = 1;
    for (size_t dim : shape) {
        num_elements *= dim;
    }
    
    std::shared_ptr<Tensor> pruned = std::make_shared<Tensor>(shape, DataType::FLOAT32);
    float* pruned_data = static_cast<float*>(pruned->get_data());
    
    std::memcpy(pruned_data, data, num_elements * sizeof(float));
    
    if (config.is_structured()) {
        if (shape.size() != 2) {
            throw std::runtime_error("Structured pruning is only supported for 2D tensors");
        }
        
        size_t rows = shape[0];
        size_t cols = shape[1];
        
        if (config.get_method() == "magnitude") {
            std::vector<std::pair<float, size_t>> col_norms;
            for (size_t j = 0; j < cols; ++j) {
                float norm = 0.0f;
                for (size_t i = 0; i < rows; ++i) {
                    norm += std::abs(data[i * cols + j]);
                }
                col_norms.push_back(std::make_pair(norm, j));
            }
            
            std::sort(col_norms.begin(), col_norms.end());
            
            size_t num_cols_to_prune = static_cast<size_t>(config.get_sparsity() * cols);
            for (size_t j = 0; j < num_cols_to_prune; ++j) {
                size_t col = col_norms[j].second;
                for (size_t i = 0; i < rows; ++i) {
                    pruned_data[i * cols + col] = 0.0f;
                }
            }
        } else {
            throw std::runtime_error("Unsupported structured pruning method: " + config.get_method());
        }
    } else {
        return prune_tensor(tensor, config.get_sparsity(), config.get_method());
    }
    
    return pruned;
}

} // namespace hardware
} // namespace phynexus
