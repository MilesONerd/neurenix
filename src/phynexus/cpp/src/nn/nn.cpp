/**
 * @file nn.cpp
 * @brief Implementation of neural network components for the Phynexus engine
 * 
 * This file contains the implementation of neural network components for the Phynexus engine,
 * including layers, activations, losses, etc.
 * 
 * @copyright Copyright (c) 2025 Neurenix
 * @license Apache License 2.0
 */

#include "phynexus/nn.h"
#include "phynexus/ops.h"
#include <random>
#include <cmath>

namespace phynexus {
namespace nn {

// Linear layer implementation

void Linear::reset_parameters() {
    // Initialize weight using Kaiming initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    
    float std_dev = std::sqrt(2.0f / in_features_);
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    if (weight_.dtype() == DataType::Float32) {
        float* weight_data = static_cast<float*>(weight_.data());
        for (size_t i = 0; i < out_features_ * in_features_; ++i) {
            weight_data[i] = dist(gen);
        }
    } else if (weight_.dtype() == DataType::Float64) {
        double* weight_data = static_cast<double*>(weight_.data());
        for (size_t i = 0; i < out_features_ * in_features_; ++i) {
            weight_data[i] = static_cast<double>(dist(gen));
        }
    }
    
    // Initialize bias to zeros
    if (use_bias_) {
        if (bias_.dtype() == DataType::Float32) {
            float* bias_data = static_cast<float*>(bias_.data());
            for (size_t i = 0; i < out_features_; ++i) {
                bias_data[i] = 0.0f;
            }
        } else if (bias_.dtype() == DataType::Float64) {
            double* bias_data = static_cast<double*>(bias_.data());
            for (size_t i = 0; i < out_features_; ++i) {
                bias_data[i] = 0.0;
            }
        }
    }
}

// Conv2d layer implementation

void Conv2d::reset_parameters() {
    // Initialize weight using Kaiming initialization
    std::random_device rd;
    std::mt19937 gen(rd());
    
    float n = in_channels_ * kernel_size_[0] * kernel_size_[1];
    float std_dev = std::sqrt(2.0f / n);
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    if (weight_.dtype() == DataType::Float32) {
        float* weight_data = static_cast<float*>(weight_.data());
        for (size_t i = 0; i < out_channels_ * (in_channels_ / groups_) * kernel_size_[0] * kernel_size_[1]; ++i) {
            weight_data[i] = dist(gen);
        }
    } else if (weight_.dtype() == DataType::Float64) {
        double* weight_data = static_cast<double*>(weight_.data());
        for (size_t i = 0; i < out_channels_ * (in_channels_ / groups_) * kernel_size_[0] * kernel_size_[1]; ++i) {
            weight_data[i] = static_cast<double>(dist(gen));
        }
    }
    
    // Initialize bias to zeros
    if (use_bias_) {
        if (bias_.dtype() == DataType::Float32) {
            float* bias_data = static_cast<float*>(bias_.data());
            for (size_t i = 0; i < out_channels_; ++i) {
                bias_data[i] = 0.0f;
            }
        } else if (bias_.dtype() == DataType::Float64) {
            double* bias_data = static_cast<double*>(bias_.data());
            for (size_t i = 0; i < out_channels_; ++i) {
                bias_data[i] = 0.0;
            }
        }
    }
}

// BatchNorm2d layer implementation

void BatchNorm2d::reset_parameters() {
    // Initialize weight to ones
    if (affine_) {
        if (weight_.dtype() == DataType::Float32) {
            float* weight_data = static_cast<float*>(weight_.data());
            for (size_t i = 0; i < num_features_; ++i) {
                weight_data[i] = 1.0f;
            }
        } else if (weight_.dtype() == DataType::Float64) {
            double* weight_data = static_cast<double*>(weight_.data());
            for (size_t i = 0; i < num_features_; ++i) {
                weight_data[i] = 1.0;
            }
        }
        
        // Initialize bias to zeros
        if (bias_.dtype() == DataType::Float32) {
            float* bias_data = static_cast<float*>(bias_.data());
            for (size_t i = 0; i < num_features_; ++i) {
                bias_data[i] = 0.0f;
            }
        } else if (bias_.dtype() == DataType::Float64) {
            double* bias_data = static_cast<double*>(bias_.data());
            for (size_t i = 0; i < num_features_; ++i) {
                bias_data[i] = 0.0;
            }
        }
    }
    
    // Initialize running stats to zeros
    if (track_running_stats_) {
        if (running_mean_.dtype() == DataType::Float32) {
            float* mean_data = static_cast<float*>(running_mean_.data());
            for (size_t i = 0; i < num_features_; ++i) {
                mean_data[i] = 0.0f;
            }
        } else if (running_mean_.dtype() == DataType::Float64) {
            double* mean_data = static_cast<double*>(running_mean_.data());
            for (size_t i = 0; i < num_features_; ++i) {
                mean_data[i] = 0.0;
            }
        }
        
        if (running_var_.dtype() == DataType::Float32) {
            float* var_data = static_cast<float*>(running_var_.data());
            for (size_t i = 0; i < num_features_; ++i) {
                var_data[i] = 1.0f;
            }
        } else if (running_var_.dtype() == DataType::Float64) {
            double* var_data = static_cast<double*>(running_var_.data());
            for (size_t i = 0; i < num_features_; ++i) {
                var_data[i] = 1.0;
            }
        }
    }
}

// Factory functions for creating modules

std::shared_ptr<Module> make_linear(size_t in_features, size_t out_features, bool bias) {
    return std::make_shared<Linear>(in_features, out_features, bias);
}

std::shared_ptr<Module> make_conv2d(size_t in_channels, size_t out_channels,
                                   const std::vector<size_t>& kernel_size,
                                   const std::vector<size_t>& stride,
                                   const std::vector<size_t>& padding,
                                   const std::vector<size_t>& dilation,
                                   size_t groups, bool bias) {
    return std::make_shared<Conv2d>(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, bias);
}

std::shared_ptr<Module> make_relu(bool inplace) {
    return std::make_shared<ReLU>(inplace);
}

std::shared_ptr<Module> make_sigmoid() {
    return std::make_shared<Sigmoid>();
}

std::shared_ptr<Module> make_tanh() {
    return std::make_shared<Tanh>();
}

std::shared_ptr<Module> make_softmax(int dim) {
    return std::make_shared<Softmax>(dim);
}

std::shared_ptr<Module> make_dropout(double p, bool inplace) {
    return std::make_shared<Dropout>(p, inplace);
}

std::shared_ptr<Module> make_batch_norm2d(size_t num_features, double eps, double momentum,
                                         bool affine, bool track_running_stats) {
    return std::make_shared<BatchNorm2d>(num_features, eps, momentum,
                                        affine, track_running_stats);
}

std::shared_ptr<Module> make_max_pool2d(const std::vector<size_t>& kernel_size,
                                       const std::vector<size_t>& stride,
                                       const std::vector<size_t>& padding,
                                       const std::vector<size_t>& dilation,
                                       bool ceil_mode) {
    return std::make_shared<MaxPool2d>(kernel_size, stride, padding, dilation, ceil_mode);
}

std::shared_ptr<Module> make_avg_pool2d(const std::vector<size_t>& kernel_size,
                                       const std::vector<size_t>& stride,
                                       const std::vector<size_t>& padding,
                                       bool ceil_mode, bool count_include_pad) {
    return std::make_shared<AvgPool2d>(kernel_size, stride, padding, ceil_mode, count_include_pad);
}

std::shared_ptr<Module> make_embedding(size_t num_embeddings, size_t embedding_dim,
                                      int padding_idx, float max_norm, float norm_type,
                                      bool scale_grad_by_freq, bool sparse) {
    return std::make_shared<Embedding>(num_embeddings, embedding_dim, padding_idx,
                                      max_norm, norm_type, scale_grad_by_freq, sparse);
}

std::shared_ptr<Module> make_lstm_cell(size_t input_size, size_t hidden_size, bool bias) {
    return std::make_shared<LSTMCell>(input_size, hidden_size, bias);
}

// Sequential module factory
std::shared_ptr<Sequential> make_sequential(const std::vector<std::shared_ptr<Module>>& modules) {
    return std::make_shared<Sequential>(modules);
}

// Add module to sequential
void add_module(std::shared_ptr<Sequential> sequential, std::shared_ptr<Module> module) {
    sequential->add(module);
}

} // namespace nn
} // namespace phynexus
