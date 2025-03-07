/**
 * @file nn.h
 * @brief Neural network components for the Phynexus engine
 * 
 * This file contains the definitions of neural network components for the Phynexus engine,
 * including layers, activations, losses, etc.
 * 
 * @copyright Copyright (c) 2025 Neurenix
 * @license Apache License 2.0
 */

#ifndef PHYNEXUS_NN_H
#define PHYNEXUS_NN_H

#include "tensor.h"
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>

namespace phynexus {
namespace nn {

/**
 * @brief Base class for all neural network modules
 */
class Module {
public:
    Module() = default;
    virtual ~Module() = default;
    virtual Tensor forward(const Tensor& input) = 0;
    virtual std::vector<Tensor> parameters() = 0;
    virtual Module& train(bool mode = true);
    virtual Module& eval();
    bool is_training() const;
    virtual std::string name() const = 0;
    
protected:
    bool training_ = true;
};

/**
 * @brief Linear layer
 */
class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::string name() const override;
    void reset_parameters();
    Tensor& weight();
    Tensor& bias();
    
private:
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;
    Tensor weight_;
    Tensor bias_;
};

/**
 * @brief 2D convolutional layer
 */
class Conv2d : public Module {
public:
    Conv2d(size_t in_channels, size_t out_channels,
           const std::vector<size_t>& kernel_size,
           const std::vector<size_t>& stride = {1, 1},
           const std::vector<size_t>& padding = {0, 0},
           const std::vector<size_t>& dilation = {1, 1},
           size_t groups = 1, bool bias = true);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::string name() const override;
    void reset_parameters();
    Tensor& weight();
    Tensor& bias();
    
private:
    size_t in_channels_;
    size_t out_channels_;
    std::vector<size_t> kernel_size_;
    std::vector<size_t> stride_;
    std::vector<size_t> padding_;
    std::vector<size_t> dilation_;
    size_t groups_;
    bool use_bias_;
    Tensor weight_;
    Tensor bias_;
};

/**
 * @brief 2D batch normalization layer
 */
class BatchNorm2d : public Module {
public:
    BatchNorm2d(size_t num_features, double eps = 1e-5, double momentum = 0.1,
               bool affine = true, bool track_running_stats = true);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::string name() const override;
    void reset_parameters();
    Tensor& weight();
    Tensor& bias();
    Tensor& running_mean();
    Tensor& running_var();
    
private:
    size_t num_features_;
    double eps_;
    double momentum_;
    bool affine_;
    bool track_running_stats_;
    Tensor weight_;
    Tensor bias_;
    Tensor running_mean_;
    Tensor running_var_;
};

/**
 * @brief ReLU activation
 */
class ReLU : public Module {
public:
    ReLU(bool inplace = false);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::string name() const override;
    
private:
    bool inplace_;
};

/**
 * @brief Sigmoid activation
 */
class Sigmoid : public Module {
public:
    Sigmoid() = default;
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::string name() const override;
};

/**
 * @brief Tanh activation
 */
class Tanh : public Module {
public:
    Tanh() = default;
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::string name() const override;
};

/**
 * @brief Softmax activation
 */
class Softmax : public Module {
public:
    Softmax(int dim = -1);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::string name() const override;
    
private:
    int dim_;
};

/**
 * @brief Dropout layer
 */
class Dropout : public Module {
public:
    Dropout(double p = 0.5, bool inplace = false);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::string name() const override;
    
private:
    double p_;
    bool inplace_;
};

/**
 * @brief Sequential module
 */
class Sequential : public Module {
public:
    Sequential() = default;
    Sequential(const std::vector<std::shared_ptr<Module>>& modules);
    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::string name() const override;
    void add(std::shared_ptr<Module> module);
    
private:
    std::vector<std::shared_ptr<Module>> modules_;
};

// Factory functions for creating modules
std::shared_ptr<Module> make_linear(size_t in_features, size_t out_features, bool bias = true);
std::shared_ptr<Module> make_conv2d(size_t in_channels, size_t out_channels,
                                   const std::vector<size_t>& kernel_size,
                                   const std::vector<size_t>& stride = {1, 1},
                                   const std::vector<size_t>& padding = {0, 0},
                                   const std::vector<size_t>& dilation = {1, 1},
                                   size_t groups = 1, bool bias = true);
std::shared_ptr<Module> make_relu(bool inplace = false);
std::shared_ptr<Module> make_sigmoid();
std::shared_ptr<Module> make_tanh();
std::shared_ptr<Module> make_softmax(int dim = -1);
std::shared_ptr<Module> make_dropout(double p = 0.5, bool inplace = false);
std::shared_ptr<Module> make_batch_norm2d(size_t num_features, double eps = 1e-5, double momentum = 0.1,
                                         bool affine = true, bool track_running_stats = true);
std::shared_ptr<Sequential> make_sequential(const std::vector<std::shared_ptr<Module>>& modules = {});
void add_module(std::shared_ptr<Sequential> sequential, std::shared_ptr<Module> module);

} // namespace nn
} // namespace phynexus

#endif // PHYNEXUS_NN_H
