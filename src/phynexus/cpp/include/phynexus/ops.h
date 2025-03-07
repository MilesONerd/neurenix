/**
 * @file ops.h
 * @brief Operations for the Phynexus engine
 * 
 * This file contains the definitions of operations for the Phynexus engine,
 * including element-wise operations, matrix multiplication, convolution, etc.
 * 
 * @copyright Copyright (c) 2025 Neurenix
 * @license Apache License 2.0
 */

#ifndef PHYNEXUS_OPS_H
#define PHYNEXUS_OPS_H

#include "tensor.h"

namespace phynexus {
namespace ops {

/**
 * @brief Element-wise addition
 * 
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor
 */
Tensor add(const Tensor& a, const Tensor& b);

/**
 * @brief Element-wise subtraction
 * 
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor
 */
Tensor sub(const Tensor& a, const Tensor& b);

/**
 * @brief Element-wise multiplication
 * 
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor
 */
Tensor mul(const Tensor& a, const Tensor& b);

/**
 * @brief Element-wise division
 * 
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor
 */
Tensor div(const Tensor& a, const Tensor& b);

/**
 * @brief Matrix multiplication
 * 
 * @param a First tensor
 * @param b Second tensor
 * @return Result tensor
 */
Tensor matmul(const Tensor& a, const Tensor& b);

/**
 * @brief Convolution
 * 
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias Bias tensor
 * @param stride Stride
 * @param padding Padding
 * @param dilation Dilation
 * @param groups Groups
 * @return Result tensor
 */
Tensor conv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              const std::vector<size_t>& stride, const std::vector<size_t>& padding,
              const std::vector<size_t>& dilation, size_t groups);

/**
 * @brief Max pooling
 * 
 * @param input Input tensor
 * @param kernel_size Kernel size
 * @param stride Stride
 * @param padding Padding
 * @param dilation Dilation
 * @param ceil_mode Whether to use ceil mode
 * @return Result tensor
 */
Tensor max_pool2d(const Tensor& input, const std::vector<size_t>& kernel_size,
                  const std::vector<size_t>& stride, const std::vector<size_t>& padding,
                  const std::vector<size_t>& dilation, bool ceil_mode);

/**
 * @brief Average pooling
 * 
 * @param input Input tensor
 * @param kernel_size Kernel size
 * @param stride Stride
 * @param padding Padding
 * @param ceil_mode Whether to use ceil mode
 * @param count_include_pad Whether to include padding in averaging
 * @return Result tensor
 */
Tensor avg_pool2d(const Tensor& input, const std::vector<size_t>& kernel_size,
                  const std::vector<size_t>& stride, const std::vector<size_t>& padding,
                  bool ceil_mode, bool count_include_pad);

/**
 * @brief ReLU activation
 * 
 * @param input Input tensor
 * @return Result tensor
 */
Tensor relu(const Tensor& input);

/**
 * @brief Sigmoid activation
 * 
 * @param input Input tensor
 * @return Result tensor
 */
Tensor sigmoid(const Tensor& input);

/**
 * @brief Tanh activation
 * 
 * @param input Input tensor
 * @return Result tensor
 */
Tensor tanh(const Tensor& input);

/**
 * @brief Softmax activation
 * 
 * @param input Input tensor
 * @param dim Dimension to apply softmax
 * @return Result tensor
 */
Tensor softmax(const Tensor& input, int dim);

/**
 * @brief Log softmax activation
 * 
 * @param input Input tensor
 * @param dim Dimension to apply log softmax
 * @return Result tensor
 */
Tensor log_softmax(const Tensor& input, int dim);

/**
 * @brief Batch normalization
 * 
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias Bias tensor
 * @param running_mean Running mean tensor
 * @param running_var Running variance tensor
 * @param training Whether in training mode
 * @param momentum Momentum
 * @param eps Epsilon
 * @return Result tensor
 */
Tensor batch_norm(const Tensor& input, const Tensor& weight, const Tensor& bias,
                  const Tensor& running_mean, const Tensor& running_var,
                  bool training, double momentum, double eps);

/**
 * @brief Dropout
 * 
 * @param input Input tensor
 * @param p Dropout probability
 * @param training Whether in training mode
 * @return Result tensor
 */
Tensor dropout(const Tensor& input, double p, bool training);

/**
 * @brief Linear transformation
 * 
 * @param input Input tensor
 * @param weight Weight tensor
 * @param bias Bias tensor
 * @return Result tensor
 */
Tensor linear(const Tensor& input, const Tensor& weight, const Tensor& bias);

/**
 * @brief Embedding
 * 
 * @param input Input tensor
 * @param weight Weight tensor
 * @param padding_idx Padding index
 * @param scale_grad_by_freq Whether to scale gradients by frequency
 * @param sparse Whether to use sparse gradients
 * @return Result tensor
 */
Tensor embedding(const Tensor& input, const Tensor& weight, int padding_idx,
                 bool scale_grad_by_freq, bool sparse);

/**
 * @brief Cross entropy loss
 * 
 * @param input Input tensor
 * @param target Target tensor
 * @param weight Weight tensor
 * @param ignore_index Ignore index
 * @param reduction Reduction type
 * @return Result tensor
 */
Tensor cross_entropy(const Tensor& input, const Tensor& target, const Tensor& weight,
                     int ignore_index, const std::string& reduction);

/**
 * @brief Mean squared error loss
 * 
 * @param input Input tensor
 * @param target Target tensor
 * @param reduction Reduction type
 * @return Result tensor
 */
Tensor mse_loss(const Tensor& input, const Tensor& target, const std::string& reduction);

/**
 * @brief Binary cross entropy loss
 * 
 * @param input Input tensor
 * @param target Target tensor
 * @param weight Weight tensor
 * @param reduction Reduction type
 * @return Result tensor
 */
Tensor binary_cross_entropy(const Tensor& input, const Tensor& target, const Tensor& weight,
                            const std::string& reduction);

/**
 * @brief Transpose tensor
 * 
 * @param input Input tensor
 * @param dim0 First dimension
 * @param dim1 Second dimension
 * @return Result tensor
 */
Tensor transpose(const Tensor& input, int dim0, int dim1);

/**
 * @brief Permute tensor
 * 
 * @param input Input tensor
 * @param dims Dimensions
 * @return Result tensor
 */
Tensor permute(const Tensor& input, const std::vector<int>& dims);

/**
 * @brief View tensor
 * 
 * @param input Input tensor
 * @param shape Shape
 * @return Result tensor
 */
Tensor view(const Tensor& input, const std::vector<size_t>& shape);

/**
 * @brief Reshape tensor
 * 
 * @param input Input tensor
 * @param shape Shape
 * @return Result tensor
 */
Tensor reshape(const Tensor& input, const std::vector<size_t>& shape);

/**
 * @brief Squeeze tensor
 * 
 * @param input Input tensor
 * @param dim Dimension
 * @return Result tensor
 */
Tensor squeeze(const Tensor& input, int dim);

/**
 * @brief Unsqueeze tensor
 * 
 * @param input Input tensor
 * @param dim Dimension
 * @return Result tensor
 */
Tensor unsqueeze(const Tensor& input, int dim);

/**
 * @brief Concatenate tensors
 * 
 * @param tensors Tensors
 * @param dim Dimension
 * @return Result tensor
 */
Tensor cat(const std::vector<Tensor>& tensors, int dim);

/**
 * @brief Stack tensors
 * 
 * @param tensors Tensors
 * @param dim Dimension
 * @return Result tensor
 */
Tensor stack(const std::vector<Tensor>& tensors, int dim);

/**
 * @brief Split tensor
 * 
 * @param input Input tensor
 * @param split_size Split size
 * @param dim Dimension
 * @return Result tensors
 */
std::vector<Tensor> split(const Tensor& input, size_t split_size, int dim);

/**
 * @brief Chunk tensor
 * 
 * @param input Input tensor
 * @param chunks Number of chunks
 * @param dim Dimension
 * @return Result tensors
 */
std::vector<Tensor> chunk(const Tensor& input, size_t chunks, int dim);

/**
 * @brief Sum tensor
 * 
 * @param input Input tensor
 * @param dim Dimension
 * @param keepdim Whether to keep dimensions
 * @return Result tensor
 */
Tensor sum(const Tensor& input, int dim, bool keepdim);

/**
 * @brief Mean tensor
 * 
 * @param input Input tensor
 * @param dim Dimension
 * @param keepdim Whether to keep dimensions
 * @return Result tensor
 */
Tensor mean(const Tensor& input, int dim, bool keepdim);

/**
 * @brief Max tensor
 * 
 * @param input Input tensor
 * @param dim Dimension
 * @param keepdim Whether to keep dimensions
 * @return Result tensor
 */
Tensor max(const Tensor& input, int dim, bool keepdim);

/**
 * @brief Min tensor
 * 
 * @param input Input tensor
 * @param dim Dimension
 * @param keepdim Whether to keep dimensions
 * @return Result tensor
 */
Tensor min(const Tensor& input, int dim, bool keepdim);

/**
 * @brief Argmax tensor
 * 
 * @param input Input tensor
 * @param dim Dimension
 * @param keepdim Whether to keep dimensions
 * @return Result tensor
 */
Tensor argmax(const Tensor& input, int dim, bool keepdim);

/**
 * @brief Argmin tensor
 * 
 * @param input Input tensor
 * @param dim Dimension
 * @param keepdim Whether to keep dimensions
 * @return Result tensor
 */
Tensor argmin(const Tensor& input, int dim, bool keepdim);

/**
 * @brief Norm tensor
 * 
 * @param input Input tensor
 * @param p Norm type
 * @param dim Dimension
 * @param keepdim Whether to keep dimensions
 * @return Result tensor
 */
Tensor norm(const Tensor& input, float p, int dim, bool keepdim);

} // namespace ops
} // namespace phynexus

#endif // PHYNEXUS_OPS_H
