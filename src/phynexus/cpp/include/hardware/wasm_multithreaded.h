/**
 * @file wasm_multithreaded.h
 * @brief WebAssembly multithreaded support for the Phynexus engine.
 *
 * This file provides multithreading capabilities for WebAssembly, enabling
 * parallel execution of computations in browser environments using Web Workers
 * and SharedArrayBuffer.
 */

#ifndef PHYNEXUS_WASM_MULTITHREADED_H
#define PHYNEXUS_WASM_MULTITHREADED_H

#include <memory>
#include <string>
#include <vector>
#include "phynexus/tensor.h"
#include "phynexus/model.h"

namespace phynexus {
namespace wasm {

/**
 * @brief Check if WebAssembly multithreading is supported in the current environment.
 * 
 * @return true if WebAssembly multithreading is supported, false otherwise.
 */
bool is_multithreading_supported();

/**
 * @brief Enable multithreading for WebAssembly if available.
 * 
 * @return true if multithreading was enabled, false otherwise.
 */
bool enable_multithreading();

/**
 * @brief Get the number of available worker threads.
 * 
 * @return Number of available worker threads, or 1 if multithreading is not supported.
 */
size_t get_num_workers();

/**
 * @brief Precision modes for WebAssembly multithreaded operations.
 */
enum class MultithreadedPrecision {
    FP32,  ///< 32-bit floating point precision
    FP16,  ///< 16-bit floating point precision
    Mixed  ///< Mixed precision (FP16 for computation, FP32 for accumulation)
};

/**
 * @brief WebAssembly multithreaded backend for parallel execution.
 */
class MultithreadedBackend {
public:
    /**
     * @brief Construct a new MultithreadedBackend object.
     */
    MultithreadedBackend();

    /**
     * @brief Destroy the MultithreadedBackend object.
     */
    ~MultithreadedBackend();

    /**
     * @brief Initialize the multithreaded backend.
     * 
     * @return true if initialization was successful, false otherwise.
     */
    bool initialize();

    /**
     * @brief Clean up resources used by the multithreaded backend.
     */
    void cleanup();

    /**
     * @brief Set the precision mode for multithreaded operations.
     * 
     * @param precision Precision mode to use.
     */
    void set_precision(MultithreadedPrecision precision);

    /**
     * @brief Get the current precision mode.
     * 
     * @return Current precision mode.
     */
    MultithreadedPrecision get_precision() const;

    /**
     * @brief Perform matrix multiplication using multiple threads.
     * 
     * @param a First tensor.
     * @param b Second tensor.
     * @return Result of matrix multiplication.
     */
    Tensor parallel_matmul(const Tensor& a, const Tensor& b);

    /**
     * @brief Perform 2D convolution using multiple threads.
     * 
     * @param input Input tensor of shape (N, C_in, H_in, W_in).
     * @param weight Weight tensor of shape (C_out, C_in/groups, H_kernel, W_kernel).
     * @param bias Optional bias tensor of shape (C_out).
     * @param stride Stride of the convolution.
     * @param padding Padding added to all sides of the input.
     * @param dilation Spacing between kernel elements.
     * @param groups Number of blocked connections from input to output channels.
     * @return Result of convolution.
     */
    Tensor parallel_conv2d(
        const Tensor& input,
        const Tensor& weight,
        const Tensor* bias,
        const std::pair<size_t, size_t>& stride,
        const std::pair<size_t, size_t>& padding,
        const std::pair<size_t, size_t>& dilation,
        size_t groups
    );

    /**
     * @brief Apply a function to each tensor in parallel.
     * 
     * @param func Function to apply.
     * @param tensors List of tensors to process.
     * @return List of processed tensors.
     */
    std::vector<Tensor> parallel_map(
        std::function<Tensor(const Tensor&)> func,
        const std::vector<Tensor>& tensors
    );

    /**
     * @brief Process multiple batches in parallel using a model.
     * 
     * @param model Model to use for processing.
     * @param batches List of input batches.
     * @return List of model outputs.
     */
    std::vector<Tensor> parallel_batch_processing(
        const std::shared_ptr<Model>& model,
        const std::vector<Tensor>& batches
    );

private:
    bool initialized_;
    MultithreadedPrecision precision_;
    void* thread_pool_;
    size_t num_threads_;
};

} // namespace wasm
} // namespace phynexus

#endif // PHYNEXUS_WASM_MULTITHREADED_H
