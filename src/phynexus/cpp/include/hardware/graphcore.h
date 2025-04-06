/**
 * @file graphcore.h
 * @brief GraphCore IPU backend for specialized hardware acceleration
 */

#ifndef PHYNEXUS_HARDWARE_GRAPHCORE_H
#define PHYNEXUS_HARDWARE_GRAPHCORE_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace phynexus {
namespace hardware {

/**
 * @brief Configuration for GraphCore IPU initialization
 */
struct GraphCoreConfig {
    int num_ipus{1};
    std::string precision{"float16"};
    float memory_proportion{0.6};
    bool enable_half_partials{true};
    bool compile_only{false};
    int device_id{0};
    bool enable_stochastic_rounding{true};
    bool enable_replicated_graphs{false};
    int num_replicas{1};
    bool enable_pipelining{false};
    int pipeline_stages{1};
    bool enable_gradient_accumulation{false};
    int gradient_accumulation_steps{1};
    bool enable_synthetic_data{false};
    bool enable_profiling{false};
    std::string profiling_output_path{"./profile"};
};

/**
 * @brief GraphCore IPU backend for specialized hardware acceleration
 */
class GraphCoreBackend {
public:
    /**
     * @brief Initialize GraphCore IPU backend
     * @param config GraphCore IPU configuration
     * @return true if initialization succeeded, false otherwise
     */
    static bool initialize(const GraphCoreConfig& config = GraphCoreConfig());

    /**
     * @brief Finalize GraphCore IPU backend
     */
    static void finalize();

    /**
     * @brief Get the number of available IPUs
     * @return Number of available IPUs
     */
    static int get_ipu_count();

    /**
     * @brief Get information about the IPU device
     * @return IPU device information
     */
    static std::unordered_map<std::string, std::string> get_ipu_info();

    /**
     * @brief Compile a model for IPU execution
     * @param model_handle Model handle
     * @param inputs Input tensors
     * @return Compiled model handle
     */
    static void* compile_model(void* model_handle, const std::unordered_map<std::string, void*>& inputs);

    /**
     * @brief Execute a compiled model on IPU
     * @param compiled_model_handle Compiled model handle
     * @param inputs Input tensors
     * @param outputs Output tensors
     */
    static void execute_model(void* compiled_model_handle, const std::unordered_map<std::string, void*>& inputs, std::unordered_map<std::string, void*>& outputs);

    /**
     * @brief Optimize a model for IPU execution
     * @param model_handle Model handle
     * @param inputs Input tensors
     * @return Optimized model handle
     */
    static void* optimize_model(void* model_handle, const std::unordered_map<std::string, void*>& inputs);

    /**
     * @brief Create a pipeline model for IPU execution
     * @param model_handles Model handles for each pipeline stage
     * @param num_stages Number of pipeline stages
     * @return Pipeline model handle
     */
    static void* create_pipeline_model(void** model_handles, int num_stages);

    /**
     * @brief Create a replicated model for IPU execution
     * @param model_handle Model handle
     * @param num_replicas Number of replicas
     * @return Replicated model handle
     */
    static void* create_replicated_model(void* model_handle, int num_replicas);

    /**
     * @brief Allocate memory on IPU
     * @param size Size in bytes
     * @param dtype Data type
     * @return Pointer to allocated memory
     */
    static void* allocate_memory(size_t size, const std::string& dtype);

    /**
     * @brief Free memory on IPU
     * @param ptr Pointer to memory
     */
    static void free_memory(void* ptr);

    /**
     * @brief Copy data to IPU
     * @param dst Destination pointer on IPU
     * @param src Source pointer on host
     * @param size Size in bytes
     */
    static void copy_to_ipu(void* dst, const void* src, size_t size);

    /**
     * @brief Copy data from IPU
     * @param dst Destination pointer on host
     * @param src Source pointer on IPU
     * @param size Size in bytes
     */
    static void copy_from_ipu(void* dst, const void* src, size_t size);

private:
    static bool initialized_;
    static GraphCoreConfig config_;
};

} // namespace hardware
} // namespace phynexus

#endif // PHYNEXUS_HARDWARE_GRAPHCORE_H
