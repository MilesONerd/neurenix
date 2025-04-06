/**
 * @file deepspeed.h
 * @brief DeepSpeed backend for distributed training
 */

#ifndef PHYNEXUS_DISTRIBUTED_DEEPSPEED_H
#define PHYNEXUS_DISTRIBUTED_DEEPSPEED_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace phynexus {
namespace distributed {

/**
 * @brief Configuration for DeepSpeed initialization
 */
struct DeepSpeedConfig {
    bool use_fp16{true};
    bool use_zero{true};
    int zero_stage{1};
    bool offload_optimizer{false};
    bool offload_param{false};
    int gradient_accumulation_steps{1};
    bool gradient_clipping{true};
    float gradient_clipping_threshold{1.0};
    bool enable_communication_overlap{true};
    bool enable_aio{false};
    int aio_block_size{1048576};
    int aio_queue_depth{8};
    int aio_thread_count{1};
    bool enable_tensorboard{false};
    std::string tensorboard_output_path{"./runs"};
};

/**
 * @brief DeepSpeed backend for distributed training
 */
class DeepSpeedBackend {
public:
    /**
     * @brief Initialize DeepSpeed backend
     * @param config DeepSpeed configuration
     * @return true if initialization succeeded, false otherwise
     */
    static bool initialize(const DeepSpeedConfig& config = DeepSpeedConfig());

    /**
     * @brief Finalize DeepSpeed backend
     */
    static void finalize();

    /**
     * @brief Get the rank of the current process
     * @return Rank of the current process
     */
    static int get_rank();

    /**
     * @brief Get the local rank of the current process
     * @return Local rank of the current process
     */
    static int get_local_rank();

    /**
     * @brief Get the total number of processes
     * @return Total number of processes
     */
    static int get_world_size();

    /**
     * @brief Get the total number of local processes
     * @return Total number of local processes
     */
    static int get_local_size();

    /**
     * @brief Barrier synchronization
     */
    static void barrier();

    /**
     * @brief Initialize optimizer for ZeRO optimization
     * @param model_size Size of the model in parameters
     * @param dtype Data type of the model parameters
     * @param learning_rate Learning rate
     * @param weight_decay Weight decay
     * @return Optimizer handle
     */
    static void* initialize_optimizer(size_t model_size, const std::string& dtype, float learning_rate, float weight_decay);

    /**
     * @brief Forward pass with DeepSpeed
     * @param model_handle Model handle
     * @param inputs Input tensors
     * @param outputs Output tensors
     * @param is_training Whether this is a training pass
     */
    static void forward(void* model_handle, const std::unordered_map<std::string, void*>& inputs, std::unordered_map<std::string, void*>& outputs, bool is_training);

    /**
     * @brief Backward pass with DeepSpeed
     * @param model_handle Model handle
     * @param gradients Gradient tensors
     */
    static void backward(void* model_handle, const std::unordered_map<std::string, void*>& gradients);

    /**
     * @brief Optimizer step with DeepSpeed
     * @param optimizer_handle Optimizer handle
     */
    static void optimizer_step(void* optimizer_handle);

    /**
     * @brief Save checkpoint
     * @param model_handle Model handle
     * @param optimizer_handle Optimizer handle
     * @param checkpoint_path Path to save checkpoint
     */
    static void save_checkpoint(void* model_handle, void* optimizer_handle, const std::string& checkpoint_path);

    /**
     * @brief Load checkpoint
     * @param model_handle Model handle
     * @param optimizer_handle Optimizer handle
     * @param checkpoint_path Path to load checkpoint from
     * @return true if loading succeeded, false otherwise
     */
    static bool load_checkpoint(void* model_handle, void* optimizer_handle, const std::string& checkpoint_path);

private:
    static bool initialized_;
    static int rank_;
    static int local_rank_;
    static int world_size_;
    static int local_size_;
};

} // namespace distributed
} // namespace phynexus

#endif // PHYNEXUS_DISTRIBUTED_DEEPSPEED_H
