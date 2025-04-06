/**
 * @file horovod.h
 * @brief Horovod backend for distributed training
 */

#ifndef PHYNEXUS_DISTRIBUTED_HOROVOD_H
#define PHYNEXUS_DISTRIBUTED_HOROVOD_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace phynexus {
namespace distributed {

/**
 * @brief Configuration for Horovod initialization
 */
struct HorovodConfig {
    bool use_gloo{true};
    bool use_mpi{true};
    bool use_nccl{true};
    int timeout_ms{30000};
    bool enable_tensor_fusion{true};
    int tensor_fusion_threshold{64 * 1024 * 1024};
    bool enable_auto_tuning{true};
    bool enable_hierarchical_allreduce{true};
    bool enable_adasum{false};
};

/**
 * @brief Horovod backend for distributed training
 */
class HorovodBackend {
public:
    /**
     * @brief Initialize Horovod backend
     * @param config Horovod configuration
     * @return true if initialization succeeded, false otherwise
     */
    static bool initialize(const HorovodConfig& config = HorovodConfig());

    /**
     * @brief Finalize Horovod backend
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
     * @brief Broadcast data from root to all processes
     * @param data Data to broadcast
     * @param count Number of elements
     * @param root Root process
     */
    template <typename T>
    static void broadcast(T* data, int count, int root);

    /**
     * @brief All-reduce operation
     * @param send_data Data to send
     * @param recv_data Buffer to receive data
     * @param count Number of elements
     * @param op Reduction operation
     */
    template <typename T>
    static void all_reduce(const T* send_data, T* recv_data, int count, const std::string& op);

    /**
     * @brief All-gather operation
     * @param send_data Data to send
     * @param recv_data Buffer to receive data
     * @param count Number of elements
     */
    template <typename T>
    static void all_gather(const T* send_data, T* recv_data, int count);

    /**
     * @brief All-to-all operation
     * @param send_data Data to send
     * @param recv_data Buffer to receive data
     * @param count Number of elements
     */
    template <typename T>
    static void all_to_all(const T* send_data, T* recv_data, int count);

private:
    static bool initialized_;
    static int rank_;
    static int local_rank_;
    static int world_size_;
    static int local_size_;
};

} // namespace distributed
} // namespace phynexus

#endif // PHYNEXUS_DISTRIBUTED_HOROVOD_H
