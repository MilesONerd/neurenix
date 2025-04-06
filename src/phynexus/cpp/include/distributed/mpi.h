/**
 * @file mpi.h
 * @brief MPI backend for distributed training
 */

#ifndef PHYNEXUS_DISTRIBUTED_MPI_H
#define PHYNEXUS_DISTRIBUTED_MPI_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace phynexus {
namespace distributed {

/**
 * @brief Configuration for MPI initialization
 */
struct MPIConfig {
    bool use_threads{true};
    std::string backend{"openmpi"};
    int timeout_ms{30000};
    bool enable_cuda_aware{false};
};

/**
 * @brief MPI backend for distributed training
 */
class MPIBackend {
public:
    /**
     * @brief Initialize MPI backend
     * @param config MPI configuration
     * @return true if initialization succeeded, false otherwise
     */
    static bool initialize(const MPIConfig& config = MPIConfig());

    /**
     * @brief Finalize MPI backend
     */
    static void finalize();

    /**
     * @brief Get the rank of the current process
     * @return Rank of the current process
     */
    static int get_rank();

    /**
     * @brief Get the total number of processes
     * @return Total number of processes
     */
    static int get_world_size();

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
     * @brief Send data to a specific process
     * @param data Data to send
     * @param count Number of elements
     * @param dest Destination process
     * @param tag Message tag
     */
    template <typename T>
    static void send(const T* data, int count, int dest, int tag);

    /**
     * @brief Receive data from a specific process
     * @param data Buffer to receive data
     * @param count Number of elements
     * @param source Source process
     * @param tag Message tag
     */
    template <typename T>
    static void recv(T* data, int count, int source, int tag);

private:
    static bool initialized_;
    static int rank_;
    static int world_size_;
};

} // namespace distributed
} // namespace phynexus

#endif // PHYNEXUS_DISTRIBUTED_MPI_H
