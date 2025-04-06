/**
 * @file fpga.h
 * @brief FPGA backend for specialized hardware acceleration
 */

#ifndef PHYNEXUS_HARDWARE_FPGA_H
#define PHYNEXUS_HARDWARE_FPGA_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

namespace phynexus {
namespace hardware {

/**
 * @brief FPGA framework type
 */
enum class FPGAFramework {
    OpenCL,
    Vitis,
    OpenVINO
};

/**
 * @brief Configuration for FPGA initialization
 */
struct FPGAConfig {
    FPGAFramework framework{FPGAFramework::OpenCL};
    int device_id{0};
    int platform_id{0};
    std::string bitstream{""};
    bool enable_profiling{false};
    bool enable_optimization{true};
    int num_compute_units{1};
    bool enable_emulation{false};
    std::string emulation_mode{"hw"};
    bool enable_debug{false};
    std::string debug_level{"info"};
    bool enable_memory_bank_mapping{false};
    std::unordered_map<std::string, int> memory_bank_mapping;
};

/**
 * @brief FPGA backend for specialized hardware acceleration
 */
class FPGABackend {
public:
    /**
     * @brief Initialize FPGA backend
     * @param config FPGA configuration
     * @return true if initialization succeeded, false otherwise
     */
    static bool initialize(const FPGAConfig& config = FPGAConfig());

    /**
     * @brief Finalize FPGA backend
     */
    static void finalize();

    /**
     * @brief Get the number of available FPGA devices
     * @return Number of available FPGA devices
     */
    static int get_device_count();

    /**
     * @brief Get information about the FPGA device
     * @return FPGA device information
     */
    static std::unordered_map<std::string, std::string> get_device_info();

    /**
     * @brief Load a bitstream to the FPGA
     * @param bitstream_path Path to the bitstream file
     * @return Bitstream handle
     */
    static void* load_bitstream(const std::string& bitstream_path);

    /**
     * @brief Create a kernel from a loaded bitstream
     * @param bitstream_handle Bitstream handle
     * @param kernel_name Name of the kernel
     * @return Kernel handle
     */
    static void* create_kernel(void* bitstream_handle, const std::string& kernel_name);

    /**
     * @brief Execute a kernel on the FPGA
     * @param kernel_handle Kernel handle
     * @param args Kernel arguments
     * @param global_work_size Global work size
     * @param local_work_size Local work size
     */
    static void execute_kernel(void* kernel_handle, const std::vector<void*>& args, const std::vector<size_t>& global_work_size, const std::vector<size_t>& local_work_size);

    /**
     * @brief Allocate memory on the FPGA
     * @param size Size in bytes
     * @param memory_bank Memory bank to allocate from
     * @return Pointer to allocated memory
     */
    static void* allocate_memory(size_t size, int memory_bank = 0);

    /**
     * @brief Free memory on the FPGA
     * @param ptr Pointer to memory
     */
    static void free_memory(void* ptr);

    /**
     * @brief Copy data to the FPGA
     * @param dst Destination pointer on FPGA
     * @param src Source pointer on host
     * @param size Size in bytes
     */
    static void copy_to_fpga(void* dst, const void* src, size_t size);

    /**
     * @brief Copy data from the FPGA
     * @param dst Destination pointer on host
     * @param src Source pointer on FPGA
     * @param size Size in bytes
     */
    static void copy_from_fpga(void* dst, const void* src, size_t size);

    /**
     * @brief Create a buffer on the FPGA
     * @param size Size in bytes
     * @param read_only Whether the buffer is read-only
     * @param memory_bank Memory bank to allocate from
     * @return Buffer handle
     */
    static void* create_buffer(size_t size, bool read_only = false, int memory_bank = 0);

    /**
     * @brief Set a kernel argument
     * @param kernel_handle Kernel handle
     * @param arg_index Argument index
     * @param arg_size Argument size
     * @param arg_value Argument value
     */
    static void set_kernel_arg(void* kernel_handle, int arg_index, size_t arg_size, const void* arg_value);

    /**
     * @brief Get the framework type
     * @return Framework type
     */
    static FPGAFramework get_framework();

private:
    static bool initialized_;
    static FPGAConfig config_;
};

} // namespace hardware
} // namespace phynexus

#endif // PHYNEXUS_HARDWARE_FPGA_H
