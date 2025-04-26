/**
 * @file arm.cpp
 * @brief ARM hardware acceleration implementation
 */

#include "phynexus/hardware/arm.h"
#include "phynexus/error.h"
#include <stdexcept>
#include <cstring>
#include <arm_neon.h>
#include <arm_sve.h>
#include <arm_compute/runtime/NEON/NEFunctions.h>
#include <arm_compute/runtime/NEON/NEScheduler.h>
#include <arm_compute/core/Types.h>
#include <arm_compute/runtime/Tensor.h>
#include <arm_compute/runtime/TensorAllocator.h>

namespace phynexus {
namespace hardware {

struct ARMDeviceInfo {
    bool has_neon;
    bool has_sve;
    bool has_ethos_u;
    size_t sve_vector_length;
    std::string device_name;
};

static ARMDeviceInfo g_arm_device_info;
static bool g_arm_initialized = false;

bool arm_init() {
    if (g_arm_initialized) {
        return true;
    }
    
    #ifdef __ARM_NEON
    g_arm_device_info.has_neon = true;
    #else
    g_arm_device_info.has_neon = false;
    #endif
    
    #ifdef __ARM_FEATURE_SVE
    g_arm_device_info.has_sve = true;
    g_arm_device_info.sve_vector_length = svcntb();
    #else
    g_arm_device_info.has_sve = false;
    g_arm_device_info.sve_vector_length = 0;
    #endif
    
    g_arm_device_info.has_ethos_u = false;
    
    g_arm_device_info.device_name = "ARM CPU";
    if (g_arm_device_info.has_neon) {
        g_arm_device_info.device_name += " with NEON";
    }
    if (g_arm_device_info.has_sve) {
        g_arm_device_info.device_name += " with SVE";
    }
    if (g_arm_device_info.has_ethos_u) {
        g_arm_device_info.device_name += " with Ethos-U NPU";
    }
    
    arm_compute::NEScheduler::get().set_num_threads(std::thread::hardware_concurrency());
    
    g_arm_initialized = true;
    return true;
}

bool arm_is_available() {
    if (!g_arm_initialized) {
        arm_init();
    }
    return g_arm_device_info.has_neon || g_arm_device_info.has_sve;
}

int arm_get_device_count() {
    return arm_is_available() ? 1 : 0;
}

bool arm_set_device(int device_id) {
    if (device_id != 0) {
        return false;
    }
    return arm_is_available();
}

bool arm_get_device_properties(int device_id, ARMDeviceProperties* props) {
    if (device_id != 0 || !arm_is_available() || !props) {
        return false;
    }
    
    props->has_neon = g_arm_device_info.has_neon;
    props->has_sve = g_arm_device_info.has_sve;
    props->has_ethos_u = g_arm_device_info.has_ethos_u;
    props->sve_vector_length = g_arm_device_info.sve_vector_length;
    strncpy(props->device_name, g_arm_device_info.device_name.c_str(), sizeof(props->device_name) - 1);
    props->device_name[sizeof(props->device_name) - 1] = '\0';
    
    return true;
}

bool arm_malloc(void** ptr, size_t size) {
    if (!ptr) {
        return false;
    }
    
    *ptr = std::aligned_alloc(64, size);  // 64-byte alignment for SIMD
    return *ptr != nullptr;
}

bool arm_free(void* ptr) {
    if (!ptr) {
        return false;
    }
    
    std::free(ptr);
    return true;
}

bool arm_memcpy_host_to_device(void* dst, const void* src, size_t size) {
    if (!dst || !src) {
        return false;
    }
    
    std::memcpy(dst, src, size);
    return true;
}

bool arm_memcpy_device_to_host(void* dst, const void* src, size_t size) {
    if (!dst || !src) {
        return false;
    }
    
    std::memcpy(dst, src, size);
    return true;
}

bool arm_memcpy_device_to_device(void* dst, const void* src, size_t size) {
    if (!dst || !src) {
        return false;
    }
    
    std::memcpy(dst, src, size);
    return true;
}

bool arm_device_synchronize() {
    return true;
}

bool arm_neon_add(const float* a, const float* b, float* c, size_t size) {
    #ifdef __ARM_NEON
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vaddq_f32(va, vb);
        vst1q_f32(c + i, vc);
    }
    for (; i < size; i++) {
        c[i] = a[i] + b[i];
    }
    return true;
    #else
    return false;
    #endif
}

bool arm_neon_multiply(const float* a, const float* b, float* c, size_t size) {
    #ifdef __ARM_NEON
    size_t i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vmulq_f32(va, vb);
        vst1q_f32(c + i, vc);
    }
    for (; i < size; i++) {
        c[i] = a[i] * b[i];
    }
    return true;
    #else
    return false;
    #endif
}

bool arm_sve_add(const float* a, const float* b, float* c, size_t size) {
    #ifdef __ARM_FEATURE_SVE
    size_t i = 0;
    svbool_t pg = svptrue_b32();
    
    for (; i + svcntw() <= size; i += svcntw()) {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        svfloat32_t vc = svadd_f32_z(pg, va, vb);
        svst1_f32(pg, c + i, vc);
    }
    
    if (i < size) {
        svbool_t pg_rem = svwhilelt_b32(i, size);
        svfloat32_t va = svld1_f32(pg_rem, a + i);
        svfloat32_t vb = svld1_f32(pg_rem, b + i);
        svfloat32_t vc = svadd_f32_z(pg_rem, va, vb);
        svst1_f32(pg_rem, c + i, vc);
    }
    return true;
    #else
    return false;
    #endif
}

bool arm_sve_multiply(const float* a, const float* b, float* c, size_t size) {
    #ifdef __ARM_FEATURE_SVE
    size_t i = 0;
    svbool_t pg = svptrue_b32();
    
    for (; i + svcntw() <= size; i += svcntw()) {
        svfloat32_t va = svld1_f32(pg, a + i);
        svfloat32_t vb = svld1_f32(pg, b + i);
        svfloat32_t vc = svmul_f32_z(pg, va, vb);
        svst1_f32(pg, c + i, vc);
    }
    
    if (i < size) {
        svbool_t pg_rem = svwhilelt_b32(i, size);
        svfloat32_t va = svld1_f32(pg_rem, a + i);
        svfloat32_t vb = svld1_f32(pg_rem, b + i);
        svfloat32_t vc = svmul_f32_z(pg_rem, va, vb);
        svst1_f32(pg_rem, c + i, vc);
    }
    return true;
    #else
    return false;
    #endif
}

bool arm_acl_conv2d(const float* input, const float* weights, const float* bias,
                    float* output, const Conv2DParams& params) {
    try {
        arm_compute::TensorShape input_shape(params.input_width, params.input_height, 
                                           params.input_channels, params.batch_size);
        arm_compute::TensorShape weights_shape(params.kernel_width, params.kernel_height,
                                             params.input_channels, params.output_channels);
        arm_compute::TensorShape bias_shape(params.output_channels);
        arm_compute::TensorShape output_shape(params.output_width, params.output_height,
                                            params.output_channels, params.batch_size);
        
        arm_compute::Tensor input_tensor;
        arm_compute::Tensor weights_tensor;
        arm_compute::Tensor bias_tensor;
        arm_compute::Tensor output_tensor;
        
        input_tensor.allocator()->init(arm_compute::TensorInfo(input_shape, 1, arm_compute::DataType::F32));
        weights_tensor.allocator()->init(arm_compute::TensorInfo(weights_shape, 1, arm_compute::DataType::F32));
        bias_tensor.allocator()->init(arm_compute::TensorInfo(bias_shape, 1, arm_compute::DataType::F32));
        output_tensor.allocator()->init(arm_compute::TensorInfo(output_shape, 1, arm_compute::DataType::F32));
        
        arm_compute::NEConvolutionLayer conv;
        arm_compute::PadStrideInfo pad_stride_info(params.stride_width, params.stride_height,
                                                  params.padding_width, params.padding_height);
        
        conv.configure(&input_tensor, &weights_tensor, &bias_tensor, &output_tensor, pad_stride_info);
        
        input_tensor.allocator()->allocate();
        weights_tensor.allocator()->allocate();
        bias_tensor.allocator()->allocate();
        output_tensor.allocator()->allocate();
        
        std::memcpy(input_tensor.buffer(), input, input_tensor.info()->total_size());
        std::memcpy(weights_tensor.buffer(), weights, weights_tensor.info()->total_size());
        std::memcpy(bias_tensor.buffer(), bias, bias_tensor.info()->total_size());
        
        conv.run();
        
        std::memcpy(output, output_tensor.buffer(), output_tensor.info()->total_size());
        
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool arm_ethos_u_conv2d(const float* input, const float* weights, const float* bias,
                        float* output, const Conv2DParams& params) {
    return false;
}

} // namespace hardware
} // namespace phynexus
