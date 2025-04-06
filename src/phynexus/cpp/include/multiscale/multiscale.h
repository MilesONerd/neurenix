/**
 * @file multiscale.h
 * @brief Header file for multi-scale models in Phynexus
 */

#ifndef PHYNEXUS_MULTISCALE_H
#define PHYNEXUS_MULTISCALE_H

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

namespace phynexus {
namespace multiscale {

/**
 * @brief Base class for multi-scale models
 */
class MultiScaleModel {
public:
    MultiScaleModel(int input_channels, int num_scales = 3, float scale_factor = 0.5);
    virtual ~MultiScaleModel() = default;
    
    /**
     * @brief Forward pass through the multi-scale model
     * @param x Input tensor
     * @return Output tensor after multi-scale processing
     */
    virtual std::vector<float> forward(const std::vector<float>& x) = 0;
    
protected:
    int input_channels_;
    int num_scales_;
    float scale_factor_;
    
    /**
     * @brief Generate inputs at different scales
     * @param x Original input tensor
     * @return List of tensors at different scales
     */
    std::vector<std::vector<float>> generate_multi_scale_inputs(const std::vector<float>& x);
};

/**
 * @brief Pyramid Network for multi-scale feature extraction
 */
class PyramidNetwork : public MultiScaleModel {
public:
    PyramidNetwork(int input_channels, const std::vector<int>& hidden_channels, 
                  int num_scales = 3, float scale_factor = 0.5);
    
    /**
     * @brief Forward pass through the pyramid network
     * @param x Input tensor
     * @return Output tensor after pyramid network processing
     */
    std::vector<float> forward(const std::vector<float>& x) override;
    
private:
    std::vector<int> hidden_channels_;
};

/**
 * @brief U-Net architecture for multi-scale processing
 */
class UNet : public MultiScaleModel {
public:
    UNet(int input_channels, const std::vector<int>& hidden_channels, 
        int output_channels, int num_scales = 4, float scale_factor = 0.5);
    
    /**
     * @brief Forward pass through the U-Net
     * @param x Input tensor
     * @return Output tensor after U-Net processing
     */
    std::vector<float> forward(const std::vector<float>& x) override;
    
private:
    std::vector<int> hidden_channels_;
    int output_channels_;
};

/**
 * @brief Base class for multi-scale pooling operations
 */
class MultiScalePooling {
public:
    MultiScalePooling(const std::pair<int, int>& output_size, const std::string& pool_type = "avg");
    virtual ~MultiScalePooling() = default;
    
    /**
     * @brief Forward pass through the multi-scale pooling module
     * @param x Input tensor
     * @return Pooled tensor
     */
    virtual std::vector<float> forward(const std::vector<float>& x);
    
protected:
    std::pair<int, int> output_size_;
    std::string pool_type_;
};

/**
 * @brief Pyramid Pooling Module (PPM) as used in PSPNet
 */
class PyramidPooling : public MultiScalePooling {
public:
    PyramidPooling(int in_channels, int out_channels, 
                  const std::vector<int>& pool_sizes = {1, 2, 3, 6}, 
                  const std::string& pool_type = "avg");
    
    /**
     * @brief Forward pass through the pyramid pooling module
     * @param x Input tensor
     * @return Concatenated multi-scale features
     */
    std::vector<float> forward(const std::vector<float>& x) override;
    
private:
    int in_channels_;
    int out_channels_;
    std::vector<int> pool_sizes_;
};

/**
 * @brief Spatial Pyramid Pooling (SPP) as introduced in SPPNet
 */
class SpatialPyramidPooling : public MultiScalePooling {
public:
    SpatialPyramidPooling(const std::vector<int>& output_sizes = {1, 2, 4}, 
                         const std::string& pool_type = "max");
    
    /**
     * @brief Forward pass through the spatial pyramid pooling module
     * @param x Input tensor
     * @return Flattened fixed-length representation
     */
    std::vector<float> forward(const std::vector<float>& x) override;
    
private:
    std::vector<int> output_sizes_;
};

/**
 * @brief Base class for feature fusion operations
 */
class FeatureFusion {
public:
    FeatureFusion(int in_channels, int out_channels);
    virtual ~FeatureFusion() = default;
    
    /**
     * @brief Forward pass through the feature fusion module
     * @param features List of feature tensors to fuse
     * @return Fused feature tensor
     */
    virtual std::vector<float> forward(const std::vector<std::vector<float>>& features);
    
protected:
    int in_channels_;
    int out_channels_;
};

/**
 * @brief Scale Fusion module for combining features from different scales
 */
class ScaleFusion : public FeatureFusion {
public:
    ScaleFusion(int in_channels, int out_channels, 
               const std::string& fusion_mode = "concat", 
               const std::string& target_scale = "largest");
    
    /**
     * @brief Forward pass through the scale fusion module
     * @param features List of feature tensors from different scales
     * @return Fused feature tensor
     */
    std::vector<float> forward(const std::vector<std::vector<float>>& features) override;
    
private:
    std::string fusion_mode_;
    std::string target_scale_;
    
    /**
     * @brief Determine the target size for feature resizing
     * @param features List of feature tensors
     * @return Target height and width
     */
    std::pair<int, int> get_target_size(const std::vector<std::vector<float>>& features);
};

/**
 * @brief Base class for multi-scale transformations
 */
class MultiScaleTransform {
public:
    MultiScaleTransform() = default;
    virtual ~MultiScaleTransform() = default;
    
    /**
     * @brief Forward pass through the multi-scale transform module
     * @param x Input tensor
     * @return List of tensors at different scales
     */
    virtual std::vector<std::vector<float>> forward(const std::vector<float>& x) = 0;
};

/**
 * @brief Rescale transform for generating multi-scale representations
 */
class Rescale : public MultiScaleTransform {
public:
    Rescale(const std::vector<float>& scales, 
           const std::string& mode = "bilinear", 
           bool align_corners = false);
    
    /**
     * @brief Forward pass through the rescale transform
     * @param x Input tensor
     * @return List of tensors at different scales
     */
    std::vector<std::vector<float>> forward(const std::vector<float>& x) override;
    
private:
    std::vector<float> scales_;
    std::string mode_;
    bool align_corners_;
};

/**
 * @brief Pyramid downsampling transform for generating multi-scale representations
 */
class PyramidDownsample : public MultiScaleTransform {
public:
    PyramidDownsample(int num_levels = 3, 
                     float downsample_factor = 0.5, 
                     const std::string& mode = "pool");
    
    /**
     * @brief Forward pass through the pyramid downsampling transform
     * @param x Input tensor
     * @return List of tensors at different scales
     */
    std::vector<std::vector<float>> forward(const std::vector<float>& x) override;
    
private:
    int num_levels_;
    float downsample_factor_;
    std::string mode_;
};

} // namespace multiscale
} // namespace phynexus

#endif // PHYNEXUS_MULTISCALE_H
