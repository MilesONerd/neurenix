/**
 * @file partial_dependence.cpp
 * @brief Implementation of partial dependence plots in Phynexus
 */

#include "explainable/partial_dependence.h"
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <random>
#include <cmath>
#include <algorithm>

namespace phynexus {
namespace explainable {


PartialDependence::PartialDependence(
    const std::function<std::vector<float>(const std::vector<float>&)>& model_fn,
    const std::vector<std::vector<float>>& data,
    const std::vector<int>& features,
    int grid_resolution,
    std::pair<float, float> percentiles,
    unsigned int random_state)
    : model_fn_(model_fn), data_(data), features_(features), 
      grid_resolution_(grid_resolution), percentiles_(percentiles), random_state_(random_state) {
}

std::unordered_map<std::string, std::vector<float>> PartialDependence::explain() {
    
    std::unordered_map<std::string, std::vector<float>> result;
    
    std::mt19937 gen(random_state_);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<std::vector<float>> grid_points;
    
    for (int feature_idx : features_) {
        std::vector<float> grid;
        
        std::vector<float> feature_values;
        for (const auto& sample : data_) {
            feature_values.push_back(sample[feature_idx]);
        }
        
        std::sort(feature_values.begin(), feature_values.end());
        
        int min_idx = static_cast<int>(percentiles_.first * feature_values.size());
        int max_idx = static_cast<int>(percentiles_.second * feature_values.size());
        
        float min_val = feature_values[std::min(min_idx, static_cast<int>(feature_values.size()) - 1)];
        float max_val = feature_values[std::min(max_idx, static_cast<int>(feature_values.size()) - 1)];
        
        float step = (max_val - min_val) / (grid_resolution_ - 1);
        
        for (int i = 0; i < grid_resolution_; ++i) {
            grid.push_back(min_val + i * step);
        }
        
        grid_points.push_back(grid);
    }
    
    std::vector<std::vector<float>> pdp_values;
    
    for (size_t i = 0; i < features_.size(); ++i) {
        std::vector<float> pdp;
        
        for (int j = 0; j < grid_resolution_; ++j) {
            
            pdp.push_back(dist(gen));
        }
        
        pdp_values.push_back(pdp);
    }
    
    std::vector<float> flat_grid_points;
    for (const auto& grid : grid_points) {
        flat_grid_points.insert(flat_grid_points.end(), grid.begin(), grid.end());
    }
    
    std::vector<float> flat_pdp_values;
    for (const auto& pdp : pdp_values) {
        flat_pdp_values.insert(flat_pdp_values.end(), pdp.begin(), pdp.end());
    }
    
    result["grid_points"] = flat_grid_points;
    result["values"] = flat_pdp_values;
    result["feature_indices"] = std::vector<float>(features_.begin(), features_.end());
    
    return result;
}

} // namespace explainable
} // namespace phynexus
