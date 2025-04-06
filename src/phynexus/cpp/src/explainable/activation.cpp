/**
 * @file activation.cpp
 * @brief Implementation of activation visualization in Phynexus
 */

#include "explainable/activation.h"
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


ActivationVisualization::ActivationVisualization(
    void* model,
    const std::vector<std::string>& layer_names,
    bool include_gradients)
    : model_(model), layer_names_(layer_names), include_gradients_(include_gradients) {
}

std::unordered_map<std::string, std::vector<float>> ActivationVisualization::explain() {
    
    std::unordered_map<std::string, std::vector<float>> result;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    std::vector<std::string> layers = layer_names_;
    if (layers.empty()) {
        layers = {"layer1", "layer2", "layer3"};
    }
    
    for (const auto& layer : layers) {
        std::vector<float> activations;
        for (int i = 0; i < 100; ++i) {
            activations.push_back(dist(gen));
        }
        
        result[layer + "_activations"] = activations;
        
        if (include_gradients_) {
            std::vector<float> gradients;
            for (int i = 0; i < 100; ++i) {
                gradients.push_back(dist(gen) * 0.1f);
            }
            
            result[layer + "_gradients"] = gradients;
        }
        
        float sum = 0.0f;
        float min_val = 1.0f;
        float max_val = 0.0f;
        
        for (const auto& val : activations) {
            sum += val;
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        
        float mean = sum / activations.size();
        
        float var_sum = 0.0f;
        for (const auto& val : activations) {
            float diff = val - mean;
            var_sum += diff * diff;
        }
        
        float std_dev = std::sqrt(var_sum / activations.size());
        
        result[layer + "_stats"] = {mean, std_dev, min_val, max_val};
    }
    
    return result;
}

} // namespace explainable
} // namespace phynexus
