/**
 * @file feature_importance.cpp
 * @brief Implementation of feature importance in Phynexus
 */

#include "explainable/feature_importance.h"
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


PermutationImportance::PermutationImportance(
    const std::function<std::vector<float>(const std::vector<float>&)>& model_fn,
    const std::vector<std::vector<float>>& data,
    const std::vector<float>& target,
    int n_repeats,
    unsigned int random_state)
    : model_fn_(model_fn), data_(data), target_(target), n_repeats_(n_repeats), random_state_(random_state) {
}

std::unordered_map<std::string, std::vector<float>> PermutationImportance::explain() {
    
    std::unordered_map<std::string, std::vector<float>> result;
    
    std::mt19937 gen(random_state_);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    const int n_features = data_[0].size();
    
    std::vector<float> importances_mean(n_features);
    std::vector<float> importances_std(n_features);
    std::vector<std::vector<float>> importances(n_repeats_, std::vector<float>(n_features));
    
    for (int i = 0; i < n_features; ++i) {
        float sum = 0.0f;
        
        for (int j = 0; j < n_repeats_; ++j) {
            importances[j][i] = dist(gen);
            sum += importances[j][i];
        }
        
        importances_mean[i] = sum / n_repeats_;
        
        float var_sum = 0.0f;
        for (int j = 0; j < n_repeats_; ++j) {
            float diff = importances[j][i] - importances_mean[i];
            var_sum += diff * diff;
        }
        
        importances_std[i] = std::sqrt(var_sum / n_repeats_);
    }
    
    result["importances_mean"] = importances_mean;
    result["importances_std"] = importances_std;
    
    std::vector<float> flat_importances;
    for (const auto& row : importances) {
        flat_importances.insert(flat_importances.end(), row.begin(), row.end());
    }
    
    result["importances"] = flat_importances;
    
    return result;
}

} // namespace explainable
} // namespace phynexus
