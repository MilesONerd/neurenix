/**
 * @file counterfactual.cpp
 * @brief Implementation of counterfactual explanations in Phynexus
 */

#include "explainable/counterfactual.h"
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


Counterfactual::Counterfactual(
    const std::function<std::vector<float>(const std::vector<float>&)>& model_fn,
    const std::vector<float>& sample,
    int target_class,
    float target_pred,
    const std::vector<int>& categorical_features,
    const std::unordered_map<int, std::pair<float, float>>& feature_ranges,
    int max_iter,
    unsigned int random_state)
    : model_fn_(model_fn), sample_(sample), target_class_(target_class), target_pred_(target_pred),
      categorical_features_(categorical_features), feature_ranges_(feature_ranges),
      max_iter_(max_iter), random_state_(random_state) {
}

std::unordered_map<std::string, std::vector<float>> Counterfactual::explain() {
    
    std::unordered_map<std::string, std::vector<float>> result;
    
    std::mt19937 gen(random_state_);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    std::vector<float> counterfactual = sample_;
    
    for (size_t i = 0; i < counterfactual.size(); ++i) {
        if (dist(gen) > 0.7f) {
            counterfactual[i] += dist(gen) * 0.1f;
        }
    }
    
    std::vector<float> counterfactual_pred = model_fn_(counterfactual);
    
    std::vector<float> diff;
    for (size_t i = 0; i < sample_.size(); ++i) {
        diff.push_back(counterfactual[i] - sample_[i]);
    }
    
    result["counterfactual"] = counterfactual;
    result["counterfactual_prediction"] = counterfactual_pred;
    result["difference"] = diff;
    
    return result;
}

} // namespace explainable
} // namespace phynexus
