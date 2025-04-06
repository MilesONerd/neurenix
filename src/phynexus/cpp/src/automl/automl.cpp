/**
 * @file automl.cpp
 * @brief Implementation of AutoML module for Neurenix C++ backend.
 */

#include "automl/automl.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <iostream>

namespace phynexus {
namespace automl {


HyperparameterSearch::HyperparameterSearch(const ParamSpace& param_space, int max_iterations)
    : param_space_(param_space), max_iterations_(max_iterations), best_score_(std::numeric_limits<double>::lowest()) {}

const std::unordered_map<std::string, double>& HyperparameterSearch::get_best_params() const {
    return best_params_;
}

double HyperparameterSearch::get_best_score() const {
    return best_score_;
}


GridSearch::GridSearch(const ParamSpace& param_space, int max_iterations)
    : HyperparameterSearch(param_space, max_iterations) {}

std::unordered_map<std::string, double> GridSearch::search(const ObjectiveFunction& objective) {
    std::vector<std::string> param_names;
    std::vector<std::vector<double>> param_values;
    
    for (const auto& param : param_space_) {
        param_names.push_back(param.first);
        param_values.push_back(param.second);
    }
    
    std::vector<size_t> indices(param_names.size(), 0);
    bool done = false;
    int iterations = 0;
    
    while (!done && iterations < max_iterations_) {
        std::unordered_map<std::string, double> params;
        for (size_t i = 0; i < param_names.size(); ++i) {
            params[param_names[i]] = param_values[i][indices[i]];
        }
        
        double score = objective(params);
        
        if (score > best_score_) {
            best_score_ = score;
            best_params_ = params;
        }
        
        size_t idx = param_names.size() - 1;
        indices[idx]++;
        
        while (indices[idx] >= param_values[idx].size()) {
            indices[idx] = 0;
            
            if (idx == 0) {
                done = true;
                break;
            }
            
            idx--;
            indices[idx]++;
        }
        
        iterations++;
    }
    
    return best_params_;
}
