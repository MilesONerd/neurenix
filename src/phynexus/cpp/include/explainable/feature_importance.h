/**
 * @file feature_importance.h
 * @brief Header file for feature importance in Phynexus
 */

#ifndef PHYNEXUS_EXPLAINABLE_FEATURE_IMPORTANCE_H
#define PHYNEXUS_EXPLAINABLE_FEATURE_IMPORTANCE_H

#include "explainable/explainable.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

namespace phynexus {
namespace explainable {

/**
 * @brief Base class for feature importance
 */
class FeatureImportance : public Explainer {
public:
    virtual ~FeatureImportance() = default;
    
    /**
     * @brief Get the name of the explainer
     * @return Name of the explainer
     */
    std::string name() const override { return "FeatureImportance"; }
};

/**
 * @brief Permutation importance implementation
 */
class PermutationImportance : public FeatureImportance {
public:
    PermutationImportance(const std::function<std::vector<float>(const std::vector<float>&)>& model_fn,
                          const std::vector<std::vector<float>>& data,
                          const std::vector<float>& target,
                          int n_repeats = 5,
                          unsigned int random_state = 0);
    
    /**
     * @brief Explain a model prediction
     * @return Explanation result as a map of strings to values
     */
    std::unordered_map<std::string, std::vector<float>> explain() override;
    
private:
    std::function<std::vector<float>(const std::vector<float>&)> model_fn_;
    std::vector<std::vector<float>> data_;
    std::vector<float> target_;
    int n_repeats_;
    unsigned int random_state_;
};

} // namespace explainable
} // namespace phynexus

#endif // PHYNEXUS_EXPLAINABLE_FEATURE_IMPORTANCE_H
