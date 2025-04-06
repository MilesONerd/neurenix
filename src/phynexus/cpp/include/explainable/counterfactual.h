/**
 * @file counterfactual.h
 * @brief Header file for counterfactual explanations in Phynexus
 */

#ifndef PHYNEXUS_EXPLAINABLE_COUNTERFACTUAL_H
#define PHYNEXUS_EXPLAINABLE_COUNTERFACTUAL_H

#include "explainable/explainable.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <utility>

namespace phynexus {
namespace explainable {

/**
 * @brief Counterfactual explanation implementation
 */
class Counterfactual : public Explainer {
public:
    Counterfactual(const std::function<std::vector<float>(const std::vector<float>&)>& model_fn,
                  const std::vector<float>& sample,
                  int target_class = -1,
                  float target_pred = -1.0f,
                  const std::vector<int>& categorical_features = {},
                  const std::unordered_map<int, std::pair<float, float>>& feature_ranges = {},
                  int max_iter = 1000,
                  unsigned int random_state = 0);
    
    /**
     * @brief Explain a model prediction
     * @return Explanation result as a map of strings to values
     */
    std::unordered_map<std::string, std::vector<float>> explain() override;
    
    /**
     * @brief Get the name of the explainer
     * @return Name of the explainer
     */
    std::string name() const override { return "Counterfactual"; }
    
private:
    std::function<std::vector<float>(const std::vector<float>&)> model_fn_;
    std::vector<float> sample_;
    int target_class_;
    float target_pred_;
    std::vector<int> categorical_features_;
    std::unordered_map<int, std::pair<float, float>> feature_ranges_;
    int max_iter_;
    unsigned int random_state_;
};

} // namespace explainable
} // namespace phynexus

#endif // PHYNEXUS_EXPLAINABLE_COUNTERFACTUAL_H
