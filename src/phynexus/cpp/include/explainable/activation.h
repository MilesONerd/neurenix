/**
 * @file activation.h
 * @brief Header file for activation visualization in Phynexus
 */

#ifndef PHYNEXUS_EXPLAINABLE_ACTIVATION_H
#define PHYNEXUS_EXPLAINABLE_ACTIVATION_H

#include "explainable/explainable.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

namespace phynexus {
namespace explainable {

/**
 * @brief Activation visualization implementation
 */
class ActivationVisualization : public Explainer {
public:
    ActivationVisualization(void* model,
                           const std::vector<std::string>& layer_names = {},
                           bool include_gradients = false);
    
    /**
     * @brief Explain a model prediction
     * @return Explanation result as a map of strings to values
     */
    std::unordered_map<std::string, std::vector<float>> explain() override;
    
    /**
     * @brief Get the name of the explainer
     * @return Name of the explainer
     */
    std::string name() const override { return "ActivationVisualization"; }
    
private:
    void* model_;
    std::vector<std::string> layer_names_;
    bool include_gradients_;
};

} // namespace explainable
} // namespace phynexus

#endif // PHYNEXUS_EXPLAINABLE_ACTIVATION_H
