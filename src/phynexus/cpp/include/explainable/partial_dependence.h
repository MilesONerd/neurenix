/**
 * @file partial_dependence.h
 * @brief Header file for partial dependence plots in Phynexus
 */

#ifndef PHYNEXUS_EXPLAINABLE_PARTIAL_DEPENDENCE_H
#define PHYNEXUS_EXPLAINABLE_PARTIAL_DEPENDENCE_H

#include "explainable/explainable.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

namespace phynexus {
namespace explainable {

/**
 * @brief Partial Dependence Plot (PDP) implementation
 */
class PartialDependence : public Explainer {
public:
    PartialDependence(const std::function<std::vector<float>(const std::vector<float>&)>& model_fn,
                      const std::vector<std::vector<float>>& data,
                      const std::vector<int>& features,
                      int grid_resolution = 20,
                      std::pair<float, float> percentiles = {0.05f, 0.95f},
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
    std::string name() const override { return "PartialDependence"; }
    
private:
    std::function<std::vector<float>(const std::vector<float>&)> model_fn_;
    std::vector<std::vector<float>> data_;
    std::vector<int> features_;
    int grid_resolution_;
    std::pair<float, float> percentiles_;
    unsigned int random_state_;
};

} // namespace explainable
} // namespace phynexus

#endif // PHYNEXUS_EXPLAINABLE_PARTIAL_DEPENDENCE_H
