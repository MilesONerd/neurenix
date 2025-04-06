/**
 * @file shap.h
 * @brief Header file for SHAP (SHapley Additive exPlanations) in Phynexus
 */

#ifndef PHYNEXUS_EXPLAINABLE_SHAP_H
#define PHYNEXUS_EXPLAINABLE_SHAP_H

#include "explainable/explainable.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

namespace phynexus {
namespace explainable {

/**
 * @brief Base class for SHAP explainers
 */
class ShapExplainer : public Explainer {
public:
    virtual ~ShapExplainer() = default;
    
    /**
     * @brief Get the name of the explainer
     * @return Name of the explainer
     */
    std::string name() const override { return "SHAP"; }
};

/**
 * @brief KernelSHAP implementation
 */
class KernelShap : public ShapExplainer {
public:
    KernelShap(const std::function<std::vector<float>(const std::vector<float>&)>& model_fn,
               const std::vector<std::vector<float>>& background_data);
    
    /**
     * @brief Explain a model prediction
     * @return Explanation result as a map of strings to values
     */
    std::unordered_map<std::string, std::vector<float>> explain() override;
    
private:
    std::function<std::vector<float>(const std::vector<float>&)> model_fn_;
    std::vector<std::vector<float>> background_data_;
};

/**
 * @brief TreeSHAP implementation
 */
class TreeShap : public ShapExplainer {
public:
    TreeShap(void* tree_model);
    
    /**
     * @brief Explain a model prediction
     * @return Explanation result as a map of strings to values
     */
    std::unordered_map<std::string, std::vector<float>> explain() override;
    
private:
    void* tree_model_;
};

/**
 * @brief DeepSHAP implementation
 */
class DeepShap : public ShapExplainer {
public:
    DeepShap(void* deep_model, const std::vector<std::vector<float>>& background_data);
    
    /**
     * @brief Explain a model prediction
     * @return Explanation result as a map of strings to values
     */
    std::unordered_map<std::string, std::vector<float>> explain() override;
    
private:
    void* deep_model_;
    std::vector<std::vector<float>> background_data_;
};

} // namespace explainable
} // namespace phynexus

#endif // PHYNEXUS_EXPLAINABLE_SHAP_H
