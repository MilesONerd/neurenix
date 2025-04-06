/**
 * @file lime.h
 * @brief Header file for LIME (Local Interpretable Model-agnostic Explanations) in Phynexus
 */

#ifndef PHYNEXUS_EXPLAINABLE_LIME_H
#define PHYNEXUS_EXPLAINABLE_LIME_H

#include "explainable/explainable.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

namespace phynexus {
namespace explainable {

/**
 * @brief Base class for LIME explainers
 */
class LimeExplainer : public Explainer {
public:
    virtual ~LimeExplainer() = default;
    
    /**
     * @brief Get the name of the explainer
     * @return Name of the explainer
     */
    std::string name() const override { return "LIME"; }
};

/**
 * @brief LIME for tabular data
 */
class LimeTabular : public LimeExplainer {
public:
    LimeTabular(const std::function<std::vector<float>(const std::vector<float>&)>& model_fn,
                const std::vector<std::string>& feature_names,
                int num_samples = 5000);
    
    /**
     * @brief Explain a model prediction
     * @return Explanation result as a map of strings to values
     */
    std::unordered_map<std::string, std::vector<float>> explain() override;
    
private:
    std::function<std::vector<float>(const std::vector<float>&)> model_fn_;
    std::vector<std::string> feature_names_;
    int num_samples_;
};

/**
 * @brief LIME for text data
 */
class LimeText : public LimeExplainer {
public:
    LimeText(const std::function<std::vector<float>(const std::string&)>& model_fn,
             int num_samples = 5000);
    
    /**
     * @brief Explain a model prediction
     * @return Explanation result as a map of strings to values
     */
    std::unordered_map<std::string, std::vector<float>> explain() override;
    
private:
    std::function<std::vector<float>(const std::string&)> model_fn_;
    int num_samples_;
};

/**
 * @brief LIME for image data
 */
class LimeImage : public LimeExplainer {
public:
    LimeImage(const std::function<std::vector<float>(const std::vector<std::vector<std::vector<float>>>&)>& model_fn,
              int num_samples = 1000);
    
    /**
     * @brief Explain a model prediction
     * @return Explanation result as a map of strings to values
     */
    std::unordered_map<std::string, std::vector<float>> explain() override;
    
private:
    std::function<std::vector<float>(const std::vector<std::vector<std::vector<float>>>&)> model_fn_;
    int num_samples_;
};

} // namespace explainable
} // namespace phynexus

#endif // PHYNEXUS_EXPLAINABLE_LIME_H
