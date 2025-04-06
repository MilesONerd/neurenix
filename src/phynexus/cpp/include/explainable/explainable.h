/**
 * @file explainable.h
 * @brief Header file for explainable AI module in Phynexus
 */

#ifndef PHYNEXUS_EXPLAINABLE_H
#define PHYNEXUS_EXPLAINABLE_H

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

namespace phynexus {
namespace explainable {

/**
 * @brief Base class for all explainers
 */
class Explainer {
public:
    virtual ~Explainer() = default;
    
    /**
     * @brief Explain a model prediction
     * @return Explanation result as a map of strings to values
     */
    virtual std::unordered_map<std::string, std::vector<float>> explain() = 0;
    
    /**
     * @brief Get the name of the explainer
     * @return Name of the explainer
     */
    virtual std::string name() const = 0;
};

} // namespace explainable
} // namespace phynexus

#endif // PHYNEXUS_EXPLAINABLE_H
