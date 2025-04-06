/**
 * @file fuzzy.cpp
 * @brief Implementation of Fuzzy Logic module for Neurenix C++ backend.
 */

#include "fuzzy/fuzzy.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <iostream>
#include <numeric>

namespace phynexus {
namespace fuzzy {


TriangularMF::TriangularMF(float a, float b, float c)
    : a_(a), b_(b), c_(c) {}

float TriangularMF::compute(float x) const {
    if (x <= a_ || x >= c_) {
        return 0.0f;
    } else if (x <= b_) {
        return (x - a_) / (b_ - a_);
    } else {
        return (c_ - x) / (c_ - b_);
    }
}

MembershipFunctionType TriangularMF::get_type() const {
    return MembershipFunctionType::TRIANGULAR;
}

std::vector<float> TriangularMF::get_parameters() const {
    return {a_, b_, c_};
}

void TriangularMF::set_parameters(const std::vector<float>& parameters) {
    if (parameters.size() != 3) {
        throw std::invalid_argument("TriangularMF requires 3 parameters");
    }
    
    a_ = parameters[0];
    b_ = parameters[1];
    c_ = parameters[2];
}

TrapezoidalMF::TrapezoidalMF(float a, float b, float c, float d)
    : a_(a), b_(b), c_(c), d_(d) {}

float TrapezoidalMF::compute(float x) const {
    if (x <= a_ || x >= d_) {
        return 0.0f;
    } else if (x >= b_ && x <= c_) {
        return 1.0f;
    } else if (x < b_) {
        return (x - a_) / (b_ - a_);
    } else {
        return (d_ - x) / (d_ - c_);
    }
}

MembershipFunctionType TrapezoidalMF::get_type() const {
    return MembershipFunctionType::TRAPEZOIDAL;
}

std::vector<float> TrapezoidalMF::get_parameters() const {
    return {a_, b_, c_, d_};
}

void TrapezoidalMF::set_parameters(const std::vector<float>& parameters) {
    if (parameters.size() != 4) {
        throw std::invalid_argument("TrapezoidalMF requires 4 parameters");
    }
    
    a_ = parameters[0];
    b_ = parameters[1];
    c_ = parameters[2];
    d_ = parameters[3];
}

GaussianMF::GaussianMF(float c, float sigma)
    : c_(c), sigma_(sigma) {}

float GaussianMF::compute(float x) const {
    float exponent = -0.5f * std::pow((x - c_) / sigma_, 2);
    return std::exp(exponent);
}

MembershipFunctionType GaussianMF::get_type() const {
    return MembershipFunctionType::GAUSSIAN;
}

std::vector<float> GaussianMF::get_parameters() const {
    return {c_, sigma_};
}

void GaussianMF::set_parameters(const std::vector<float>& parameters) {
    if (parameters.size() != 2) {
        throw std::invalid_argument("GaussianMF requires 2 parameters");
    }
    
    c_ = parameters[0];
    sigma_ = parameters[1];
}

SigmoidMF::SigmoidMF(float a, float c)
    : a_(a), c_(c) {}

float SigmoidMF::compute(float x) const {
    return 1.0f / (1.0f + std::exp(-a_ * (x - c_)));
}

MembershipFunctionType SigmoidMF::get_type() const {
    return MembershipFunctionType::SIGMOID;
}

std::vector<float> SigmoidMF::get_parameters() const {
    return {a_, c_};
}

void SigmoidMF::set_parameters(const std::vector<float>& parameters) {
    if (parameters.size() != 2) {
        throw std::invalid_argument("SigmoidMF requires 2 parameters");
    }
    
    a_ = parameters[0];
    c_ = parameters[1];
}

BellMF::BellMF(float a, float b, float c)
    : a_(a), b_(b), c_(c) {}

float BellMF::compute(float x) const {
    return 1.0f / (1.0f + std::pow(std::abs((x - c_) / a_), 2 * b_));
}

MembershipFunctionType BellMF::get_type() const {
    return MembershipFunctionType::BELL;
}

std::vector<float> BellMF::get_parameters() const {
    return {a_, b_, c_};
}

void BellMF::set_parameters(const std::vector<float>& parameters) {
    if (parameters.size() != 3) {
        throw std::invalid_argument("BellMF requires 3 parameters");
    }
    
    a_ = parameters[0];
    b_ = parameters[1];
    c_ = parameters[2];
}

CustomMF::CustomMF(std::function<float(float, const std::vector<float>&)> function,
                 const std::vector<float>& parameters)
    : function_(function), parameters_(parameters) {}

float CustomMF::compute(float x) const {
    return function_(x, parameters_);
}

MembershipFunctionType CustomMF::get_type() const {
    return MembershipFunctionType::CUSTOM;
}

std::vector<float> CustomMF::get_parameters() const {
    return parameters_;
}

void CustomMF::set_parameters(const std::vector<float>& parameters) {
    parameters_ = parameters;
}


LinguisticVariable::LinguisticVariable(const std::string& name, float min_value, float max_value)
    : name_(name), min_value_(min_value), max_value_(max_value) {}

void LinguisticVariable::add_term(const std::string& term_name, std::shared_ptr<MembershipFunction> mf) {
    terms_[term_name] = mf;
}

const std::string& LinguisticVariable::get_name() const {
    return name_;
}

float LinguisticVariable::get_min_value() const {
    return min_value_;
}

float LinguisticVariable::get_max_value() const {
    return max_value_;
}

float LinguisticVariable::get_membership_degree(const std::string& term_name, float value) const {
    auto it = terms_.find(term_name);
    
    if (it == terms_.end()) {
        throw std::invalid_argument("Term '" + term_name + "' not found in variable '" + name_ + "'");
    }
    
    return it->second->compute(value);
}

std::shared_ptr<MembershipFunction> LinguisticVariable::get_membership_function(const std::string& term_name) const {
    auto it = terms_.find(term_name);
    
    if (it == terms_.end()) {
        throw std::invalid_argument("Term '" + term_name + "' not found in variable '" + name_ + "'");
    }
    
    return it->second;
}

std::vector<std::string> LinguisticVariable::get_terms() const {
    std::vector<std::string> term_names;
    
    for (const auto& term : terms_) {
        term_names.push_back(term.first);
    }
    
    return term_names;
}


SimpleFuzzyAntecedent::SimpleFuzzyAntecedent(const LinguisticVariable& variable,
                                         const std::string& term_name,
                                         float value)
    : variable_(variable), term_name_(term_name), value_(value) {}

float SimpleFuzzyAntecedent::evaluate() const {
    return variable_.get_membership_degree(term_name_, value_);
}

CompositeFuzzyAntecedent::CompositeFuzzyAntecedent(std::shared_ptr<FuzzyAntecedent> antecedent1,
                                               std::shared_ptr<FuzzyAntecedent> antecedent2,
                                               FuzzyOperator op)
    : antecedent1_(antecedent1), antecedent2_(antecedent2), op_(op) {}

float CompositeFuzzyAntecedent::evaluate() const {
    float a = antecedent1_->evaluate();
    float b = antecedent2_->evaluate();
    
    return apply_operator(a, b, op_);
}

float CompositeFuzzyAntecedent::apply_operator(float a, float b, FuzzyOperator op) const {
    switch (op) {
        case FuzzyOperator::MIN:
            return std::min(a, b);
        case FuzzyOperator::PRODUCT:
            return a * b;
        case FuzzyOperator::MAX:
            return std::max(a, b);
        case FuzzyOperator::PROBABILISTIC_SUM:
            return a + b - a * b;
        case FuzzyOperator::BOUNDED_SUM:
            return std::min(1.0f, a + b);
        case FuzzyOperator::BOUNDED_DIFFERENCE:
            return std::max(0.0f, a + b - 1.0f);
        case FuzzyOperator::DRASTIC_PRODUCT:
            if (a == 1.0f) return b;
            if (b == 1.0f) return a;
            return 0.0f;
        case FuzzyOperator::DRASTIC_SUM:
            if (a == 0.0f) return b;
            if (b == 0.0f) return a;
            return 1.0f;
        case FuzzyOperator::EINSTEIN_PRODUCT:
            return (a * b) / (2.0f - (a + b - a * b));
        case FuzzyOperator::EINSTEIN_SUM:
            return (a + b) / (1.0f + a * b);
        case FuzzyOperator::HAMACHER_PRODUCT:
            if (a == 0.0f && b == 0.0f) return 0.0f;
            return (a * b) / (a + b - a * b);
        case FuzzyOperator::HAMACHER_SUM:
            if (a == 0.0f && b == 0.0f) return 0.0f;
            return (a + b - 2.0f * a * b) / (1.0f - a * b);
        default:
            return std::min(a, b);
    }
}


FuzzyConsequent::FuzzyConsequent(const LinguisticVariable& variable,
                               const std::string& term_name)
    : variable_(variable), term_name_(term_name) {}

const LinguisticVariable& FuzzyConsequent::get_variable() const {
    return variable_;
}

const std::string& FuzzyConsequent::get_term_name() const {
    return term_name_;
}

std::shared_ptr<MembershipFunction> FuzzyConsequent::get_membership_function() const {
    return variable_.get_membership_function(term_name_);
}


FuzzyRule::FuzzyRule(std::shared_ptr<FuzzyAntecedent> antecedent,
                   std::shared_ptr<FuzzyConsequent> consequent,
                   float weight)
    : antecedent_(antecedent), consequent_(consequent), weight_(weight) {}

std::shared_ptr<FuzzyAntecedent> FuzzyRule::get_antecedent() const {
    return antecedent_;
}

std::shared_ptr<FuzzyConsequent> FuzzyRule::get_consequent() const {
    return consequent_;
}

float FuzzyRule::get_weight() const {
    return weight_;
}

void FuzzyRule::set_weight(float weight) {
    weight_ = weight;
}

float FuzzyRule::evaluate() const {
    return antecedent_->evaluate() * weight_;
}


FuzzyInferenceSystem::FuzzyInferenceSystem(const std::string& name,
                                       FuzzyOperator and_op,
                                       FuzzyOperator or_op,
                                       FuzzyOperator implication_op,
                                       FuzzyOperator aggregation_op,
                                       DefuzzificationMethod defuzzification_method)
    : name_(name), and_op_(and_op), or_op_(or_op),
      implication_op_(implication_op), aggregation_op_(aggregation_op),
      defuzzification_method_(defuzzification_method) {}

void FuzzyInferenceSystem::add_variable(const LinguisticVariable& variable) {
    variables_[variable.get_name()] = variable;
}

void FuzzyInferenceSystem::add_rule(const FuzzyRule& rule) {
    rules_.push_back(rule);
}

const LinguisticVariable& FuzzyInferenceSystem::get_variable(const std::string& name) const {
    auto it = variables_.find(name);
    
    if (it == variables_.end()) {
        throw std::invalid_argument("Variable '" + name + "' not found");
    }
    
    return it->second;
}

std::vector<LinguisticVariable> FuzzyInferenceSystem::get_variables() const {
    std::vector<LinguisticVariable> variables;
    
    for (const auto& variable : variables_) {
        variables.push_back(variable.second);
    }
    
    return variables;
}

std::vector<FuzzyRule> FuzzyInferenceSystem::get_rules() const {
    return rules_;
}

void FuzzyInferenceSystem::set_input(const std::string& variable_name, float value) {
    auto it = variables_.find(variable_name);
    
    if (it == variables_.end()) {
        throw std::invalid_argument("Variable '" + variable_name + "' not found");
    }
    
    inputs_[variable_name] = value;
}

float FuzzyInferenceSystem::get_output(const std::string& variable_name) const {
    auto it = outputs_.find(variable_name);
    
    if (it == outputs_.end()) {
        throw std::invalid_argument("Output for variable '" + variable_name + "' not found");
    }
    
    return it->second;
}

void FuzzyInferenceSystem::evaluate() {
    outputs_.clear();
    
    std::unordered_map<std::string, std::vector<std::pair<float, std::shared_ptr<MembershipFunction>>>> rule_outputs;
    
    for (const auto& rule : rules_) {
        float activation = rule.evaluate();
        
        if (activation > 0.0f) {
            const auto& consequent = rule.get_consequent();
            const auto& variable = consequent->get_variable();
            const auto& term_name = consequent->get_term_name();
            
            rule_outputs[variable.get_name()].push_back(
                std::make_pair(activation, variable.get_membership_function(term_name)));
        }
    }
    
    for (const auto& output : rule_outputs) {
        const std::string& variable_name = output.first;
        const auto& variable = variables_.at(variable_name);
        
        int num_points = 100;
        float step = (variable.get_max_value() - variable.get_min_value()) / (num_points - 1);
        std::vector<float> fuzzy_set(num_points, 0.0f);
        
        for (int i = 0; i < num_points; ++i) {
            float x = variable.get_min_value() + i * step;
            
            for (const auto& rule_output : output.second) {
                float activation = rule_output.first;
                const auto& mf = rule_output.second;
                
                float y = apply_operator(activation, mf->compute(x), implication_op_);
                fuzzy_set[i] = apply_operator(fuzzy_set[i], y, aggregation_op_);
            }
        }
        
        outputs_[variable_name] = defuzzify(variable, fuzzy_set, defuzzification_method_);
    }
}

float FuzzyInferenceSystem::apply_operator(float a, float b, FuzzyOperator op) const {
    switch (op) {
        case FuzzyOperator::MIN:
            return std::min(a, b);
        case FuzzyOperator::PRODUCT:
            return a * b;
        case FuzzyOperator::MAX:
            return std::max(a, b);
        case FuzzyOperator::PROBABILISTIC_SUM:
            return a + b - a * b;
        case FuzzyOperator::BOUNDED_SUM:
            return std::min(1.0f, a + b);
        case FuzzyOperator::BOUNDED_DIFFERENCE:
            return std::max(0.0f, a + b - 1.0f);
        case FuzzyOperator::DRASTIC_PRODUCT:
            if (a == 1.0f) return b;
            if (b == 1.0f) return a;
            return 0.0f;
        case FuzzyOperator::DRASTIC_SUM:
            if (a == 0.0f) return b;
            if (b == 0.0f) return a;
            return 1.0f;
        case FuzzyOperator::EINSTEIN_PRODUCT:
            return (a * b) / (2.0f - (a + b - a * b));
        case FuzzyOperator::EINSTEIN_SUM:
            return (a + b) / (1.0f + a * b);
        case FuzzyOperator::HAMACHER_PRODUCT:
            if (a == 0.0f && b == 0.0f) return 0.0f;
            return (a * b) / (a + b - a * b);
        case FuzzyOperator::HAMACHER_SUM:
            if (a == 0.0f && b == 0.0f) return 0.0f;
            return (a + b - 2.0f * a * b) / (1.0f - a * b);
        default:
            return std::min(a, b);
    }
}

float FuzzyInferenceSystem::defuzzify(const LinguisticVariable& variable,
                                    const std::vector<float>& fuzzy_set,
                                    DefuzzificationMethod method) const {
    int num_points = fuzzy_set.size();
    float step = (variable.get_max_value() - variable.get_min_value()) / (num_points - 1);
    
    switch (method) {
        case DefuzzificationMethod::CENTROID: {
            float sum_product = 0.0f;
            float sum_membership = 0.0f;
            
            for (int i = 0; i < num_points; ++i) {
                float x = variable.get_min_value() + i * step;
                sum_product += x * fuzzy_set[i];
                sum_membership += fuzzy_set[i];
            }
            
            if (sum_membership == 0.0f) {
                return (variable.get_min_value() + variable.get_max_value()) / 2.0f;
            }
            
            return sum_product / sum_membership;
        }
        
        case DefuzzificationMethod::BISECTOR: {
            float sum_membership = 0.0f;
            
            for (int i = 0; i < num_points; ++i) {
                sum_membership += fuzzy_set[i];
            }
            
            float half_sum = sum_membership / 2.0f;
            float current_sum = 0.0f;
            
            for (int i = 0; i < num_points; ++i) {
                current_sum += fuzzy_set[i];
                
                if (current_sum >= half_sum) {
                    return variable.get_min_value() + i * step;
                }
            }
            
            return variable.get_max_value();
        }
        
        case DefuzzificationMethod::MEAN_OF_MAXIMUM: {
            float max_membership = 0.0f;
            
            for (int i = 0; i < num_points; ++i) {
                max_membership = std::max(max_membership, fuzzy_set[i]);
            }
            
            if (max_membership == 0.0f) {
                return (variable.get_min_value() + variable.get_max_value()) / 2.0f;
            }
            
            float sum_x = 0.0f;
            int count = 0;
            
            for (int i = 0; i < num_points; ++i) {
                if (fuzzy_set[i] == max_membership) {
                    sum_x += variable.get_min_value() + i * step;
                    count++;
                }
            }
            
            return sum_x / count;
        }
        
        case DefuzzificationMethod::SMALLEST_OF_MAXIMUM: {
            float max_membership = 0.0f;
            
            for (int i = 0; i < num_points; ++i) {
                max_membership = std::max(max_membership, fuzzy_set[i]);
            }
            
            if (max_membership == 0.0f) {
                return variable.get_min_value();
            }
            
            for (int i = 0; i < num_points; ++i) {
                if (fuzzy_set[i] == max_membership) {
                    return variable.get_min_value() + i * step;
                }
            }
            
            return variable.get_min_value();
        }
        
        case DefuzzificationMethod::LARGEST_OF_MAXIMUM: {
            float max_membership = 0.0f;
            
            for (int i = 0; i < num_points; ++i) {
                max_membership = std::max(max_membership, fuzzy_set[i]);
            }
            
            if (max_membership == 0.0f) {
                return variable.get_max_value();
            }
            
            for (int i = num_points - 1; i >= 0; --i) {
                if (fuzzy_set[i] == max_membership) {
                    return variable.get_min_value() + i * step;
                }
            }
            
            return variable.get_max_value();
        }
        
        case DefuzzificationMethod::WEIGHTED_AVERAGE: {
            float sum_product = 0.0f;
            float sum_weight = 0.0f;
            
            for (int i = 0; i < num_points; ++i) {
                float x = variable.get_min_value() + i * step;
                sum_product += x * fuzzy_set[i] * fuzzy_set[i];
                sum_weight += fuzzy_set[i] * fuzzy_set[i];
            }
            
            if (sum_weight == 0.0f) {
                return (variable.get_min_value() + variable.get_max_value()) / 2.0f;
            }
            
            return sum_product / sum_weight;
        }
        
        default:
            return (variable.get_min_value() + variable.get_max_value()) / 2.0f;
    }
}

} // namespace fuzzy
} // namespace phynexus
