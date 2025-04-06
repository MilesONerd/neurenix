/**
 * @file fuzzy.h
 * @brief Header file for Fuzzy Logic module in Neurenix C++ backend.
 */

#ifndef PHYNEXUS_FUZZY_H
#define PHYNEXUS_FUZZY_H

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <functional>
#include <cmath>

namespace phynexus {
namespace fuzzy {

/**
 * @brief Enumeration of fuzzy membership function types.
 */
enum class MembershipFunctionType {
    TRIANGULAR,
    TRAPEZOIDAL,
    GAUSSIAN,
    SIGMOID,
    BELL,
    CUSTOM
};

/**
 * @brief Base class for fuzzy membership functions.
 */
class MembershipFunction {
public:
    virtual ~MembershipFunction() = default;
    
    /**
     * @brief Compute the membership degree for a given value.
     * @param x Input value.
     * @return Membership degree in [0, 1].
     */
    virtual float compute(float x) const = 0;
    
    /**
     * @brief Get the type of the membership function.
     * @return Type of the membership function.
     */
    virtual MembershipFunctionType get_type() const = 0;
    
    /**
     * @brief Get the parameters of the membership function.
     * @return Vector of parameters.
     */
    virtual std::vector<float> get_parameters() const = 0;
    
    /**
     * @brief Set the parameters of the membership function.
     * @param parameters Vector of parameters.
     */
    virtual void set_parameters(const std::vector<float>& parameters) = 0;
};

/**
 * @brief Triangular membership function.
 */
class TriangularMF : public MembershipFunction {
public:
    /**
     * @brief Constructor.
     * @param a Left point.
     * @param b Center point.
     * @param c Right point.
     */
    TriangularMF(float a, float b, float c);
    
    /**
     * @brief Compute the membership degree for a given value.
     * @param x Input value.
     * @return Membership degree in [0, 1].
     */
    float compute(float x) const override;
    
    /**
     * @brief Get the type of the membership function.
     * @return Type of the membership function.
     */
    MembershipFunctionType get_type() const override;
    
    /**
     * @brief Get the parameters of the membership function.
     * @return Vector of parameters.
     */
    std::vector<float> get_parameters() const override;
    
    /**
     * @brief Set the parameters of the membership function.
     * @param parameters Vector of parameters.
     */
    void set_parameters(const std::vector<float>& parameters) override;
    
private:
    float a_;
    float b_;
    float c_;
};

/**
 * @brief Trapezoidal membership function.
 */
class TrapezoidalMF : public MembershipFunction {
public:
    /**
     * @brief Constructor.
     * @param a Left point.
     * @param b Left shoulder.
     * @param c Right shoulder.
     * @param d Right point.
     */
    TrapezoidalMF(float a, float b, float c, float d);
    
    /**
     * @brief Compute the membership degree for a given value.
     * @param x Input value.
     * @return Membership degree in [0, 1].
     */
    float compute(float x) const override;
    
    /**
     * @brief Get the type of the membership function.
     * @return Type of the membership function.
     */
    MembershipFunctionType get_type() const override;
    
    /**
     * @brief Get the parameters of the membership function.
     * @return Vector of parameters.
     */
    std::vector<float> get_parameters() const override;
    
    /**
     * @brief Set the parameters of the membership function.
     * @param parameters Vector of parameters.
     */
    void set_parameters(const std::vector<float>& parameters) override;
    
private:
    float a_;
    float b_;
    float c_;
    float d_;
};

/**
 * @brief Gaussian membership function.
 */
class GaussianMF : public MembershipFunction {
public:
    /**
     * @brief Constructor.
     * @param c Center.
     * @param sigma Width.
     */
    GaussianMF(float c, float sigma);
    
    /**
     * @brief Compute the membership degree for a given value.
     * @param x Input value.
     * @return Membership degree in [0, 1].
     */
    float compute(float x) const override;
    
    /**
     * @brief Get the type of the membership function.
     * @return Type of the membership function.
     */
    MembershipFunctionType get_type() const override;
    
    /**
     * @brief Get the parameters of the membership function.
     * @return Vector of parameters.
     */
    std::vector<float> get_parameters() const override;
    
    /**
     * @brief Set the parameters of the membership function.
     * @param parameters Vector of parameters.
     */
    void set_parameters(const std::vector<float>& parameters) override;
    
private:
    float c_;
    float sigma_;
};

/**
 * @brief Sigmoid membership function.
 */
class SigmoidMF : public MembershipFunction {
public:
    /**
     * @brief Constructor.
     * @param a Slope.
     * @param c Inflection point.
     */
    SigmoidMF(float a, float c);
    
    /**
     * @brief Compute the membership degree for a given value.
     * @param x Input value.
     * @return Membership degree in [0, 1].
     */
    float compute(float x) const override;
    
    /**
     * @brief Get the type of the membership function.
     * @return Type of the membership function.
     */
    MembershipFunctionType get_type() const override;
    
    /**
     * @brief Get the parameters of the membership function.
     * @return Vector of parameters.
     */
    std::vector<float> get_parameters() const override;
    
    /**
     * @brief Set the parameters of the membership function.
     * @param parameters Vector of parameters.
     */
    void set_parameters(const std::vector<float>& parameters) override;
    
private:
    float a_;
    float c_;
};

/**
 * @brief Bell-shaped membership function.
 */
class BellMF : public MembershipFunction {
public:
    /**
     * @brief Constructor.
     * @param a Width.
     * @param b Slope.
     * @param c Center.
     */
    BellMF(float a, float b, float c);
    
    /**
     * @brief Compute the membership degree for a given value.
     * @param x Input value.
     * @return Membership degree in [0, 1].
     */
    float compute(float x) const override;
    
    /**
     * @brief Get the type of the membership function.
     * @return Type of the membership function.
     */
    MembershipFunctionType get_type() const override;
    
    /**
     * @brief Get the parameters of the membership function.
     * @return Vector of parameters.
     */
    std::vector<float> get_parameters() const override;
    
    /**
     * @brief Set the parameters of the membership function.
     * @param parameters Vector of parameters.
     */
    void set_parameters(const std::vector<float>& parameters) override;
    
private:
    float a_;
    float b_;
    float c_;
};

/**
 * @brief Custom membership function.
 */
class CustomMF : public MembershipFunction {
public:
    /**
     * @brief Constructor.
     * @param function Custom function.
     * @param parameters Parameters of the function.
     */
    CustomMF(std::function<float(float, const std::vector<float>&)> function,
            const std::vector<float>& parameters);
    
    /**
     * @brief Compute the membership degree for a given value.
     * @param x Input value.
     * @return Membership degree in [0, 1].
     */
    float compute(float x) const override;
    
    /**
     * @brief Get the type of the membership function.
     * @return Type of the membership function.
     */
    MembershipFunctionType get_type() const override;
    
    /**
     * @brief Get the parameters of the membership function.
     * @return Vector of parameters.
     */
    std::vector<float> get_parameters() const override;
    
    /**
     * @brief Set the parameters of the membership function.
     * @param parameters Vector of parameters.
     */
    void set_parameters(const std::vector<float>& parameters) override;
    
private:
    std::function<float(float, const std::vector<float>&)> function_;
    std::vector<float> parameters_;
};

/**
 * @brief Fuzzy linguistic variable.
 */
class LinguisticVariable {
public:
    /**
     * @brief Constructor.
     * @param name Name of the variable.
     * @param min_value Minimum value of the variable.
     * @param max_value Maximum value of the variable.
     */
    LinguisticVariable(const std::string& name, float min_value, float max_value);
    
    /**
     * @brief Add a term to the variable.
     * @param term_name Name of the term.
     * @param mf Membership function.
     */
    void add_term(const std::string& term_name, std::shared_ptr<MembershipFunction> mf);
    
    /**
     * @brief Get the name of the variable.
     * @return Name of the variable.
     */
    const std::string& get_name() const;
    
    /**
     * @brief Get the minimum value of the variable.
     * @return Minimum value of the variable.
     */
    float get_min_value() const;
    
    /**
     * @brief Get the maximum value of the variable.
     * @return Maximum value of the variable.
     */
    float get_max_value() const;
    
    /**
     * @brief Get the membership degree of a value for a term.
     * @param term_name Name of the term.
     * @param value Input value.
     * @return Membership degree in [0, 1].
     */
    float get_membership_degree(const std::string& term_name, float value) const;
    
    /**
     * @brief Get the membership function of a term.
     * @param term_name Name of the term.
     * @return Membership function.
     */
    std::shared_ptr<MembershipFunction> get_membership_function(const std::string& term_name) const;
    
    /**
     * @brief Get all terms of the variable.
     * @return Vector of term names.
     */
    std::vector<std::string> get_terms() const;
    
private:
    std::string name_;
    float min_value_;
    float max_value_;
    std::unordered_map<std::string, std::shared_ptr<MembershipFunction>> terms_;
};

/**
 * @brief Enumeration of fuzzy operators.
 */
enum class FuzzyOperator {
    MIN,
    PRODUCT,
    MAX,
    PROBABILISTIC_SUM,
    BOUNDED_SUM,
    BOUNDED_DIFFERENCE,
    DRASTIC_PRODUCT,
    DRASTIC_SUM,
    EINSTEIN_PRODUCT,
    EINSTEIN_SUM,
    HAMACHER_PRODUCT,
    HAMACHER_SUM
};

/**
 * @brief Fuzzy rule antecedent.
 */
class FuzzyAntecedent {
public:
    virtual ~FuzzyAntecedent() = default;
    
    /**
     * @brief Evaluate the antecedent.
     * @return Activation degree in [0, 1].
     */
    virtual float evaluate() const = 0;
};

/**
 * @brief Simple fuzzy antecedent.
 */
class SimpleFuzzyAntecedent : public FuzzyAntecedent {
public:
    /**
     * @brief Constructor.
     * @param variable Linguistic variable.
     * @param term_name Term name.
     * @param value Input value.
     */
    SimpleFuzzyAntecedent(const LinguisticVariable& variable,
                        const std::string& term_name,
                        float value);
    
    /**
     * @brief Evaluate the antecedent.
     * @return Activation degree in [0, 1].
     */
    float evaluate() const override;
    
private:
    const LinguisticVariable& variable_;
    std::string term_name_;
    float value_;
};

/**
 * @brief Composite fuzzy antecedent.
 */
class CompositeFuzzyAntecedent : public FuzzyAntecedent {
public:
    /**
     * @brief Constructor.
     * @param antecedent1 First antecedent.
     * @param antecedent2 Second antecedent.
     * @param op Fuzzy operator.
     */
    CompositeFuzzyAntecedent(std::shared_ptr<FuzzyAntecedent> antecedent1,
                           std::shared_ptr<FuzzyAntecedent> antecedent2,
                           FuzzyOperator op);
    
    /**
     * @brief Evaluate the antecedent.
     * @return Activation degree in [0, 1].
     */
    float evaluate() const override;
    
private:
    std::shared_ptr<FuzzyAntecedent> antecedent1_;
    std::shared_ptr<FuzzyAntecedent> antecedent2_;
    FuzzyOperator op_;
    
    /**
     * @brief Apply a fuzzy operator to two values.
     * @param a First value.
     * @param b Second value.
     * @param op Fuzzy operator.
     * @return Result of the operation.
     */
    float apply_operator(float a, float b, FuzzyOperator op) const;
};

/**
 * @brief Fuzzy rule consequent.
 */
class FuzzyConsequent {
public:
    /**
     * @brief Constructor.
     * @param variable Linguistic variable.
     * @param term_name Term name.
     */
    FuzzyConsequent(const LinguisticVariable& variable,
                  const std::string& term_name);
    
    /**
     * @brief Get the linguistic variable.
     * @return Linguistic variable.
     */
    const LinguisticVariable& get_variable() const;
    
    /**
     * @brief Get the term name.
     * @return Term name.
     */
    const std::string& get_term_name() const;
    
    /**
     * @brief Get the membership function.
     * @return Membership function.
     */
    std::shared_ptr<MembershipFunction> get_membership_function() const;
    
private:
    const LinguisticVariable& variable_;
    std::string term_name_;
};

/**
 * @brief Fuzzy rule.
 */
class FuzzyRule {
public:
    /**
     * @brief Constructor.
     * @param antecedent Rule antecedent.
     * @param consequent Rule consequent.
     * @param weight Rule weight.
     */
    FuzzyRule(std::shared_ptr<FuzzyAntecedent> antecedent,
             std::shared_ptr<FuzzyConsequent> consequent,
             float weight = 1.0f);
    
    /**
     * @brief Get the antecedent.
     * @return Rule antecedent.
     */
    std::shared_ptr<FuzzyAntecedent> get_antecedent() const;
    
    /**
     * @brief Get the consequent.
     * @return Rule consequent.
     */
    std::shared_ptr<FuzzyConsequent> get_consequent() const;
    
    /**
     * @brief Get the weight.
     * @return Rule weight.
     */
    float get_weight() const;
    
    /**
     * @brief Set the weight.
     * @param weight Rule weight.
     */
    void set_weight(float weight);
    
    /**
     * @brief Evaluate the rule.
     * @return Activation degree in [0, 1].
     */
    float evaluate() const;
    
private:
    std::shared_ptr<FuzzyAntecedent> antecedent_;
    std::shared_ptr<FuzzyConsequent> consequent_;
    float weight_;
};

/**
 * @brief Enumeration of defuzzification methods.
 */
enum class DefuzzificationMethod {
    CENTROID,
    BISECTOR,
    MEAN_OF_MAXIMUM,
    SMALLEST_OF_MAXIMUM,
    LARGEST_OF_MAXIMUM,
    WEIGHTED_AVERAGE
};

/**
 * @brief Fuzzy inference system.
 */
class FuzzyInferenceSystem {
public:
    /**
     * @brief Constructor.
     * @param name Name of the system.
     * @param and_op Fuzzy AND operator.
     * @param or_op Fuzzy OR operator.
     * @param implication_op Fuzzy implication operator.
     * @param aggregation_op Fuzzy aggregation operator.
     * @param defuzzification_method Defuzzification method.
     */
    FuzzyInferenceSystem(const std::string& name,
                       FuzzyOperator and_op = FuzzyOperator::MIN,
                       FuzzyOperator or_op = FuzzyOperator::MAX,
                       FuzzyOperator implication_op = FuzzyOperator::MIN,
                       FuzzyOperator aggregation_op = FuzzyOperator::MAX,
                       DefuzzificationMethod defuzzification_method = DefuzzificationMethod::CENTROID);
    
    /**
     * @brief Add a linguistic variable to the system.
     * @param variable Linguistic variable.
     */
    void add_variable(const LinguisticVariable& variable);
    
    /**
     * @brief Add a rule to the system.
     * @param rule Fuzzy rule.
     */
    void add_rule(const FuzzyRule& rule);
    
    /**
     * @brief Get a linguistic variable by name.
     * @param name Name of the variable.
     * @return Linguistic variable.
     */
    const LinguisticVariable& get_variable(const std::string& name) const;
    
    /**
     * @brief Get all linguistic variables.
     * @return Vector of linguistic variables.
     */
    std::vector<LinguisticVariable> get_variables() const;
    
    /**
     * @brief Get all rules.
     * @return Vector of fuzzy rules.
     */
    std::vector<FuzzyRule> get_rules() const;
    
    /**
     * @brief Set the input value of a variable.
     * @param variable_name Name of the variable.
     * @param value Input value.
     */
    void set_input(const std::string& variable_name, float value);
    
    /**
     * @brief Get the output value of a variable.
     * @param variable_name Name of the variable.
     * @return Output value.
     */
    float get_output(const std::string& variable_name) const;
    
    /**
     * @brief Evaluate the system.
     */
    void evaluate();
    
private:
    std::string name_;
    FuzzyOperator and_op_;
    FuzzyOperator or_op_;
    FuzzyOperator implication_op_;
    FuzzyOperator aggregation_op_;
    DefuzzificationMethod defuzzification_method_;
    std::unordered_map<std::string, LinguisticVariable> variables_;
    std::vector<FuzzyRule> rules_;
    std::unordered_map<std::string, float> inputs_;
    std::unordered_map<std::string, float> outputs_;
    
    /**
     * @brief Apply a fuzzy operator to two values.
     * @param a First value.
     * @param b Second value.
     * @param op Fuzzy operator.
     * @return Result of the operation.
     */
    float apply_operator(float a, float b, FuzzyOperator op) const;
    
    /**
     * @brief Defuzzify a fuzzy set.
     * @param variable Linguistic variable.
     * @param fuzzy_set Fuzzy set.
     * @param method Defuzzification method.
     * @return Crisp value.
     */
    float defuzzify(const LinguisticVariable& variable,
                  const std::vector<float>& fuzzy_set,
                  DefuzzificationMethod method) const;
};

} // namespace fuzzy
} // namespace phynexus

#endif // PHYNEXUS_FUZZY_H
