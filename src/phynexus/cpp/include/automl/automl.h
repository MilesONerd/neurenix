/**
 * @file automl.h
 * @brief AutoML module for Neurenix C++ backend.
 * 
 * This file provides the C++ implementation of the AutoML module,
 * including hyperparameter search, neural architecture search,
 * model selection, and pipeline construction.
 */

#ifndef PHYNEXUS_AUTOML_H
#define PHYNEXUS_AUTOML_H

#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <memory>
#include <random>
#include <algorithm>
#include <limits>
#include <cmath>

#include "phynexus/tensor.h"
#include "phynexus/error.h"

namespace phynexus {
namespace automl {

/**
 * @brief Base class for hyperparameter search algorithms.
 */
class HyperparameterSearch {
public:
    using ParamSpace = std::unordered_map<std::string, std::vector<double>>;
    using ObjectiveFunction = std::function<double(const std::unordered_map<std::string, double>&)>;

    HyperparameterSearch(const ParamSpace& param_space, int max_iterations = 100);
    virtual ~HyperparameterSearch() = default;

    virtual std::unordered_map<std::string, double> search(const ObjectiveFunction& objective) = 0;
    
    const std::unordered_map<std::string, double>& get_best_params() const;
    double get_best_score() const;

protected:
    ParamSpace param_space_;
    int max_iterations_;
    std::unordered_map<std::string, double> best_params_;
    double best_score_;
};

/**
 * @brief Grid search for hyperparameter optimization.
 */
class GridSearch : public HyperparameterSearch {
public:
    GridSearch(const ParamSpace& param_space, int max_iterations = 100);
    ~GridSearch() override = default;

    std::unordered_map<std::string, double> search(const ObjectiveFunction& objective) override;
};

/**
 * @brief Random search for hyperparameter optimization.
 */
class RandomSearch : public HyperparameterSearch {
public:
    RandomSearch(const ParamSpace& param_space, int max_iterations = 100, unsigned int seed = 0);
    ~RandomSearch() override = default;

    std::unordered_map<std::string, double> search(const ObjectiveFunction& objective) override;

private:
    std::mt19937 rng_;
};

/**
 * @brief Bayesian optimization for hyperparameter search.
 */
class BayesianOptimization : public HyperparameterSearch {
public:
    BayesianOptimization(const ParamSpace& param_space, int max_iterations = 100, 
                         double exploration_weight = 0.1, unsigned int seed = 0);
    ~BayesianOptimization() override = default;

    std::unordered_map<std::string, double> search(const ObjectiveFunction& objective) override;

private:
    double exploration_weight_;
    std::mt19937 rng_;
    
    // Observed points and their values
    std::vector<std::unordered_map<std::string, double>> observed_points_;
    std::vector<double> observed_values_;
    
    // Helper methods for Bayesian optimization
    double acquisition_function(const std::unordered_map<std::string, double>& point);
    double expected_improvement(const std::unordered_map<std::string, double>& point);
    double gaussian_process_predict(const std::unordered_map<std::string, double>& point);
    double gaussian_process_variance(const std::unordered_map<std::string, double>& point);
};

/**
 * @brief Evolutionary search for hyperparameter optimization.
 */
class EvolutionarySearch : public HyperparameterSearch {
public:
    EvolutionarySearch(const ParamSpace& param_space, int max_iterations = 100, 
                      int population_size = 10, double mutation_rate = 0.1, 
                      double crossover_rate = 0.5, unsigned int seed = 0);
    ~EvolutionarySearch() override = default;

    std::unordered_map<std::string, double> search(const ObjectiveFunction& objective) override;

private:
    int population_size_;
    double mutation_rate_;
    double crossover_rate_;
    std::mt19937 rng_;
    
    // Helper methods for evolutionary search
    std::vector<std::unordered_map<std::string, double>> initialize_population();
    std::unordered_map<std::string, double> mutate(const std::unordered_map<std::string, double>& individual);
    std::unordered_map<std::string, double> crossover(
        const std::unordered_map<std::string, double>& parent1,
        const std::unordered_map<std::string, double>& parent2);
    std::vector<std::unordered_map<std::string, double>> select_parents(
        const std::vector<std::unordered_map<std::string, double>>& population,
        const std::vector<double>& fitness);
};

/**
 * @brief Base class for neural architecture search algorithms.
 */
class NeuralArchitectureSearch {
public:
    using Architecture = std::unordered_map<std::string, std::string>;
    using SearchSpace = std::unordered_map<std::string, std::vector<std::string>>;
    using ObjectiveFunction = std::function<double(const Architecture&)>;

    NeuralArchitectureSearch(const SearchSpace& search_space, int max_iterations = 100);
    virtual ~NeuralArchitectureSearch() = default;

    virtual Architecture search(const ObjectiveFunction& objective) = 0;
    
    const Architecture& get_best_architecture() const;
    double get_best_score() const;

protected:
    SearchSpace search_space_;
    int max_iterations_;
    Architecture best_architecture_;
    double best_score_;
};

/**
 * @brief Efficient Neural Architecture Search (ENAS).
 */
class ENAS : public NeuralArchitectureSearch {
public:
    ENAS(const SearchSpace& search_space, int max_iterations = 100,
         int controller_hidden_size = 64, double controller_temperature = 5.0,
         double controller_tanh_constant = 2.5, double controller_entropy_weight = 0.0001,
         unsigned int seed = 0);
    ~ENAS() override = default;

    Architecture search(const ObjectiveFunction& objective) override;

private:
    int controller_hidden_size_;
    double controller_temperature_;
    double controller_tanh_constant_;
    double controller_entropy_weight_;
    std::mt19937 rng_;
    
    // Helper methods for ENAS
    Architecture sample_architecture();
    void update_controller(const std::vector<Architecture>& architectures, 
                          const std::vector<double>& rewards);
};

/**
 * @brief Differentiable Architecture Search (DARTS).
 */
class DARTS : public NeuralArchitectureSearch {
public:
    DARTS(const SearchSpace& search_space, int max_iterations = 100,
          bool unrolled = true, double alpha_lr = 0.0003, 
          double alpha_weight_decay = 0.001, unsigned int seed = 0);
    ~DARTS() override = default;

    Architecture search(const ObjectiveFunction& objective) override;

private:
    bool unrolled_;
    double alpha_lr_;
    double alpha_weight_decay_;
    std::mt19937 rng_;
    
    // Helper methods for DARTS
    std::unordered_map<std::string, std::vector<double>> initialize_alphas();
    Architecture derive_architecture(const std::unordered_map<std::string, std::vector<double>>& alphas);
};

/**
 * @brief Progressive Neural Architecture Search (PNAS).
 */
class PNAS : public NeuralArchitectureSearch {
public:
    PNAS(const SearchSpace& search_space, int max_iterations = 100,
         int num_init_architectures = 10, int num_expansions = 5,
         int k_best = 5, unsigned int seed = 0);
    ~PNAS() override = default;

    Architecture search(const ObjectiveFunction& objective) override;

private:
    int num_init_architectures_;
    int num_expansions_;
    int k_best_;
    std::mt19937 rng_;
    
    // Helper methods for PNAS
    std::vector<Architecture> initialize_architectures();
    std::vector<Architecture> expand_architectures(const std::vector<Architecture>& architectures);
};

/**
 * @brief Base class for model selection algorithms.
 */
class ModelSelection {
public:
    using Model = std::function<void()>;
    using ModelFactory = std::function<Model(const std::unordered_map<std::string, double>&)>;
    using ParamSpace = std::unordered_map<std::string, std::vector<double>>;
    using ObjectiveFunction = std::function<double(const Model&)>;

    ModelSelection(const std::vector<ModelFactory>& model_factories,
                  const std::unordered_map<ModelFactory, ParamSpace>& hyperparams,
                  int max_trials = 10);
    virtual ~ModelSelection() = default;

    virtual Model select(const ObjectiveFunction& objective) = 0;
    
    const Model& get_best_model() const;
    double get_best_score() const;

protected:
    std::vector<ModelFactory> model_factories_;
    std::unordered_map<ModelFactory, ParamSpace> hyperparams_;
    int max_trials_;
    Model best_model_;
    double best_score_;
};

/**
 * @brief Cross-validation for model evaluation.
 */
class CrossValidation {
public:
    CrossValidation(int n_splits = 5, bool shuffle = true, unsigned int seed = 0);
    ~CrossValidation() = default;

    std::vector<std::pair<std::vector<int>, std::vector<int>>> split(int n_samples);
    
    template<typename X, typename Y>
    double evaluate(const std::function<Model()>& model_factory, 
                   const X& x, const Y& y);

private:
    int n_splits_;
    bool shuffle_;
    std::mt19937 rng_;
};

/**
 * @brief Nested cross-validation for model selection and evaluation.
 */
class NestedCrossValidation {
public:
    NestedCrossValidation(int outer_splits = 3, int inner_splits = 3,
                         bool shuffle = true, unsigned int seed = 0);
    ~NestedCrossValidation() = default;

    template<typename X, typename Y>
    std::pair<double, std::vector<std::unordered_map<std::string, double>>> 
    evaluate(const ModelFactory& model_factory, const ParamSpace& param_grid,
             const X& x, const Y& y);

private:
    CrossValidation outer_cv_;
    CrossValidation inner_cv_;
};

/**
 * @brief Base class for feature selection algorithms.
 */
class FeatureSelection {
public:
    FeatureSelection(int n_features_to_select = -1);
    virtual ~FeatureSelection() = default;

    virtual void fit(const Tensor& x, const Tensor& y) = 0;
    virtual Tensor transform(const Tensor& x) const = 0;
    virtual Tensor fit_transform(const Tensor& x, const Tensor& y);

protected:
    int n_features_to_select_;
    std::vector<int> selected_features_;
};

/**
 * @brief Variance threshold for feature selection.
 */
class VarianceThreshold : public FeatureSelection {
public:
    VarianceThreshold(double threshold = 0.0);
    ~VarianceThreshold() override = default;

    void fit(const Tensor& x, const Tensor& y) override;
    Tensor transform(const Tensor& x) const override;

private:
    double threshold_;
};

/**
 * @brief Select K best features based on a score function.
 */
class SelectKBest : public FeatureSelection {
public:
    using ScoreFunction = std::function<std::vector<double>(const Tensor&, const Tensor&)>;

    SelectKBest(const ScoreFunction& score_func, int k = -1);
    ~SelectKBest() override = default;

    void fit(const Tensor& x, const Tensor& y) override;
    Tensor transform(const Tensor& x) const override;

private:
    ScoreFunction score_func_;
    std::vector<double> scores_;
};

/**
 * @brief Base class for data preprocessing algorithms.
 */
class DataPreprocessing {
public:
    DataPreprocessing() = default;
    virtual ~DataPreprocessing() = default;

    virtual void fit(const Tensor& x) = 0;
    virtual Tensor transform(const Tensor& x) const = 0;
    virtual Tensor fit_transform(const Tensor& x);
};

/**
 * @brief Standardize features by removing the mean and scaling to unit variance.
 */
class StandardScaler : public DataPreprocessing {
public:
    StandardScaler();
    ~StandardScaler() override = default;

    void fit(const Tensor& x) override;
    Tensor transform(const Tensor& x) const override;

private:
    Tensor mean_;
    Tensor std_;
};

/**
 * @brief Scale features to a given range.
 */
class MinMaxScaler : public DataPreprocessing {
public:
    MinMaxScaler(double feature_min = 0.0, double feature_max = 1.0);
    ~MinMaxScaler() override = default;

    void fit(const Tensor& x) override;
    Tensor transform(const Tensor& x) const override;

private:
    double feature_min_;
    double feature_max_;
    Tensor min_;
    Tensor max_;
    Tensor scale_;
};

/**
 * @brief Pipeline for chaining multiple transformers.
 */
class Pipeline {
public:
    using Transformer = std::function<Tensor(const Tensor&)>;

    Pipeline();
    ~Pipeline() = default;

    void add_step(const std::string& name, const Transformer& transformer);
    void fit(const Tensor& x, const Tensor& y = Tensor());
    Tensor transform(const Tensor& x) const;
    Tensor fit_transform(const Tensor& x, const Tensor& y = Tensor());

private:
    std::vector<std::pair<std::string, Transformer>> steps_;
};

} // namespace automl
} // namespace phynexus

#endif // PHYNEXUS_AUTOML_H
