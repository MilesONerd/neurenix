/**
 * @file federated.h
 * @brief Header file for Federated Learning module in Neurenix C++ backend.
 */

#ifndef PHYNEXUS_FEDERATED_H
#define PHYNEXUS_FEDERATED_H

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <functional>
#include <cmath>
#include <random>

namespace phynexus {
namespace federated {

/**
 * @brief Enumeration of federated learning aggregation methods.
 */
enum class AggregationMethod {
    FEDAVG,
    FEDPROX,
    FEDOPT,
    FEDADAGRAD,
    FEDADAM,
    FEDYOGI,
    SCAFFOLD,
    FEDBN,
    FEDENSEMBLE,
    FEDMA
};

/**
 * @brief Enumeration of client selection strategies.
 */
enum class ClientSelectionStrategy {
    RANDOM,
    ROUND_ROBIN,
    POWER_OF_CHOICE,
    IMPORTANCE_SAMPLING,
    OORT,
    TIFL
};

/**
 * @brief Enumeration of privacy mechanisms.
 */
enum class PrivacyMechanism {
    NONE,
    DIFFERENTIAL_PRIVACY,
    SECURE_AGGREGATION,
    HOMOMORPHIC_ENCRYPTION,
    SPLIT_LEARNING,
    FEDERATED_DROPOUT
};

/**
 * @brief Enumeration of compression methods.
 */
enum class CompressionMethod {
    NONE,
    QUANTIZATION,
    SPARSIFICATION,
    LOW_RANK,
    SKETCHING,
    TERNARY_GRADIENTS,
    SIGNSGD
};

/**
 * @brief Configuration for federated learning.
 */
struct FederatedConfig {
    int num_rounds;
    int num_clients;
    int clients_per_round;
    float learning_rate;
    float client_learning_rate;
    float momentum;
    float weight_decay;
    int local_epochs;
    int batch_size;
    AggregationMethod aggregation_method;
    ClientSelectionStrategy client_selection;
    PrivacyMechanism privacy_mechanism;
    CompressionMethod compression_method;
    float compression_ratio;
    float dp_epsilon;
    float dp_delta;
    float proximal_mu;
    bool use_adaptive_lr;
    bool use_client_normalization;
    bool use_server_momentum;
    
    FederatedConfig()
        : num_rounds(100),
          num_clients(10),
          clients_per_round(5),
          learning_rate(0.01f),
          client_learning_rate(0.01f),
          momentum(0.9f),
          weight_decay(0.0001f),
          local_epochs(5),
          batch_size(32),
          aggregation_method(AggregationMethod::FEDAVG),
          client_selection(ClientSelectionStrategy::RANDOM),
          privacy_mechanism(PrivacyMechanism::NONE),
          compression_method(CompressionMethod::NONE),
          compression_ratio(0.1f),
          dp_epsilon(1.0f),
          dp_delta(1e-5f),
          proximal_mu(0.01f),
          use_adaptive_lr(false),
          use_client_normalization(false),
          use_server_momentum(false) {}
};

/**
 * @brief Base class for federated learning models.
 */
class FederatedModel {
public:
    virtual ~FederatedModel() = default;
    
    /**
     * @brief Get the number of parameters in the model.
     * @return Number of parameters.
     */
    virtual int num_parameters() const = 0;
    
    /**
     * @brief Get the parameters of the model.
     * @return Vector of parameters.
     */
    virtual std::vector<float> get_parameters() const = 0;
    
    /**
     * @brief Set the parameters of the model.
     * @param parameters Vector of parameters.
     */
    virtual void set_parameters(const std::vector<float>& parameters) = 0;
    
    /**
     * @brief Train the model on a batch of data.
     * @param batch_x Batch of input data.
     * @param batch_y Batch of target data.
     * @return Loss value.
     */
    virtual float train_batch(const std::vector<std::vector<float>>& batch_x,
                            const std::vector<std::vector<float>>& batch_y) = 0;
    
    /**
     * @brief Evaluate the model on a batch of data.
     * @param batch_x Batch of input data.
     * @param batch_y Batch of target data.
     * @return Loss value.
     */
    virtual float evaluate_batch(const std::vector<std::vector<float>>& batch_x,
                               const std::vector<std::vector<float>>& batch_y) const = 0;
    
    /**
     * @brief Predict using the model.
     * @param x Input data.
     * @return Predicted output.
     */
    virtual std::vector<float> predict(const std::vector<float>& x) const = 0;
};

/**
 * @brief Class representing a client in federated learning.
 */
class FederatedClient {
public:
    /**
     * @brief Constructor.
     * @param client_id Client ID.
     * @param model Federated model.
     * @param config Federated configuration.
     */
    FederatedClient(int client_id,
                  std::shared_ptr<FederatedModel> model,
                  const FederatedConfig& config);
    
    /**
     * @brief Train the client model.
     * @param data_x Client data inputs.
     * @param data_y Client data targets.
     * @return Training metrics.
     */
    std::unordered_map<std::string, float> train(
        const std::vector<std::vector<float>>& data_x,
        const std::vector<std::vector<float>>& data_y);
    
    /**
     * @brief Evaluate the client model.
     * @param data_x Client data inputs.
     * @param data_y Client data targets.
     * @return Evaluation metrics.
     */
    std::unordered_map<std::string, float> evaluate(
        const std::vector<std::vector<float>>& data_x,
        const std::vector<std::vector<float>>& data_y) const;
    
    /**
     * @brief Get the client model parameters.
     * @return Vector of parameters.
     */
    std::vector<float> get_parameters() const;
    
    /**
     * @brief Set the client model parameters.
     * @param parameters Vector of parameters.
     */
    void set_parameters(const std::vector<float>& parameters);
    
    /**
     * @brief Get the client ID.
     * @return Client ID.
     */
    int get_client_id() const;
    
    /**
     * @brief Get the number of samples.
     * @return Number of samples.
     */
    int get_num_samples() const;
    
    /**
     * @brief Set the number of samples.
     * @param num_samples Number of samples.
     */
    void set_num_samples(int num_samples);
    
private:
    int client_id_;
    std::shared_ptr<FederatedModel> model_;
    FederatedConfig config_;
    int num_samples_;
    std::vector<float> initial_parameters_;
    
    /**
     * @brief Create batches from data.
     * @param data_x Data inputs.
     * @param data_y Data targets.
     * @return Vector of batches.
     */
    std::vector<std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>> create_batches(
        const std::vector<std::vector<float>>& data_x,
        const std::vector<std::vector<float>>& data_y) const;
};

/**
 * @brief Class for differential privacy in federated learning.
 */
class DifferentialPrivacy {
public:
    /**
     * @brief Constructor.
     * @param epsilon Privacy parameter.
     * @param delta Privacy parameter.
     * @param sensitivity Sensitivity of the function.
     */
    DifferentialPrivacy(float epsilon = 1.0f, float delta = 1e-5f, float sensitivity = 1.0f);
    
    /**
     * @brief Add noise to parameters.
     * @param parameters Vector of parameters.
     * @return Noisy parameters.
     */
    std::vector<float> add_noise(const std::vector<float>& parameters) const;
    
private:
    float epsilon_;
    float delta_;
    float sensitivity_;
    
    /**
     * @brief Compute the scale of the noise to add.
     * @return Noise scale.
     */
    float compute_noise_scale() const;
};

/**
 * @brief Class for secure aggregation in federated learning.
 */
class SecureAggregation {
public:
    /**
     * @brief Constructor.
     * @param num_clients Number of clients.
     * @param seed Random seed.
     */
    SecureAggregation(int num_clients, int seed = 42);
    
    /**
     * @brief Mask parameters for a client.
     * @param client_id Client ID.
     * @param parameters Vector of parameters.
     * @return Masked parameters.
     */
    std::vector<float> mask_parameters(int client_id, const std::vector<float>& parameters) const;
    
    /**
     * @brief Unmask aggregated parameters.
     * @param aggregated_parameters Vector of aggregated parameters.
     * @param participating_clients Vector of participating client IDs.
     * @return Unmasked parameters.
     */
    std::vector<float> unmask_aggregated_parameters(
        const std::vector<float>& aggregated_parameters,
        const std::vector<int>& participating_clients) const;
    
private:
    int num_clients_;
    int seed_;
    
    /**
     * @brief Generate a mask for a client.
     * @param client_id Client ID.
     * @param size Size of the mask.
     * @return Mask vector.
     */
    std::vector<float> generate_mask(int client_id, int size) const;
};

/**
 * @brief Class for model compression in federated learning.
 */
class ModelCompressor {
public:
    /**
     * @brief Constructor.
     * @param method Compression method.
     * @param ratio Compression ratio.
     */
    ModelCompressor(CompressionMethod method = CompressionMethod::QUANTIZATION, float ratio = 0.1f);
    
    /**
     * @brief Compress parameters.
     * @param parameters Vector of parameters.
     * @return Compressed parameters.
     */
    std::vector<float> compress(const std::vector<float>& parameters) const;
    
    /**
     * @brief Decompress parameters.
     * @param compressed_parameters Vector of compressed parameters.
     * @param original_size Original size of the parameters.
     * @return Decompressed parameters.
     */
    std::vector<float> decompress(const std::vector<float>& compressed_parameters, int original_size) const;
    
private:
    CompressionMethod method_;
    float ratio_;
    
    /**
     * @brief Quantize parameters.
     * @param parameters Vector of parameters.
     * @return Quantized parameters.
     */
    std::vector<float> quantize(const std::vector<float>& parameters) const;
    
    /**
     * @brief Dequantize parameters.
     * @param quantized_parameters Vector of quantized parameters.
     * @return Dequantized parameters.
     */
    std::vector<float> dequantize(const std::vector<float>& quantized_parameters) const;
    
    /**
     * @brief Sparsify parameters.
     * @param parameters Vector of parameters.
     * @return Sparsified parameters.
     */
    std::vector<float> sparsify(const std::vector<float>& parameters) const;
    
    /**
     * @brief Desparsify parameters.
     * @param sparsified_parameters Vector of sparsified parameters.
     * @param original_size Original size of the parameters.
     * @return Desparsified parameters.
     */
    std::vector<float> desparsify(const std::vector<float>& sparsified_parameters, int original_size) const;
};

/**
 * @brief Class for federated learning server.
 */
class FederatedServer {
public:
    /**
     * @brief Constructor.
     * @param model Federated model.
     * @param config Federated configuration.
     */
    FederatedServer(std::shared_ptr<FederatedModel> model, const FederatedConfig& config);
    
    /**
     * @brief Add a client to the server.
     * @param client Federated client.
     */
    void add_client(std::shared_ptr<FederatedClient> client);
    
    /**
     * @brief Run federated learning.
     * @param client_data_x Vector of client data inputs.
     * @param client_data_y Vector of client data targets.
     * @return Training metrics.
     */
    std::unordered_map<std::string, std::vector<float>> run_federated_learning(
        const std::vector<std::vector<std::vector<float>>>& client_data_x,
        const std::vector<std::vector<std::vector<float>>>& client_data_y);
    
    /**
     * @brief Evaluate the global model.
     * @param data_x Data inputs.
     * @param data_y Data targets.
     * @return Evaluation metrics.
     */
    std::unordered_map<std::string, float> evaluate_global_model(
        const std::vector<std::vector<float>>& data_x,
        const std::vector<std::vector<float>>& data_y) const;
    
    /**
     * @brief Get the global model parameters.
     * @return Vector of parameters.
     */
    std::vector<float> get_global_parameters() const;
    
    /**
     * @brief Set the global model parameters.
     * @param parameters Vector of parameters.
     */
    void set_global_parameters(const std::vector<float>& parameters);
    
    /**
     * @brief Get the federated model.
     * @return Federated model.
     */
    std::shared_ptr<FederatedModel> get_model() const;
    
private:
    std::shared_ptr<FederatedModel> model_;
    FederatedConfig config_;
    std::vector<std::shared_ptr<FederatedClient>> clients_;
    std::vector<float> global_parameters_;
    std::vector<float> server_momentum_;
    std::unique_ptr<DifferentialPrivacy> dp_;
    std::unique_ptr<SecureAggregation> secure_agg_;
    std::unique_ptr<ModelCompressor> compressor_;
    
    /**
     * @brief Select clients for a round.
     * @param round Current round.
     * @return Vector of selected client indices.
     */
    std::vector<int> select_clients(int round);
    
    /**
     * @brief Aggregate client parameters.
     * @param client_parameters Vector of client parameters.
     * @param client_weights Vector of client weights.
     * @param client_indices Vector of client indices.
     * @return Aggregated parameters.
     */
    std::vector<float> aggregate_parameters(
        const std::vector<std::vector<float>>& client_parameters,
        const std::vector<float>& client_weights,
        const std::vector<int>& client_indices);
    
    /**
     * @brief Apply server optimization.
     * @param aggregated_parameters Vector of aggregated parameters.
     * @param round Current round.
     * @return Updated parameters.
     */
    std::vector<float> apply_server_optimization(
        const std::vector<float>& aggregated_parameters,
        int round);
};

} // namespace federated
} // namespace phynexus

#endif // PHYNEXUS_FEDERATED_H
