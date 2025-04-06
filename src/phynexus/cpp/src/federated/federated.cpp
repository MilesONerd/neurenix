/**
 * @file federated.cpp
 * @brief Implementation of Federated Learning module for Neurenix C++ backend.
 */

#include "federated/federated.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <iostream>
#include <numeric>

namespace phynexus {
namespace federated {


FederatedClient::FederatedClient(int client_id,
                               std::shared_ptr<FederatedModel> model,
                               const FederatedConfig& config)
    : client_id_(client_id), model_(model), config_(config), num_samples_(0) {
    
    initial_parameters_ = model_->get_parameters();
}

std::unordered_map<std::string, float> FederatedClient::train(
    const std::vector<std::vector<float>>& data_x,
    const std::vector<std::vector<float>>& data_y) {
    
    if (config_.aggregation_method == AggregationMethod::FEDPROX) {
        initial_parameters_ = model_->get_parameters();
    }
    
    auto batches = create_batches(data_x, data_y);
    
    float total_loss = 0.0f;
    int total_batches = 0;
    
    for (int epoch = 0; epoch < config_.local_epochs; ++epoch) {
        for (const auto& batch : batches) {
            const auto& batch_x = batch.first;
            const auto& batch_y = batch.second;
            
            float batch_loss = model_->train_batch(batch_x, batch_y);
            
            if (config_.aggregation_method == AggregationMethod::FEDPROX && config_.proximal_mu > 0.0f) {
                std::vector<float> current_params = model_->get_parameters();
                float proximal_term = 0.0f;
                
                for (size_t i = 0; i < current_params.size(); ++i) {
                    float diff = current_params[i] - initial_parameters_[i];
                    proximal_term += diff * diff;
                }
                
                proximal_term *= 0.5f * config_.proximal_mu;
                batch_loss += proximal_term;
            }
            
            total_loss += batch_loss;
            total_batches++;
        }
    }
    
    float avg_loss = total_batches > 0 ? total_loss / total_batches : 0.0f;
    
    std::unordered_map<std::string, float> metrics;
    metrics["loss"] = avg_loss;
    metrics["num_samples"] = static_cast<float>(num_samples_);
    
    return metrics;
}

std::vector<std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>> FederatedClient::create_batches(
    const std::vector<std::vector<float>>& data_x,
    const std::vector<std::vector<float>>& data_y) const {
    
    if (data_x.empty() || data_y.empty() || data_x.size() != data_y.size()) {
        return {};
    }
    
    std::vector<std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>>> batches;
    
    int num_samples = data_x.size();
    int batch_size = config_.batch_size;
    int num_batches = (num_samples + batch_size - 1) / batch_size;
    
    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);
    
    for (int i = 0; i < num_batches; ++i) {
        int start_idx = i * batch_size;
        int end_idx = std::min(start_idx + batch_size, num_samples);
        
        std::vector<std::vector<float>> batch_x;
        std::vector<std::vector<float>> batch_y;
        
        for (int j = start_idx; j < end_idx; ++j) {
            int idx = indices[j];
            batch_x.push_back(data_x[idx]);
            batch_y.push_back(data_y[idx]);
        }
        
        batches.push_back(std::make_pair(batch_x, batch_y));
    }
    
    return batches;
}


DifferentialPrivacy::DifferentialPrivacy(float epsilon, float delta, float sensitivity)
    : epsilon_(epsilon), delta_(delta), sensitivity_(sensitivity) {}

std::vector<float> DifferentialPrivacy::add_noise(const std::vector<float>& parameters) const {
    std::vector<float> noisy_parameters = parameters;
    
    float noise_scale = compute_noise_scale();
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, noise_scale);
    
    for (auto& param : noisy_parameters) {
        param += dist(gen);
    }
    
    return noisy_parameters;
}

float DifferentialPrivacy::compute_noise_scale() const {
    return sensitivity_ * std::sqrt(2.0f * std::log(1.25f / delta_)) / epsilon_;
}


SecureAggregation::SecureAggregation(int num_clients, int seed)
    : num_clients_(num_clients), seed_(seed) {}

std::vector<float> SecureAggregation::mask_parameters(int client_id, const std::vector<float>& parameters) const {
    std::vector<float> masked_parameters = parameters;
    
    for (int other_id = 0; other_id < num_clients_; ++other_id) {
        if (client_id == other_id) {
            continue;
        }
        
        std::vector<float> mask = generate_mask(client_id * num_clients_ + other_id, parameters.size());
        
        if (client_id < other_id) {
            for (size_t i = 0; i < masked_parameters.size(); ++i) {
                masked_parameters[i] += mask[i];
            }
        } else {
            for (size_t i = 0; i < masked_parameters.size(); ++i) {
                masked_parameters[i] -= mask[i];
            }
        }
    }
    
    return masked_parameters;
}

std::vector<float> SecureAggregation::generate_mask(int client_id, int size) const {
    std::vector<float> mask(size, 0.0f);
    
    std::mt19937 gen(seed_ + client_id);
    std::normal_distribution<float> dist(0.0f, 0.01f);
    
    for (int i = 0; i < size; ++i) {
        mask[i] = dist(gen);
    }
    
    return mask;
}


ModelCompressor::ModelCompressor(CompressionMethod method, float ratio)
    : method_(method), ratio_(ratio) {}

std::vector<float> ModelCompressor::compress(const std::vector<float>& parameters) const {
    switch (method_) {
        case CompressionMethod::QUANTIZATION:
            return quantize(parameters);
        case CompressionMethod::SPARSIFICATION:
            return sparsify(parameters);
        default:
            return parameters;
    }
}

std::vector<float> ModelCompressor::quantize(const std::vector<float>& parameters) const {
    if (parameters.empty()) {
        return {};
    }
    
    float min_val = *std::min_element(parameters.begin(), parameters.end());
    float max_val = *std::max_element(parameters.begin(), parameters.end());
    
    float range = max_val - min_val;
    int num_levels = static_cast<int>(1.0f / ratio_);
    float step = range / num_levels;
    
    std::vector<float> quantized_parameters;
    quantized_parameters.push_back(min_val);
    quantized_parameters.push_back(max_val);
    
    for (float param : parameters) {
        int level = static_cast<int>((param - min_val) / step);
        level = std::max(0, std::min(level, num_levels - 1));
        quantized_parameters.push_back(static_cast<float>(level));
    }
    
    return quantized_parameters;
}

std::vector<float> ModelCompressor::sparsify(const std::vector<float>& parameters) const {
    if (parameters.empty()) {
        return {};
    }
    
    std::vector<float> abs_parameters = parameters;
    for (auto& param : abs_parameters) {
        param = std::abs(param);
    }
    
    std::sort(abs_parameters.begin(), abs_parameters.end());
    int threshold_idx = static_cast<int>(parameters.size() * ratio_);
    float threshold = abs_parameters[threshold_idx];
    
    std::vector<float> sparsified_parameters;
    sparsified_parameters.push_back(static_cast<float>(parameters.size()));
    
    for (size_t i = 0; i < parameters.size(); ++i) {
        if (std::abs(parameters[i]) >= threshold) {
            sparsified_parameters.push_back(static_cast<float>(i));
            sparsified_parameters.push_back(parameters[i]);
        }
    }
    
    return sparsified_parameters;
}


FederatedServer::FederatedServer(std::shared_ptr<FederatedModel> model, const FederatedConfig& config)
    : model_(model), config_(config) {
    
    global_parameters_ = model_->get_parameters();
    
    if (config_.use_server_momentum) {
        server_momentum_.resize(global_parameters_.size(), 0.0f);
    }
    
    if (config_.privacy_mechanism == PrivacyMechanism::DIFFERENTIAL_PRIVACY) {
        dp_ = std::make_unique<DifferentialPrivacy>(config_.dp_epsilon, config_.dp_delta);
    }
    
    if (config_.privacy_mechanism == PrivacyMechanism::SECURE_AGGREGATION) {
        secure_agg_ = std::make_unique<SecureAggregation>(config_.num_clients);
    }
    
    if (config_.compression_method != CompressionMethod::NONE) {
        compressor_ = std::make_unique<ModelCompressor>(config_.compression_method, config_.compression_ratio);
    }
}

void FederatedServer::add_client(std::shared_ptr<FederatedClient> client) {
    clients_.push_back(client);
}

std::vector<int> FederatedServer::select_clients(int round) {
    int num_clients = clients_.size();
    int clients_per_round = std::min(config_.clients_per_round, num_clients);
    
    std::vector<int> selected_clients;
    
    std::vector<int> all_clients(num_clients);
    std::iota(all_clients.begin(), all_clients.end(), 0);
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(all_clients.begin(), all_clients.end(), g);
    
    selected_clients.assign(all_clients.begin(), all_clients.begin() + clients_per_round);
    
    return selected_clients;
}

} // namespace federated
} // namespace phynexus
