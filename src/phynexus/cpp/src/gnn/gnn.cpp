/**
 * @file gnn.cpp
 * @brief Implementation of Graph Neural Network module for Neurenix C++ backend.
 */

#include "gnn/gnn.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <limits>
#include <iostream>
#include <numeric>

namespace phynexus {
namespace gnn {

int Graph::add_node(const Node& node) {
    int id = node.id;
    
    if (node_id_to_index_.find(id) != node_id_to_index_.end()) {
        return -1;
    }
    
    node_id_to_index_[id] = nodes_.size();
    nodes_.push_back(node);
    
    return id;
}

void Graph::add_edge(const Edge& edge) {
    if (node_id_to_index_.find(edge.source_id) == node_id_to_index_.end() ||
        node_id_to_index_.find(edge.target_id) == node_id_to_index_.end()) {
        return;
    }
    
    edges_.push_back(edge);
}

const Node& Graph::get_node(int id) const {
    if (node_id_to_index_.find(id) == node_id_to_index_.end()) {
        throw std::runtime_error("Node with ID " + std::to_string(id) + " not found");
    }
    
    return nodes_[node_id_to_index_.at(id)];
}

const std::vector<Node>& Graph::get_nodes() const {
    return nodes_;
}

const std::vector<Edge>& Graph::get_edges() const {
    return edges_;
}

std::vector<Edge> Graph::get_node_edges(int node_id) const {
    std::vector<Edge> node_edges;
    
    for (const auto& edge : edges_) {
        if (edge.source_id == node_id || edge.target_id == node_id) {
            node_edges.push_back(edge);
        }
    }
    
    return node_edges;
}

std::vector<int> Graph::get_neighbors(int node_id) const {
    std::vector<int> neighbors;
    
    for (const auto& edge : edges_) {
        if (edge.source_id == node_id) {
            neighbors.push_back(edge.target_id);
        } else if (edge.target_id == node_id) {
            neighbors.push_back(edge.source_id);
        }
    }
    
    return neighbors;
}

std::vector<std::vector<float>> Graph::get_adjacency_matrix() const {
    int num_nodes = nodes_.size();
    std::vector<std::vector<float>> adjacency_matrix(num_nodes, std::vector<float>(num_nodes, 0.0f));
    
    for (const auto& edge : edges_) {
        int source_idx = node_id_to_index_.at(edge.source_id);
        int target_idx = node_id_to_index_.at(edge.target_id);
        
        adjacency_matrix[source_idx][target_idx] = edge.weight;
        adjacency_matrix[target_idx][source_idx] = edge.weight; // Assuming undirected graph
    }
    
    return adjacency_matrix;
}

std::vector<std::vector<float>> Graph::get_feature_matrix() const {
    std::vector<std::vector<float>> feature_matrix;
    
    for (const auto& node : nodes_) {
        feature_matrix.push_back(node.features);
    }
    
    return feature_matrix;
}

GCNLayer::GCNLayer(int in_features, int out_features, bool use_bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(use_bias) {
    
    weight_.resize(in_features_, std::vector<float>(out_features_, 0.0f));
    
    float std_dev = std::sqrt(2.0f / (in_features_ + out_features_));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    for (int i = 0; i < in_features_; ++i) {
        for (int j = 0; j < out_features_; ++j) {
            weight_[i][j] = dist(gen);
        }
    }
    
    if (use_bias_) {
        bias_.resize(out_features_, 0.0f);
    }
}

std::vector<std::vector<float>> GCNLayer::forward(
    const Graph& graph,
    const std::vector<std::vector<float>>& node_features) const {
    
    int num_nodes = node_features.size();
    std::vector<std::vector<float>> output(num_nodes, std::vector<float>(out_features_, 0.0f));
    
    std::vector<std::vector<float>> adjacency_matrix = graph.get_adjacency_matrix();
    
    for (int i = 0; i < num_nodes; ++i) {
        adjacency_matrix[i][i] += 1.0f;
    }
    
    std::vector<float> degree(num_nodes, 0.0f);
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            degree[i] += adjacency_matrix[i][j];
        }
        degree[i] = 1.0f / std::sqrt(degree[i]);
    }
    
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < num_nodes; ++j) {
            adjacency_matrix[i][j] = adjacency_matrix[i][j] * degree[i] * degree[j];
        }
    }
    
    std::vector<std::vector<float>> transformed_features(num_nodes, std::vector<float>(out_features_, 0.0f));
    
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < out_features_; ++j) {
            for (int k = 0; k < in_features_; ++k) {
                transformed_features[i][j] += node_features[i][k] * weight_[k][j];
            }
            
            if (use_bias_) {
                transformed_features[i][j] += bias_[j];
            }
        }
    }
    
    for (int i = 0; i < num_nodes; ++i) {
        for (int j = 0; j < out_features_; ++j) {
            for (int k = 0; k < num_nodes; ++k) {
                output[i][j] += adjacency_matrix[i][k] * transformed_features[k][j];
            }
            
            output[i][j] = std::max(0.0f, output[i][j]);
        }
    }
    
    return output;
}

int GCNLayer::num_parameters() const {
    int num_params = in_features_ * out_features_;
    
    if (use_bias_) {
        num_params += out_features_;
    }
    
    return num_params;
}

std::vector<float> GCNLayer::get_parameters() const {
    std::vector<float> parameters;
    
    for (int i = 0; i < in_features_; ++i) {
        for (int j = 0; j < out_features_; ++j) {
            parameters.push_back(weight_[i][j]);
        }
    }
    
    if (use_bias_) {
        for (int i = 0; i < out_features_; ++i) {
            parameters.push_back(bias_[i]);
        }
    }
    
    return parameters;
}

void GCNLayer::set_parameters(const std::vector<float>& parameters) {
    int idx = 0;
    
    for (int i = 0; i < in_features_; ++i) {
        for (int j = 0; j < out_features_; ++j) {
            weight_[i][j] = parameters[idx++];
        }
    }
    
    if (use_bias_) {
        for (int i = 0; i < out_features_; ++i) {
            bias_[i] = parameters[idx++];
        }
    }
}

GNNModel::GNNModel() {}

void GNNModel::add_layer(std::shared_ptr<GNNLayer> layer) {
    layers_.push_back(layer);
}

void GNNModel::add_pooling_layer(std::shared_ptr<GraphPoolingLayer> pooling_layer) {
    pooling_layers_.push_back(pooling_layer);
}

std::vector<float> GNNModel::forward(const Graph& graph) const {
    std::vector<std::vector<float>> node_features = graph.get_feature_matrix();
    
    for (const auto& layer : layers_) {
        node_features = layer->forward(graph, node_features);
    }
    
    if (!pooling_layers_.empty()) {
        return pooling_layers_[0]->forward(graph, node_features);
    }
    
    std::vector<float> flattened;
    for (const auto& features : node_features) {
        flattened.insert(flattened.end(), features.begin(), features.end());
    }
    
    return flattened;
}

int GNNModel::num_parameters() const {
    int num_params = 0;
    
    for (const auto& layer : layers_) {
        num_params += layer->num_parameters();
    }
    
    return num_params;
}

std::vector<float> GNNModel::get_parameters() const {
    std::vector<float> parameters;
    
    for (const auto& layer : layers_) {
        std::vector<float> layer_params = layer->get_parameters();
        parameters.insert(parameters.end(), layer_params.begin(), layer_params.end());
    }
    
    return parameters;
}

void GNNModel::set_parameters(const std::vector<float>& parameters) {
    int idx = 0;
    
    for (const auto& layer : layers_) {
        int num_params = layer->num_parameters();
        std::vector<float> layer_params(parameters.begin() + idx, parameters.begin() + idx + num_params);
        layer->set_parameters(layer_params);
        idx += num_params;
    }
}

} // namespace gnn
} // namespace phynexus
