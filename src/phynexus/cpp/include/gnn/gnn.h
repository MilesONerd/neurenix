/**
 * @file gnn.h
 * @brief Header file for Graph Neural Network module in Neurenix C++ backend.
 */

#ifndef PHYNEXUS_GNN_H
#define PHYNEXUS_GNN_H

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <functional>
#include <cmath>

namespace phynexus {
namespace gnn {

/**
 * @brief Structure representing a node in a graph.
 */
struct Node {
    int id;
    std::vector<float> features;
    std::unordered_map<std::string, float> attributes;
};

/**
 * @brief Structure representing an edge in a graph.
 */
struct Edge {
    int source_id;
    int target_id;
    float weight;
    std::unordered_map<std::string, float> attributes;
};

/**
 * @brief Class representing a graph data structure.
 */
class Graph {
public:
    Graph() = default;
    
    /**
     * @brief Add a node to the graph.
     * @param node Node to add.
     * @return ID of the added node.
     */
    int add_node(const Node& node);
    
    /**
     * @brief Add an edge to the graph.
     * @param edge Edge to add.
     */
    void add_edge(const Edge& edge);
    
    /**
     * @brief Get a node by its ID.
     * @param id ID of the node.
     * @return Node with the specified ID.
     */
    const Node& get_node(int id) const;
    
    /**
     * @brief Get all nodes in the graph.
     * @return Vector of all nodes.
     */
    const std::vector<Node>& get_nodes() const;
    
    /**
     * @brief Get all edges in the graph.
     * @return Vector of all edges.
     */
    const std::vector<Edge>& get_edges() const;
    
    /**
     * @brief Get all edges connected to a node.
     * @param node_id ID of the node.
     * @return Vector of edges connected to the node.
     */
    std::vector<Edge> get_node_edges(int node_id) const;
    
    /**
     * @brief Get all neighbors of a node.
     * @param node_id ID of the node.
     * @return Vector of neighbor node IDs.
     */
    std::vector<int> get_neighbors(int node_id) const;
    
    /**
     * @brief Get the adjacency matrix of the graph.
     * @return Adjacency matrix as a vector of vectors.
     */
    std::vector<std::vector<float>> get_adjacency_matrix() const;
    
    /**
     * @brief Get the feature matrix of the graph.
     * @return Feature matrix as a vector of vectors.
     */
    std::vector<std::vector<float>> get_feature_matrix() const;
    
private:
    std::vector<Node> nodes_;
    std::vector<Edge> edges_;
    std::unordered_map<int, int> node_id_to_index_;
};

/**
 * @brief Base class for graph neural network layers.
 */
class GNNLayer {
public:
    virtual ~GNNLayer() = default;
    
    /**
     * @brief Forward pass through the layer.
     * @param graph Input graph.
     * @param node_features Input node features.
     * @return Output node features.
     */
    virtual std::vector<std::vector<float>> forward(
        const Graph& graph,
        const std::vector<std::vector<float>>& node_features) const = 0;
    
    /**
     * @brief Get the number of parameters in the layer.
     * @return Number of parameters.
     */
    virtual int num_parameters() const = 0;
    
    /**
     * @brief Get the parameters of the layer.
     * @return Vector of parameters.
     */
    virtual std::vector<float> get_parameters() const = 0;
    
    /**
     * @brief Set the parameters of the layer.
     * @param parameters Vector of parameters.
     */
    virtual void set_parameters(const std::vector<float>& parameters) = 0;
};

/**
 * @brief Graph Convolutional Network (GCN) layer.
 */
class GCNLayer : public GNNLayer {
public:
    /**
     * @brief Constructor.
     * @param in_features Number of input features.
     * @param out_features Number of output features.
     * @param use_bias Whether to use bias.
     */
    GCNLayer(int in_features, int out_features, bool use_bias = true);
    
    /**
     * @brief Forward pass through the layer.
     * @param graph Input graph.
     * @param node_features Input node features.
     * @return Output node features.
     */
    std::vector<std::vector<float>> forward(
        const Graph& graph,
        const std::vector<std::vector<float>>& node_features) const override;
    
    /**
     * @brief Get the number of parameters in the layer.
     * @return Number of parameters.
     */
    int num_parameters() const override;
    
    /**
     * @brief Get the parameters of the layer.
     * @return Vector of parameters.
     */
    std::vector<float> get_parameters() const override;
    
    /**
     * @brief Set the parameters of the layer.
     * @param parameters Vector of parameters.
     */
    void set_parameters(const std::vector<float>& parameters) override;
    
private:
    int in_features_;
    int out_features_;
    bool use_bias_;
    std::vector<std::vector<float>> weight_;
    std::vector<float> bias_;
};

/**
 * @brief Graph Attention Network (GAT) layer.
 */
class GATLayer : public GNNLayer {
public:
    /**
     * @brief Constructor.
     * @param in_features Number of input features.
     * @param out_features Number of output features.
     * @param num_heads Number of attention heads.
     * @param concat Whether to concatenate or average the outputs of the attention heads.
     * @param alpha Negative slope of the leaky ReLU activation.
     * @param dropout Dropout probability.
     */
    GATLayer(int in_features, int out_features, int num_heads = 1,
            bool concat = true, float alpha = 0.2f, float dropout = 0.0f);
    
    /**
     * @brief Forward pass through the layer.
     * @param graph Input graph.
     * @param node_features Input node features.
     * @return Output node features.
     */
    std::vector<std::vector<float>> forward(
        const Graph& graph,
        const std::vector<std::vector<float>>& node_features) const override;
    
    /**
     * @brief Get the number of parameters in the layer.
     * @return Number of parameters.
     */
    int num_parameters() const override;
    
    /**
     * @brief Get the parameters of the layer.
     * @return Vector of parameters.
     */
    std::vector<float> get_parameters() const override;
    
    /**
     * @brief Set the parameters of the layer.
     * @param parameters Vector of parameters.
     */
    void set_parameters(const std::vector<float>& parameters) override;
    
private:
    int in_features_;
    int out_features_;
    int num_heads_;
    bool concat_;
    float alpha_;
    float dropout_;
    std::vector<std::vector<std::vector<float>>> weight_;
    std::vector<std::vector<float>> att_weight_;
};

/**
 * @brief Graph SAGE (GraphSAGE) layer.
 */
class GraphSAGELayer : public GNNLayer {
public:
    /**
     * @brief Constructor.
     * @param in_features Number of input features.
     * @param out_features Number of output features.
     * @param aggregator Aggregation function.
     * @param use_bias Whether to use bias.
     */
    GraphSAGELayer(int in_features, int out_features,
                 std::string aggregator = "mean", bool use_bias = true);
    
    /**
     * @brief Forward pass through the layer.
     * @param graph Input graph.
     * @param node_features Input node features.
     * @return Output node features.
     */
    std::vector<std::vector<float>> forward(
        const Graph& graph,
        const std::vector<std::vector<float>>& node_features) const override;
    
    /**
     * @brief Get the number of parameters in the layer.
     * @return Number of parameters.
     */
    int num_parameters() const override;
    
    /**
     * @brief Get the parameters of the layer.
     * @return Vector of parameters.
     */
    std::vector<float> get_parameters() const override;
    
    /**
     * @brief Set the parameters of the layer.
     * @param parameters Vector of parameters.
     */
    void set_parameters(const std::vector<float>& parameters) override;
    
private:
    int in_features_;
    int out_features_;
    std::string aggregator_;
    bool use_bias_;
    std::vector<std::vector<float>> weight_self_;
    std::vector<std::vector<float>> weight_neigh_;
    std::vector<float> bias_;
    
    /**
     * @brief Aggregate neighbor features.
     * @param node_id ID of the node.
     * @param graph Input graph.
     * @param node_features Input node features.
     * @return Aggregated neighbor features.
     */
    std::vector<float> aggregate_neighbors(
        int node_id,
        const Graph& graph,
        const std::vector<std::vector<float>>& node_features) const;
};

/**
 * @brief Graph Pooling layer.
 */
class GraphPoolingLayer {
public:
    /**
     * @brief Constructor.
     * @param pooling_type Type of pooling operation.
     */
    GraphPoolingLayer(std::string pooling_type = "mean");
    
    /**
     * @brief Forward pass through the layer.
     * @param graph Input graph.
     * @param node_features Input node features.
     * @return Pooled graph representation.
     */
    std::vector<float> forward(
        const Graph& graph,
        const std::vector<std::vector<float>>& node_features) const;
    
private:
    std::string pooling_type_;
};

/**
 * @brief Graph Neural Network model.
 */
class GNNModel {
public:
    /**
     * @brief Constructor.
     */
    GNNModel();
    
    /**
     * @brief Add a layer to the model.
     * @param layer Layer to add.
     */
    void add_layer(std::shared_ptr<GNNLayer> layer);
    
    /**
     * @brief Add a pooling layer to the model.
     * @param pooling_layer Pooling layer to add.
     */
    void add_pooling_layer(std::shared_ptr<GraphPoolingLayer> pooling_layer);
    
    /**
     * @brief Forward pass through the model.
     * @param graph Input graph.
     * @return Output features.
     */
    std::vector<float> forward(const Graph& graph) const;
    
    /**
     * @brief Get the number of parameters in the model.
     * @return Number of parameters.
     */
    int num_parameters() const;
    
    /**
     * @brief Get the parameters of the model.
     * @return Vector of parameters.
     */
    std::vector<float> get_parameters() const;
    
    /**
     * @brief Set the parameters of the model.
     * @param parameters Vector of parameters.
     */
    void set_parameters(const std::vector<float>& parameters);
    
private:
    std::vector<std::shared_ptr<GNNLayer>> layers_;
    std::vector<std::shared_ptr<GraphPoolingLayer>> pooling_layers_;
};

} // namespace gnn
} // namespace phynexus

#endif // PHYNEXUS_GNN_H
