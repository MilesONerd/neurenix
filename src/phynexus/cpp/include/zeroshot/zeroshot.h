/**
 * @file zeroshot.h
 * @brief Zero-shot learning module for Phynexus C++ backend.
 * 
 * This file provides the C++ implementation of zero-shot learning capabilities,
 * enabling models to recognize objects or classes that were not seen during training.
 */

#ifndef PHYNEXUS_ZEROSHOT_H
#define PHYNEXUS_ZEROSHOT_H

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

#include "phynexus/tensor.h"
#include "phynexus/error.h"

namespace phynexus {
namespace zeroshot {

/**
 * @brief Base class for zero-shot learning models.
 * 
 * Zero-shot learning models can recognize objects or classes that were not seen
 * during training by leveraging semantic information about classes.
 */
class ZeroShotModel {
public:
    /**
     * @brief Construct a new Zero Shot Model object
     * 
     * @param visual_dim Dimension of visual features
     * @param semantic_dim Dimension of semantic features (class embeddings)
     * @param hidden_dim Dimension of hidden layers
     * @param dropout Dropout probability
     */
    ZeroShotModel(int visual_dim, int semantic_dim, int hidden_dim = 512, float dropout = 0.2);
    
    /**
     * @brief Destroy the Zero Shot Model object
     */
    virtual ~ZeroShotModel();
    
    /**
     * @brief Forward pass through the zero-shot model.
     * 
     * @param visual_features Visual features of shape (batch_size, visual_dim)
     * @param semantic_features Semantic features of shape (num_classes, semantic_dim)
     * @return std::pair<Tensor, Tensor> Compatibility scores and visual embeddings
     */
    virtual std::pair<Tensor, Tensor> forward(const Tensor& visual_features, const Tensor& semantic_features);
    
    /**
     * @brief Compute visual embeddings only.
     * 
     * @param visual_features Visual features of shape (batch_size, visual_dim)
     * @return Tensor Visual embeddings of shape (batch_size, hidden_dim)
     */
    virtual Tensor embed_visual(const Tensor& visual_features);
    
    /**
     * @brief Compute semantic embeddings only.
     * 
     * @param semantic_features Semantic features of shape (num_classes, semantic_dim)
     * @return Tensor Semantic embeddings of shape (num_classes, hidden_dim)
     */
    virtual Tensor embed_semantic(const Tensor& semantic_features);
    
    /**
     * @brief Predict class labels for visual features.
     * 
     * @param visual_features Visual features of shape (batch_size, visual_dim)
     * @param semantic_features Semantic features of shape (num_classes, semantic_dim)
     * @return std::vector<int> Predicted class indices
     */
    virtual std::vector<int> predict(const Tensor& visual_features, const Tensor& semantic_features);

protected:
    int visual_dim_;
    int semantic_dim_;
    int hidden_dim_;
    float dropout_;
    
    // Visual embedding network
    std::unique_ptr<nn::Module> visual_embedding_;
    
    // Semantic embedding network
    std::unique_ptr<nn::Module> semantic_embedding_;
};

/**
 * @brief Transformer-based zero-shot learning model.
 * 
 * This model uses transformer architecture for cross-modal alignment
 * between visual and semantic features.
 */
class ZeroShotTransformer : public ZeroShotModel {
public:
    /**
     * @brief Construct a new Zero Shot Transformer object
     * 
     * @param visual_dim Dimension of visual features
     * @param semantic_dim Dimension of semantic features (class embeddings)
     * @param hidden_dim Dimension of hidden layers
     * @param num_heads Number of attention heads
     * @param num_layers Number of transformer layers
     * @param dropout Dropout probability
     */
    ZeroShotTransformer(int visual_dim, int semantic_dim, int hidden_dim = 512, 
                        int num_heads = 8, int num_layers = 2, float dropout = 0.1);
    
    /**
     * @brief Destroy the Zero Shot Transformer object
     */
    virtual ~ZeroShotTransformer();
    
    /**
     * @brief Forward pass through the transformer-based zero-shot model.
     * 
     * @param visual_features Visual features of shape (batch_size, visual_dim)
     * @param semantic_features Semantic features of shape (num_classes, semantic_dim)
     * @return std::pair<Tensor, Tensor> Compatibility scores and visual embeddings
     */
    std::pair<Tensor, Tensor> forward(const Tensor& visual_features, const Tensor& semantic_features) override;
    
    /**
     * @brief Compute visual embeddings only.
     * 
     * @param visual_features Visual features of shape (batch_size, visual_dim)
     * @return Tensor Visual embeddings of shape (batch_size, hidden_dim)
     */
    Tensor embed_visual(const Tensor& visual_features) override;
    
    /**
     * @brief Compute semantic embeddings only.
     * 
     * @param semantic_features Semantic features of shape (num_classes, semantic_dim)
     * @return Tensor Semantic embeddings of shape (num_classes, hidden_dim)
     */
    Tensor embed_semantic(const Tensor& semantic_features) override;

private:
    int num_heads_;
    int num_layers_;
    
    // Visual transformer
    std::unique_ptr<nn::Module> visual_transformer_;
    
    // Semantic transformer
    std::unique_ptr<nn::Module> semantic_transformer_;
};

/**
 * @brief Base class for embedding models.
 * 
 * Embedding models encode inputs into a shared embedding space for zero-shot learning.
 */
class EmbeddingModel {
public:
    /**
     * @brief Construct a new Embedding Model object
     * 
     * @param input_dim Input dimension
     * @param embedding_dim Embedding dimension
     * @param normalize Whether to L2 normalize embeddings
     */
    EmbeddingModel(int input_dim, int embedding_dim, bool normalize = true);
    
    /**
     * @brief Destroy the Embedding Model object
     */
    virtual ~EmbeddingModel();
    
    /**
     * @brief Forward pass through the embedding model.
     * 
     * @param inputs Input tensor of shape (batch_size, input_dim)
     * @return Tensor Embeddings of shape (batch_size, embedding_dim)
     */
    virtual Tensor forward(const Tensor& inputs);
    
    /**
     * @brief Get the embedding dimension
     * 
     * @return int Embedding dimension
     */
    int embedding_dim() const;
    
    /**
     * @brief Set whether to normalize embeddings
     * 
     * @param normalize Whether to L2 normalize embeddings
     */
    void set_normalize(bool normalize);

protected:
    int input_dim_;
    int embedding_dim_;
    bool normalize_;
    
    // Embedding network
    std::unique_ptr<nn::Module> network_;
};

/**
 * @brief Text encoder for zero-shot learning.
 * 
 * This model encodes text inputs into embeddings for zero-shot learning.
 */
class TextEncoder : public EmbeddingModel {
public:
    /**
     * @brief Construct a new Text Encoder object
     * 
     * @param vocab_size Vocabulary size
     * @param embedding_dim Embedding dimension
     * @param hidden_dim Hidden dimension
     * @param num_layers Number of layers
     * @param normalize Whether to L2 normalize embeddings
     */
    TextEncoder(int vocab_size, int embedding_dim, int hidden_dim = 512, 
                int num_layers = 2, bool normalize = true);
    
    /**
     * @brief Destroy the Text Encoder object
     */
    virtual ~TextEncoder();
    
    /**
     * @brief Forward pass through the text encoder.
     * 
     * @param token_ids Token IDs of shape (batch_size, sequence_length)
     * @param attention_mask Attention mask of shape (batch_size, sequence_length)
     * @return Tensor Text embeddings of shape (batch_size, embedding_dim)
     */
    Tensor forward(const Tensor& token_ids, const Tensor& attention_mask);

private:
    int vocab_size_;
    int hidden_dim_;
    int num_layers_;
    
    // Word embedding layer
    std::unique_ptr<nn::Module> word_embedding_;
    
    // Encoder network
    std::unique_ptr<nn::Module> encoder_;
    
    // Pooling layer
    std::unique_ptr<nn::Module> pooling_;
};

/**
 * @brief Image encoder for zero-shot learning.
 * 
 * This model encodes image inputs into embeddings for zero-shot learning.
 */
class ImageEncoder : public EmbeddingModel {
public:
    /**
     * @brief Construct a new Image Encoder object
     * 
     * @param input_channels Input channels
     * @param embedding_dim Embedding dimension
     * @param backbone_type Backbone type (e.g., "resnet", "vit")
     * @param normalize Whether to L2 normalize embeddings
     */
    ImageEncoder(int input_channels, int embedding_dim, const std::string& backbone_type = "resnet", 
                 bool normalize = true);
    
    /**
     * @brief Destroy the Image Encoder object
     */
    virtual ~ImageEncoder();
    
    /**
     * @brief Forward pass through the image encoder.
     * 
     * @param images Image tensor of shape (batch_size, channels, height, width)
     * @return Tensor Image embeddings of shape (batch_size, embedding_dim)
     */
    Tensor forward(const Tensor& images);

private:
    int input_channels_;
    std::string backbone_type_;
    
    // Backbone network
    std::unique_ptr<nn::Module> backbone_;
    
    // Projection layer
    std::unique_ptr<nn::Module> projection_;
};

/**
 * @brief Cross-modal encoder for zero-shot learning.
 * 
 * This model encodes both text and image inputs into a shared embedding space.
 */
class CrossModalEncoder {
public:
    /**
     * @brief Construct a new Cross Modal Encoder object
     * 
     * @param text_encoder Text encoder
     * @param image_encoder Image encoder
     * @param projection_dim Projection dimension
     */
    CrossModalEncoder(std::unique_ptr<TextEncoder> text_encoder, 
                      std::unique_ptr<ImageEncoder> image_encoder,
                      int projection_dim = 512);
    
    /**
     * @brief Destroy the Cross Modal Encoder object
     */
    virtual ~CrossModalEncoder();
    
    /**
     * @brief Encode text inputs.
     * 
     * @param token_ids Token IDs of shape (batch_size, sequence_length)
     * @param attention_mask Attention mask of shape (batch_size, sequence_length)
     * @return Tensor Text embeddings of shape (batch_size, projection_dim)
     */
    Tensor encode_text(const Tensor& token_ids, const Tensor& attention_mask);
    
    /**
     * @brief Encode image inputs.
     * 
     * @param images Image tensor of shape (batch_size, channels, height, width)
     * @return Tensor Image embeddings of shape (batch_size, projection_dim)
     */
    Tensor encode_image(const Tensor& images);
    
    /**
     * @brief Compute similarity between text and image embeddings.
     * 
     * @param text_embeddings Text embeddings of shape (batch_size_text, projection_dim)
     * @param image_embeddings Image embeddings of shape (batch_size_image, projection_dim)
     * @return Tensor Similarity matrix of shape (batch_size_text, batch_size_image)
     */
    Tensor compute_similarity(const Tensor& text_embeddings, const Tensor& image_embeddings);

private:
    std::unique_ptr<TextEncoder> text_encoder_;
    std::unique_ptr<ImageEncoder> image_encoder_;
    int projection_dim_;
    
    // Text projection layer
    std::unique_ptr<nn::Module> text_projection_;
    
    // Image projection layer
    std::unique_ptr<nn::Module> image_projection_;
};

/**
 * @brief Base class for zero-shot classifiers.
 * 
 * Zero-shot classifiers can classify inputs into classes that were not seen
 * during training by leveraging semantic information about classes.
 */
class ZeroShotClassifier {
public:
    /**
     * @brief Construct a new Zero Shot Classifier object
     * 
     * @param model Zero-shot model
     * @param class_names Class names
     * @param class_embeddings Class embeddings
     */
    ZeroShotClassifier(std::unique_ptr<ZeroShotModel> model,
                       const std::vector<std::string>& class_names = {},
                       const Tensor& class_embeddings = Tensor());
    
    /**
     * @brief Destroy the Zero Shot Classifier object
     */
    virtual ~ZeroShotClassifier();
    
    /**
     * @brief Forward pass through the zero-shot classifier.
     * 
     * @param visual_features Visual features of shape (batch_size, visual_dim)
     * @return std::pair<Tensor, Tensor> Classification logits and visual embeddings
     */
    virtual std::pair<Tensor, Tensor> forward(const Tensor& visual_features);
    
    /**
     * @brief Predict class labels for visual features.
     * 
     * @param visual_features Visual features of shape (batch_size, visual_dim)
     * @return std::pair<std::vector<int>, std::vector<std::string>> Predicted class indices and names
     */
    virtual std::pair<std::vector<int>, std::vector<std::string>> predict(const Tensor& visual_features);
    
    /**
     * @brief Set class embeddings for the classifier.
     * 
     * @param class_embeddings Class embeddings of shape (num_classes, semantic_dim)
     * @param class_names Class names
     */
    void set_class_embeddings(const Tensor& class_embeddings, 
                              const std::vector<std::string>& class_names = {});

protected:
    std::unique_ptr<ZeroShotModel> model_;
    std::vector<std::string> class_names_;
    Tensor class_embeddings_;
};

/**
 * @brief Attribute-based zero-shot classifier.
 * 
 * This classifier uses class-attribute mappings for zero-shot classification.
 */
class AttributeClassifier : public ZeroShotClassifier {
public:
    /**
     * @brief Construct a new Attribute Classifier object
     * 
     * @param model Zero-shot model
     * @param class_names Class names
     * @param attribute_names Attribute names
     * @param class_attribute_matrix Binary matrix indicating which attributes are present in each class
     */
    AttributeClassifier(std::unique_ptr<ZeroShotModel> model,
                        const std::vector<std::string>& class_names = {},
                        const std::vector<std::string>& attribute_names = {},
                        const Tensor& class_attribute_matrix = Tensor());
    
    /**
     * @brief Destroy the Attribute Classifier object
     */
    virtual ~AttributeClassifier();
    
    /**
     * @brief Set class-attribute mappings for the classifier.
     * 
     * @param class_attribute_matrix Binary matrix indicating which attributes are present in each class
     * @param class_names Class names
     * @param attribute_names Attribute names
     */
    void set_class_attributes(const Tensor& class_attribute_matrix,
                              const std::vector<std::string>& class_names = {},
                              const std::vector<std::string>& attribute_names = {});
    
    /**
     * @brief Predict attributes for visual features.
     * 
     * @param visual_features Visual features of shape (batch_size, visual_dim)
     * @return std::pair<Tensor, std::vector<std::vector<std::string>>> Attribute scores and predicted attributes
     */
    std::pair<Tensor, std::vector<std::vector<std::string>>> predict_attributes(const Tensor& visual_features);

private:
    std::vector<std::string> attribute_names_;
    Tensor class_attribute_matrix_;
};

/**
 * @brief Semantic-based zero-shot classifier.
 * 
 * This classifier uses semantic embeddings (e.g., word embeddings) for zero-shot classification.
 */
class SemanticClassifier : public ZeroShotClassifier {
public:
    /**
     * @brief Construct a new Semantic Classifier object
     * 
     * @param model Zero-shot model
     * @param class_names Class names
     * @param class_embeddings Class embeddings
     * @param text_encoder Text encoder for computing class embeddings
     */
    SemanticClassifier(std::unique_ptr<ZeroShotModel> model,
                       const std::vector<std::string>& class_names = {},
                       const Tensor& class_embeddings = Tensor(),
                       std::unique_ptr<TextEncoder> text_encoder = nullptr);
    
    /**
     * @brief Destroy the Semantic Classifier object
     */
    virtual ~SemanticClassifier();
    
    /**
     * @brief Compute class embeddings from class names using the text encoder.
     */
    void compute_class_embeddings();
    
    /**
     * @brief Set the text encoder for computing class embeddings.
     * 
     * @param text_encoder Text encoder
     */
    void set_text_encoder(std::unique_ptr<TextEncoder> text_encoder);

private:
    std::unique_ptr<TextEncoder> text_encoder_;
};

/**
 * @brief Compute semantic similarity between two sets of embeddings.
 * 
 * @param embeddings1 First set of embeddings of shape (batch_size1, embedding_dim)
 * @param embeddings2 Second set of embeddings of shape (batch_size2, embedding_dim)
 * @return Tensor Similarity matrix of shape (batch_size1, batch_size2)
 */
Tensor semantic_similarity(const Tensor& embeddings1, const Tensor& embeddings2);

/**
 * @brief Create a class-attribute binary matrix from a mapping dictionary.
 * 
 * @param class_names Class names
 * @param attribute_names Attribute names
 * @param class_attribute_map Dictionary mapping class names to lists of attribute names
 * @return Tensor Binary matrix of shape (num_classes, num_attributes)
 */
Tensor attribute_mapping(const std::vector<std::string>& class_names,
                         const std::vector<std::string>& attribute_names,
                         const std::unordered_map<std::string, std::vector<std::string>>& class_attribute_map);

/**
 * @brief Compute class embeddings from class names using a text encoder.
 * 
 * @param class_names Class names
 * @param text_encoder Text encoder
 * @param tokenizer Tokenizer for preprocessing class names
 * @return Tensor Class embeddings of shape (num_classes, embedding_dim)
 */
Tensor class_embedding(const std::vector<std::string>& class_names,
                       TextEncoder& text_encoder,
                       std::function<std::pair<Tensor, Tensor>(const std::vector<std::string>&)> tokenizer = nullptr);

/**
 * @brief Generate class descriptions from class names using a template.
 * 
 * @param class_names Class names
 * @param template_str Template string with {} placeholder for class name
 * @return std::vector<std::string> Class descriptions
 */
std::vector<std::string> generate_class_descriptions(const std::vector<std::string>& class_names,
                                                    const std::string& template_str = "A photo of a {}.");

/**
 * @brief Compute the importance of each attribute for classification.
 * 
 * @param model Zero-shot model
 * @param visual_features Visual features of shape (batch_size, visual_dim)
 * @param attribute_matrix Binary matrix of shape (num_classes, num_attributes)
 * @return Tensor Importance scores of shape (batch_size, num_attributes)
 */
Tensor compute_attribute_importance(ZeroShotModel& model,
                                   const Tensor& visual_features,
                                   const Tensor& attribute_matrix);

} // namespace zeroshot
} // namespace phynexus

#endif // PHYNEXUS_ZEROSHOT_H
