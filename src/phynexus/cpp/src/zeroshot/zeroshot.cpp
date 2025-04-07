/**
 * @file zeroshot.cpp
 * @brief Zero-shot learning module for Phynexus C++ backend.
 * 
 * This file provides the C++ implementation of zero-shot learning capabilities,
 * enabling models to recognize objects or classes that were not seen during training.
 */

#include "zeroshot/zeroshot.h"
#include "phynexus/error.h"
#include "phynexus/tensor.h"
#include "phynexus/nn.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <memory>

namespace phynexus {
namespace zeroshot {


ZeroShotModel::ZeroShotModel(int visual_dim, int semantic_dim, int hidden_dim, float dropout)
    : visual_dim_(visual_dim), semantic_dim_(semantic_dim), hidden_dim_(hidden_dim), dropout_(dropout) {
    
    auto visual_linear1 = std::make_unique<nn::Linear>(visual_dim, hidden_dim);
    auto visual_bn = std::make_unique<nn::BatchNorm1d>(hidden_dim);
    auto visual_relu = std::make_unique<nn::ReLU>();
    auto visual_dropout = std::make_unique<nn::Dropout>(dropout);
    auto visual_linear2 = std::make_unique<nn::Linear>(hidden_dim, hidden_dim);
    
    auto visual_seq = std::make_unique<nn::Sequential>();
    visual_seq->add(std::move(visual_linear1));
    visual_seq->add(std::move(visual_bn));
    visual_seq->add(std::move(visual_relu));
    visual_seq->add(std::move(visual_dropout));
    visual_seq->add(std::move(visual_linear2));
    
    visual_embedding_ = std::move(visual_seq);
    
    auto semantic_linear1 = std::make_unique<nn::Linear>(semantic_dim, hidden_dim);
    auto semantic_bn = std::make_unique<nn::BatchNorm1d>(hidden_dim);
    auto semantic_relu = std::make_unique<nn::ReLU>();
    auto semantic_dropout = std::make_unique<nn::Dropout>(dropout);
    auto semantic_linear2 = std::make_unique<nn::Linear>(hidden_dim, hidden_dim);
    
    auto semantic_seq = std::make_unique<nn::Sequential>();
    semantic_seq->add(std::move(semantic_linear1));
    semantic_seq->add(std::move(semantic_bn));
    semantic_seq->add(std::move(semantic_relu));
    semantic_seq->add(std::move(semantic_dropout));
    semantic_seq->add(std::move(semantic_linear2));
    
    semantic_embedding_ = std::move(semantic_seq);
}

ZeroShotModel::~ZeroShotModel() = default;

std::pair<Tensor, Tensor> ZeroShotModel::forward(const Tensor& visual_features, const Tensor& semantic_features) {
    Tensor visual_embeddings = embed_visual(visual_features);
    
    Tensor semantic_embeddings = embed_semantic(semantic_features);
    
    Tensor visual_embeddings_unsqueezed = visual_embeddings.unsqueeze(1);
    Tensor semantic_embeddings_unsqueezed = semantic_embeddings.unsqueeze(0);
    
    Tensor compatibility = nn::functional::cosine_similarity(
        visual_embeddings_unsqueezed,
        semantic_embeddings_unsqueezed,
        -1
    );
    
    return {compatibility, visual_embeddings};
}

Tensor ZeroShotModel::embed_visual(const Tensor& visual_features) {
    return visual_embedding_->forward(visual_features);
}

Tensor ZeroShotModel::embed_semantic(const Tensor& semantic_features) {
    return semantic_embedding_->forward(semantic_features);
}

std::vector<int> ZeroShotModel::predict(const Tensor& visual_features, const Tensor& semantic_features) {
    auto [compatibility, _] = forward(visual_features, semantic_features);
    
    std::vector<int> predictions;
    predictions.reserve(compatibility.size(0));
    
    for (int i = 0; i < compatibility.size(0); ++i) {
        Tensor row = compatibility[i];
        int max_idx = 0;
        float max_val = row[0].item<float>();
        
        for (int j = 1; j < row.size(0); ++j) {
            float val = row[j].item<float>();
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        
        predictions.push_back(max_idx);
    }
    
    return predictions;
}


ZeroShotTransformer::ZeroShotTransformer(int visual_dim, int semantic_dim, int hidden_dim, 
                                         int num_heads, int num_layers, float dropout)
    : ZeroShotModel(visual_dim, semantic_dim, hidden_dim, dropout),
      num_heads_(num_heads), num_layers_(num_layers) {
    
    auto encoder_layer = std::make_unique<nn::TransformerEncoderLayer>(
        hidden_dim, num_heads, hidden_dim * 4, dropout
    );
    
    visual_transformer_ = std::make_unique<nn::TransformerEncoder>(
        std::move(encoder_layer), num_layers
    );
    
    auto encoder_layer2 = std::make_unique<nn::TransformerEncoderLayer>(
        hidden_dim, num_heads, hidden_dim * 4, dropout
    );
    
    semantic_transformer_ = std::make_unique<nn::TransformerEncoder>(
        std::move(encoder_layer2), num_layers
    );
}

ZeroShotTransformer::~ZeroShotTransformer() = default;

std::pair<Tensor, Tensor> ZeroShotTransformer::forward(const Tensor& visual_features, const Tensor& semantic_features) {
    Tensor visual_embeddings = embed_visual(visual_features);
    
    Tensor semantic_embeddings = embed_semantic(semantic_features);
    
    Tensor visual_embeddings_unsqueezed = visual_embeddings.unsqueeze(1);
    Tensor semantic_embeddings_unsqueezed = semantic_embeddings.unsqueeze(0);
    
    Tensor compatibility = nn::functional::cosine_similarity(
        visual_embeddings_unsqueezed,
        semantic_embeddings_unsqueezed,
        -1
    );
    
    return {compatibility, visual_embeddings};
}

Tensor ZeroShotTransformer::embed_visual(const Tensor& visual_features) {
    Tensor embeddings = ZeroShotModel::embed_visual(visual_features);
    
    Tensor embeddings_unsqueezed = embeddings.unsqueeze(1);
    
    Tensor transformed = visual_transformer_->forward(embeddings_unsqueezed);
    
    return transformed.squeeze(1);
}

Tensor ZeroShotTransformer::embed_semantic(const Tensor& semantic_features) {
    Tensor embeddings = ZeroShotModel::embed_semantic(semantic_features);
    
    Tensor embeddings_unsqueezed = embeddings.unsqueeze(1);
    
    Tensor transformed = semantic_transformer_->forward(embeddings_unsqueezed);
    
    return transformed.squeeze(1);
}


EmbeddingModel::EmbeddingModel(int input_dim, int embedding_dim, bool normalize)
    : input_dim_(input_dim), embedding_dim_(embedding_dim), normalize_(normalize) {
    
    auto linear1 = std::make_unique<nn::Linear>(input_dim, embedding_dim * 2);
    auto bn = std::make_unique<nn::BatchNorm1d>(embedding_dim * 2);
    auto relu = std::make_unique<nn::ReLU>();
    auto linear2 = std::make_unique<nn::Linear>(embedding_dim * 2, embedding_dim);
    
    auto seq = std::make_unique<nn::Sequential>();
    seq->add(std::move(linear1));
    seq->add(std::move(bn));
    seq->add(std::move(relu));
    seq->add(std::move(linear2));
    
    network_ = std::move(seq);
}

EmbeddingModel::~EmbeddingModel() = default;

Tensor EmbeddingModel::forward(const Tensor& inputs) {
    Tensor embeddings = network_->forward(inputs);
    
    if (normalize_) {
        Tensor norm = embeddings.norm(2, 1, true);
        embeddings = embeddings.div(norm.add_scalar(1e-10));
    }
    
    return embeddings;
}

int EmbeddingModel::embedding_dim() const {
    return embedding_dim_;
}

void EmbeddingModel::set_normalize(bool normalize) {
    normalize_ = normalize;
}


TextEncoder::TextEncoder(int vocab_size, int embedding_dim, int hidden_dim, 
                         int num_layers, bool normalize)
    : EmbeddingModel(hidden_dim, embedding_dim, normalize),
      vocab_size_(vocab_size), hidden_dim_(hidden_dim), num_layers_(num_layers) {
    
    word_embedding_ = std::make_unique<nn::Embedding>(vocab_size, hidden_dim);
    
    encoder_ = std::make_unique<nn::LSTM>(hidden_dim, hidden_dim, num_layers, true, 0.1);
    
    pooling_ = std::make_unique<nn::AdaptiveAvgPool1d>(1);
}

TextEncoder::~TextEncoder() = default;

Tensor TextEncoder::forward(const Tensor& token_ids, const Tensor& attention_mask) {
    Tensor embedded = word_embedding_->forward(token_ids);
    
    embedded = embedded * attention_mask.unsqueeze(-1);
    
    auto [output, _] = encoder_->forward(embedded);
    
    output = output * attention_mask.unsqueeze(-1);
    
    Tensor pooled = output.sum(1) / attention_mask.sum(1, true);
    
    return EmbeddingModel::forward(pooled);
}


ImageEncoder::ImageEncoder(int input_channels, int embedding_dim, const std::string& backbone_type, 
                           bool normalize)
    : EmbeddingModel(2048, embedding_dim, normalize),  // Assuming 2048 features from backbone
      input_channels_(input_channels), backbone_type_(backbone_type) {
    
    if (backbone_type == "resnet") {
        backbone_ = std::make_unique<nn::ResNet>(input_channels, 50);  // ResNet-50
    } else if (backbone_type == "vit") {
        backbone_ = std::make_unique<nn::VisionTransformer>(input_channels, 16, 768, 12, 12);
    } else {
        throw std::invalid_argument("Unsupported backbone type: " + backbone_type);
    }
    
    projection_ = std::make_unique<nn::Linear>(2048, embedding_dim);
}

ImageEncoder::~ImageEncoder() = default;

Tensor ImageEncoder::forward(const Tensor& images) {
    Tensor features = backbone_->forward(images);
    
    Tensor projected = projection_->forward(features);
    
    return EmbeddingModel::forward(projected);
}


CrossModalEncoder::CrossModalEncoder(std::unique_ptr<TextEncoder> text_encoder, 
                                     std::unique_ptr<ImageEncoder> image_encoder,
                                     int projection_dim)
    : text_encoder_(std::move(text_encoder)), 
      image_encoder_(std::move(image_encoder)),
      projection_dim_(projection_dim) {
    
    text_projection_ = std::make_unique<nn::Linear>(
        text_encoder_->embedding_dim(), projection_dim
    );
    
    image_projection_ = std::make_unique<nn::Linear>(
        image_encoder_->embedding_dim(), projection_dim
    );
}

CrossModalEncoder::~CrossModalEncoder() = default;

Tensor CrossModalEncoder::encode_text(const Tensor& token_ids, const Tensor& attention_mask) {
    Tensor text_embeddings = text_encoder_->forward(token_ids, attention_mask);
    
    Tensor projected = text_projection_->forward(text_embeddings);
    
    Tensor norm = projected.norm(2, 1, true);
    return projected.div(norm.add_scalar(1e-10));
}

Tensor CrossModalEncoder::encode_image(const Tensor& images) {
    Tensor image_embeddings = image_encoder_->forward(images);
    
    Tensor projected = image_projection_->forward(image_embeddings);
    
    Tensor norm = projected.norm(2, 1, true);
    return projected.div(norm.add_scalar(1e-10));
}

Tensor CrossModalEncoder::compute_similarity(const Tensor& text_embeddings, const Tensor& image_embeddings) {
    return nn::functional::cosine_similarity(
        text_embeddings.unsqueeze(1),
        image_embeddings.unsqueeze(0),
        -1
    );
}


ZeroShotClassifier::ZeroShotClassifier(std::unique_ptr<ZeroShotModel> model,
                                       const std::vector<std::string>& class_names,
                                       const Tensor& class_embeddings)
    : model_(std::move(model)), class_names_(class_names), class_embeddings_(class_embeddings) {
}

ZeroShotClassifier::~ZeroShotClassifier() = default;

std::pair<Tensor, Tensor> ZeroShotClassifier::forward(const Tensor& visual_features) {
    if (class_embeddings_.numel() == 0) {
        throw std::runtime_error("Class embeddings must be set before inference");
    }
    
    return model_->forward(visual_features, class_embeddings_);
}

std::pair<std::vector<int>, std::vector<std::string>> ZeroShotClassifier::predict(const Tensor& visual_features) {
    auto [logits, _] = forward(visual_features);
    
    std::vector<int> predictions;
    predictions.reserve(logits.size(0));
    
    for (int i = 0; i < logits.size(0); ++i) {
        Tensor row = logits[i];
        int max_idx = 0;
        float max_val = row[0].item<float>();
        
        for (int j = 1; j < row.size(0); ++j) {
            float val = row[j].item<float>();
            if (val > max_val) {
                max_val = val;
                max_idx = j;
            }
        }
        
        predictions.push_back(max_idx);
    }
    
    std::vector<std::string> predicted_classes;
    if (!class_names_.empty()) {
        predicted_classes.reserve(predictions.size());
        for (int idx : predictions) {
            if (idx >= 0 && idx < static_cast<int>(class_names_.size())) {
                predicted_classes.push_back(class_names_[idx]);
            } else {
                predicted_classes.push_back("unknown");
            }
        }
    }
    
    return {predictions, predicted_classes};
}

void ZeroShotClassifier::set_class_embeddings(const Tensor& class_embeddings, 
                                             const std::vector<std::string>& class_names) {
    class_embeddings_ = class_embeddings;
    if (!class_names.empty()) {
        class_names_ = class_names;
    }
}


AttributeClassifier::AttributeClassifier(std::unique_ptr<ZeroShotModel> model,
                                         const std::vector<std::string>& class_names,
                                         const std::vector<std::string>& attribute_names,
                                         const Tensor& class_attribute_matrix)
    : ZeroShotClassifier(std::move(model), class_names),
      attribute_names_(attribute_names), class_attribute_matrix_(class_attribute_matrix) {
    
    if (class_attribute_matrix.numel() > 0) {
        set_class_embeddings(class_attribute_matrix);
    }
}

AttributeClassifier::~AttributeClassifier() = default;

void AttributeClassifier::set_class_attributes(const Tensor& class_attribute_matrix,
                                              const std::vector<std::string>& class_names,
                                              const std::vector<std::string>& attribute_names) {
    class_attribute_matrix_ = class_attribute_matrix;
    set_class_embeddings(class_attribute_matrix, class_names);
    
    if (!attribute_names.empty()) {
        attribute_names_ = attribute_names;
    }
}

std::pair<Tensor, std::vector<std::vector<std::string>>> AttributeClassifier::predict_attributes(const Tensor& visual_features) {
    Tensor visual_embeddings = model_->embed_visual(visual_features);
    
    int num_attributes = class_attribute_matrix_.size(1);
    Tensor attribute_embeddings = Tensor::eye(num_attributes);
    
    Tensor attribute_scores = nn::functional::cosine_similarity(
        visual_embeddings.unsqueeze(1),
        attribute_embeddings.unsqueeze(0),
        -1
    );
    
    Tensor predicted_attributes = attribute_scores > 0.5;
    
    std::vector<std::vector<std::string>> attribute_lists;
    if (!attribute_names_.empty()) {
        attribute_lists.reserve(predicted_attributes.size(0));
        
        for (int i = 0; i < predicted_attributes.size(0); ++i) {
            std::vector<std::string> attrs;
            
            for (int j = 0; j < num_attributes; ++j) {
                if (predicted_attributes[i][j].item<bool>()) {
                    attrs.push_back(attribute_names_[j]);
                }
            }
            
            attribute_lists.push_back(attrs);
        }
    }
    
    return {attribute_scores, attribute_lists};
}


SemanticClassifier::SemanticClassifier(std::unique_ptr<ZeroShotModel> model,
                                       const std::vector<std::string>& class_names,
                                       const Tensor& class_embeddings,
                                       std::unique_ptr<TextEncoder> text_encoder)
    : ZeroShotClassifier(std::move(model), class_names, class_embeddings),
      text_encoder_(std::move(text_encoder)) {
    
    if (class_names_.size() > 0 && text_encoder_ && class_embeddings_.numel() == 0) {
        compute_class_embeddings();
    }
}

SemanticClassifier::~SemanticClassifier() = default;

void SemanticClassifier::compute_class_embeddings() {
    if (class_names_.empty() || !text_encoder_) {
        throw std::runtime_error("Class names and text encoder must be set");
    }
    
    std::vector<std::vector<int>> tokens;
    for (const auto& name : class_names_) {
        std::vector<int> token_ids;
        for (char c : name) {
            token_ids.push_back(static_cast<int>(c));
        }
        tokens.push_back(token_ids);
    }
    
    int max_length = 0;
    for (const auto& t : tokens) {
        max_length = std::max(max_length, static_cast<int>(t.size()));
    }
    
    std::vector<std::vector<int>> padded_tokens;
    std::vector<std::vector<int>> masks;
    
    for (const auto& t : tokens) {
        std::vector<int> padded = t;
        padded.resize(max_length, 0);
        padded_tokens.push_back(padded);
        
        std::vector<int> mask(t.size(), 1);
        mask.resize(max_length, 0);
        masks.push_back(mask);
    }
    
    Tensor token_tensor = Tensor::from_vector(padded_tokens);
    Tensor mask_tensor = Tensor::from_vector(masks);
    
    class_embeddings_ = text_encoder_->forward(token_tensor, mask_tensor);
}

void SemanticClassifier::set_text_encoder(std::unique_ptr<TextEncoder> text_encoder) {
    text_encoder_ = std::move(text_encoder);
    
    if (class_names_.size() > 0 && class_embeddings_.numel() == 0) {
        compute_class_embeddings();
    }
}


Tensor semantic_similarity(const Tensor& embeddings1, const Tensor& embeddings2) {
    Tensor norm1 = embeddings1.norm(2, 1, true);
    Tensor norm2 = embeddings2.norm(2, 1, true);
    
    Tensor embeddings1_normalized = embeddings1.div(norm1.add_scalar(1e-10));
    Tensor embeddings2_normalized = embeddings2.div(norm2.add_scalar(1e-10));
    
    return nn::functional::cosine_similarity(
        embeddings1_normalized.unsqueeze(1),
        embeddings2_normalized.unsqueeze(0),
        -1
    );
}

Tensor attribute_mapping(const std::vector<std::string>& class_names,
                         const std::vector<std::string>& attribute_names,
                         const std::unordered_map<std::string, std::vector<std::string>>& class_attribute_map) {
    int num_classes = class_names.size();
    int num_attributes = attribute_names.size();
    
    std::unordered_map<std::string, int> attribute_indices;
    for (int i = 0; i < num_attributes; ++i) {
        attribute_indices[attribute_names[i]] = i;
    }
    
    Tensor class_attribute_matrix = Tensor::zeros({num_classes, num_attributes});
    
    for (int i = 0; i < num_classes; ++i) {
        const auto& class_name = class_names[i];
        auto it = class_attribute_map.find(class_name);
        
        if (it != class_attribute_map.end()) {
            for (const auto& attr : it->second) {
                auto attr_it = attribute_indices.find(attr);
                if (attr_it != attribute_indices.end()) {
                    class_attribute_matrix.set_item({i, attr_it->second}, 1.0f);
                }
            }
        }
    }
    
    return class_attribute_matrix;
}

Tensor class_embedding(const std::vector<std::string>& class_names,
                       TextEncoder& text_encoder,
                       std::function<std::pair<Tensor, Tensor>(const std::vector<std::string>&)> tokenizer) {
    Tensor token_tensor;
    Tensor mask_tensor;
    
    if (tokenizer) {
        auto [tokens, masks] = tokenizer(class_names);
        token_tensor = tokens;
        mask_tensor = masks;
    } else {
        std::vector<std::vector<int>> tokens;
        for (const auto& name : class_names) {
            std::vector<int> token_ids;
            for (char c : name) {
                token_ids.push_back(static_cast<int>(c));
            }
            tokens.push_back(token_ids);
        }
        
        int max_length = 0;
        for (const auto& t : tokens) {
            max_length = std::max(max_length, static_cast<int>(t.size()));
        }
        
        std::vector<std::vector<int>> padded_tokens;
        std::vector<std::vector<int>> masks;
        
        for (const auto& t : tokens) {
            std::vector<int> padded = t;
            padded.resize(max_length, 0);
            padded_tokens.push_back(padded);
            
            std::vector<int> mask(t.size(), 1);
            mask.resize(max_length, 0);
            masks.push_back(mask);
        }
        
        token_tensor = Tensor::from_vector(padded_tokens);
        mask_tensor = Tensor::from_vector(masks);
    }
    
    return text_encoder.forward(token_tensor, mask_tensor);
}

std::vector<std::string> generate_class_descriptions(const std::vector<std::string>& class_names,
                                                    const std::string& template_str) {
    std::vector<std::string> descriptions;
    descriptions.reserve(class_names.size());
    
    for (const auto& name : class_names) {
        std::string desc = template_str;
        size_t pos = desc.find("{}");
        if (pos != std::string::npos) {
            desc.replace(pos, 2, name);
        }
        descriptions.push_back(desc);
    }
    
    return descriptions;
}

Tensor compute_attribute_importance(ZeroShotModel& model,
                                   const Tensor& visual_features,
                                   const Tensor& attribute_matrix) {
    Tensor visual_embeddings = model.embed_visual(visual_features);
    
    int num_attributes = attribute_matrix.size(1);
    int batch_size = visual_features.size(0);
    
    Tensor attribute_importance = Tensor::zeros({batch_size, num_attributes});
    
    for (int i = 0; i < num_attributes; ++i) {
        Tensor attribute_mask = attribute_matrix.slice(1, i, i + 1, 1);
        
        Tensor similarity = nn::functional::cosine_similarity(
            visual_embeddings.unsqueeze(1),
            attribute_mask.unsqueeze(0),
            -1
        );
        
        Tensor mean_similarity = similarity.mean(1);
        
        for (int j = 0; j < batch_size; ++j) {
            attribute_importance.set_item({j, i}, mean_similarity[j].item<float>());
        }
    }
    
    return attribute_importance;
}

} // namespace zeroshot
} // namespace phynexus
