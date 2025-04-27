# Zero-Shot Learning API Documentation

## Overview

The Zero-Shot Learning module provides tools and techniques for enabling models to recognize objects or classes that were not seen during training. This capability allows AI systems to generalize beyond their training data, making predictions about novel categories by leveraging semantic relationships and transferring knowledge from seen to unseen classes.

Zero-shot learning is particularly valuable in scenarios where collecting labeled examples for all possible classes is impractical or impossible, such as rare object recognition, emerging categories in e-commerce, or rapidly evolving domains. By learning to map between visual features and semantic descriptions, zero-shot models can make reasonable predictions about new classes based on their semantic attributes or descriptions.

## Key Concepts

### Semantic Embeddings

Semantic embeddings represent classes in a semantic space:

- **Class Embeddings**: Vector representations of class names or descriptions
- **Attribute Embeddings**: Vector representations of class attributes
- **Word Embeddings**: Distributed representations of words that capture semantic relationships
- **Sentence Embeddings**: Vector representations of sentences or descriptions

### Cross-Modal Mapping

Cross-modal mapping connects visual and semantic spaces:

- **Visual-Semantic Embedding**: Maps visual features to semantic space
- **Semantic-Visual Embedding**: Maps semantic features to visual space
- **Joint Embedding Space**: Common space for both visual and semantic features
- **Compatibility Functions**: Measure similarity between visual and semantic embeddings

### Zero-Shot Classification

Zero-shot classification recognizes unseen classes:

- **Seen Classes**: Classes with examples in the training data
- **Unseen Classes**: Novel classes not present in the training data
- **Semantic Similarity**: Measuring similarity between class embeddings
- **Attribute-Based Classification**: Classification based on class attributes

### Zero-Shot Models

Different model architectures for zero-shot learning:

- **Embedding Models**: Learn mappings between visual and semantic spaces
- **Generative Models**: Generate synthetic examples for unseen classes
- **Attribute Classifiers**: Predict attributes rather than classes
- **Transformer-Based Models**: Leverage pre-trained transformers for zero-shot tasks

## API Reference

### Zero-Shot Models

```python
neurenix.zeroshot.ZeroShotModel(
    visual_encoder: neurenix.nn.Module,
    semantic_encoder: neurenix.nn.Module,
    embedding_dim: int = 512,
    compatibility_function: str = "cosine"
)
```

Creates a zero-shot learning model that maps between visual and semantic spaces.

**Parameters:**
- `visual_encoder`: Neural network for encoding visual inputs
- `semantic_encoder`: Neural network for encoding semantic inputs
- `embedding_dim`: Dimension of the joint embedding space
- `compatibility_function`: Function to measure compatibility between embeddings ("cosine", "dot", "euclidean")

**Methods:**
- `forward(visual_input, semantic_input=None)`: Process inputs through the model
- `predict(visual_input, class_embeddings)`: Predict class for visual input
- `compute_compatibility(visual_embedding, semantic_embedding)`: Compute compatibility between embeddings
- `train(visual_inputs, semantic_inputs, optimizer, epochs)`: Train the model

**Example:**
```python
import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU
from neurenix.zeroshot import ZeroShotModel

# Create visual encoder
visual_encoder = Sequential(
    Linear(2048, 1024),
    ReLU(),
    Linear(1024, 512)
)

# Create semantic encoder
semantic_encoder = Sequential(
    Linear(300, 512),
    ReLU(),
    Linear(512, 512)
)

# Create zero-shot model
model = ZeroShotModel(
    visual_encoder=visual_encoder,
    semantic_encoder=semantic_encoder,
    embedding_dim=512,
    compatibility_function="cosine"
)

# Process inputs
visual_input = nx.Tensor(shape=(1, 2048))  # Visual features
semantic_input = nx.Tensor(shape=(1, 300))  # Semantic features
embedding = model(visual_input, semantic_input)

# Predict class
class_embeddings = nx.Tensor(shape=(10, 512))  # Embeddings for 10 classes
prediction = model.predict(visual_input, class_embeddings)
```

```python
neurenix.zeroshot.ZeroShotTransformer(
    vision_model: str = "resnet50",
    text_model: str = "bert-base-uncased",
    embedding_dim: int = 512,
    pretrained: bool = True
)
```

Creates a transformer-based zero-shot learning model.

**Parameters:**
- `vision_model`: Pre-trained vision model architecture
- `text_model`: Pre-trained text model architecture
- `embedding_dim`: Dimension of the joint embedding space
- `pretrained`: Whether to use pre-trained weights

**Methods:**
- `forward(images, texts=None)`: Process inputs through the model
- `predict(images, class_names)`: Predict classes for images
- `compute_similarity(image_features, text_features)`: Compute similarity between features
- `encode_images(images)`: Encode images into the joint space
- `encode_text(texts)`: Encode text into the joint space

**Example:**
```python
from neurenix.zeroshot import ZeroShotTransformer

# Create a transformer-based zero-shot model
model = ZeroShotTransformer(
    vision_model="resnet50",
    text_model="bert-base-uncased",
    embedding_dim=512,
    pretrained=True
)

# Predict classes for an image
image = load_image("cat.jpg")
class_names = ["cat", "dog", "bird", "fish", "horse"]
prediction = model.predict(image, class_names)
print(f"Prediction: {prediction}")
```

### Embedding Models

```python
neurenix.zeroshot.EmbeddingModel(
    input_dim: int,
    embedding_dim: int,
    hidden_dims: List[int] = [1024, 512],
    dropout: float = 0.2
)
```

Creates a model for embedding inputs into a joint space.

**Parameters:**
- `input_dim`: Dimension of the input features
- `embedding_dim`: Dimension of the output embedding
- `hidden_dims`: Dimensions of hidden layers
- `dropout`: Dropout rate for regularization

**Methods:**
- `forward(x)`: Embed input into the joint space
- `normalize(x)`: Normalize embeddings
- `compute_similarity(x1, x2)`: Compute similarity between embeddings

**Example:**
```python
from neurenix.zeroshot import EmbeddingModel

# Create an embedding model for visual features
visual_embedding_model = EmbeddingModel(
    input_dim=2048,  # ResNet features
    embedding_dim=512,
    hidden_dims=[1024, 768],
    dropout=0.2
)

# Create an embedding model for semantic features
semantic_embedding_model = EmbeddingModel(
    input_dim=300,  # Word2Vec features
    embedding_dim=512,
    hidden_dims=[512, 512],
    dropout=0.2
)

# Embed inputs
visual_features = load_visual_features()
semantic_features = load_semantic_features()

visual_embedding = visual_embedding_model(visual_features)
semantic_embedding = semantic_embedding_model(semantic_features)

# Compute similarity
similarity = visual_embedding_model.compute_similarity(visual_embedding, semantic_embedding)
```

```python
neurenix.zeroshot.TextEncoder(
    model_name: str = "bert-base-uncased",
    embedding_dim: int = 512,
    pooling: str = "mean",
    pretrained: bool = True
)
```

Creates a model for encoding text into embeddings.

**Parameters:**
- `model_name`: Name of the pre-trained text model
- `embedding_dim`: Dimension of the output embedding
- `pooling`: Pooling method for sentence embeddings ("mean", "max", "cls")
- `pretrained`: Whether to use pre-trained weights

**Methods:**
- `forward(texts)`: Encode texts into embeddings
- `encode_classes(class_names)`: Encode class names into embeddings
- `encode_attributes(attribute_descriptions)`: Encode attribute descriptions into embeddings

```python
neurenix.zeroshot.ImageEncoder(
    model_name: str = "resnet50",
    embedding_dim: int = 512,
    pretrained: bool = True
)
```

Creates a model for encoding images into embeddings.

**Parameters:**
- `model_name`: Name of the pre-trained image model
- `embedding_dim`: Dimension of the output embedding
- `pretrained`: Whether to use pre-trained weights

**Methods:**
- `forward(images)`: Encode images into embeddings
- `extract_features(images)`: Extract features from images
- `fine_tune(images, labels, optimizer, epochs)`: Fine-tune the encoder

```python
neurenix.zeroshot.CrossModalEncoder(
    visual_encoder: neurenix.nn.Module,
    text_encoder: neurenix.nn.Module,
    embedding_dim: int = 512,
    fusion_method: str = "concat"
)
```

Creates a model for cross-modal encoding.

**Parameters:**
- `visual_encoder`: Encoder for visual inputs
- `text_encoder`: Encoder for text inputs
- `embedding_dim`: Dimension of the output embedding
- `fusion_method`: Method for fusing visual and text features ("concat", "sum", "attention")

**Methods:**
- `forward(images, texts)`: Encode image-text pairs into embeddings
- `encode_image(image)`: Encode an image into the joint space
- `encode_text(text)`: Encode text into the joint space
- `compute_similarity(image, text)`: Compute similarity between image and text

### Zero-Shot Classifiers

```python
neurenix.zeroshot.ZeroShotClassifier(
    embedding_model: neurenix.zeroshot.EmbeddingModel,
    class_embeddings: neurenix.Tensor,
    class_names: List[str],
    similarity_metric: str = "cosine"
)
```

Creates a zero-shot classifier using pre-computed class embeddings.

**Parameters:**
- `embedding_model`: Model for embedding inputs
- `class_embeddings`: Pre-computed embeddings for classes
- `class_names`: Names of the classes
- `similarity_metric`: Metric for computing similarity ("cosine", "dot", "euclidean")

**Methods:**
- `predict(inputs)`: Predict classes for inputs
- `predict_proba(inputs)`: Predict class probabilities for inputs
- `add_class(class_name, class_embedding)`: Add a new class to the classifier
- `remove_class(class_name)`: Remove a class from the classifier

**Example:**
```python
from neurenix.zeroshot import ZeroShotClassifier, TextEncoder

# Create a text encoder
text_encoder = TextEncoder(
    model_name="bert-base-uncased",
    embedding_dim=512,
    pooling="mean",
    pretrained=True
)

# Encode class names
class_names = ["cat", "dog", "bird", "fish", "horse"]
class_embeddings = text_encoder.encode_classes(class_names)

# Create a zero-shot classifier
classifier = ZeroShotClassifier(
    embedding_model=visual_embedding_model,
    class_embeddings=class_embeddings,
    class_names=class_names,
    similarity_metric="cosine"
)

# Predict class for an image
image_features = extract_features(image)
prediction = classifier.predict(image_features)
print(f"Predicted class: {prediction}")

# Add a new class
new_class_name = "elephant"
new_class_embedding = text_encoder.encode_classes([new_class_name])[0]
classifier.add_class(new_class_name, new_class_embedding)

# Predict with the updated classifier
prediction = classifier.predict(image_features)
```

```python
neurenix.zeroshot.AttributeClassifier(
    attribute_model: neurenix.nn.Module,
    class_attributes: Dict[str, List[float]],
    class_names: List[str]
)
```

Creates a zero-shot classifier based on attributes.

**Parameters:**
- `attribute_model`: Model for predicting attributes
- `class_attributes`: Mapping from class names to attribute values
- `class_names`: Names of the classes

**Methods:**
- `predict(inputs)`: Predict classes for inputs
- `predict_attributes(inputs)`: Predict attribute values for inputs
- `add_class(class_name, attributes)`: Add a new class to the classifier
- `remove_class(class_name)`: Remove a class from the classifier

```python
neurenix.zeroshot.SemanticClassifier(
    visual_encoder: neurenix.nn.Module,
    semantic_encoder: neurenix.nn.Module,
    class_descriptions: List[str],
    class_names: List[str]
)
```

Creates a zero-shot classifier based on semantic descriptions.

**Parameters:**
- `visual_encoder`: Encoder for visual inputs
- `semantic_encoder`: Encoder for semantic inputs
- `class_descriptions`: Textual descriptions of classes
- `class_names`: Names of the classes

**Methods:**
- `predict(inputs)`: Predict classes for inputs
- `predict_proba(inputs)`: Predict class probabilities for inputs
- `add_class(class_name, description)`: Add a new class to the classifier
- `remove_class(class_name)`: Remove a class from the classifier

### Utility Functions

```python
neurenix.zeroshot.semantic_similarity(
    embedding1: neurenix.Tensor,
    embedding2: neurenix.Tensor,
    metric: str = "cosine"
) -> neurenix.Tensor
```

Computes semantic similarity between embeddings.

**Parameters:**
- `embedding1`: First embedding
- `embedding2`: Second embedding
- `metric`: Similarity metric ("cosine", "dot", "euclidean")

**Returns:**
- Similarity score

**Example:**
```python
from neurenix.zeroshot import semantic_similarity, TextEncoder

# Create a text encoder
text_encoder = TextEncoder(
    model_name="bert-base-uncased",
    embedding_dim=512,
    pooling="mean",
    pretrained=True
)

# Encode texts
text1 = "A cat sitting on a mat"
text2 = "A kitten on a rug"
embedding1 = text_encoder.forward([text1])[0]
embedding2 = text_encoder.forward([text2])[0]

# Compute similarity
similarity = semantic_similarity(embedding1, embedding2, metric="cosine")
print(f"Similarity: {similarity.item()}")
```

```python
neurenix.zeroshot.attribute_mapping(
    class_names: List[str],
    attributes: List[str],
    values: Optional[np.ndarray] = None
) -> Dict[str, List[float]]
```

Creates a mapping from class names to attribute values.

**Parameters:**
- `class_names`: Names of the classes
- `attributes`: Names of the attributes
- `values`: Matrix of attribute values (if None, will be filled with zeros)

**Returns:**
- Dictionary mapping class names to attribute values

```python
neurenix.zeroshot.class_embedding(
    class_name: str,
    embedding_model: Callable,
    use_attributes: bool = False,
    attributes: Optional[List[str]] = None
) -> neurenix.Tensor
```

Computes an embedding for a class name.

**Parameters:**
- `class_name`: Name of the class
- `embedding_model`: Model for computing embeddings
- `use_attributes`: Whether to use attributes for embedding
- `attributes`: List of attributes (if use_attributes is True)

**Returns:**
- Embedding for the class

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Zero-Shot Learning** | Comprehensive API with multiple models | Limited built-in support |
| **Cross-Modal Embedding** | Native support | Requires custom implementation |
| **Transformer Integration** | Seamless integration with pre-trained models | Available through TF-Hub |
| **Attribute-Based Classification** | Built-in support | Requires custom implementation |
| **Semantic Similarity** | Multiple similarity metrics | Basic similarity functions |
| **Class Addition** | Dynamic class addition at inference time | Requires model retraining |

Neurenix provides more comprehensive zero-shot learning capabilities compared to TensorFlow, with built-in support for various zero-shot models, cross-modal embeddings, and attribute-based classification. TensorFlow requires more custom implementation for most zero-shot features, making it less accessible for users interested in zero-shot learning.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Zero-Shot Learning** | Comprehensive API with multiple models | Limited built-in support |
| **Cross-Modal Embedding** | Native support | Available through third-party libraries |
| **Transformer Integration** | Seamless integration with pre-trained models | Available through HuggingFace |
| **Attribute-Based Classification** | Built-in support | Requires custom implementation |
| **Semantic Similarity** | Multiple similarity metrics | Basic similarity functions |
| **Class Addition** | Dynamic class addition at inference time | Requires model retraining |

While PyTorch provides flexibility for implementing zero-shot models, it lacks native support for most zero-shot features. Neurenix's integrated zero-shot module offers a more cohesive experience, with seamless integration with the rest of the framework and built-in support for various zero-shot learning approaches.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Zero-Shot Learning** | Comprehensive support | No support |
| **Cross-Modal Embedding** | Native support | No support |
| **Transformer Integration** | Seamless integration | No support |
| **Attribute-Based Classification** | Built-in support | No direct support |
| **Semantic Similarity** | Multiple similarity metrics | Basic similarity functions |
| **Class Addition** | Dynamic class addition at inference time | Requires model retraining |

Scikit-Learn does not provide support for zero-shot learning, focusing instead on traditional supervised learning approaches. Neurenix fills this gap with its comprehensive zero-shot module, enabling the recognition of unseen classes without requiring examples for all possible classes.

## Best Practices

### Embedding Space Design

Design an effective joint embedding space:

```python
from neurenix.zeroshot import ZeroShotModel
from neurenix.nn import Sequential, Linear, ReLU, BatchNorm1d

# Create a visual encoder with normalization
visual_encoder = Sequential(
    Linear(2048, 1024),
    BatchNorm1d(1024),  # Normalize activations
    ReLU(),
    Linear(1024, 512),
    BatchNorm1d(512),
    ReLU(),
    Linear(512, 512)
)

# Create a semantic encoder with normalization
semantic_encoder = Sequential(
    Linear(300, 512),
    BatchNorm1d(512),  # Normalize activations
    ReLU(),
    Linear(512, 512),
    BatchNorm1d(512),
    ReLU(),
    Linear(512, 512)
)

# Create a zero-shot model with normalized embeddings
model = ZeroShotModel(
    visual_encoder=visual_encoder,
    semantic_encoder=semantic_encoder,
    embedding_dim=512,
    compatibility_function="cosine"  # Use cosine similarity for normalized embeddings
)
```

### Class Description Design

Create effective class descriptions:

```python
# Simple class names
simple_class_names = ["cat", "dog", "bird"]

# More descriptive class names
descriptive_class_names = ["domestic cat", "domestic dog", "wild bird"]

# Full class descriptions
class_descriptions = [
    "A small domesticated carnivorous mammal with soft fur, a short snout, and retractable claws.",
    "A domesticated carnivorous mammal that typically has a long snout, an acute sense of smell, and a barking, howling, or whining voice.",
    "A warm-blooded egg-laying vertebrate animal distinguished by the possession of feathers, wings, a beak, and typically by being able to fly."
]

# Template-based descriptions
template = "A photo of a {}"
template_descriptions = [template.format(name) for name in simple_class_names]
```

### Handling Unseen Classes

Effectively handle unseen classes:

```python
from neurenix.zeroshot import ZeroShotClassifier, TextEncoder

# Create a text encoder
text_encoder = TextEncoder(
    model_name="bert-base-uncased",
    embedding_dim=512,
    pooling="mean",
    pretrained=True
)

# Encode seen class names
seen_class_names = ["cat", "dog", "bird"]
seen_class_embeddings = text_encoder.encode_classes(seen_class_names)

# Create a zero-shot classifier
classifier = ZeroShotClassifier(
    embedding_model=visual_embedding_model,
    class_embeddings=seen_class_embeddings,
    class_names=seen_class_names,
    similarity_metric="cosine"
)

# Add unseen classes dynamically
unseen_class_names = ["elephant", "giraffe", "zebra"]
unseen_class_embeddings = text_encoder.encode_classes(unseen_class_names)

for name, embedding in zip(unseen_class_names, unseen_class_embeddings):
    classifier.add_class(name, embedding)

# Now the classifier can recognize both seen and unseen classes
```

### Calibration

Calibrate confidence scores for better reliability:

```python
from neurenix.zeroshot import ZeroShotClassifier
from neurenix.calibration import TemperatureScaling

# Create a zero-shot classifier
classifier = ZeroShotClassifier(
    embedding_model=visual_embedding_model,
    class_embeddings=class_embeddings,
    class_names=class_names,
    similarity_metric="cosine"
)

# Create a calibrator
calibrator = TemperatureScaling()

# Calibrate the classifier using validation data
validation_features = extract_features(validation_images)
validation_labels = get_labels(validation_images)
calibrator.fit(classifier, validation_features, validation_labels)

# Get calibrated predictions
image_features = extract_features(test_image)
calibrated_probs = calibrator.calibrate(classifier.predict_proba(image_features))
```

## Tutorials

### Zero-Shot Image Classification

```python
import neurenix as nx
from neurenix.zeroshot import ZeroShotTransformer
from neurenix.data import ImageDataset, DataLoader
from neurenix.vision import load_image, transform_image

# Create a zero-shot transformer model
model = ZeroShotTransformer(
    vision_model="resnet50",
    text_model="bert-base-uncased",
    embedding_dim=512,
    pretrained=True
)

# Define seen and unseen classes
seen_classes = ["cat", "dog", "horse", "elephant", "bear"]
unseen_classes = ["tiger", "lion", "zebra", "giraffe", "panda"]
all_classes = seen_classes + unseen_classes

# Load test images
test_dataset = ImageDataset("path/to/test/images", transform=transform_image)
test_loader = DataLoader(test_dataset, batch_size=16)

# Evaluate on test images
correct_seen = 0
correct_unseen = 0
total_seen = 0
total_unseen = 0

for images, labels in test_loader:
    # Get class names for the labels
    true_classes = [all_classes[label.item()] for label in labels]
    
    # Predict classes
    predicted_classes = model.predict(images, all_classes)
    
    # Count correct predictions
    for true_class, pred_class in zip(true_classes, predicted_classes):
        if true_class in seen_classes:
            total_seen += 1
            if true_class == pred_class:
                correct_seen += 1
        else:
            total_unseen += 1
            if true_class == pred_class:
                correct_unseen += 1

# Calculate accuracy
seen_accuracy = correct_seen / total_seen if total_seen > 0 else 0
unseen_accuracy = correct_unseen / total_unseen if total_unseen > 0 else 0
overall_accuracy = (correct_seen + correct_unseen) / (total_seen + total_unseen)

print(f"Seen classes accuracy: {seen_accuracy:.4f}")
print(f"Unseen classes accuracy: {unseen_accuracy:.4f}")
print(f"Overall accuracy: {overall_accuracy:.4f}")

# Test on a single image
test_image = load_image("path/to/test/image.jpg")
transformed_image = transform_image(test_image)
prediction = model.predict(transformed_image, all_classes)
print(f"Predicted class: {prediction}")
```

### Attribute-Based Zero-Shot Learning

```python
import neurenix as nx
import numpy as np
from neurenix.nn import Sequential, Linear, ReLU, Sigmoid
from neurenix.optim import Adam
from neurenix.zeroshot import AttributeClassifier, attribute_mapping

# Define classes and attributes
animal_classes = ["cat", "dog", "horse", "cow", "sheep", "pig"]
attributes = ["has_fur", "has_tail", "is_carnivore", "is_domestic", "can_fly"]

# Create attribute values for each class (1 = yes, 0 = no)
attribute_values = np.array([
    [1, 1, 1, 1, 0],  # cat
    [1, 1, 1, 1, 0],  # dog
    [1, 1, 0, 1, 0],  # horse
    [1, 1, 0, 1, 0],  # cow
    [1, 1, 0, 1, 0],  # sheep
    [1, 1, 0, 1, 0],  # pig
])

# Create attribute mapping
class_attributes = attribute_mapping(animal_classes, attributes, attribute_values)

# Create an attribute prediction model
attribute_model = Sequential(
    Linear(2048, 1024),
    ReLU(),
    Linear(1024, 512),
    ReLU(),
    Linear(512, len(attributes)),
    Sigmoid()
)

# Create a zero-shot attribute classifier
classifier = AttributeClassifier(
    attribute_model=attribute_model,
    class_attributes=class_attributes,
    class_names=animal_classes
)

# Train the attribute model
def train_attribute_model(model, train_loader, optimizer, epochs):
    for epoch in range(epochs):
        for images, attribute_labels in train_loader:
            # Forward pass
            attribute_preds = model(images)
            loss = nx.nn.BCELoss()(attribute_preds, attribute_labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Assume we have a DataLoader for training data
optimizer = Adam(attribute_model.parameters(), lr=0.001)
train_attribute_model(attribute_model, train_loader, optimizer, epochs=10)

# Add a new class with attributes
new_class = "tiger"
new_attributes = [1, 1, 1, 0, 0]  # has_fur, has_tail, is_carnivore, is_domestic, can_fly
classifier.add_class(new_class, new_attributes)

# Test on an image
image_features = extract_features(test_image)
predicted_attributes = classifier.predict_attributes(image_features)
predicted_class = classifier.predict(image_features)

print(f"Predicted attributes: {predicted_attributes}")
print(f"Predicted class: {predicted_class}")
```

### Cross-Modal Retrieval

```python
import neurenix as nx
from neurenix.zeroshot import CrossModalEncoder, TextEncoder, ImageEncoder
from neurenix.data import ImageTextDataset, DataLoader
from neurenix.vision import load_image, transform_image
from neurenix.text import tokenize_text

# Create image and text encoders
image_encoder = ImageEncoder(
    model_name="resnet50",
    embedding_dim=512,
    pretrained=True
)

text_encoder = TextEncoder(
    model_name="bert-base-uncased",
    embedding_dim=512,
    pooling="mean",
    pretrained=True
)

# Create a cross-modal encoder
cross_modal_encoder = CrossModalEncoder(
    visual_encoder=image_encoder,
    text_encoder=text_encoder,
    embedding_dim=512,
    fusion_method="concat"
)

# Train the cross-modal encoder
def train_cross_modal_encoder(model, train_loader, optimizer, epochs):
    for epoch in range(epochs):
        for images, texts in train_loader:
            # Forward pass
            image_embeddings = model.encode_image(images)
            text_embeddings = model.encode_text(texts)
            
            # Compute contrastive loss
            loss = contrastive_loss(image_embeddings, text_embeddings)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Assume we have a DataLoader for training data
optimizer = Adam(cross_modal_encoder.parameters(), lr=0.001)
train_cross_modal_encoder(cross_modal_encoder, train_loader, optimizer, epochs=10)

# Image-to-text retrieval
def retrieve_text(image, text_database, top_k=5):
    # Encode the query image
    image_embedding = cross_modal_encoder.encode_image(image)
    
    # Compute similarities with all texts
    similarities = []
    for text, text_embedding in text_database:
        similarity = nx.nn.functional.cosine_similarity(
            image_embedding.unsqueeze(0),
            text_embedding.unsqueeze(0)
        ).item()
        similarities.append((text, similarity))
    
    # Sort by similarity and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Text-to-image retrieval
def retrieve_image(text, image_database, top_k=5):
    # Encode the query text
    text_embedding = cross_modal_encoder.encode_text(text)
    
    # Compute similarities with all images
    similarities = []
    for image_path, image_embedding in image_database:
        similarity = nx.nn.functional.cosine_similarity(
            text_embedding.unsqueeze(0),
            image_embedding.unsqueeze(0)
        ).item()
        similarities.append((image_path, similarity))
    
    # Sort by similarity and return top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

# Test image-to-text retrieval
query_image = load_image("path/to/query/image.jpg")
transformed_image = transform_image(query_image)
retrieved_texts = retrieve_text(transformed_image, text_database, top_k=5)

print("Retrieved texts:")
for text, similarity in retrieved_texts:
    print(f"- {text} (similarity: {similarity:.4f})")

# Test text-to-image retrieval
query_text = "A cat sitting on a mat"
tokenized_text = tokenize_text(query_text)
retrieved_images = retrieve_image(tokenized_text, image_database, top_k=5)

print("Retrieved images:")
for image_path, similarity in retrieved_images:
    print(f"- {image_path} (similarity: {similarity:.4f})")
```

This documentation provides a comprehensive overview of the Zero-Shot Learning module in Neurenix, including key concepts, API reference, framework comparisons, best practices, and tutorials for enabling models to recognize unseen classes.
