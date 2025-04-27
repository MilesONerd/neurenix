# Transfer Learning Module

## Overview

The Transfer Learning module in Neurenix provides a comprehensive suite of tools and techniques for leveraging pre-trained models to solve new tasks with limited data. Built on Neurenix's high-performance multi-language architecture, this module enables efficient knowledge transfer across different domains and tasks.

Transfer learning is a machine learning technique where a model developed for one task is reused as the starting point for a model on a second task. This approach significantly reduces the amount of data and computational resources required to train models for new tasks, making it particularly valuable in scenarios with limited labeled data.

The Neurenix Transfer Learning module supports various transfer learning paradigms, including feature extraction, fine-tuning, domain adaptation, and multi-task learning. It provides pre-trained models for different domains, such as computer vision, natural language processing, and audio processing, along with tools for adapting these models to new tasks.

## Key Concepts

### Pre-trained Models

The Transfer Learning module provides access to a wide range of pre-trained models:

- **Computer Vision Models**: Models pre-trained on large-scale image datasets
  - Classification models: ResNet, EfficientNet, Vision Transformer
  - Detection models: YOLO, Faster R-CNN, RetinaNet
  - Segmentation models: U-Net, DeepLab, Mask R-CNN

- **Natural Language Processing Models**: Models pre-trained on large-scale text corpora
  - Language models: BERT, GPT, T5, RoBERTa
  - Embedding models: Word2Vec, GloVe, FastText
  - Translation models: Transformer-based NMT

- **Audio Processing Models**: Models pre-trained on large-scale audio datasets
  - Speech recognition models: Wav2Vec, DeepSpeech
  - Audio classification models: VGGish, PANNs
  - Music processing models: JukeBox, MusicTransformer

### Transfer Learning Techniques

The module supports various transfer learning techniques:

- **Feature Extraction**: Using pre-trained models as fixed feature extractors
  - Layer freezing: Keeping pre-trained weights fixed
  - Feature caching: Computing and storing features for efficiency
  - Feature selection: Choosing relevant features for the target task

- **Fine-tuning**: Adapting pre-trained models to new tasks by updating weights
  - Full fine-tuning: Updating all model parameters
  - Partial fine-tuning: Updating only specific layers
  - Progressive fine-tuning: Gradually unfreezing layers
  - Layer-wise fine-tuning: Using different learning rates for different layers

- **Parameter-Efficient Fine-tuning**: Adapting models with minimal parameter updates
  - Adapters: Adding small adapter modules between layers
  - Prompt tuning: Learning continuous prompts for language models
  - LoRA: Low-rank adaptation of large language models

- **Domain Adaptation**: Adapting models to new domains
  - Adversarial domain adaptation: Using adversarial training to align domains
  - Self-supervised adaptation: Using self-supervised objectives for adaptation
  - Gradual domain adaptation: Adapting through intermediate domains

### Model Adaptation

The module provides tools for adapting pre-trained models to new tasks:

- **Task-Specific Heads**: Adding task-specific output layers
  - Classification heads: For classification tasks
  - Regression heads: For regression tasks
  - Detection heads: For object detection tasks
  - Segmentation heads: For segmentation tasks

- **Architecture Modification**: Modifying model architectures for new tasks
  - Layer addition: Adding new layers to the model
  - Layer removal: Removing unnecessary layers
  - Layer replacement: Replacing layers with task-specific ones

- **Input Adaptation**: Adapting models to different input formats
  - Modality conversion: Converting between different modalities
  - Resolution adaptation: Handling different input resolutions
  - Channel adaptation: Handling different numbers of input channels

## API Reference

### Pre-trained Model Loading

```python
import neurenix
from neurenix.transfer import load_pretrained_model

# Load a pre-trained computer vision model
vision_model = load_pretrained_model(
    model_name="resnet50",             # Model architecture
    domain="vision",                   # Model domain
    weights="imagenet",                # Pre-trained weights
    input_shape=(224, 224, 3),         # Input shape
    include_top=False,                 # Whether to include top layers
    pooling="avg",                     # Pooling mode
    device="cuda" if neurenix.cuda.is_available() else "cpu"  # Device to load the model on
)

# Load a pre-trained NLP model
nlp_model = load_pretrained_model(
    model_name="bert-base-uncased",    # Model architecture
    domain="nlp",                      # Model domain
    weights="bert-base-uncased",       # Pre-trained weights
    max_length=512,                    # Maximum sequence length
    include_top=False,                 # Whether to include top layers
    pooling="cls",                     # Pooling mode
    device="cuda" if neurenix.cuda.is_available() else "cpu"  # Device to load the model on
)
```

### Feature Extraction

```python
from neurenix.transfer import FeatureExtractor

# Create a feature extractor from a pre-trained model
feature_extractor = FeatureExtractor(
    model=vision_model,                # Pre-trained model
    layers=["layer1", "layer2", "layer3"],  # Layers to extract features from
    pooling="avg",                     # Pooling mode
    flatten=True,                      # Whether to flatten the features
    normalize=True                     # Whether to normalize the features
)

# Extract features from input data
features = feature_extractor(input_data)

# Cache features for efficiency
feature_extractor.cache_features(
    dataset=train_dataset,             # Dataset to extract features from
    batch_size=32,                     # Batch size for feature extraction
    num_workers=4,                     # Number of workers for data loading
    cache_dir="./feature_cache"        # Directory to store cached features
)
```

### Fine-tuning

```python
from neurenix.transfer import Finetuner

# Create a fine-tuner for a pre-trained model
finetuner = Finetuner(
    model=vision_model,                # Pre-trained model
    num_classes=10,                    # Number of target classes
    freeze_layers=["conv1", "layer1"],  # Layers to freeze
    dropout_rate=0.5,                  # Dropout rate for the new head
    head_hidden_sizes=[256],           # Hidden layer sizes for the new head
    head_activation="relu",            # Activation function for the new head
    learning_rate=0.001,               # Learning rate for fine-tuning
    weight_decay=1e-5,                 # Weight decay for fine-tuning
    optimizer="adam",                  # Optimizer for fine-tuning
    loss="cross_entropy",              # Loss function for fine-tuning
    metrics=["accuracy", "f1_score"]   # Metrics for evaluation
)

# Fine-tune the model
finetuner.fit(
    train_dataset=train_dataset,       # Training dataset
    val_dataset=val_dataset,           # Validation dataset
    batch_size=32,                     # Batch size for training
    epochs=10,                         # Number of training epochs
    callbacks=[early_stopping, model_checkpoint],  # Training callbacks
    num_workers=4                      # Number of workers for data loading
)
```

### Parameter-Efficient Fine-tuning

```python
from neurenix.transfer import AdapterTuner, PromptTuner, LoraTuner

# Create an adapter tuner for a pre-trained model
adapter_tuner = AdapterTuner(
    model=nlp_model,                   # Pre-trained model
    adapter_size=64,                   # Size of adapter modules
    adapter_type="houlsby",            # Type of adapter (houlsby or pfeiffer)
    adapter_init_scale=1e-3,           # Initialization scale for adapters
    learning_rate=0.001,               # Learning rate for tuning
    weight_decay=1e-5,                 # Weight decay for tuning
    optimizer="adam",                  # Optimizer for tuning
    loss="cross_entropy",              # Loss function for tuning
    metrics=["accuracy", "f1_score"]   # Metrics for evaluation
)
```

## Framework Comparison

### Neurenix Transfer Learning vs. TensorFlow Transfer Learning

| Feature | Neurenix Transfer Learning | TensorFlow Transfer Learning |
|---------|----------------------------|------------------------------|
| Performance | Multi-language implementation with Rust/C++ backends | Python implementation with TensorFlow backend |
| Hardware Support | Comprehensive support for various hardware | Primarily focused on TPUs and GPUs |
| Pre-trained Model Variety | Extensive collection of models across domains | Good selection of models, primarily vision and NLP |
| Parameter-Efficient Fine-tuning | Comprehensive support for various techniques | Limited support through separate libraries |
| Domain Adaptation | Built-in support for various domain adaptation methods | Limited built-in support |
| Knowledge Distillation | Comprehensive support for various distillation techniques | Basic support for vanilla distillation |
| Edge Device Support | Native support for edge devices | Limited through TensorFlow Lite |

Neurenix's Transfer Learning module provides better performance through its multi-language implementation and offers more comprehensive hardware support, especially for edge devices. It also provides a wider variety of pre-trained models, more advanced parameter-efficient fine-tuning techniques, and better support for domain adaptation and knowledge distillation.

### Neurenix Transfer Learning vs. PyTorch Transfer Learning

| Feature | Neurenix Transfer Learning | PyTorch Transfer Learning |
|---------|----------------------------|---------------------------|
| Performance | Multi-language implementation with Rust/C++ backends | Python implementation with PyTorch backend |
| Hardware Support | Comprehensive support for various hardware | Primarily focused on CUDA devices |
| Pre-trained Model Variety | Extensive collection of models across domains | Good selection of models through separate libraries |
| Parameter-Efficient Fine-tuning | Comprehensive support for various techniques | Good support through separate libraries |
| Domain Adaptation | Built-in support for various domain adaptation methods | Limited built-in support |
| Knowledge Distillation | Comprehensive support for various distillation techniques | Good support through separate libraries |
| Edge Device Support | Native support for edge devices | Limited through separate tools |

While PyTorch has good transfer learning capabilities through various libraries, Neurenix's Transfer Learning module offers better performance through its multi-language implementation and provides more comprehensive hardware support, especially for edge devices. It also offers more integrated support for domain adaptation and transfer learning evaluation.

### Neurenix Transfer Learning vs. Scikit-Learn Transfer Learning

| Feature | Neurenix Transfer Learning | Scikit-Learn Transfer Learning |
|---------|----------------------------|--------------------------------|
| Deep Learning Support | Full support for deep transfer learning | Limited to classical machine learning |
| Hardware Acceleration | Native support for various hardware accelerators | Limited hardware acceleration |
| Pre-trained Model Variety | Extensive collection of models across domains | Limited pre-trained model support |
| Parameter-Efficient Fine-tuning | Comprehensive support for various techniques | No support for deep learning techniques |
| Domain Adaptation | Built-in support for various domain adaptation methods | Basic support for classical methods |
| Knowledge Distillation | Comprehensive support for various distillation techniques | No built-in support |
| Edge Device Support | Native support for edge devices | Limited edge support |

Scikit-Learn's transfer learning capabilities are primarily focused on classical machine learning, while Neurenix's Transfer Learning module is designed for both classical and deep transfer learning. Neurenix provides better hardware acceleration, more comprehensive support for various transfer learning techniques, and better integration with the deep learning ecosystem.

## Best Practices

### Choosing the Right Pre-trained Model

1. **Consider Task Similarity**: Choose a pre-trained model trained on a task similar to your target task.

```python
# For image classification, use a model pre-trained on ImageNet
if task_type == "image_classification":
    model = neurenix.transfer.load_pretrained_model(
        model_name="resnet50",
        domain="vision",
        weights="imagenet"
    )
# For text classification, use a model pre-trained on a large text corpus
elif task_type == "text_classification":
    model = neurenix.transfer.load_pretrained_model(
        model_name="bert-base-uncased",
        domain="nlp",
        weights="bert-base-uncased"
    )
```

2. **Consider Model Size and Complexity**: Balance model performance with computational constraints.

```python
# For resource-constrained environments, use a smaller model
if resources == "limited":
    model = neurenix.transfer.load_pretrained_model(
        model_name="mobilenetv2",
        domain="vision",
        weights="imagenet"
    )
# For high-performance environments, use a larger model
elif resources == "abundant":
    model = neurenix.transfer.load_pretrained_model(
        model_name="efficientnet-b7",
        domain="vision",
        weights="imagenet"
    )
```

3. **Consider Data Availability**: Choose the transfer learning approach based on data availability.

```python
# For very limited data, use feature extraction
if data_size == "very_small":
    feature_extractor = neurenix.transfer.FeatureExtractor(
        model=model,
        layers=["layer4"],
        pooling="avg"
    )
    features = feature_extractor(input_data)
    # Train a simple classifier on extracted features
    
# For moderate data, use fine-tuning with frozen layers
elif data_size == "moderate":
    finetuner = neurenix.transfer.Finetuner(
        model=model,
        num_classes=num_classes,
        freeze_layers=["conv1", "layer1", "layer2"]
    )
    
# For abundant data, use full fine-tuning
elif data_size == "large":
    finetuner = neurenix.transfer.Finetuner(
        model=model,
        num_classes=num_classes,
        freeze_layers=[]  # No frozen layers
    )
```

### Fine-tuning Strategies

1. **Start with Feature Extraction**: Begin with feature extraction before moving to fine-tuning.

2. **Use Progressive Unfreezing**: Gradually unfreeze layers during fine-tuning.

```python
# Step 1: Fine-tune only the top layers
finetuner = neurenix.transfer.Finetuner(
    model=model,
    num_classes=num_classes,
    freeze_layers=["conv1", "layer1", "layer2", "layer3"]
)
finetuner.fit(train_dataset, val_dataset, epochs=5)

# Step 2: Unfreeze more layers and continue fine-tuning
finetuner.unfreeze_layers(["layer3"])
finetuner.fit(train_dataset, val_dataset, epochs=5)

# Step 3: Unfreeze all layers and continue fine-tuning
finetuner.unfreeze_layers(["layer2", "layer1", "conv1"])
finetuner.fit(train_dataset, val_dataset, epochs=5)
```

3. **Use Discriminative Learning Rates**: Apply different learning rates to different layers.

```python
# Use lower learning rates for early layers and higher rates for later layers
finetuner = neurenix.transfer.Finetuner(
    model=model,
    num_classes=num_classes,
    layer_learning_rates={
        "conv1": 1e-5,
        "layer1": 3e-5,
        "layer2": 1e-4,
        "layer3": 3e-4,
        "layer4": 1e-3,
        "fc": 1e-3
    }
)
finetuner.fit(train_dataset, val_dataset)
```

## Tutorials

### Image Classification with Transfer Learning

```python
import neurenix
from neurenix.transfer import load_pretrained_model, Finetuner
from neurenix.data import ImageFolder, DataLoader
from neurenix.vision import transforms

# Define data transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = ImageFolder(
    root="./data/flowers/train",
    transform=train_transform
)

val_dataset = ImageFolder(
    root="./data/flowers/val",
    transform=val_transform
)

# Create data loaders
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4
)

# Load a pre-trained model
model = load_pretrained_model(
    model_name="resnet50",
    domain="vision",
    weights="imagenet",
    input_shape=(224, 224, 3),
    include_top=False,
    pooling="avg",
    device="cuda" if neurenix.cuda.is_available() else "cpu"
)

# Create a fine-tuner
finetuner = Finetuner(
    model=model,
    num_classes=len(train_dataset.classes),
    freeze_layers=["conv1", "layer1"],
    dropout_rate=0.5,
    learning_rate=0.001,
    weight_decay=1e-5,
    optimizer="adam",
    loss="cross_entropy",
    metrics=["accuracy", "top5_accuracy"]
)

# Define callbacks
early_stopping = neurenix.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    mode="max"
)

model_checkpoint = neurenix.callbacks.ModelCheckpoint(
    filepath="./checkpoints/resnet50_flowers",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

# Fine-tune the model
finetuner.fit(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    batch_size=32,
    epochs=20,
    callbacks=[early_stopping, model_checkpoint],
    num_workers=4
)

# Evaluate the fine-tuned model
metrics = finetuner.evaluate(
    test_dataset=val_dataset,
    batch_size=32,
    num_workers=4
)

print(f"Test accuracy: {metrics['accuracy']:.4f}")
print(f"Test top-5 accuracy: {metrics['top5_accuracy']:.4f}")
```
