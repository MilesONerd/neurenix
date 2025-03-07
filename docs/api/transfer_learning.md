# Transfer Learning Documentation

## Overview

The Transfer Learning module in Neurenix provides tools and utilities for leveraging pre-trained models and adapting them to new tasks with minimal data. Transfer learning is a technique that allows developers to reuse knowledge gained from solving one problem and apply it to a different but related problem, significantly reducing the amount of data and computation required for training.

Neurenix's transfer learning capabilities are implemented using a multi-language architecture, where the high-performance operations are implemented in the Rust/C++ Phynexus engine, while the Python API provides a user-friendly interface. This architecture enables Neurenix to deliver optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

## Key Concepts

### Pre-trained Models

Pre-trained models are neural networks that have been trained on large datasets for general-purpose tasks. These models have learned useful feature representations that can be transferred to new tasks. Neurenix allows you to use pre-trained models as feature extractors or as a starting point for fine-tuning.

### Feature Extraction

Feature extraction involves using a pre-trained model to transform input data into feature representations, which are then used as input to a new model trained for a specific task. In this approach, the pre-trained model's weights remain fixed, and only the new model is trained.

### Fine-tuning

Fine-tuning involves taking a pre-trained model and further training it on a new dataset for a specific task. This process typically involves freezing some layers of the pre-trained model while allowing others to be updated during training. Neurenix provides flexible tools for controlling which layers are frozen or fine-tuned.

### Layer Freezing and Unfreezing

Freezing layers means preventing their weights from being updated during training, while unfreezing allows them to be updated. Neurenix provides functions to freeze and unfreeze specific layers or the entire model, giving you fine-grained control over the transfer learning process.

## API Reference

### TransferModel

```python
neurenix.transfer.TransferModel(base_model, new_layers, freeze_base=True, fine_tune_layers=None)
```

A model for transfer learning that combines a pre-trained base model with new task-specific layers.

**Parameters:**
- `base_model`: The pre-trained model to use as a feature extractor.
- `new_layers`: New layers to add on top of the base model for the target task.
- `freeze_base`: Whether to freeze the parameters of the base model.
- `fine_tune_layers`: List of layer names in the base model to fine-tune (unfreeze). Only used if freeze_base is True.

**Methods:**
- `forward(x)`: Forward pass through the transfer learning model.
- `get_base_model()`: Get the base model.
- `get_new_layers()`: Get the new layers.
- `freeze_base_model()`: Freeze all parameters in the base model.
- `unfreeze_base_model()`: Unfreeze all parameters in the base model.
- `unfreeze_layers(layer_names)`: Unfreeze specific layers in the base model.

**Example:**
```python
import neurenix
from neurenix.nn import Linear, Sequential, ReLU
from neurenix.transfer import TransferModel

# Load a pre-trained model (e.g., a model trained on ImageNet)
base_model = neurenix.models.resnet18(pretrained=True)

# Create new layers for the target task
new_layers = Sequential(
    Linear(512, 256),
    ReLU(),
    Linear(256, 10)  # 10 classes for the target task
)

# Create a transfer learning model
transfer_model = TransferModel(
    base_model=base_model,
    new_layers=new_layers,
    freeze_base=True,
    fine_tune_layers=["layer4"]  # Fine-tune only the last layer of ResNet
)

# Use the model
input_tensor = neurenix.Tensor.randn((1, 3, 224, 224))
output = transfer_model(input_tensor)
```

### Fine-tuning Functions

```python
neurenix.transfer.freeze_layers(model, layer_names=None)
```

Freeze specific layers in a model or the entire model.

**Parameters:**
- `model`: The model to freeze layers in.
- `layer_names`: List of layer names to freeze. If None, freeze all layers.

```python
neurenix.transfer.unfreeze_layers(model, layer_names=None)
```

Unfreeze specific layers in a model or the entire model.

**Parameters:**
- `model`: The model to unfreeze layers in.
- `layer_names`: List of layer names to unfreeze. If None, unfreeze all layers.

```python
neurenix.transfer.fine_tune(model, optimizer, train_data, train_labels, val_data=None, val_labels=None, epochs=10, batch_size=32, loss_fn=None, callbacks=None, early_stopping=False, patience=3)
```

Fine-tune a model on a new dataset.

**Parameters:**
- `model`: The model to fine-tune (either a TransferModel or a regular Module).
- `optimizer`: The optimizer to use for training.
- `train_data`: List of training data tensors.
- `train_labels`: List of training label tensors.
- `val_data`: List of validation data tensors.
- `val_labels`: List of validation label tensors.
- `epochs`: Number of epochs to train for.
- `batch_size`: Batch size for training.
- `loss_fn`: Loss function to use. If None, uses cross-entropy for classification and mean squared error for regression.
- `callbacks`: List of callback functions to call after each epoch.
- `early_stopping`: Whether to use early stopping.
- `patience`: Number of epochs to wait for improvement before stopping.

**Returns:**
- Dictionary containing training history (loss and metrics for each epoch).

**Example:**
```python
import neurenix
from neurenix.nn import Linear, Sequential, ReLU
from neurenix.optim import Adam
from neurenix.transfer import TransferModel, fine_tune

# Create a transfer learning model
base_model = neurenix.models.resnet18(pretrained=True)
new_layers = Sequential(
    Linear(512, 256),
    ReLU(),
    Linear(256, 10)
)
transfer_model = TransferModel(base_model, new_layers, freeze_base=True)

# Create an optimizer
optimizer = Adam(transfer_model.parameters(), lr=0.001)

# Prepare data
train_data = [neurenix.Tensor.randn((3, 224, 224)) for _ in range(100)]
train_labels = [neurenix.Tensor.randint(0, 10, (1,)) for _ in range(100)]
val_data = [neurenix.Tensor.randn((3, 224, 224)) for _ in range(20)]
val_labels = [neurenix.Tensor.randint(0, 10, (1,)) for _ in range(20)]

# Fine-tune the model
history = fine_tune(
    model=transfer_model,
    optimizer=optimizer,
    train_data=train_data,
    train_labels=train_labels,
    val_data=val_data,
    val_labels=val_labels,
    epochs=5,
    batch_size=16,
    early_stopping=True,
    patience=2
)

# Print training history
print(f"Training loss: {history['train_loss']}")
print(f"Validation loss: {history['val_loss']}")
```

### Utility Functions

```python
neurenix.transfer.utils.get_layer_outputs(model, input_tensor, layer_names)
```

Get the outputs of specific layers in a model for a given input.

**Parameters:**
- `model`: The model to extract layer outputs from.
- `input_tensor`: Input tensor to pass through the model.
- `layer_names`: Names of layers to extract outputs from.

**Returns:**
- Dictionary mapping layer names to their output tensors.

```python
neurenix.transfer.utils.visualize_layer_activations(model, input_tensor, layer_name)
```

Visualize the activations of a specific layer in a model.

**Parameters:**
- `model`: The model to visualize activations for.
- `input_tensor`: Input tensor to pass through the model.
- `layer_name`: Name of the layer to visualize.

```python
neurenix.transfer.utils.get_model_feature_extractor(model, output_layer)
```

Create a feature extractor from a model by truncating it at a specific layer.

**Parameters:**
- `model`: The model to create a feature extractor from.
- `output_layer`: Name of the layer to use as the output.

**Returns:**
- A new model that outputs the activations of the specified layer.

```python
neurenix.transfer.utils.compare_model_features(model1, model2, input_tensor, layer_pairs)
```

Compare the features extracted by two models at specific layer pairs.

**Parameters:**
- `model1`: First model.
- `model2`: Second model.
- `input_tensor`: Input tensor to pass through both models.
- `layer_pairs`: List of (layer_name1, layer_name2) tuples to compare.

**Returns:**
- Dictionary mapping layer pairs to similarity scores.

**Example:**
```python
import neurenix
from neurenix.transfer.utils import get_layer_outputs, get_model_feature_extractor

# Load a pre-trained model
model = neurenix.models.resnet18(pretrained=True)

# Get outputs from specific layers
input_tensor = neurenix.Tensor.randn((1, 3, 224, 224))
layer_outputs = get_layer_outputs(model, input_tensor, ["layer1", "layer2", "layer3"])

# Create a feature extractor that outputs from a specific layer
feature_extractor = get_model_feature_extractor(model, "layer3")
features = feature_extractor(input_tensor)
```

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Transfer Learning API** | Unified TransferModel class with fine-tuning utilities | Multiple approaches (Keras applications, TF Hub, model subclassing) |
| **Layer Freezing** | Simple API for freezing/unfreezing specific layers | Requires setting trainable attribute for each layer |
| **Fine-tuning Process** | Integrated fine_tune function with early stopping | Manual implementation of training loops |
| **Edge Optimization** | Native optimization for edge devices | TensorFlow Lite for edge devices |
| **Feature Extraction** | Built-in utilities for extracting intermediate features | Requires custom model creation or Keras functional API |
| **Multi-Language Architecture** | Rust/C++ core with Python interface | C++ core with Python interface |

Neurenix's transfer learning capabilities offer a more unified and intuitive API compared to TensorFlow's multiple approaches. The TransferModel class provides a clean abstraction for combining pre-trained models with new layers, while the fine_tune function simplifies the training process with built-in features like early stopping. The native optimization for edge devices in Neurenix provides better performance on resource-constrained hardware compared to TensorFlow Lite, which is an add-on component rather than being integrated into the core architecture.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Transfer Learning API** | Unified TransferModel class | No specific class, typically done through model composition |
| **Layer Freezing** | Simple API for freezing/unfreezing specific layers | Requires setting requires_grad for each parameter |
| **Fine-tuning Process** | Integrated fine_tune function | Manual implementation of training loops |
| **Edge Optimization** | Native optimization for edge devices | PyTorch Mobile for edge devices |
| **Feature Extraction** | Built-in utilities for extracting intermediate features | Requires hooks or custom forward methods |
| **Hardware Support** | CPU, CUDA, ROCm, WebGPU | CPU, CUDA |

Neurenix's transfer learning capabilities provide a more structured approach compared to PyTorch's flexible but manual approach. The TransferModel class and fine_tune function simplify common transfer learning workflows, while PyTorch requires more manual implementation. Neurenix also extends hardware support to include ROCm and WebGPU, making it more versatile across different hardware platforms. The native edge device optimization in Neurenix provides advantages over PyTorch Mobile, particularly for AI agent applications.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Transfer Learning Support** | Comprehensive deep learning transfer capabilities | Limited transfer learning support |
| **Neural Network Support** | Full neural network architecture | Limited neural network support through MLPClassifier/MLPRegressor |
| **GPU Acceleration** | Native support for multiple GPU types | No native GPU support |
| **Edge Device Support** | Native optimization for edge devices | No specific edge device support |
| **Feature Extraction** | Built-in utilities for neural network feature extraction | Pipeline and FeatureUnion for traditional ML features |
| **Fine-tuning Process** | Integrated fine_tune function for neural networks | No equivalent for neural networks |

Neurenix provides much more comprehensive transfer learning capabilities compared to Scikit-Learn, which primarily focuses on traditional machine learning algorithms with limited neural network support. While Scikit-Learn offers feature extraction through its Pipeline and FeatureUnion components, these are designed for traditional machine learning features rather than deep neural network features. Neurenix's TransferModel and fine-tuning utilities provide a full suite of transfer learning capabilities for deep learning models, along with GPU acceleration and edge device optimization.

## Best Practices

### Choosing Layers to Fine-tune

When deciding which layers to fine-tune in a pre-trained model, consider these guidelines:

1. **Task Similarity**: If the new task is very similar to the original task, freeze more layers. If it's quite different, unfreeze more layers.
2. **Data Size**: With small datasets, freeze more layers to prevent overfitting. With larger datasets, you can unfreeze more layers.
3. **Computational Resources**: Freezing more layers reduces computational requirements during training.
4. **Layer Depth**: Earlier layers capture more general features, while later layers capture more task-specific features. It's common to freeze earlier layers and fine-tune later layers.

```python
import neurenix
from neurenix.transfer import TransferModel

# Load a pre-trained model
base_model = neurenix.models.resnet18(pretrained=True)
new_layers = neurenix.nn.Linear(512, 10)

# For a task similar to the original with a small dataset
model_similar_task = TransferModel(
    base_model=base_model,
    new_layers=new_layers,
    freeze_base=True,
    fine_tune_layers=["layer4"]  # Fine-tune only the last layer
)

# For a different task with a larger dataset
model_different_task = TransferModel(
    base_model=base_model,
    new_layers=new_layers,
    freeze_base=True,
    fine_tune_layers=["layer2", "layer3", "layer4"]  # Fine-tune more layers
)
```

### Setting Learning Rates for Fine-tuning

When fine-tuning a pre-trained model, it's often beneficial to use different learning rates for different parts of the model:

1. **Lower Learning Rates for Pre-trained Layers**: Use smaller learning rates for fine-tuned layers from the pre-trained model to avoid destroying learned features.
2. **Higher Learning Rates for New Layers**: Use larger learning rates for newly added layers that need to learn from scratch.
3. **Learning Rate Schedules**: Gradually decrease learning rates during fine-tuning to refine the model.

```python
import neurenix
from neurenix.transfer import TransferModel
from neurenix.optim import SGD

# Create a transfer learning model
base_model = neurenix.models.resnet18(pretrained=True)
new_layers = neurenix.nn.Sequential(
    neurenix.nn.Linear(512, 256),
    neurenix.nn.ReLU(),
    neurenix.nn.Linear(256, 10)
)
model = TransferModel(base_model, new_layers, freeze_base=True, fine_tune_layers=["layer4"])

# Create an optimizer with different learning rates
optimizer = SGD([
    {"params": model.get_base_model().parameters(), "lr": 0.0001},  # Lower learning rate for pre-trained layers
    {"params": model.get_new_layers().parameters(), "lr": 0.001}    # Higher learning rate for new layers
], momentum=0.9)
```

### Data Augmentation for Transfer Learning

Data augmentation is particularly important for transfer learning with small datasets:

1. **Basic Augmentations**: Use simple transformations like flips, rotations, and crops to increase dataset diversity.
2. **Task-Specific Augmentations**: Choose augmentations that make sense for your specific task and domain.
3. **Consistency**: Apply the same preprocessing steps used during the pre-trained model's training.

```python
import neurenix
from neurenix.data import DataLoader, Dataset
from neurenix.data.transforms import Compose, RandomHorizontalFlip, RandomRotation, RandomCrop, Normalize

# Define data augmentation for training
train_transforms = Compose([
    RandomHorizontalFlip(),
    RandomRotation(10),
    RandomCrop(224, padding=4),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet statistics
])

# Define minimal preprocessing for validation
val_transforms = Compose([
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet statistics
])

# Create datasets with augmentation
train_dataset = Dataset(train_data, train_labels, transform=train_transforms)
val_dataset = Dataset(val_data, val_labels, transform=val_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
```

### Optimizing for Edge Devices

When deploying transfer learning models to edge devices, consider these optimizations:

1. **Model Pruning**: Remove unnecessary layers or channels from the pre-trained model.
2. **Quantization**: Reduce the precision of model weights to decrease memory usage.
3. **Knowledge Distillation**: Train a smaller model to mimic the behavior of the larger pre-trained model.
4. **Feature Selection**: Use only the most informative features from the pre-trained model.

```python
import neurenix
from neurenix import Device, DeviceType
from neurenix.transfer import TransferModel
from neurenix.transfer.utils import get_model_feature_extractor

# Create a smaller feature extractor from a pre-trained model
base_model = neurenix.models.resnet18(pretrained=True)
feature_extractor = get_model_feature_extractor(base_model, "layer3")  # Use earlier layer for smaller model

# Create a compact transfer learning model
new_layers = neurenix.nn.Linear(256, 10)  # Smaller input size from earlier layer
model = TransferModel(feature_extractor, new_layers, freeze_base=True)

# Use the most efficient available device
devices = neurenix.get_available_devices()
edge_device = None

# Prioritize WebGPU for browser-based edge devices
for device in devices:
    if device.type == DeviceType.WEBGPU:
        edge_device = device
        break

# Fall back to CPU if no accelerator is available
if edge_device is None:
    edge_device = Device(DeviceType.CPU)

# Move model to the selected device
for param in model.parameters():
    param.to(edge_device, inplace=True)
```

## Tutorials

### Basic Transfer Learning for Image Classification

```python
import neurenix
from neurenix.nn import Linear, Sequential, ReLU
from neurenix.optim import Adam
from neurenix.transfer import TransferModel, fine_tune

# Step 1: Load a pre-trained model
base_model = neurenix.models.resnet18(pretrained=True)

# Step 2: Create new layers for the target task
new_layers = Sequential(
    Linear(512, 256),
    ReLU(),
    Linear(256, 10)  # 10 classes for the target task
)

# Step 3: Create a transfer learning model
transfer_model = TransferModel(
    base_model=base_model,
    new_layers=new_layers,
    freeze_base=True  # Freeze the base model to use it as a feature extractor
)

# Step 4: Prepare data (simplified example)
train_data = [neurenix.Tensor.randn((3, 224, 224)) for _ in range(100)]
train_labels = [neurenix.Tensor.randint(0, 10, (1,)) for _ in range(100)]
val_data = [neurenix.Tensor.randn((3, 224, 224)) for _ in range(20)]
val_labels = [neurenix.Tensor.randint(0, 10, (1,)) for _ in range(20)]

# Step 5: Create an optimizer
optimizer = Adam(transfer_model.parameters(), lr=0.001)

# Step 6: Fine-tune the model
history = fine_tune(
    model=transfer_model,
    optimizer=optimizer,
    train_data=train_data,
    train_labels=train_labels,
    val_data=val_data,
    val_labels=val_labels,
    epochs=10,
    batch_size=16,
    early_stopping=True,
    patience=3
)

# Step 7: Evaluate the model
transfer_model.train(False)  # Set to evaluation mode
with neurenix.no_grad():
    correct = 0
    total = 0
    for data, label in zip(val_data, val_labels):
        output = transfer_model(data.unsqueeze(0))
        predicted = output.argmax(dim=1)
        total += 1
        correct += (predicted == label).sum().item()
    
    accuracy = correct / total
    print(f"Validation accuracy: {accuracy:.4f}")
```

### Progressive Fine-tuning

```python
import neurenix
from neurenix.nn import Linear
from neurenix.optim import SGD
from neurenix.transfer import TransferModel, fine_tune

# Step 1: Load a pre-trained model
base_model = neurenix.models.resnet18(pretrained=True)
new_layers = Linear(512, 10)

# Step 2: First phase - train only the new layers
transfer_model = TransferModel(
    base_model=base_model,
    new_layers=new_layers,
    freeze_base=True
)

# Create an optimizer for the first phase
optimizer_phase1 = SGD(transfer_model.get_new_layers().parameters(), lr=0.01)

# Fine-tune only the new layers
history_phase1 = fine_tune(
    model=transfer_model,
    optimizer=optimizer_phase1,
    train_data=train_data,
    train_labels=train_labels,
    val_data=val_data,
    val_labels=val_labels,
    epochs=5,
    batch_size=16
)

# Step 3: Second phase - unfreeze and fine-tune the last few layers of the base model
transfer_model.unfreeze_layers(["layer4"])

# Create an optimizer for the second phase with different learning rates
optimizer_phase2 = SGD([
    {"params": transfer_model.get_base_model().parameters(), "lr": 0.0001},  # Lower learning rate for pre-trained layers
    {"params": transfer_model.get_new_layers().parameters(), "lr": 0.001}    # Higher learning rate for new layers
], momentum=0.9)

# Fine-tune the model with some layers unfrozen
history_phase2 = fine_tune(
    model=transfer_model,
    optimizer=optimizer_phase2,
    train_data=train_data,
    train_labels=train_labels,
    val_data=val_data,
    val_labels=val_labels,
    epochs=10,
    batch_size=16
)

# Print training history
print("Phase 1 (new layers only):")
print(f"  Final training loss: {history_phase1['train_loss'][-1]:.4f}")
print(f"  Final validation loss: {history_phase1['val_loss'][-1]:.4f}")

print("Phase 2 (with fine-tuning):")
print(f"  Final training loss: {history_phase2['train_loss'][-1]:.4f}")
print(f"  Final validation loss: {history_phase2['val_loss'][-1]:.4f}")
```

### Feature Extraction and Visualization

```python
import neurenix
from neurenix.transfer.utils import get_layer_outputs, visualize_layer_activations, get_model_feature_extractor

# Step 1: Load a pre-trained model
model = neurenix.models.resnet18(pretrained=True)

# Step 2: Prepare input data
input_tensor = neurenix.Tensor.randn((1, 3, 224, 224))

# Step 3: Extract features from different layers
layer_outputs = get_layer_outputs(model, input_tensor, ["layer1", "layer2", "layer3", "layer4"])

# Step 4: Visualize activations from a specific layer
visualize_layer_activations(model, input_tensor, "layer3")

# Step 5: Create a feature extractor model
feature_extractor = get_model_feature_extractor(model, "layer3")

# Step 6: Use the feature extractor for a new task
features = feature_extractor(input_tensor)
print(f"Extracted features shape: {features.shape}")

# Step 7: Create a classifier using the extracted features
classifier = neurenix.nn.Linear(features.shape[1], 10)
output = classifier(features)
print(f"Classifier output shape: {output.shape}")
```

## Conclusion

The Transfer Learning module of Neurenix provides a comprehensive set of tools for leveraging pre-trained models and adapting them to new tasks with minimal data. Its multi-language architecture with a high-performance Rust/C++ core enables optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

Compared to other frameworks like TensorFlow, PyTorch, and Scikit-Learn, Neurenix's Transfer Learning module offers advantages in terms of API design, hardware support, and edge device optimization. The unified TransferModel class and integrated fine-tuning utilities simplify common transfer learning workflows, while the native optimization for edge devices makes Neurenix particularly well-suited for AI agent development and edge computing applications.
