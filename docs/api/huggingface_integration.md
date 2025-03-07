# Hugging Face Integration Documentation

## Overview

The Hugging Face Integration module in Neurenix provides seamless integration with the Hugging Face ecosystem, allowing users to leverage pre-trained models from the Hugging Face Hub within the Neurenix framework. This integration enables access to state-of-the-art models for natural language processing, computer vision, and other domains, while maintaining compatibility with Neurenix's tensor operations and neural network components.

Neurenix's Hugging Face integration is implemented using a multi-language architecture, where the high-performance operations are implemented in the Rust/C++ Phynexus engine, while the Python API provides a user-friendly interface to Hugging Face models. This architecture enables Neurenix to deliver optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

## Key Concepts

### Model Integration

Neurenix provides wrapper classes that allow Hugging Face models to be used as Neurenix modules. These wrappers handle the conversion between Neurenix tensors and PyTorch tensors, making the integration transparent to the user. The main wrapper class is `HuggingFaceModel`, with specialized subclasses for text models (`HuggingFaceTextModel`) and vision models (`HuggingFaceVisionModel`).

### Training and Fine-tuning

The Hugging Face Integration module includes training utilities that enable fine-tuning of pre-trained models on custom datasets. The `Trainer` class provides a wrapper around Hugging Face's training functionality, while the `FineTuningTrainer` class adds specialized features for transfer learning, such as layer freezing and gradual unfreezing.

### Model Loading and Saving

Neurenix's Hugging Face integration supports loading models from the Hugging Face Hub, as well as saving fine-tuned models locally or pushing them back to the Hub. This enables seamless sharing and reuse of models within the Neurenix ecosystem.

### Task-specific Models

The integration supports loading models for specific tasks using Hugging Face's pipeline API. This allows users to quickly set up models for common tasks such as text classification, named entity recognition, question answering, image classification, and more.

## API Reference

### HuggingFaceModel

```python
neurenix.huggingface.HuggingFaceModel(
    model_name,
    task=None,
    config=None,
    cache_dir=None,
    device="cpu",
    name="HuggingFaceModel"
)
```

Base class for Hugging Face model integration in Neurenix.

**Parameters:**
- `model_name`: Name of the model to load from Hugging Face Hub
- `task`: Task for which to load the model (optional)
- `config`: Model configuration (optional)
- `cache_dir`: Directory to cache models (optional)
- `device`: Device to load the model on (default: "cpu")
- `name`: Model name (default: "HuggingFaceModel")

**Methods:**
- `forward(x)`: Forward pass through the model
- `to(device)`: Move model to device
- `save(path)`: Save model to disk
- `load(path)`: Load model from disk

### HuggingFaceTextModel

```python
neurenix.huggingface.HuggingFaceTextModel(
    model_name,
    task=None,
    config=None,
    cache_dir=None,
    device="cpu",
    tokenizer_name=None,
    max_length=512,
    name="HuggingFaceTextModel"
)
```

Specialized class for Hugging Face text model integration in Neurenix.

**Parameters:**
- `model_name`: Name of the model to load from Hugging Face Hub
- `task`: Task for which to load the model (optional)
- `config`: Model configuration (optional)
- `cache_dir`: Directory to cache models (optional)
- `device`: Device to load the model on (default: "cpu")
- `tokenizer_name`: Name of the tokenizer to load (optional)
- `max_length`: Maximum sequence length (default: 512)
- `name`: Model name (default: "HuggingFaceTextModel")

**Methods:**
- `forward(x)`: Forward pass through the model
- `encode(x)`: Encode text using the tokenizer
- `decode(x)`: Decode token IDs to text
- `to(device)`: Move model to device
- `save(path)`: Save model and tokenizer to disk
- `load(path)`: Load model and tokenizer from disk

### HuggingFaceVisionModel

```python
neurenix.huggingface.HuggingFaceVisionModel(
    model_name,
    task=None,
    config=None,
    cache_dir=None,
    device="cpu",
    processor_name=None,
    name="HuggingFaceVisionModel"
)
```

Specialized class for Hugging Face vision model integration in Neurenix.

**Parameters:**
- `model_name`: Name of the model to load from Hugging Face Hub
- `task`: Task for which to load the model (optional)
- `config`: Model configuration (optional)
- `cache_dir`: Directory to cache models (optional)
- `device`: Device to load the model on (default: "cpu")
- `processor_name`: Name of the processor to load (optional)
- `name`: Model name (default: "HuggingFaceVisionModel")

**Methods:**
- `forward(x)`: Forward pass through the model
- `preprocess(x)`: Preprocess images using the processor
- `to(device)`: Move model to device
- `save(path)`: Save model and processor to disk
- `load(path)`: Load model and processor from disk

### Trainer

```python
neurenix.huggingface.Trainer(
    model,
    args=None,
    train_dataset=None,
    eval_dataset=None,
    compute_metrics=None,
    optimizers=None,
    callbacks=None,
    name="HuggingFaceTrainer"
)
```

Trainer for Hugging Face models in Neurenix.

**Parameters:**
- `model`: Model to train (HuggingFaceModel or transformers.PreTrainedModel)
- `args`: Training arguments (optional)
- `train_dataset`: Training dataset (optional)
- `eval_dataset`: Evaluation dataset (optional)
- `compute_metrics`: Function to compute metrics (optional)
- `optimizers`: Tuple of (optimizer, scheduler) (optional)
- `callbacks`: List of callbacks (optional)
- `name`: Trainer name (default: "HuggingFaceTrainer")

**Methods:**
- `train()`: Train the model
- `evaluate()`: Evaluate the model
- `predict(test_dataset)`: Make predictions with the model
- `save_model(path)`: Save model to disk
- `push_to_hub(repo_id, **kwargs)`: Push model to Hugging Face Hub

### FineTuningTrainer

```python
neurenix.huggingface.FineTuningTrainer(
    model,
    args=None,
    train_dataset=None,
    eval_dataset=None,
    compute_metrics=None,
    optimizers=None,
    callbacks=None,
    freeze_base_model=False,
    freeze_layers=None,
    name="FineTuningTrainer"
)
```

Fine-tuning trainer for Hugging Face models in Neurenix.

**Parameters:**
- `model`: Model to train (HuggingFaceModel or transformers.PreTrainedModel)
- `args`: Training arguments (optional)
- `train_dataset`: Training dataset (optional)
- `eval_dataset`: Evaluation dataset (optional)
- `compute_metrics`: Function to compute metrics (optional)
- `optimizers`: Tuple of (optimizer, scheduler) (optional)
- `callbacks`: List of callbacks (optional)
- `freeze_base_model`: Whether to freeze the base model (default: False)
- `freeze_layers`: List of layer names to freeze (optional)
- `name`: Trainer name (default: "FineTuningTrainer")

**Methods:**
- `train()`: Train the model
- `evaluate()`: Evaluate the model
- `predict(test_dataset)`: Make predictions with the model
- `save_model(path)`: Save model to disk
- `push_to_hub(repo_id, **kwargs)`: Push model to Hugging Face Hub
- `unfreeze_layers(layer_names=None)`: Unfreeze specific layers of the model
- `train_with_gradual_unfreezing(layer_groups, epochs_per_group=1)`: Train the model with gradual unfreezing

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Hugging Face Integration** | Native integration with Neurenix tensors and modules | Requires TensorFlow-specific Hugging Face models |
| **Model Conversion** | Automatic conversion between Neurenix and PyTorch tensors | Requires manual conversion or TensorFlow-specific models |
| **Fine-tuning Capabilities** | Advanced fine-tuning with layer freezing and gradual unfreezing | Basic fine-tuning through Keras API or custom training loops |
| **Edge Device Support** | Native optimization for edge devices | TensorFlow Lite for edge devices |
| **Hardware Acceleration** | Multi-device support (CPU, CUDA, ROCm, WebGPU) | Primarily optimized for CPU and CUDA |
| **API Design** | Consistent API across all model types | Varies between different model types and versions |

Neurenix's Hugging Face integration offers a more seamless experience compared to TensorFlow, with automatic conversion between tensor types and a consistent API across different model types. The integration is designed to work with Neurenix's multi-language architecture, enabling optimal performance across a wide range of devices, including edge devices. Additionally, Neurenix provides advanced fine-tuning capabilities, such as layer freezing and gradual unfreezing, which are not directly available in TensorFlow.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Hugging Face Integration** | Wrapped as Neurenix modules with tensor conversion | Native integration (Hugging Face is built on PyTorch) |
| **Multi-language Architecture** | Rust/C++ core with Python API | Python with C++ extensions |
| **Edge Device Support** | Native optimization for edge devices | PyTorch Mobile for edge devices |
| **Hardware Support** | CPU, CUDA, ROCm, WebGPU | CPU, CUDA |
| **Fine-tuning Capabilities** | Advanced fine-tuning with layer freezing and gradual unfreezing | Basic fine-tuning through Hugging Face Trainer |
| **Integration with Framework** | Integrated with Neurenix's tensor operations and neural networks | Native integration with PyTorch's ecosystem |

While PyTorch has a native advantage in Hugging Face integration (as Hugging Face is built on PyTorch), Neurenix provides a more comprehensive solution for deploying models across a wider range of devices, including edge devices and hardware platforms not directly supported by PyTorch. Neurenix's multi-language architecture with a high-performance Rust/C++ core enables optimal performance on resource-constrained devices. Additionally, Neurenix's advanced fine-tuning capabilities provide more control over the training process, making it easier to adapt pre-trained models to specific tasks.

### Neurenix vs. Hugging Face Transformers

| Feature | Neurenix | Hugging Face Transformers |
|---------|----------|---------------------------|
| **Framework Integration** | Integrated with Neurenix's tensor operations and neural networks | Primarily designed for PyTorch, with limited TensorFlow support |
| **Multi-language Architecture** | Rust/C++ core with Python API | Python with C++ extensions (via PyTorch) |
| **Edge Device Support** | Native optimization for edge devices | Limited support through PyTorch Mobile or TensorFlow Lite |
| **Hardware Support** | CPU, CUDA, ROCm, WebGPU | CPU, CUDA (via PyTorch) |
| **Fine-tuning Capabilities** | Advanced fine-tuning with layer freezing and gradual unfreezing | Basic fine-tuning through Trainer API |
| **Integration with Other Components** | Seamless integration with other Neurenix components | Requires additional libraries for integration with other components |

Neurenix's Hugging Face integration provides a more comprehensive solution for deploying models across a wider range of devices and hardware platforms. While Hugging Face Transformers is primarily designed for PyTorch, Neurenix's integration enables the use of Hugging Face models within a framework that supports multiple hardware platforms, including edge devices. Additionally, Neurenix's advanced fine-tuning capabilities provide more control over the training process, making it easier to adapt pre-trained models to specific tasks.

## Best Practices

### Choosing the Right Model

When selecting a Hugging Face model for use with Neurenix, consider the following factors:

1. **Task Compatibility**: Choose a model that is pre-trained for a task similar to your target task
2. **Model Size**: Consider the computational resources available, especially for edge devices
3. **Language/Domain**: Select a model trained on data from a similar domain or language
4. **Performance**: Check the model's performance on benchmarks relevant to your task

### Optimizing for Edge Devices

When deploying Hugging Face models to edge devices with Neurenix, consider these optimizations:

1. **Model Distillation**: Use smaller, distilled versions of models (e.g., DistilBERT instead of BERT)
2. **Quantization**: Quantize model weights to reduce memory usage
3. **Pruning**: Remove unnecessary connections in neural networks
4. **Layer Freezing**: Freeze parts of the model to reduce computational requirements
5. **Batch Size**: Use smaller batch sizes to reduce memory usage

### Fine-tuning Strategies

When fine-tuning Hugging Face models with Neurenix, consider these strategies:

1. **Layer Freezing**: Freeze the base model and only train task-specific layers
2. **Gradual Unfreezing**: Start with most layers frozen, then gradually unfreeze them
3. **Learning Rate**: Use a smaller learning rate for pre-trained layers and a larger one for new layers
4. **Early Stopping**: Monitor validation performance and stop training when it plateaus
5. **Data Augmentation**: Use data augmentation to improve generalization

## Tutorials

### Loading and Using a Pre-trained Text Model

```python
import neurenix
from neurenix.huggingface import HuggingFaceTextModel

# Load a pre-trained BERT model for sentiment analysis
model = HuggingFaceTextModel(
    model_name="distilbert-base-uncased-finetuned-sst-2-english",
    task="sentiment-analysis",
    device="cuda" if neurenix.is_cuda_available() else "cpu",
)

# Analyze sentiment of text
texts = [
    "I love this product!",
    "This is terrible.",
    "The movie was okay, but not great.",
]

# Forward pass
results = model(texts)

# Print results
for text, result in zip(texts, results):
    label = result["label"]
    score = result["score"]
    print(f"Text: {text}")
    print(f"Sentiment: {label} (score: {score:.4f})")
    print()
```

### Fine-tuning a Pre-trained Model

```python
import neurenix
from neurenix.huggingface import HuggingFaceTextModel, FineTuningTrainer
from datasets import load_dataset

# Load a pre-trained BERT model
model = HuggingFaceTextModel(
    model_name="distilbert-base-uncased",
    device="cuda" if neurenix.is_cuda_available() else "cpu",
)

# Load a dataset
dataset = load_dataset("glue", "sst2")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

# Define training arguments
training_args = {
    "output_dir": "./results",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_dir": "./logs",
    "logging_steps": 10,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
}

# Define compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# Create trainer
trainer = FineTuningTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    freeze_base_model=True,  # Freeze the base model
)

# Train model
trainer.train()

# Evaluate model
metrics = trainer.evaluate()
print(f"Evaluation metrics: {metrics}")

# Save model
trainer.save_model("./fine_tuned_model")

# Push to Hugging Face Hub (if desired)
# trainer.push_to_hub("username/model-name")
```

### Using a Pre-trained Vision Model

```python
import neurenix
import numpy as np
from PIL import Image
from neurenix.huggingface import HuggingFaceVisionModel

# Load a pre-trained vision model for image classification
model = HuggingFaceVisionModel(
    model_name="google/vit-base-patch16-224",
    task="image-classification",
    device="cuda" if neurenix.is_cuda_available() else "cpu",
)

# Load an image
image = Image.open("cat.jpg")

# Forward pass
result = model(image)

# Print result
print(f"Predicted class: {result[0]['label']}")
print(f"Confidence: {result[0]['score']:.4f}")
```

### Gradual Unfreezing for Fine-tuning

```python
import neurenix
from neurenix.huggingface import HuggingFaceTextModel, FineTuningTrainer
from datasets import load_dataset

# Load a pre-trained BERT model
model = HuggingFaceTextModel(
    model_name="bert-base-uncased",
    device="cuda" if neurenix.is_cuda_available() else "cpu",
)

# Load a dataset
dataset = load_dataset("glue", "mrpc")
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

# Define training arguments
training_args = {
    "output_dir": "./results",
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "logging_dir": "./logs",
    "logging_steps": 10,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
}

# Define compute_metrics function
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

# Create trainer
trainer = FineTuningTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# Define layer groups for gradual unfreezing
layer_groups = [
    ["classifier"],  # First unfreeze the classifier
    ["encoder.layer.11"],  # Then unfreeze the last encoder layer
    ["encoder.layer.10"],  # Then unfreeze the second-to-last encoder layer
    ["encoder.layer.9"],  # And so on...
    ["encoder.layer.8"],
]

# Train with gradual unfreezing
metrics = trainer.train_with_gradual_unfreezing(
    layer_groups=layer_groups,
    epochs_per_group=1,
)

# Print metrics
for group, group_metrics in metrics.items():
    print(f"Metrics for {group}: {group_metrics}")

# Save model
trainer.save_model("./fine_tuned_model")
```

## Conclusion

The Hugging Face Integration module of Neurenix provides seamless access to state-of-the-art pre-trained models from the Hugging Face ecosystem, while maintaining compatibility with Neurenix's tensor operations and neural network components. The integration enables users to leverage these models for a wide range of tasks, from natural language processing to computer vision, and to fine-tune them on custom datasets.

Compared to other frameworks like TensorFlow, PyTorch, and Hugging Face Transformers, Neurenix's Hugging Face integration offers advantages in terms of API design, hardware support, and edge device optimization. The multi-language architecture with a high-performance Rust/C++ core enables optimal performance across a wide range of devices, from edge devices to multi-GPU clusters. Additionally, the advanced fine-tuning capabilities provide more control over the training process, making it easier to adapt pre-trained models to specific tasks.
