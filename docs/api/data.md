# Data Module

## Overview

The Data module in Neurenix provides comprehensive tools and utilities for data loading, preprocessing, augmentation, and management in machine learning workflows. It enables efficient handling of various data types, including images, text, time series, and structured data, with a focus on performance and scalability.

Built on Neurenix's multi-language architecture, the Data module leverages high-performance operations implemented in Rust and C++ while providing a user-friendly Python API. This design ensures optimal performance for data-intensive tasks while maintaining ease of use for researchers and developers.

The module integrates seamlessly with other Neurenix components, providing a consistent interface for data handling across the framework. It supports both standard datasets and custom data sources, making it suitable for a wide range of machine learning applications.

## Key Concepts

### Dataset Abstraction

The Data module provides a unified dataset abstraction that handles various data types and sources:

- **Standard Datasets**: Pre-configured datasets like MNIST, CIFAR, ImageNet
- **Custom Datasets**: User-defined datasets from files or in-memory data
- **Streaming Datasets**: Efficient handling of large datasets that don't fit in memory
- **Distributed Datasets**: Datasets distributed across multiple nodes or devices

Datasets implement a common interface, allowing them to be used interchangeably with data loaders and preprocessing pipelines.

### Data Loaders

Data loaders handle the efficient loading and batching of data from datasets:

- **Batch Processing**: Efficient creation of mini-batches for training
- **Shuffling**: Random sampling for better training convergence
- **Prefetching**: Asynchronous data loading to overlap computation and I/O
- **Multi-processing**: Parallel data loading using multiple worker processes
- **Pin Memory**: Optimized memory management for GPU training

### Preprocessing Pipelines

Preprocessing pipelines transform raw data into a format suitable for model training:

- **Composable Transforms**: Chains of transformations applied sequentially
- **Lazy Evaluation**: Transformations applied on-demand to minimize memory usage
- **Hardware Acceleration**: GPU-accelerated preprocessing when available
- **Deterministic Pipelines**: Reproducible preprocessing with fixed random seeds
- **Adaptive Preprocessing**: Dynamic adjustments based on data characteristics

### Data Augmentation

Data augmentation techniques increase the diversity of training data:

- **Image Augmentations**: Rotations, flips, crops, color jittering, etc.
- **Text Augmentations**: Synonym replacement, back-translation, word dropout, etc.
- **Time Series Augmentations**: Time warping, magnitude warping, jittering, etc.
- **Structured Data Augmentations**: Feature perturbation, synthetic sampling, etc.
- **Advanced Augmentations**: Mixup, CutMix, AutoAugment, RandAugment, etc.

### Data Versioning and Management

Tools for tracking and managing datasets throughout the machine learning lifecycle:

- **Dataset Versioning**: Track changes to datasets over time
- **Metadata Management**: Store and retrieve dataset metadata
- **Data Validation**: Verify data quality and consistency
- **Data Splitting**: Create train/validation/test splits
- **Data Caching**: Efficient storage and retrieval of processed data

## API Reference

### Dataset Classes

#### `Dataset`

Base class for all datasets in Neurenix.

```python
import neurenix
from neurenix.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        super().__init__()
        self.data_path = data_path
        self.transform = transform
        self.data = self._load_data()
        
    def _load_data(self):
        # Load data from self.data_path
        return data
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        if self.transform:
            item = self.transform(item)
        return item
```

#### `ImageFolder`

Dataset for loading images organized in folders by class.

```python
from neurenix.data import ImageFolder, transforms

# Create a dataset from images organized in class folders
dataset = ImageFolder(
    root="/path/to/images",
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
)
```

#### `TensorDataset`

Dataset wrapping tensors.

```python
import neurenix
from neurenix.data import TensorDataset

# Create a dataset from tensors
features = neurenix.tensor([[1, 2], [3, 4], [5, 6]])
labels = neurenix.tensor([0, 1, 0])
dataset = TensorDataset(features, labels)
```

#### `CSVDataset`

Dataset for loading data from CSV files.

```python
from neurenix.data import CSVDataset

# Create a dataset from a CSV file
dataset = CSVDataset(
    file_path="/path/to/data.csv",
    target_column="label",
    feature_columns=["feature1", "feature2", "feature3"],
    categorical_columns=["feature1"],
    transform=None
)
```

#### `StreamingDataset`

Dataset for streaming data that doesn't fit in memory.

```python
from neurenix.data import StreamingDataset

# Create a streaming dataset
dataset = StreamingDataset(
    data_source="/path/to/large/dataset",
    batch_size=32,
    shuffle_buffer_size=10000,
    prefetch_size=5,
    transform=transform_fn
)
```

### DataLoader

Handles efficient loading and batching of data.

```python
from neurenix.data import DataLoader

# Create a data loader
dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    prefetch_factor=2
)

# Iterate through batches
for batch_idx, (inputs, targets) in enumerate(dataloader):
    # Process batch
    pass
```

### Transforms

#### Basic Transforms

```python
from neurenix.data import transforms

# Compose multiple transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Apply transform to an image
transformed_image = transform(image)
```

#### Image Transforms

```python
from neurenix.data import transforms

# Image transforms
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

#### Text Transforms

```python
from neurenix.data import text_transforms

# Text transforms
text_transform = text_transforms.Compose([
    text_transforms.Tokenize(),
    text_transforms.RemoveStopwords(),
    text_transforms.Lowercase(),
    text_transforms.PadTruncate(max_length=512),
    text_transforms.ToTensor()
])
```

#### Time Series Transforms

```python
from neurenix.data import time_series_transforms

# Time series transforms
ts_transform = time_series_transforms.Compose([
    time_series_transforms.Resample(freq='1H'),
    time_series_transforms.Normalize(),
    time_series_transforms.SlidingWindow(window_size=24, stride=1),
    time_series_transforms.ToTensor()
])
```

### Data Augmentation

#### Image Augmentation

```python
from neurenix.data import transforms, RandAugment, AutoAugment

# RandAugment
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# AutoAugment
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    AutoAugment(policy='imagenet'),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

#### Advanced Augmentation

```python
from neurenix.data import Mixup, CutMix

# Mixup augmentation
mixup = Mixup(alpha=0.2)
inputs, targets = mixup(inputs, targets)

# CutMix augmentation
cutmix = CutMix(alpha=1.0)
inputs, targets = cutmix(inputs, targets)
```

### Data Management

#### Dataset Versioning

```python
from neurenix.data import DatasetVersion

# Create a versioned dataset
versioned_dataset = DatasetVersion(
    dataset=dataset,
    version="1.0.0",
    description="Initial version of the dataset",
    metadata={"source": "web scraping", "date": "2023-01-01"}
)

# Save the versioned dataset
versioned_dataset.save("/path/to/versioned/dataset")

# Load a specific version
loaded_dataset = DatasetVersion.load(
    path="/path/to/versioned/dataset",
    version="1.0.0"
)
```

#### Data Splitting

```python
from neurenix.data import random_split, stratified_split

# Random split
train_dataset, val_dataset, test_dataset = random_split(
    dataset=dataset,
    ratios=[0.7, 0.15, 0.15],
    seed=42
)

# Stratified split
train_dataset, val_dataset, test_dataset = stratified_split(
    dataset=dataset,
    target_column="label",
    ratios=[0.7, 0.15, 0.15],
    seed=42
)
```

#### Data Caching

```python
from neurenix.data import CachedDataset

# Create a cached dataset
cached_dataset = CachedDataset(
    dataset=dataset,
    cache_path="/path/to/cache",
    transform=transform,
    force_refresh=False
)
```

## Framework Comparison

### Neurenix Data vs. TensorFlow Data

| Feature | Neurenix Data | TensorFlow Data |
|---------|---------------|-----------------|
| API Design | Unified API for all data types | Separate APIs for different data types |
| Performance | Multi-language implementation for optimal performance | C++ implementation with Python bindings |
| Flexibility | Highly customizable with composable components | Somewhat rigid with predefined patterns |
| Hardware Acceleration | Comprehensive support for various hardware | Primarily optimized for TPUs and GPUs |
| Edge Device Support | Native support for edge devices | Limited through TensorFlow Lite |
| Streaming Data | First-class support for streaming datasets | Available through tf.data.Dataset |
| Data Versioning | Built-in versioning and management | Requires external tools |
| Multi-modal Data | Integrated support for multi-modal datasets | Requires custom implementation |

Neurenix Data provides a more unified and flexible API compared to TensorFlow Data, with better support for edge devices and multi-modal data. Its multi-language implementation ensures optimal performance across different hardware platforms.

### Neurenix Data vs. PyTorch Data

| Feature | Neurenix Data | PyTorch Data |
|---------|---------------|--------------|
| API Design | Comprehensive API with advanced features | Simple and flexible API |
| Performance | Multi-language implementation for optimal performance | Python implementation with C++ extensions |
| Preprocessing | Integrated preprocessing pipelines | Basic transforms with limited pipeline support |
| Hardware Acceleration | Support for various hardware accelerators | Primarily focused on CUDA |
| Data Management | Built-in versioning and management | Requires external tools |
| Streaming Data | Native support for streaming datasets | Limited support through IterableDataset |
| Distributed Data | Integrated with distributed training | Requires additional setup with DistributedSampler |
| Augmentation | Advanced augmentation techniques built-in | Basic augmentations with limited advanced options |

While PyTorch Data offers flexibility, Neurenix Data provides more comprehensive features for data management, preprocessing, and augmentation. Its multi-language implementation ensures better performance, especially for data-intensive tasks.

### Neurenix Data vs. Scikit-Learn Data

| Feature | Neurenix Data | Scikit-Learn Data |
|---------|---------------|-------------------|
| API Design | Comprehensive API for deep learning | Focused on traditional ML workflows |
| Performance | Multi-language implementation for optimal performance | Python implementation with limited C extensions |
| Hardware Acceleration | Support for various hardware accelerators | Limited hardware acceleration |
| Streaming Data | Native support for streaming datasets | Limited support through partial_fit |
| Data Augmentation | Advanced augmentation techniques | Limited augmentation capabilities |
| Preprocessing | Comprehensive preprocessing pipelines | Strong preprocessing for tabular data |
| Scalability | Scales to large datasets and distributed systems | Limited scalability for very large datasets |
| Integration | Seamless integration with deep learning models | Primarily designed for traditional ML models |

Scikit-Learn provides excellent tools for traditional machine learning workflows, but Neurenix Data offers more comprehensive features for deep learning applications, with better support for large datasets, hardware acceleration, and advanced augmentation techniques.

## Best Practices

### Efficient Data Loading

1. **Use Appropriate Batch Size**: Balance between memory usage and throughput.

```python
# Adjust batch size based on available memory and model complexity
dataloader = DataLoader(dataset, batch_size=32 if is_large_model else 128)
```

2. **Enable Prefetching**: Overlap data loading with computation.

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Adjust based on CPU cores
    prefetch_factor=2,  # Prefetch 2 batches per worker
    pin_memory=True  # Faster data transfer to GPU
)
```

3. **Use Memory Mapping for Large Datasets**: Avoid loading entire datasets into memory.

```python
from neurenix.data import MemmapDataset

dataset = MemmapDataset(
    data_path="/path/to/large/data.npy",
    shape=(10000, 3, 224, 224),
    dtype="float32"
)
```

4. **Implement Custom Collate Functions**: Optimize batch creation for specific data types.

```python
def custom_collate(batch):
    # Custom logic to create batches
    return processed_batch

dataloader = DataLoader(dataset, batch_size=32, collate_fn=custom_collate)
```

5. **Cache Preprocessed Data**: Avoid redundant preprocessing.

```python
from neurenix.data import CachedDataset

cached_dataset = CachedDataset(
    dataset=dataset,
    cache_path="/path/to/cache",
    transform=expensive_transform
)
```

### Effective Data Augmentation

1. **Balance Augmentation Strength**: Too much augmentation can hinder convergence.

```python
# Adjust augmentation strength based on dataset size
if len(dataset) < 10000:
    transform = strong_augmentation
else:
    transform = moderate_augmentation
```

2. **Use Different Augmentations for Different Phases**: Separate augmentations for training and validation.

```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
```

3. **Implement Adaptive Augmentation**: Adjust augmentation based on training progress.

```python
from neurenix.data import AdaptiveAugmentation

augmenter = AdaptiveAugmentation(
    initial_strength=0.5,
    adaptation_metric="validation_loss",
    increase_threshold=0.01,
    decrease_threshold=0.05
)

# During training
augmentation_strength = augmenter.update(validation_loss)
```

4. **Combine Multiple Augmentation Techniques**: Use complementary augmentations.

```python
from neurenix.data import transforms, Mixup, CutMix

# Combine spatial, color, and mixing augmentations
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

mixup = Mixup(alpha=0.2)
cutmix = CutMix(alpha=1.0)

# During training
if np.random.rand() < 0.5:
    inputs, targets = mixup(inputs, targets)
else:
    inputs, targets = cutmix(inputs, targets)
```

### Data Management Best Practices

1. **Version Control Your Datasets**: Track changes to datasets over time.

```python
from neurenix.data import DatasetVersion

# Create a new version when dataset changes
versioned_dataset = DatasetVersion(
    dataset=updated_dataset,
    version="1.1.0",
    description="Added 1000 new samples",
    metadata={"update_date": "2023-02-15"}
)
```

2. **Implement Data Validation**: Verify data quality before training.

```python
from neurenix.data import DataValidator

validator = DataValidator(
    checks=[
        "missing_values",
        "outliers",
        "class_imbalance",
        "data_leakage"
    ]
)

validation_results = validator.validate(dataset)
if not validation_results.is_valid:
    print(f"Data validation failed: {validation_results.issues}")
```

3. **Use Appropriate Data Formats**: Choose formats based on access patterns.

```python
# For random access patterns
from neurenix.data import HDF5Dataset

dataset = HDF5Dataset(
    file_path="/path/to/data.h5",
    dataset_name="images",
    transform=transform
)

# For sequential access patterns
from neurenix.data import TFRecordDataset

dataset = TFRecordDataset(
    file_pattern="/path/to/data-*.tfrecord",
    feature_description={
        "image": "image",
        "label": "int64"
    },
    transform=transform
)
```

## Tutorials

### Creating a Custom Dataset and DataLoader

```python
import neurenix
from neurenix.data import Dataset, DataLoader, transforms
import os
import numpy as np
from PIL import Image

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((
                        os.path.join(class_dir, filename),
                        self.class_to_idx[class_name]
                    ))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Create transforms
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = CustomImageDataset(
    root_dir="/path/to/train",
    transform=train_transform
)

val_dataset = CustomImageDataset(
    root_dir="/path/to/val",
    transform=val_transform
)

# Create data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Training loop
model = neurenix.nn.Sequential(...)
optimizer = neurenix.optim.Adam(model.parameters(), lr=0.001)
loss_fn = neurenix.nn.CrossEntropyLoss()

for epoch in range(10):
    # Training
    model.train()
    for inputs, targets in train_loader:
        inputs = inputs.to("cuda")
        targets = targets.to("cuda")
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
    
    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with neurenix.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to("cuda")
            targets = targets.to("cuda")
            
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            val_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    print(f"Epoch {epoch+1}, Val Loss: {val_loss/len(val_loader)}, Accuracy: {100*correct/total}%")
```

### Implementing Advanced Data Augmentation

```python
import neurenix
from neurenix.data import transforms, Mixup, CutMix, RandAugment
import numpy as np

# Create a strong augmentation pipeline
transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create dataset and data loader
train_dataset = neurenix.data.ImageFolder(
    root="/path/to/train",
    transform=transform
)

train_loader = neurenix.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Create mixing augmentations
mixup = Mixup(alpha=0.2)
cutmix = CutMix(alpha=1.0)

# Training loop with advanced augmentation
model = neurenix.nn.Sequential(...)
optimizer = neurenix.optim.Adam(model.parameters(), lr=0.001)
loss_fn = neurenix.nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for inputs, targets in train_loader:
        inputs = inputs.to("cuda")
        targets = targets.to("cuda")
        
        # Apply mixing augmentation with 50% probability
        if np.random.rand() < 0.5:
            if np.random.rand() < 0.5:
                inputs, targets = mixup(inputs, targets)
            else:
                inputs, targets = cutmix(inputs, targets)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        # For mixed targets, use soft targets
        if len(targets.shape) > 1:  # Mixed targets are one-hot encoded
            loss = neurenix.nn.cross_entropy_with_soft_targets(outputs, targets)
        else:
            loss = loss_fn(outputs, targets)
            
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1} completed")
```

### Working with Streaming Data

```python
import neurenix
from neurenix.data import StreamingDataset, DataLoader, transforms

# Create a streaming dataset for large-scale data
streaming_dataset = StreamingDataset(
    data_source="/path/to/large/dataset/*.tfrecord",
    feature_description={
        "image": "image",
        "label": "int64"
    },
    transform=transforms.Compose([
        transforms.Decode(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    shuffle_buffer_size=10000,
    prefetch_size=5,
    repeat=True
)

# Create a data loader with minimal buffering
loader = DataLoader(
    streaming_dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True,
    # No need for shuffle as the dataset handles it internally
    shuffle=False
)

# Training loop for streaming data
model = neurenix.nn.Sequential(...)
optimizer = neurenix.optim.Adam(model.parameters(), lr=0.001)
loss_fn = neurenix.nn.CrossEntropyLoss()

# Train for a fixed number of steps instead of epochs
steps_per_epoch = 1000
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for step, (inputs, targets) in enumerate(loader):
        if step >= steps_per_epoch:
            break
            
        inputs = inputs.to("cuda")
        targets = targets.to("cuda")
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item()}")
    
    print(f"Epoch {epoch+1} completed")
```

## Conclusion

The Data module in Neurenix provides a comprehensive set of tools for efficient data handling in machine learning workflows. Its multi-language architecture ensures optimal performance for data-intensive tasks, while the user-friendly Python API makes it accessible to researchers and developers.

Compared to other frameworks like TensorFlow, PyTorch, and Scikit-Learn, Neurenix Data offers advantages in terms of API design, performance, and flexibility. Its integrated support for data versioning, advanced augmentation techniques, and hardware acceleration makes it particularly well-suited for modern machine learning applications.

By following the best practices and tutorials outlined in this documentation, users can leverage the full power of the Neurenix Data module to improve the efficiency and effectiveness of their machine learning workflows.
