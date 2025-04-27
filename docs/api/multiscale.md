# Multi-Scale Models API Documentation

## Overview

The Multi-Scale Models module provides tools and techniques for working with data at multiple scales and resolutions, enabling more efficient and effective model training and inference. Multi-scale approaches are particularly valuable for tasks where features exist at different scales, such as image segmentation, object detection, and hierarchical data processing.

By processing data at multiple scales simultaneously, models can capture both fine-grained details and high-level structures, leading to improved performance across a wide range of tasks. This module implements various multi-scale architectures, pooling mechanisms, feature fusion techniques, and transformations to support multi-scale processing in Neurenix.

## Key Concepts

### Multi-Scale Architectures

Multi-scale architectures process data at different resolutions simultaneously:

- **Pyramid Networks**: Process data at different scales in a hierarchical manner
- **U-Net**: Encoder-decoder architecture with skip connections between corresponding scales
- **Feature Pyramid Networks**: Build feature pyramids with strong semantics at all scales

### Multi-Scale Pooling

Multi-scale pooling aggregates features across different scales:

- **Pyramid Pooling**: Pools features at multiple scales and concatenates them
- **Spatial Pyramid Pooling**: Partitions features into bins of different sizes
- **Multi-Scale Pooling**: Applies pooling operations with different kernel sizes

### Feature Fusion

Feature fusion combines features from different scales:

- **Scale Fusion**: Merges features from different scales
- **Attention Fusion**: Uses attention mechanisms to selectively combine features
- **Feature Fusion**: General techniques for combining features from different sources

### Multi-Scale Transforms

Multi-scale transforms modify data to create multi-scale representations:

- **Rescale**: Changes the resolution of data
- **Pyramid Downsample**: Creates a pyramid of downsampled versions of the data
- **Multi-Scale Transform**: General transformations for creating multi-scale representations

## API Reference

### Multi-Scale Models

```python
neurenix.multiscale.MultiScaleModel(
    scales: List[float] = [1.0, 0.5, 0.25],
    base_model: Optional[neurenix.nn.Module] = None,
    fusion_method: str = "concat"
)
```

Creates a model that processes input at multiple scales.

**Parameters:**
- `scales`: List of scales to process the input at
- `base_model`: Base model to apply at each scale
- `fusion_method`: Method to fuse features from different scales ("concat", "sum", "attention")

**Methods:**
- `forward(x)`: Process input at multiple scales and fuse the results
- `add_scale(scale)`: Add a new scale to the model
- `remove_scale(scale)`: Remove a scale from the model

**Example:**
```python
import neurenix as nx
from neurenix.nn import Conv2d, ReLU, Sequential
from neurenix.multiscale import MultiScaleModel

# Create a base model
base_model = Sequential(
    Conv2d(3, 64, kernel_size=3, padding=1),
    ReLU(),
    Conv2d(64, 64, kernel_size=3, padding=1),
    ReLU()
)

# Create a multi-scale model
model = MultiScaleModel(
    scales=[1.0, 0.5, 0.25],
    base_model=base_model,
    fusion_method="concat"
)

# Process input
input_tensor = nx.Tensor(shape=(1, 3, 224, 224))
output = model(input_tensor)
```

```python
neurenix.multiscale.PyramidNetwork(
    in_channels: int,
    out_channels: int,
    num_levels: int = 3,
    base_channels: int = 64
)
```

Creates a pyramid network that processes data at multiple levels.

**Parameters:**
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels
- `num_levels`: Number of levels in the pyramid
- `base_channels`: Number of channels in the first level

**Methods:**
- `forward(x)`: Process input through the pyramid network
- `get_features()`: Get features from all levels of the pyramid

```python
neurenix.multiscale.UNet(
    in_channels: int,
    out_channels: int,
    features: List[int] = [64, 128, 256, 512]
)
```

Creates a U-Net model for multi-scale processing with skip connections.

**Parameters:**
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels
- `features`: List of feature dimensions for each level

**Methods:**
- `forward(x)`: Process input through the U-Net
- `get_encoder_features()`: Get features from the encoder path
- `get_decoder_features()`: Get features from the decoder path

### Multi-Scale Pooling

```python
neurenix.multiscale.MultiScalePooling(
    pool_sizes: List[int] = [1, 2, 3, 6],
    pool_type: str = "avg"
)
```

Applies pooling at multiple scales and concatenates the results.

**Parameters:**
- `pool_sizes`: List of pooling kernel sizes
- `pool_type`: Type of pooling to apply ("avg", "max")

**Methods:**
- `forward(x)`: Apply multi-scale pooling to the input

```python
neurenix.multiscale.PyramidPooling(
    in_channels: int,
    out_channels: int,
    pool_sizes: List[int] = [1, 2, 3, 6]
)
```

Implements pyramid pooling module for capturing context at different scales.

**Parameters:**
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels
- `pool_sizes`: List of pooling sizes

**Methods:**
- `forward(x)`: Apply pyramid pooling to the input

```python
neurenix.multiscale.SpatialPyramidPooling(
    num_levels: int = 3,
    pool_type: str = "max"
)
```

Implements spatial pyramid pooling for fixed-size output regardless of input dimensions.

**Parameters:**
- `num_levels`: Number of levels in the pyramid
- `pool_type`: Type of pooling to apply ("avg", "max")

**Methods:**
- `forward(x)`: Apply spatial pyramid pooling to the input

### Feature Fusion

```python
neurenix.multiscale.FeatureFusion(
    in_channels: List[int],
    out_channels: int,
    fusion_method: str = "concat"
)
```

Fuses features from different sources.

**Parameters:**
- `in_channels`: List of input channel dimensions for each feature
- `out_channels`: Number of output channels
- `fusion_method`: Method to fuse features ("concat", "sum", "attention")

**Methods:**
- `forward(features)`: Fuse the input features

```python
neurenix.multiscale.ScaleFusion(
    scales: List[float],
    in_channels: int,
    out_channels: int,
    fusion_method: str = "concat"
)
```

Fuses features from different scales.

**Parameters:**
- `scales`: List of scales to fuse
- `in_channels`: Number of input channels
- `out_channels`: Number of output channels
- `fusion_method`: Method to fuse features ("concat", "sum", "attention")

**Methods:**
- `forward(features)`: Fuse the input features from different scales

```python
neurenix.multiscale.AttentionFusion(
    in_channels: List[int],
    out_channels: int
)
```

Fuses features using attention mechanisms.

**Parameters:**
- `in_channels`: List of input channel dimensions for each feature
- `out_channels`: Number of output channels

**Methods:**
- `forward(features)`: Fuse the input features using attention

### Multi-Scale Transforms

```python
neurenix.multiscale.MultiScaleTransform(
    scales: List[float] = [1.0, 0.5, 0.25],
    mode: str = "bilinear"
)
```

Transforms input to multiple scales.

**Parameters:**
- `scales`: List of scales to transform the input to
- `mode`: Interpolation mode ("nearest", "bilinear", "bicubic")

**Methods:**
- `forward(x)`: Transform the input to multiple scales
- `inverse(x, original_size)`: Transform back to the original scale

```python
neurenix.multiscale.Rescale(
    scale_factor: float = 0.5,
    mode: str = "bilinear"
)
```

Rescales input by a given factor.

**Parameters:**
- `scale_factor`: Factor to scale the input by
- `mode`: Interpolation mode ("nearest", "bilinear", "bicubic")

**Methods:**
- `forward(x)`: Rescale the input
- `inverse(x)`: Rescale back to the original size

```python
neurenix.multiscale.PyramidDownsample(
    num_levels: int = 3,
    mode: str = "bilinear"
)
```

Creates a pyramid of downsampled versions of the input.

**Parameters:**
- `num_levels`: Number of levels in the pyramid
- `mode`: Interpolation mode ("nearest", "bilinear", "bicubic")

**Methods:**
- `forward(x)`: Create a pyramid from the input
- `get_level(level)`: Get a specific level of the pyramid

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Multi-Scale Models** | Native support for various architectures | Limited built-in support |
| **Pyramid Networks** | Built-in implementation | Requires custom implementation |
| **U-Net** | Native implementation | Available in TF-Keras but less flexible |
| **Feature Fusion** | Multiple fusion methods | Limited built-in fusion methods |
| **Spatial Pyramid Pooling** | Native implementation | Requires custom implementation |
| **Multi-Scale Transforms** | Comprehensive transform API | Basic resize operations |

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Multi-Scale Models** | Native support for various architectures | Requires custom implementation |
| **Pyramid Networks** | Built-in implementation | Requires custom implementation |
| **U-Net** | Native implementation | Available in torchvision but less flexible |
| **Feature Fusion** | Multiple fusion methods | Requires custom implementation |
| **Spatial Pyramid Pooling** | Native implementation | Available in some libraries |
| **Multi-Scale Transforms** | Comprehensive transform API | Basic resize operations |

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Multi-Scale Models** | Comprehensive support | No support |
| **Pyramid Networks** | Built-in implementation | No support |
| **U-Net** | Native implementation | No support |
| **Feature Fusion** | Multiple fusion methods | No support |
| **Spatial Pyramid Pooling** | Native implementation | No support |
| **Multi-Scale Transforms** | Comprehensive transform API | Limited image processing |

## Best Practices

### Choosing Scales

When working with multi-scale models, choose appropriate scales:

```python
from neurenix.multiscale import MultiScaleModel

# For fine-grained tasks (e.g., segmentation), use more scales
fine_model = MultiScaleModel(
    scales=[1.0, 0.75, 0.5, 0.25],
    base_model=base_model,
    fusion_method="concat"
)

# For coarse tasks (e.g., classification), fewer scales may suffice
coarse_model = MultiScaleModel(
    scales=[1.0, 0.5],
    base_model=base_model,
    fusion_method="concat"
)
```

### Feature Fusion

Choose appropriate fusion methods based on your task:

```python
from neurenix.multiscale import FeatureFusion

# For preserving all information, use concatenation
concat_fusion = FeatureFusion(
    in_channels=[64, 128, 256],
    out_channels=512,
    fusion_method="concat"
)

# For lightweight models, use summation
sum_fusion = FeatureFusion(
    in_channels=[128, 128, 128],
    out_channels=128,
    fusion_method="sum"
)

# For selective feature combination, use attention
attention_fusion = FeatureFusion(
    in_channels=[64, 128, 256],
    out_channels=128,
    fusion_method="attention"
)
```

### Memory Efficiency

Optimize memory usage when working with multi-scale models:

```python
import neurenix as nx
from neurenix.multiscale import PyramidNetwork

# Create a memory-efficient pyramid network
model = PyramidNetwork(
    in_channels=3,
    out_channels=1,
    num_levels=4,
    base_channels=32  # Use smaller base channels for memory efficiency
)

# Process input with gradient checkpointing for memory efficiency
with nx.enable_gradient_checkpointing():
    output = model(input_tensor)
```

### Scale Selection

Adapt scale selection to the input data:

```python
import neurenix as nx
from neurenix.multiscale import MultiScaleModel

# For high-resolution images
high_res_model = MultiScaleModel(
    scales=[1.0, 0.5, 0.25, 0.125],
    base_model=base_model,
    fusion_method="concat"
)

# For low-resolution images
low_res_model = MultiScaleModel(
    scales=[1.0, 0.75, 0.5],
    base_model=base_model,
    fusion_method="concat"
)
```

## Tutorials

### Image Segmentation with U-Net

```python
import neurenix as nx
from neurenix.multiscale import UNet
from neurenix.nn import BCELoss
from neurenix.optim import Adam
from neurenix.data import DataLoader

# Create a U-Net model for segmentation
model = UNet(
    in_channels=3,
    out_channels=1,
    features=[64, 128, 256, 512]
)

# Define loss function and optimizer
criterion = BCELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Assume we have a DataLoader for segmentation data
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for batch_idx, (images, masks) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

# Save the trained model
nx.save(model, "unet_segmentation_model.nx")
```

### Multi-Scale Feature Extraction

```python
import neurenix as nx
from neurenix.multiscale import PyramidNetwork, FeatureFusion
from neurenix.nn import Sequential, Conv2d, ReLU, Linear

# Create a pyramid network for feature extraction
feature_extractor = PyramidNetwork(
    in_channels=3,
    out_channels=128,
    num_levels=4,
    base_channels=64
)

# Create a feature fusion module
fusion = FeatureFusion(
    in_channels=[64, 128, 256, 512],  # Channels at each level of the pyramid
    out_channels=256,
    fusion_method="attention"
)

# Create a classifier
classifier = Sequential(
    Linear(256, 128),
    ReLU(),
    Linear(128, 10)
)

# Define a forward function for the full model
def forward(x):
    # Extract features at multiple scales
    _ = feature_extractor(x)
    features = feature_extractor.get_features()
    
    # Fuse features from different scales
    fused_features = fusion(features)
    
    # Global average pooling
    pooled_features = nx.mean(fused_features, dim=[2, 3])
    
    # Classify
    output = classifier(pooled_features)
    return output

# Process an input image
input_image = nx.Tensor(shape=(1, 3, 224, 224))
output = forward(input_image)
```
