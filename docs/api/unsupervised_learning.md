# Unsupervised Learning Documentation

## Overview

The Unsupervised Learning module in Neurenix provides tools and utilities for learning patterns and representations from unlabeled data. Unsupervised learning is a powerful approach for discovering hidden structures in data without the need for explicit labels or supervision.

Neurenix's unsupervised learning capabilities are implemented using a multi-language architecture, where the high-performance operations are implemented in the Rust/C++ Phynexus engine, while the Python API provides a user-friendly interface. This architecture enables Neurenix to deliver optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

## Key Concepts

### Autoencoders

Autoencoders are neural networks that learn to compress data into a lower-dimensional representation (encoding) and then reconstruct it (decoding). They are useful for dimensionality reduction, feature learning, and generative modeling. Neurenix provides several types of autoencoders:

- **Standard Autoencoders**: Learn to compress and reconstruct data
- **Variational Autoencoders (VAEs)**: Learn a probabilistic mapping between the input space and a latent space
- **Denoising Autoencoders**: Learn to reconstruct clean inputs from corrupted ones

### Clustering

Clustering algorithms group similar data points together based on their features or characteristics. Neurenix provides several clustering algorithms:

- **K-Means**: Partitions data into k clusters by minimizing the sum of squared distances between data points and their assigned cluster centers
- **DBSCAN**: Groups together points that are close to each other and marks points in low-density regions as outliers
- **Spectral Clustering**: Uses the eigenvalues of a similarity matrix to reduce dimensionality before clustering in fewer dimensions

### Dimensionality Reduction

Dimensionality reduction techniques transform high-dimensional data into a lower-dimensional representation while preserving important properties of the original data. Neurenix provides several dimensionality reduction algorithms:

- **PCA (Principal Component Analysis)**: Finds the directions of maximum variance in the data
- **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: Preserves local relationships in the data
- **UMAP (Uniform Manifold Approximation and Projection)**: Preserves both local and global structure in the data

### Contrastive Learning

Contrastive learning is a self-supervised learning technique that learns representations by contrasting positive pairs against negative pairs. Neurenix provides several contrastive learning algorithms:

- **SimCLR**: Learns representations by maximizing agreement between differently augmented views of the same data example
- **BYOL**: Learns representations by predicting the representations of one augmented view from another augmented view
- **MoCo**: Learns representations by matching an encoded query to a dictionary of encoded keys using a contrastive loss

## API Reference

### Autoencoders

#### Autoencoder

```python
neurenix.unsupervised.Autoencoder(input_dim, hidden_dims, latent_dim, activation='relu')
```

Basic autoencoder implementation that compresses the input data into a lower-dimensional latent space and then reconstructs the original input.

**Parameters:**
- `input_dim`: Dimensionality of the input data
- `hidden_dims`: List of hidden layer dimensions for the encoder and decoder
- `latent_dim`: Dimensionality of the latent space
- `activation`: Activation function to use ('relu', 'sigmoid', or 'tanh')

**Methods:**
- `encode(x)`: Encode input data into the latent space
- `decode(z)`: Decode latent representation back to the input space
- `forward(x)`: Forward pass through the autoencoder
- `get_latent_representation(x)`: Get the latent representation for input data

#### VAE (Variational Autoencoder)

```python
neurenix.unsupervised.VAE(input_dim, hidden_dims, latent_dim, activation='relu')
```

Variational Autoencoder implementation that learns a probabilistic mapping between the input space and a latent space, allowing for generative modeling.

**Parameters:**
- `input_dim`: Dimensionality of the input data
- `hidden_dims`: List of hidden layer dimensions for the encoder and decoder
- `latent_dim`: Dimensionality of the latent space
- `activation`: Activation function to use ('relu', 'sigmoid', or 'tanh')

**Methods:**
- `encode(x)`: Encode input data into the latent space, returning mean and log variance
- `reparameterize(mu, logvar)`: Sample from the latent distribution using the reparameterization trick
- `decode(z)`: Decode latent representation back to the input space
- `forward(x)`: Forward pass through the VAE, returning reconstructed input, mean, and log variance
- `loss_function(recon_x, x, mu, logvar, beta=1.0)`: Compute the VAE loss function
- `sample(num_samples, device=None)`: Generate samples from the latent space

#### DenoisingAutoencoder

```python
neurenix.unsupervised.DenoisingAutoencoder(input_dim, hidden_dims, latent_dim, noise_factor=0.3, activation='relu')
```

Denoising Autoencoder implementation that is trained to reconstruct clean inputs from corrupted ones, making it more robust and better at learning useful features.

**Parameters:**
- `input_dim`: Dimensionality of the input data
- `hidden_dims`: List of hidden layer dimensions for the encoder and decoder
- `latent_dim`: Dimensionality of the latent space
- `noise_factor`: Amount of noise to add to the input during training
- `activation`: Activation function to use ('relu', 'sigmoid', or 'tanh')

**Methods:**
- `add_noise(x)`: Add noise to the input
- `forward(x, add_noise=True)`: Forward pass through the denoising autoencoder

### Clustering

#### KMeans

```python
neurenix.unsupervised.KMeans(n_clusters=8, max_iter=300, tol=1e-4, random_state=None)
```

K-Means clustering algorithm implementation that partitions data into k clusters by minimizing the sum of squared distances between data points and their assigned cluster centers.

**Parameters:**
- `n_clusters`: Number of clusters to form
- `max_iter`: Maximum number of iterations
- `tol`: Tolerance for convergence
- `random_state`: Random seed for reproducibility

**Methods:**
- `fit(X)`: Fit K-Means clustering on the data
- `predict(X)`: Predict the closest cluster for each sample in X
- `fit_predict(X)`: Fit the model and predict cluster labels

**Attributes:**
- `cluster_centers_`: Coordinates of cluster centers
- `labels_`: Labels of each point
- `inertia_`: Sum of squared distances to closest centroid

#### DBSCAN

```python
neurenix.unsupervised.DBSCAN(eps=0.5, min_samples=5, metric='euclidean')
```

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) implementation that groups together points that are close to each other and marks points in low-density regions as outliers.

**Parameters:**
- `eps`: Maximum distance between two samples for them to be considered neighbors
- `min_samples`: Minimum number of samples in a neighborhood for a point to be a core point
- `metric`: Distance metric to use ('euclidean' or 'cosine')

**Methods:**
- `fit(X)`: Fit DBSCAN clustering on the data
- `fit_predict(X)`: Fit the model and predict cluster labels

**Attributes:**
- `labels_`: Labels of each point (-1 for noise points)
- `core_sample_indices_`: Indices of core samples

#### SpectralClustering

```python
neurenix.unsupervised.SpectralClustering(n_clusters=8, affinity='rbf', gamma=1.0, random_state=None)
```

Spectral Clustering implementation that uses the eigenvalues of a similarity matrix to reduce dimensionality before clustering in fewer dimensions.

**Parameters:**
- `n_clusters`: Number of clusters to form
- `affinity`: How to construct the affinity matrix ('rbf' or 'nearest_neighbors')
- `gamma`: Kernel coefficient for RBF kernel
- `random_state`: Random seed for reproducibility

**Methods:**
- `fit(X)`: Fit Spectral Clustering on the data
- `fit_predict(X)`: Fit the model and predict cluster labels

**Attributes:**
- `labels_`: Labels of each point
- `affinity_matrix_`: Affinity matrix used for clustering

### Dimensionality Reduction

#### PCA (Principal Component Analysis)

```python
neurenix.unsupervised.PCA(n_components=None, whiten=False, random_state=None)
```

Principal Component Analysis (PCA) implementation that finds the directions of maximum variance in the data and projects the data onto a lower-dimensional subspace.

**Parameters:**
- `n_components`: Number of components to keep. If None, keep all components.
- `whiten`: Whether to whiten the data (scale to unit variance)
- `random_state`: Random seed for reproducibility

**Methods:**
- `fit(X)`: Fit PCA on the data
- `transform(X)`: Apply dimensionality reduction to X
- `fit_transform(X)`: Fit the model with X and apply dimensionality reduction
- `inverse_transform(X)`: Transform data back to its original space

**Attributes:**
- `components_`: Principal axes in feature space
- `explained_variance_`: Amount of variance explained by each component
- `explained_variance_ratio_`: Percentage of variance explained by each component

#### TSNE (t-Distributed Stochastic Neighbor Embedding)

```python
neurenix.unsupervised.TSNE(n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000, random_state=None)
```

t-Distributed Stochastic Neighbor Embedding (t-SNE) implementation that is particularly well-suited for visualizing high-dimensional data in a low-dimensional space.

**Parameters:**
- `n_components`: Dimension of the embedded space
- `perplexity`: Related to the number of nearest neighbors used in manifold learning
- `learning_rate`: Learning rate for gradient descent
- `n_iter`: Number of iterations for optimization
- `random_state`: Random seed for reproducibility

**Methods:**
- `fit_transform(X)`: Fit t-SNE on the data and return the embedding

**Attributes:**
- `embedding_`: Embedding of the training data

#### UMAP (Uniform Manifold Approximation and Projection)

```python
neurenix.unsupervised.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=None)
```

Uniform Manifold Approximation and Projection (UMAP) implementation that can be used for visualization and general non-linear dimension reduction.

**Parameters:**
- `n_components`: Dimension of the embedded space
- `n_neighbors`: Number of neighbors to consider for each point
- `min_dist`: Minimum distance between points in the embedding
- `metric`: Distance metric to use
- `random_state`: Random seed for reproducibility

**Methods:**
- `fit_transform(X)`: Fit UMAP on the data and return the embedding

**Attributes:**
- `embedding_`: Embedding of the training data

### Contrastive Learning

#### SimCLR (Simple Contrastive Learning of Representations)

```python
neurenix.unsupervised.SimCLR(encoder, projection_dim=128, temperature=0.5)
```

Simple Contrastive Learning of Representations (SimCLR) implementation that learns representations by maximizing agreement between differently augmented views of the same data example.

**Parameters:**
- `encoder`: Base encoder network
- `projection_dim`: Dimensionality of the projection head output
- `temperature`: Temperature parameter for the contrastive loss

**Methods:**
- `forward(x)`: Forward pass through the SimCLR model
- `contrastive_loss(z_i, z_j)`: Compute the contrastive loss between two sets of projected features

#### BYOL (Bootstrap Your Own Latent)

```python
neurenix.unsupervised.BYOL(encoder, projection_dim=256, hidden_dim=4096, momentum=0.99)
```

Bootstrap Your Own Latent (BYOL) implementation that learns representations by predicting the representations of one augmented view from another augmented view of the same image.

**Parameters:**
- `encoder`: Base encoder network
- `projection_dim`: Dimensionality of the projection head output
- `hidden_dim`: Dimensionality of the hidden layer in the projection head
- `momentum`: Momentum for updating the target network

**Methods:**
- `forward(x)`: Forward pass through the BYOL model, returning online prediction and target projection
- `loss_function(online_prediction, target_projection)`: Compute the BYOL loss
- `update_target()`: Update the target network after each training step

#### MoCo (Momentum Contrast)

```python
neurenix.unsupervised.MoCo(encoder, dim=128, K=65536, m=0.999, T=0.07)
```

Momentum Contrast (MoCo) implementation that learns representations by matching an encoded query to a dictionary of encoded keys using a contrastive loss.

**Parameters:**
- `encoder`: Base encoder network
- `dim`: Feature dimension
- `K`: Queue size
- `m`: Momentum for updating the key encoder
- `T`: Temperature for the contrastive loss

**Methods:**
- `forward(im_q, im_k)`: Forward pass through the MoCo model, returning logits, labels, and queue
- `contrastive_loss(logits, labels)`: Compute the contrastive loss

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Autoencoder API** | Unified API with standard, variational, and denoising autoencoders | Requires custom implementation or third-party libraries |
| **Clustering Algorithms** | Built-in KMeans, DBSCAN, and Spectral Clustering | Limited clustering support through TensorFlow Probability |
| **Dimensionality Reduction** | Native PCA, t-SNE, and UMAP implementations | Limited support through TensorFlow Extended (TFX) |
| **Contrastive Learning** | Built-in SimCLR, BYOL, and MoCo implementations | Available through third-party libraries |
| **Edge Device Support** | Native optimization for edge devices | TensorFlow Lite for edge devices |
| **Hardware Acceleration** | Multi-device support (CPU, CUDA, ROCm, WebGPU) | Primarily optimized for CPU and CUDA |

Neurenix's unsupervised learning capabilities offer a more unified and integrated approach compared to TensorFlow, which often requires custom implementations or third-party libraries for many unsupervised learning algorithms. The native implementation of multiple unsupervised learning algorithms in Neurenix provides a consistent API and seamless integration with other framework components. Additionally, Neurenix's multi-language architecture and edge device optimization make it particularly well-suited for deploying unsupervised learning models in resource-constrained environments.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Autoencoder API** | Unified API with standard, variational, and denoising autoencoders | Requires custom implementation |
| **Clustering Algorithms** | Built-in KMeans, DBSCAN, and Spectral Clustering | No built-in clustering (requires scikit-learn integration) |
| **Dimensionality Reduction** | Native PCA, t-SNE, and UMAP implementations | Limited built-in support (requires scikit-learn integration) |
| **Contrastive Learning** | Built-in SimCLR, BYOL, and MoCo implementations | Available through third-party libraries |
| **Hardware Support** | CPU, CUDA, ROCm, WebGPU | CPU, CUDA |
| **Edge Device Optimization** | Native optimization for edge devices | PyTorch Mobile for edge devices |

Neurenix provides a more comprehensive and integrated unsupervised learning solution compared to PyTorch, which requires custom implementations or third-party libraries for many unsupervised learning algorithms. While PyTorch's dynamic computation graph makes it flexible for implementing custom unsupervised learning algorithms, Neurenix's built-in implementations offer a more streamlined experience with less boilerplate code. Neurenix also extends hardware support to include ROCm and WebGPU, making it more versatile across different hardware platforms.

### Neurenix vs. Scikit-Learn

| Feature | Neurenix | Scikit-Learn |
|---------|----------|--------------|
| **Neural Network Support** | Full neural network architecture | Limited neural network support |
| **Autoencoder Support** | Standard, variational, and denoising autoencoders | No built-in autoencoder support |
| **Clustering Algorithms** | KMeans, DBSCAN, and Spectral Clustering with tensor support | Comprehensive clustering algorithms but limited to NumPy arrays |
| **Dimensionality Reduction** | PCA, t-SNE, and UMAP with tensor support | Comprehensive dimensionality reduction but limited to NumPy arrays |
| **Contrastive Learning** | Built-in SimCLR, BYOL, and MoCo implementations | No built-in contrastive learning support |
| **GPU Acceleration** | Native support for multiple GPU types | No native GPU support |
| **Edge Device Support** | Native optimization for edge devices | No specific edge device support |

Neurenix provides a more comprehensive unsupervised learning solution for deep learning applications compared to Scikit-Learn, which excels at traditional machine learning algorithms but lacks built-in support for modern deep learning-based unsupervised learning techniques like autoencoders and contrastive learning. While Scikit-Learn offers a wide range of clustering and dimensionality reduction algorithms, Neurenix's implementations are designed to work seamlessly with tensors and neural networks, and they benefit from GPU acceleration and edge device optimization.

## Best Practices

### Choosing the Right Unsupervised Learning Algorithm

Different unsupervised learning algorithms have different strengths and weaknesses:

1. **Autoencoders**:
   - Use standard autoencoders for dimensionality reduction and feature learning
   - Use variational autoencoders (VAEs) for generative modeling and probabilistic representations
   - Use denoising autoencoders for learning robust features and noise reduction

2. **Clustering**:
   - Use K-Means for well-separated, spherical clusters and when the number of clusters is known
   - Use DBSCAN for irregularly shaped clusters and when the number of clusters is unknown
   - Use Spectral Clustering for complex, non-linear cluster boundaries

3. **Dimensionality Reduction**:
   - Use PCA for linear dimensionality reduction and when preserving global structure is important
   - Use t-SNE for visualization and when preserving local structure is important
   - Use UMAP for a balance between preserving local and global structure

4. **Contrastive Learning**:
   - Use SimCLR for simplicity and when you have strong data augmentation
   - Use BYOL when you want to avoid negative examples
   - Use MoCo when you need a large number of negative examples

### Optimizing for Edge Devices

When deploying unsupervised learning models to edge devices, consider these optimizations:

1. **Model Size**: Use smaller models with fewer parameters
2. **Quantization**: Quantize model weights to reduce memory usage
3. **Pruning**: Remove unnecessary connections in neural networks
4. **Efficient Architectures**: Use architectures specifically designed for edge devices
5. **Batch Processing**: Process data in small batches to reduce memory usage

## Tutorials

### Dimensionality Reduction with PCA

```python
import neurenix
from neurenix.unsupervised import PCA
import matplotlib.pyplot as plt

# Load data
# For example, MNIST digits (28x28 = 784 dimensions)
X = neurenix.Tensor(...)  # Shape: (n_samples, 784)

# Create a PCA model
pca = PCA(n_components=2)

# Fit the model and transform the data
X_reduced = pca.fit_transform(X)

# Plot the results
plt.figure(figsize=(10, 8))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], alpha=0.5)
plt.title('PCA of MNIST Digits')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

# Check explained variance
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

### Image Reconstruction with Variational Autoencoder

```python
import neurenix
from neurenix.nn import Module, Sequential, Linear, ReLU, Sigmoid
from neurenix.unsupervised import VAE
from neurenix.optim import Adam
import matplotlib.pyplot as plt

# Load data
# For example, MNIST digits (28x28 = 784 dimensions)
X_train = neurenix.Tensor(...)  # Shape: (n_samples, 784)

# Create a VAE model
vae = VAE(
    input_dim=784,
    hidden_dims=[256, 128],
    latent_dim=20,
    activation='relu'
)

# Create an optimizer
optimizer = Adam(vae.parameters(), lr=0.001)

# Train the VAE
num_epochs = 50
batch_size = 128

for epoch in range(num_epochs):
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    # Training loop
    for i in range(0, len(X_train), batch_size):
        batch_x = X_train[i:i+batch_size]
        
        # Forward pass
        recon_x, mu, logvar = vae(batch_x)
        
        # Compute loss
        loss = vae.loss_function(recon_x, batch_x, mu, logvar, beta=1.0)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.shape[0]
        total_recon_loss += vae.reconstruction_loss * batch_x.shape[0]
        total_kl_loss += vae.kl_divergence * batch_x.shape[0]
    
    # Print epoch statistics
    avg_loss = total_loss / len(X_train)
    avg_recon_loss = total_recon_loss / len(X_train)
    avg_kl_loss = total_kl_loss / len(X_train)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}")

# Generate new samples
num_samples = 16
samples = vae.sample(num_samples)

# Reshape samples for visualization
samples = samples.reshape(num_samples, 28, 28)

# Plot the generated samples
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(samples[i], cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

### Self-Supervised Learning with SimCLR

```python
import neurenix
from neurenix.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear, ReLU
from neurenix.unsupervised import SimCLR
from neurenix.optim import Adam
from neurenix.data import DataLoader, RandomResizedCrop, ColorJitter, RandomHorizontalFlip, ToTensor, Normalize

# Define data augmentation
def get_augmentation():
    return Sequential([
        RandomResizedCrop(size=32),
        RandomHorizontalFlip(),
        ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
        ToTensor(),
        Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    ])

# Create a base encoder network
def get_encoder():
    return Sequential([
        Conv2d(3, 32, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(2),
        Conv2d(32, 64, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(2),
        Conv2d(64, 128, kernel_size=3, padding=1),
        ReLU(),
        MaxPool2d(2),
        Flatten(),
        Linear(128 * 4 * 4, 512),
        ReLU(),
    ])

# Create a SimCLR model
simclr = SimCLR(
    encoder=get_encoder(),
    projection_dim=128,
    temperature=0.5
)

# Create an optimizer
optimizer = Adam(simclr.parameters(), lr=0.001)

# Load data
# For example, CIFAR-10
train_dataset = ...  # CIFAR-10 dataset
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

# Train the SimCLR model
num_epochs = 100

for epoch in range(num_epochs):
    total_loss = 0.0
    
    for batch_x, _ in train_loader:  # Ignore labels in self-supervised learning
        # Generate two augmented views
        augmentation = get_augmentation()
        x_i = augmentation(batch_x)
        x_j = augmentation(batch_x)
        
        # Forward pass
        z_i = simclr(x_i)
        z_j = simclr(x_j)
        
        # Compute contrastive loss
        loss = simclr.contrastive_loss(z_i, z_j)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.shape[0]
    
    # Print epoch statistics
    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save the encoder for downstream tasks
encoder = simclr.encoder
neurenix.save(encoder, "simclr_encoder.pt")
```

## Conclusion

The Unsupervised Learning module of Neurenix provides a comprehensive set of tools for learning patterns and representations from unlabeled data. Its multi-language architecture with a high-performance Rust/C++ core enables optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

Compared to other frameworks like TensorFlow, PyTorch, and Scikit-Learn, Neurenix's Unsupervised Learning module offers advantages in terms of API design, hardware support, and edge device optimization. The unified API and implementations of multiple unsupervised learning algorithms (autoencoders, clustering, dimensionality reduction, and contrastive learning) provide a consistent and integrated experience, making Neurenix particularly well-suited for unsupervised learning tasks and AI agent development.
