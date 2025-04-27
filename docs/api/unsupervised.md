# Unsupervised Learning Module

## Overview

The Unsupervised Learning module in Neurenix provides a comprehensive suite of algorithms and tools for learning patterns and structures from unlabeled data. Built on Neurenix's high-performance multi-language architecture, this module delivers efficient implementations of classical and state-of-the-art unsupervised learning techniques.

The module features a unified interface that works seamlessly with Neurenix's tensor and neural network modules, enabling users to easily switch between different unsupervised learning algorithms without changing their data processing code. It supports various unsupervised learning paradigms, including dimensionality reduction, clustering, density estimation, generative modeling, and self-supervised learning.

Implemented with a combination of Rust and C++ for performance-critical components and Python for the user-friendly interface, the Unsupervised Learning module ensures both computational efficiency and ease of use. It provides native support for various hardware accelerators, including GPUs, TPUs, and specialized AI hardware, with automatic optimization for the available hardware.

## Key Concepts

### Dimensionality Reduction

The module includes various dimensionality reduction techniques for visualizing and preprocessing high-dimensional data:

- **Linear Methods**: Techniques based on linear transformations
  - Principal Component Analysis (PCA): Finding orthogonal directions of maximum variance
  - Linear Discriminant Analysis (LDA): Finding directions that maximize class separability
  - Factor Analysis (FA): Modeling observed variables as linear combinations of factors

- **Manifold Learning**: Techniques for non-linear dimensionality reduction
  - t-Distributed Stochastic Neighbor Embedding (t-SNE): Preserving local similarities in low dimensions
  - Uniform Manifold Approximation and Projection (UMAP): Preserving both local and global structure
  - Isomap: Preserving geodesic distances in low dimensions

- **Autoencoder-Based Methods**: Neural network approaches to dimensionality reduction
  - Vanilla Autoencoders: Encoding and decoding data through a bottleneck
  - Variational Autoencoders (VAEs): Probabilistic autoencoders with regularized latent space
  - Sparse Autoencoders: Autoencoders with sparsity constraints on activations

### Clustering

The module includes various clustering algorithms for grouping similar data points:

- **Centroid-Based Methods**: Techniques based on cluster centers
  - K-Means: Partitioning data into k clusters with minimum within-cluster variance
  - K-Medoids: Using actual data points as cluster centers
  - Mean Shift: Finding modes of the underlying density function

- **Connectivity-Based Methods**: Techniques based on connectivity between data points
  - Hierarchical Clustering: Building a hierarchy of clusters
  - DBSCAN: Density-based spatial clustering of applications with noise
  - Spectral Clustering: Clustering based on the spectrum of the similarity matrix

- **Density-Based Methods**: Techniques based on density estimation
  - DBSCAN: Finding regions of high density separated by regions of low density
  - HDBSCAN: Hierarchical DBSCAN with varying density thresholds
  - DENCLUE: Clustering based on density distribution functions

### Generative Models

The module includes various generative models for learning data distributions and generating new samples:

- **Classical Generative Models**: Traditional approaches to generative modeling
  - Gaussian Mixture Models (GMMs): Modeling data as a mixture of Gaussian distributions
  - Hidden Markov Models (HMMs): Modeling sequential data with hidden states
  - Restricted Boltzmann Machines (RBMs): Energy-based models with visible and hidden units

- **Deep Generative Models**: Neural network approaches to generative modeling
  - Variational Autoencoders (VAEs): Probabilistic autoencoders with regularized latent space
  - Generative Adversarial Networks (GANs): Adversarial training of generator and discriminator
  - Flow-Based Models: Invertible transformations for exact likelihood computation
  - Diffusion Models: Gradually adding and removing noise for generation

### Self-Supervised Learning

The module includes various self-supervised learning techniques for learning representations without labeled data:

- **Contrastive Learning**: Learning representations by contrasting positive and negative pairs
  - SimCLR: Simple framework for contrastive learning of visual representations
  - MoCo: Momentum contrast for unsupervised visual representation learning
  - BYOL: Bootstrap your own latent for self-supervised learning

- **Predictive Learning**: Learning representations by predicting parts of the data
  - Masked Language Modeling (MLM): Predicting masked tokens in text
  - Masked Image Modeling (MIM): Predicting masked patches in images
  - Next Token Prediction: Predicting the next token in a sequence

## API Reference

### Dimensionality Reduction

```python
import neurenix
from neurenix.unsupervised import PCA, TSNE, UMAP, Autoencoder

# Principal Component Analysis
pca = PCA(
    n_components=2,                    # Number of components to keep
    whiten=False,                      # Whether to whiten the data
    svd_solver="auto",                 # SVD solver to use
    random_state=42                    # Random state for reproducibility
)

# Fit PCA to data
pca.fit(X_train)

# Transform data
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# t-SNE
tsne = TSNE(
    n_components=2,                    # Number of components to keep
    perplexity=30.0,                   # Perplexity parameter
    early_exaggeration=12.0,           # Early exaggeration factor
    learning_rate="auto",              # Learning rate
    n_iter=1000,                       # Number of iterations
    random_state=42                    # Random state for reproducibility
)

# Fit and transform data with t-SNE
X_tsne = tsne.fit_transform(X)
```

### Clustering

```python
from neurenix.unsupervised import KMeans, DBSCAN, SpectralClustering, GaussianMixture

# K-Means
kmeans = KMeans(
    n_clusters=5,                      # Number of clusters
    init="k-means++",                  # Initialization method
    n_init=10,                         # Number of initializations
    max_iter=300,                      # Maximum number of iterations
    tol=1e-4,                          # Tolerance for convergence
    random_state=42                    # Random state for reproducibility
)

# Fit K-Means to data
kmeans.fit(X)

# Get cluster assignments
labels = kmeans.predict(X)

# Get cluster centers
centers = kmeans.cluster_centers_

# DBSCAN
dbscan = DBSCAN(
    eps=0.5,                           # Maximum distance between points
    min_samples=5,                     # Minimum number of samples in a neighborhood
    metric="euclidean",                # Distance metric
    algorithm="auto",                  # Algorithm to use
    n_jobs=-1                          # Number of jobs to run in parallel
)

# Fit DBSCAN to data
dbscan.fit(X)

# Get cluster assignments
labels = dbscan.labels_
```

### Generative Models

```python
from neurenix.unsupervised import VAE, GAN, NormalizingFlow, DiffusionModel

# Variational Autoencoder
vae = VAE(
    input_dim=X.shape[1],              # Input dimension
    hidden_dims=[128, 64],             # Hidden layer dimensions
    latent_dim=10,                     # Latent dimension
    activation="relu",                 # Activation function
    dropout_rate=0.2,                  # Dropout rate
    learning_rate=0.001,               # Learning rate
    weight_decay=1e-5,                 # Weight decay
    optimizer="adam",                  # Optimizer
    reconstruction_loss="mse",         # Reconstruction loss function
    kl_weight=1.0,                     # Weight for KL divergence term
    device="cuda" if neurenix.cuda.is_available() else "cpu"  # Device to use
)

# Fit VAE to data
vae.fit(
    X_train=X_train,
    X_val=X_val,
    batch_size=32,
    epochs=100,
    callbacks=[early_stopping, model_checkpoint]
)

# Generate samples
samples = vae.generate(n_samples=100)
```

## Framework Comparison

### Neurenix Unsupervised Learning vs. TensorFlow Unsupervised Learning

| Feature | Neurenix Unsupervised Learning | TensorFlow Unsupervised Learning |
|---------|--------------------------------|----------------------------------|
| Performance | Multi-language implementation with Rust/C++ backends | Python implementation with TensorFlow backend |
| Hardware Support | Comprehensive support for various hardware | Primarily focused on TPUs and GPUs |
| Algorithm Variety | Extensive collection of classical and modern algorithms | Good selection of common algorithms |
| Dimensionality Reduction | Comprehensive support for various techniques | Limited built-in support |
| Clustering | Comprehensive support for various algorithms | Limited built-in support |
| Generative Models | Extensive collection of generative models | Good support for common generative models |
| Self-Supervised Learning | Comprehensive support for various techniques | Limited built-in support |
| Edge Device Support | Native support for edge devices | Limited through TensorFlow Lite |

Neurenix's Unsupervised Learning module provides better performance through its multi-language implementation and offers more comprehensive hardware support, especially for edge devices. It also provides a wider variety of unsupervised learning algorithms, more advanced dimensionality reduction and clustering techniques, and better support for self-supervised learning and anomaly detection.

### Neurenix Unsupervised Learning vs. PyTorch Unsupervised Learning

| Feature | Neurenix Unsupervised Learning | PyTorch Unsupervised Learning |
|---------|--------------------------------|-------------------------------|
| Performance | Multi-language implementation with Rust/C++ backends | Python implementation with PyTorch backend |
| Hardware Support | Comprehensive support for various hardware | Primarily focused on CUDA devices |
| Algorithm Variety | Extensive collection of classical and modern algorithms | Good selection through separate libraries |
| Dimensionality Reduction | Comprehensive support for various techniques | Limited built-in support |
| Clustering | Comprehensive support for various algorithms | Limited built-in support |
| Generative Models | Extensive collection of generative models | Good support through separate libraries |
| Self-Supervised Learning | Comprehensive support for various techniques | Good support through separate libraries |
| Edge Device Support | Native support for edge devices | Limited through separate tools |

While PyTorch has good unsupervised learning capabilities through various libraries, Neurenix's Unsupervised Learning module offers better performance through its multi-language implementation and provides more comprehensive hardware support, especially for edge devices. It also offers more integrated support for dimensionality reduction, clustering, and anomaly detection.

### Neurenix Unsupervised Learning vs. Scikit-Learn Unsupervised Learning

| Feature | Neurenix Unsupervised Learning | Scikit-Learn Unsupervised Learning |
|---------|--------------------------------|------------------------------------|
| Performance | Multi-language implementation with Rust/C++ backends | Python implementation with limited C++ backends |
| Hardware Acceleration | Native support for various hardware accelerators | Limited hardware acceleration |
| Algorithm Variety | Extensive collection of classical and modern algorithms | Good selection of classical algorithms |
| Deep Learning Integration | Seamless integration with deep learning | Limited deep learning integration |
| Dimensionality Reduction | Comprehensive support for various techniques | Good support for classical techniques |
| Clustering | Comprehensive support for various algorithms | Good support for classical algorithms |
| Generative Models | Extensive collection of generative models | Limited support for generative models |
| Edge Device Support | Native support for edge devices | Limited edge support |

Scikit-Learn has good support for classical unsupervised learning algorithms, but Neurenix's Unsupervised Learning module offers better performance through its multi-language implementation and provides more comprehensive hardware support. It also offers better integration with deep learning, more advanced generative models, and better support for self-supervised learning.

## Best Practices

### Choosing the Right Algorithm

1. **Consider Data Characteristics**: Choose algorithms based on data characteristics.

```python
# For high-dimensional data, use dimensionality reduction first
if X.shape[1] > 100:
    # Use PCA for linear relationships
    if linear_relationships:
        pca = neurenix.unsupervised.PCA(n_components=min(50, X.shape[1]))
        X_reduced = pca.fit_transform(X)
    # Use t-SNE or UMAP for non-linear relationships
    else:
        umap = neurenix.unsupervised.UMAP(n_components=2)
        X_reduced = umap.fit_transform(X)
    
    # Use the reduced data for further analysis
    X = X_reduced
```

2. **Consider Task Requirements**: Choose algorithms based on task requirements.

```python
# For visualization, use dimensionality reduction
if task == "visualization":
    # For preserving global structure
    if preserve_global_structure:
        pca = neurenix.unsupervised.PCA(n_components=2)
        X_viz = pca.fit_transform(X)
    # For preserving local structure
    else:
        tsne = neurenix.unsupervised.TSNE(n_components=2)
        X_viz = tsne.fit_transform(X)
```

### Data Preprocessing

1. **Scale Features**: Scale features to ensure equal contribution.

```python
from neurenix.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Use StandardScaler for normally distributed data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use MinMaxScaler for bounded data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
```

## Tutorials

### Dimensionality Reduction and Visualization

```python
import neurenix
import numpy as np
import matplotlib.pyplot as plt
from neurenix.datasets import load_digits
from neurenix.unsupervised import PCA, TSNE, UMAP
from neurenix.preprocessing import StandardScaler

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Apply UMAP
umap = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = umap.fit_transform(X_scaled)

# Create a figure for visualization
plt.figure(figsize=(18, 6))

# Plot PCA results
plt.subplot(1, 3, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", alpha=0.7)
plt.colorbar(scatter, label="Digit")
plt.title("PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.grid(True, linestyle="--", alpha=0.7)

plt.tight_layout()
plt.show()
```
