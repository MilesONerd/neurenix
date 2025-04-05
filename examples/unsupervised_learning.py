"""
Example of unsupervised learning with the Neurenix framework.

This example demonstrates how to:
1. Create and train autoencoders for representation learning
2. Use dimensionality reduction techniques for visualization
3. Apply clustering algorithms to discover patterns in data
4. Implement contrastive learning for self-supervised learning
"""

import numpy as np
import matplotlib.pyplot as plt

import neurenix as nx
from neurenix.nn import Sequential, Linear, ReLU, Sigmoid
from neurenix.optim import Adam
from neurenix.unsupervised import Autoencoder, VAE, KMeans, PCA, TSNE

def generate_synthetic_data(n_samples=1000, n_features=50, n_clusters=5):
    """
    Generate synthetic high-dimensional data with cluster structure.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features (dimensionality)
        n_clusters: Number of clusters in the data
        
    Returns:
        Tuple of (data, cluster_labels)
    """
    # Generate cluster centers
    centers = np.random.randn(n_clusters, n_features) * 5
    
    # Generate samples for each cluster
    samples_per_cluster = n_samples // n_clusters
    
    data = []
    labels = []
    
    for i in range(n_clusters):
        # Generate samples for this cluster
        cluster_samples = np.random.randn(samples_per_cluster, n_features) + centers[i]
        
        data.append(cluster_samples)
        labels.extend([i] * samples_per_cluster)
    
    # Concatenate all samples
    data = np.vstack(data).astype(np.float32)
    labels = np.array(labels)
    
    # Shuffle the data
    indices = np.random.permutation(n_samples)
    data = data[indices]
    labels = labels[indices]
    
    return data, labels

def main():
    # Initialize Neurenix
    nx.init({"device": "cpu", "log_level": "info"})
    
    print("Neurenix Unsupervised Learning Example")
    print("======================================")
    
    # Generate synthetic data
    print("Generating synthetic data...")
    data, true_labels = generate_synthetic_data(n_samples=1000, n_features=50, n_clusters=5)
    
    # Convert to Neurenix tensors
    data_tensor = nx.Tensor(data)
    
    # Split into train and test sets
    train_size = int(0.8 * len(data))
    train_data = data_tensor[:train_size]
    test_data = data_tensor[train_size:]
    
    # Part 1: Autoencoder for dimensionality reduction
    print("\nPart 1: Training an Autoencoder")
    print("-------------------------------")
    
    # Create an autoencoder
    input_dim = data.shape[1]
    hidden_dims = [128, 64]
    latent_dim = 16
    
    autoencoder = Autoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        activation='relu',
    )
    
    # Create optimizer
    optimizer = Adam(autoencoder.parameters(), lr=0.001)
    
    # Train the autoencoder
    print("Training autoencoder...")
    batch_size = 64
    epochs = 10
    
    for epoch in range(epochs):
        # Shuffle the data
        indices = np.random.permutation(len(train_data))
        train_data_shuffled = train_data[indices]
        
        # Train in batches
        epoch_loss = 0.0
        num_batches = (len(train_data) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(train_data))
            batch = train_data_shuffled[start_idx:end_idx]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = autoencoder(batch)
            
            # Compute loss (mean squared error)
            loss = ((reconstructed - batch) ** 2).mean()
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item() * len(batch)
        
        # Average loss for the epoch
        epoch_loss /= len(train_data)
        
        print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")
    
    # Get latent representations
    print("Extracting latent representations...")
    latent_train = autoencoder.get_latent_representation(train_data)
    latent_test = autoencoder.get_latent_representation(test_data)
    
    # Part 2: Dimensionality reduction for visualization
    print("\nPart 2: Dimensionality Reduction")
    print("-------------------------------")
    
    # PCA
    print("Applying PCA...")
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_tensor)
    
    # t-SNE
    print("Applying t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30.0)
    tsne_result = tsne.fit_transform(data_tensor)
    
    # Part 3: Clustering
    print("\nPart 3: Clustering")
    print("----------------")
    
    # K-means clustering
    print("Applying K-means clustering...")
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(data_tensor)
    
    from sklearn.metrics import adjusted_rand_score
    
    pred_labels = cluster_labels.numpy()
    
    ari = adjusted_rand_score(true_labels, pred_labels)
    
    from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score
    
    homogeneity = homogeneity_score(true_labels, pred_labels)
    completeness = completeness_score(true_labels, pred_labels)
    v_measure = v_measure_score(true_labels, pred_labels)
    
    print("Clustering evaluation:")
    print(f"- Number of clusters: {kmeans.n_clusters}")
    print(f"- Adjusted Rand Index: {ari:.4f}")
    print(f"- Homogeneity: {homogeneity:.4f}")
    print(f"- Completeness: {completeness:.4f}")
    print(f"- V-measure: {v_measure:.4f}")
    print("- Cluster sizes:", [np.sum(pred_labels == i) for i in range(kmeans.n_clusters)])
    
    # Part 4: Variational Autoencoder (VAE)
    print("\nPart 4: Variational Autoencoder")
    print("-----------------------------")
    
    # Create a VAE
    vae = VAE(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        activation='relu',
    )
    
    # Create optimizer
    vae_optimizer = Adam(vae.parameters(), lr=0.001)
    
    # Train the VAE
    print("Training VAE...")
    
    for epoch in range(5):  # Fewer epochs for brevity
        # Shuffle the data
        indices = np.random.permutation(len(train_data))
        train_data_shuffled = train_data[indices]
        
        # Train in batches
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        num_batches = (len(train_data) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            # Get batch
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(train_data))
            batch = train_data_shuffled[start_idx:end_idx]
            
            # Zero gradients
            vae_optimizer.zero_grad()
            
            # Forward pass
            reconstructed, mu, logvar = vae(batch)
            
            # Compute loss
            loss = vae.loss_function(reconstructed, batch, mu, logvar, beta=1.0)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            vae_optimizer.step()
            
            # Accumulate losses
            epoch_loss += loss.item() * len(batch)
            epoch_recon_loss += vae.reconstruction_loss * len(batch)
            epoch_kl_loss += vae.kl_divergence * len(batch)
        
        # Average losses for the epoch
        epoch_loss /= len(train_data)
        epoch_recon_loss /= len(train_data)
        epoch_kl_loss /= len(train_data)
        
        print(f"Epoch {epoch+1}/5 - loss: {epoch_loss:.4f} - recon_loss: {epoch_recon_loss:.4f} - kl_loss: {epoch_kl_loss:.4f}")
    
    # Generate samples from the VAE
    print("Generating samples from the VAE...")
    num_samples = 10
    generated_samples = vae.sample(num_samples)
    
    print("\nUnsupervised learning example completed successfully!")

if __name__ == "__main__":
    main()
