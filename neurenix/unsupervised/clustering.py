"""
Clustering algorithms for unsupervised learning in the Neurenix framework.

Clustering is an unsupervised learning technique that groups similar data points
together based on their features or characteristics.
"""

from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import numpy as np

from neurenix.tensor import Tensor
from neurenix.device import Device

class KMeans:
    """
    K-Means clustering algorithm implementation.
    
    K-Means partitions data into k clusters by minimizing the sum of squared
    distances between data points and their assigned cluster centers.
    """
    
    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        """
        Initialize K-Means clustering.
        
        Args:
            n_clusters: Number of clusters to form
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0
    
    def fit(self, X: Tensor) -> 'KMeans':
        """
        Fit K-Means clustering on the data.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Self
        """
        n_samples, n_features = X.shape
        
        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X_np = X.numpy()
        centers = []
        
        first_idx = np.random.randint(n_samples)
        centers.append(X_np[first_idx])
        
        for _ in range(1, self.n_clusters):
            distances = np.zeros(n_samples)
            for i in range(n_samples):
                min_dist = float('inf')
                for center in centers:
                    dist = np.sum((X_np[i] - center) ** 2)
                    min_dist = min(min_dist, dist)
                distances[i] = min_dist
            
            distances_sum = np.sum(distances)
            if distances_sum > 0:
                probabilities = distances / distances_sum
                next_idx = np.random.choice(n_samples, p=probabilities)
            else:
                next_idx = np.random.randint(n_samples)
            
            centers.append(X_np[next_idx])
        
        self.cluster_centers_ = Tensor(np.array(centers))
        
        # Initialize labels and inertia
        self.labels_ = Tensor.zeros(n_samples, dtype=Tensor.int64)
        prev_inertia = float('inf')
        
        # Main loop
        for i in range(self.max_iter):
            # Assign samples to nearest cluster center
            distances = Tensor.zeros((n_samples, self.n_clusters))
            
            for j in range(self.n_clusters):
                # Compute squared Euclidean distance
                diff = X - self.cluster_centers_[j]
                distances[:, j] = (diff ** 2).sum(dim=1)
            
            # Assign to nearest cluster
            self.labels_ = distances.argmin(dim=1)
            
            # Update cluster centers
            new_centers = Tensor.zeros_like(self.cluster_centers_)
            counts = Tensor.zeros(self.n_clusters)
            
            for j in range(n_samples):
                label = self.labels_[j].item()
                new_centers[label] += X[j]
                counts[label] += 1
            
            # Avoid division by zero
            for j in range(self.n_clusters):
                if counts[j] > 0:
                    new_centers[j] /= counts[j]
                else:
                    # If a cluster is empty, reinitialize it
                    new_centers[j] = X[np.random.randint(n_samples)]
            
            # Compute inertia (sum of squared distances to nearest centroid)
            self.inertia_ = 0.0
            for j in range(n_samples):
                label = self.labels_[j].item()
                diff = X[j] - self.cluster_centers_[label]
                self.inertia_ += (diff ** 2).sum().item()
            
            # Check for convergence
            center_shift = ((new_centers - self.cluster_centers_) ** 2).sum().sqrt().item()
            self.cluster_centers_ = new_centers
            
            if center_shift < self.tol or abs(prev_inertia - self.inertia_) < self.tol:
                break
            
            prev_inertia = self.inertia_
        
        self.n_iter_ = i + 1
        return self
    
    def predict(self, X: Tensor) -> Tensor:
        """
        Predict the closest cluster for each sample in X.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Cluster labels
        """
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
        
        n_samples = X.shape[0]
        distances = Tensor.zeros((n_samples, self.n_clusters))
        
        for j in range(self.n_clusters):
            diff = X - self.cluster_centers_[j]
            distances[:, j] = (diff ** 2).sum(dim=1)
        
        return distances.argmin(dim=1)
    
    def fit_predict(self, X: Tensor) -> Tensor:
        """
        Fit the model and predict cluster labels.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Cluster labels
        """
        self.fit(X)
        return self.labels_

class DBSCAN:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise) implementation.
    
    DBSCAN groups together points that are close to each other and marks points
    in low-density regions as outliers.
    """
    
    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = 'euclidean',
    ):
        """
        Initialize DBSCAN clustering.
        
        Args:
            eps: Maximum distance between two samples for them to be considered neighbors
            min_samples: Minimum number of samples in a neighborhood for a point to be a core point
            metric: Distance metric to use ('euclidean' or 'cosine')
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        
        self.labels_ = None
        self.core_sample_indices_ = None
    
    def _compute_distances(self, X: Tensor) -> Tensor:
        """
        Compute pairwise distances between all points.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Distance matrix of shape (n_samples, n_samples)
        """
        n_samples = X.shape[0]
        distances = Tensor.zeros((n_samples, n_samples))
        
        if self.metric == 'euclidean':
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    dist = ((X[i] - X[j]) ** 2).sum().sqrt().item()
                    distances[i, j] = dist
                    distances[j, i] = dist
        elif self.metric == 'cosine':
            # Normalize vectors for cosine similarity
            X_norm = X / X.norm(dim=1, keepdim=True)
            
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    # Cosine similarity
                    sim = (X_norm[i] * X_norm[j]).sum().item()
                    # Convert to distance (1 - similarity)
                    dist = 1.0 - sim
                    distances[i, j] = dist
                    distances[j, i] = dist
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")
        
        return distances
    
    def fit(self, X: Tensor) -> 'DBSCAN':
        """
        Fit DBSCAN clustering on the data.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Self
        """
        n_samples = X.shape[0]
        
        # Compute pairwise distances
        distances = self._compute_distances(X)
        
        # Initialize labels
        self.labels_ = Tensor.full((n_samples,), -1, dtype=Tensor.int64)
        
        # Find neighbors for each point
        neighbors = [Tensor.where(distances[i] <= self.eps)[0] for i in range(n_samples)]
        
        # Find core samples
        core_samples = [i for i, neigh in enumerate(neighbors) if len(neigh) >= self.min_samples]
        self.core_sample_indices_ = Tensor.tensor(core_samples, dtype=Tensor.int64)
        
        # Assign cluster labels to core samples
        cluster_label = 0
        
        for i in core_samples:
            if self.labels_[i] != -1:
                continue
            
            # Start a new cluster
            self.labels_[i] = cluster_label
            
            # Process neighbors
            process_queue = [j for j in neighbors[i] if j != i]
            
            while process_queue:
                j = process_queue.pop(0)
                
                if self.labels_[j] == -1:
                    self.labels_[j] = cluster_label
                    
                    # If j is a core point, add its neighbors to the queue
                    if len(neighbors[j]) >= self.min_samples:
                        process_queue.extend([k for k in neighbors[j] if k != j and self.labels_[k] == -1])
            
            cluster_label += 1
        
        return self
    
    def fit_predict(self, X: Tensor) -> Tensor:
        """
        Fit the model and predict cluster labels.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Cluster labels
        """
        self.fit(X)
        return self.labels_

class SpectralClustering:
    """
    Spectral Clustering implementation.
    
    Spectral clustering uses the eigenvalues of a similarity matrix to reduce
    dimensionality before clustering in fewer dimensions.
    """
    
    def __init__(
        self,
        n_clusters: int = 8,
        affinity: str = 'rbf',
        gamma: float = 1.0,
        random_state: Optional[int] = None,
    ):
        """
        Initialize Spectral Clustering.
        
        Args:
            n_clusters: Number of clusters to form
            affinity: How to construct the affinity matrix ('rbf' or 'nearest_neighbors')
            gamma: Kernel coefficient for RBF kernel
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
        
        self.labels_ = None
        self.affinity_matrix_ = None
    
    def _compute_affinity_matrix(self, X: Tensor) -> Tensor:
        """
        Compute the affinity matrix.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Affinity matrix of shape (n_samples, n_samples)
        """
        n_samples = X.shape[0]
        
        if self.affinity == 'rbf':
            # Compute RBF kernel
            affinity = Tensor.zeros((n_samples, n_samples))
            
            for i in range(n_samples):
                for j in range(i, n_samples):
                    # Squared Euclidean distance
                    dist_sq = ((X[i] - X[j]) ** 2).sum().item()
                    # RBF kernel
                    sim = np.exp(-self.gamma * dist_sq)
                    affinity[i, j] = sim
                    affinity[j, i] = sim
            
            return affinity
        
        elif self.affinity == 'nearest_neighbors':
            # Compute k-nearest neighbors graph
            # For simplicity, we'll use a fixed number of neighbors (10)
            k = min(10, n_samples - 1)
            
            # Compute pairwise distances
            distances = Tensor.zeros((n_samples, n_samples))
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    dist = ((X[i] - X[j]) ** 2).sum().sqrt().item()
                    distances[i, j] = dist
                    distances[j, i] = dist
            
            # Construct affinity matrix
            affinity = Tensor.zeros((n_samples, n_samples))
            
            for i in range(n_samples):
                # Find k nearest neighbors
                _, indices = distances[i].topk(k + 1, largest=False)
                
                # Skip the point itself (first index)
                for j in indices[1:]:
                    affinity[i, j] = 1.0
                    affinity[j, i] = 1.0
            
            return affinity
        
        else:
            raise ValueError(f"Unsupported affinity: {self.affinity}")
    
    def fit(self, X: Tensor) -> 'SpectralClustering':
        """
        Fit Spectral Clustering on the data.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Self
        """
        # Compute affinity matrix
        self.affinity_matrix_ = self._compute_affinity_matrix(X)
        
        # Compute Laplacian matrix
        # For simplicity, we'll use the normalized Laplacian
        # L = D^(-1/2) * (D - A) * D^(-1/2)
        # where D is the degree matrix and A is the affinity matrix
        
        # Compute degree matrix
        degrees = self.affinity_matrix_.sum(dim=1)
        
        # Compute normalized Laplacian
        n_samples = X.shape[0]
        laplacian = Tensor.eye(n_samples) - self.affinity_matrix_ / degrees.unsqueeze(1)
        
        try:
            import torch
            laplacian_torch = torch.tensor(laplacian.numpy())
            eigenvalues, eigenvectors_torch = torch.linalg.eigh(laplacian_torch)
            
            eigenvectors = Tensor.from_torch(eigenvectors_torch[:, 1:self.n_clusters+1])
        except (ImportError, AttributeError):
            try:
                eigenvalues, eigenvectors_np = np.linalg.eigh(laplacian.numpy())
                
                eigenvectors = Tensor(eigenvectors_np[:, 1:self.n_clusters+1])
            except:
                n_eigenvectors = self.n_clusters
                eigenvectors_np = np.zeros((n_samples, n_eigenvectors))
                
                vectors = np.random.randn(n_samples, n_eigenvectors)
                
                q, _ = np.linalg.qr(vectors)
                
                for i in range(n_eigenvectors):
                    v = q[:, i].copy()
                    
                    for _ in range(100):
                        v_new = laplacian.numpy() @ v
                        v_new = v_new / np.linalg.norm(v_new)
                        
                        # Check for convergence
                        if np.abs(np.dot(v, v_new)) > 0.9999:
                            break
                        
                        v = v_new
                    
                    eigenvectors_np[:, i] = v
                
                eigenvectors = Tensor(eigenvectors_np)
        
        row_norms = eigenvectors.norm(dim=1, keepdim=True)
        eigenvectors = eigenvectors / row_norms
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        
        self.labels_ = kmeans.fit_predict(eigenvectors)
        
        return self
    
    def fit_predict(self, X: Tensor) -> Tensor:
        """
        Fit the model and predict cluster labels.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Cluster labels
        """
        self.fit(X)
        return self.labels_
