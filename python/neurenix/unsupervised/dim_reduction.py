"""
Dimensionality reduction algorithms for unsupervised learning in the Neurenix framework.

Dimensionality reduction techniques transform high-dimensional data into a lower-dimensional
representation while preserving important properties of the original data.
"""

from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import numpy as np

from neurenix.tensor import Tensor
from neurenix.device import Device

class PCA:
    """
    Principal Component Analysis (PCA) implementation.
    
    PCA finds the directions of maximum variance in the data and projects
    the data onto a lower-dimensional subspace.
    """
    
    def __init__(
        self,
        n_components: Optional[int] = None,
        whiten: bool = False,
        random_state: Optional[int] = None,
    ):
        """
        Initialize PCA.
        
        Args:
            n_components: Number of components to keep. If None, keep all components.
            whiten: Whether to whiten the data (scale to unit variance)
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_samples_ = None
        self.n_features_ = None
    
    def fit(self, X: Tensor) -> 'PCA':
        """
        Fit PCA on the data.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Self
        """
        self.n_samples_, self.n_features_ = X.shape
        
        # Center the data
        self.mean_ = X.mean(dim=0)
        X_centered = X - self.mean_
        
        # Compute the covariance matrix
        cov_matrix = X_centered.t().matmul(X_centered) / (self.n_samples_ - 1)
        
        try:
            import torch
            cov_matrix_torch = torch.tensor(cov_matrix.numpy())
            U, S, V = torch.linalg.svd(cov_matrix_torch, full_matrices=False)
            eigenvalues = Tensor.from_torch(S)
            eigenvectors = Tensor.from_torch(V.T)
        except (ImportError, AttributeError):
            try:
                U, S, V = np.linalg.svd(cov_matrix.numpy(), full_matrices=False)
                eigenvalues = Tensor(S)
                eigenvectors = Tensor(V.T)
            except:
                eigenvalues = []
                eigenvectors = []
                
                A = cov_matrix.numpy()
                n = A.shape[0]
                
                for _ in range(min(n, self.n_components or n)):
                    v = np.random.randn(n)
                    v = v / np.linalg.norm(v)
                    
                    for _ in range(100):  # Usually converges quickly
                        v_new = A @ v
                        v_new = v_new / np.linalg.norm(v_new)
                        
                        if np.abs(np.dot(v, v_new)) > 0.9999:
                            break
                        
                        v = v_new
                    
                    eigenvalue = v.T @ A @ v
                    eigenvalues.append(eigenvalue)
                    eigenvectors.append(v)
                    
                    A = A - eigenvalue * np.outer(v, v)
                
                eigenvalues = Tensor(np.array(eigenvalues))
                eigenvectors = Tensor(np.column_stack(eigenvectors))
        
        # Sort eigenvalues and eigenvectors in descending order
        indices = Tensor.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[indices]
        eigenvectors = eigenvectors[:, indices]
        
        # Determine number of components to keep
        if self.n_components is None:
            n_components = self.n_features_
        else:
            n_components = min(self.n_components, self.n_features_)
        
        # Store results
        self.components_ = eigenvectors[:, :n_components].t()
        self.explained_variance_ = eigenvalues[:n_components]
        self.explained_variance_ratio_ = eigenvalues[:n_components] / eigenvalues.sum()
        self.singular_values_ = Tensor.sqrt(eigenvalues[:n_components] * (self.n_samples_ - 1))
        
        return self
    
    def transform(self, X: Tensor) -> Tensor:
        """
        Apply dimensionality reduction to X.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call 'fit' first.")
        
        # Center the data
        X_centered = X - self.mean_
        
        # Project data onto principal components
        X_transformed = X_centered.matmul(self.components_.t())
        
        # Whiten if requested
        if self.whiten:
            X_transformed /= Tensor.sqrt(self.explained_variance_).unsqueeze(0)
        
        return X_transformed
    
    def fit_transform(self, X: Tensor) -> Tensor:
        """
        Fit the model with X and apply dimensionality reduction.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Transformed data of shape (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: Tensor) -> Tensor:
        """
        Transform data back to its original space.
        
        Args:
            X: Transformed data tensor of shape (n_samples, n_components)
            
        Returns:
            Data in original space of shape (n_samples, n_features)
        """
        if self.components_ is None:
            raise ValueError("PCA not fitted. Call 'fit' first.")
        
        # Unwhiten if necessary
        if self.whiten:
            X = X * Tensor.sqrt(self.explained_variance_).unsqueeze(0)
        
        # Project back to original space
        X_original = X.matmul(self.components_)
        
        # Add the mean back
        X_original = X_original + self.mean_
        
        return X_original

class TSNE:
    """
    t-Distributed Stochastic Neighbor Embedding (t-SNE) implementation.
    
    t-SNE is a nonlinear dimensionality reduction technique that is particularly
    well-suited for visualizing high-dimensional data in a low-dimensional space.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        n_iter: int = 1000,
        random_state: Optional[int] = None,
    ):
        """
        Initialize t-SNE.
        
        Args:
            n_components: Dimension of the embedded space
            perplexity: Related to the number of nearest neighbors used in manifold learning
            learning_rate: Learning rate for gradient descent
            n_iter: Number of iterations for optimization
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.random_state = random_state
        
        self.embedding_ = None
    
    def _compute_joint_probabilities(self, distances: np.ndarray, perplexity: float) -> np.ndarray:
        """
        Compute joint probabilities p_ij from distances using perplexity-based scaling.
        
        Args:
            distances: Pairwise squared distances
            perplexity: Perplexity parameter
            
        Returns:
            Joint probability matrix P
        """
        n_samples = distances.shape[0]
        P = np.zeros((n_samples, n_samples))
        
        target_entropy = np.log(perplexity)
        
        for i in range(n_samples):
            distances_i = distances[i].copy()
            distances_i[i] = np.inf
            
            beta_min = -np.inf
            beta_max = np.inf
            beta = 1.0
            
            max_iter = 50
            tol = 1e-5
            
            for _ in range(max_iter):
                exp_distances = np.exp(-distances_i * beta)
                sum_exp_distances = np.sum(exp_distances)
                
                if sum_exp_distances == 0:
                    p_conditional = np.zeros_like(distances_i)
                else:
                    p_conditional = exp_distances / sum_exp_distances
                
                entropy = -np.sum(p_conditional * np.log(np.maximum(p_conditional, 1e-10)))
                
                entropy_diff = entropy - target_entropy
                if np.abs(entropy_diff) < tol:
                    break
                
                if entropy_diff > 0:
                    beta_min = beta
                    if beta_max == np.inf:
                        beta *= 2
                    else:
                        beta = (beta + beta_max) / 2
                else:
                    beta_max = beta
                    if beta_min == -np.inf:
                        beta /= 2
                    else:
                        beta = (beta + beta_min) / 2
            
            P[i] = p_conditional
        
        P = (P + P.T) / (2 * n_samples)
        
        P = np.maximum(P, 1e-12)
        
        return P
    
    def _compute_q_distribution(self, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the Q distribution (Student's t-distribution) in the low-dimensional space.
        
        Args:
            Y: Current embedding
            
        Returns:
            Q distribution and distances in the embedding space
        """
        n_samples = Y.shape[0]
        
        distances_Y = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                distances_Y[i, j] = np.sum((Y[i] - Y[j])**2)
                distances_Y[j, i] = distances_Y[i, j]
        
        Q = 1.0 / (1.0 + distances_Y)
        
        np.fill_diagonal(Q, 0)
        
        Q_sum = np.sum(Q)
        if Q_sum == 0:
            Q = np.ones_like(Q) / (n_samples**2 - n_samples)
            np.fill_diagonal(Q, 0)
        else:
            Q /= Q_sum
        
        return Q, distances_Y
    
    def _compute_gradients(self, P: np.ndarray, Q: np.ndarray, Y: np.ndarray, distances_Y: np.ndarray) -> np.ndarray:
        """
        Compute the gradients of the KL divergence between P and Q.
        
        Args:
            P: Joint probability matrix P
            Q: Joint probability matrix Q
            Y: Current embedding
            distances_Y: Distances in the embedding space
            
        Returns:
            Gradients
        """
        n_samples = Y.shape[0]
        
        PQ_diff = P - Q
        
        grads = np.zeros_like(Y)
        
        for i in range(n_samples):
            grad_i = np.zeros_like(Y[i])
            
            for j in range(n_samples):
                if i != j:
                    q_ij = Q[i, j]
                    p_ij = P[i, j]
                    
                    grad_i += 4 * (p_ij - q_ij) * (Y[i] - Y[j]) * (1 + distances_Y[i, j])**(-1)
            
            grads[i] = grad_i
        
        return grads
    
    def fit_transform(self, X: Tensor) -> Tensor:
        """
        Fit t-SNE on the data and return the embedding.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Embedding of shape (n_samples, n_components)
        """
        n_samples = X.shape[0]
        
        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Initialize embedding
        self.embedding_ = Tensor.randn(n_samples, self.n_components) * 0.0001
        
        X_np = X.numpy()
        distances = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                distances[i, j] = np.sum((X_np[i] - X_np[j])**2)
                distances[j, i] = distances[i, j]
        
        P = self._compute_joint_probabilities(distances, self.perplexity)
        
        Y = self.embedding_.numpy()
        
        for iteration in range(self.n_iter):
            Q, distances_Y = self._compute_q_distribution(Y)
            
            grads = self._compute_gradients(P, Q, Y, distances_Y)
            
            Y = Y - self.learning_rate * grads
            
            Y = Y - np.mean(Y, axis=0)
            
            if iteration == 250:
                self.learning_rate = self.learning_rate / 2
        
        self.embedding_ = Tensor(Y)
        
        return self.embedding_

class UMAP:
    """
    Uniform Manifold Approximation and Projection (UMAP) implementation.
    
    UMAP is a dimensionality reduction technique that can be used for visualization
    and general non-linear dimension reduction.
    """
    
    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = 'euclidean',
        random_state: Optional[int] = None,
    ):
        """
        Initialize UMAP.
        
        Args:
            n_components: Dimension of the embedded space
            n_neighbors: Number of neighbors to consider for each point
            min_dist: Minimum distance between points in the embedding
            metric: Distance metric to use
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        
        self.embedding_ = None
    
    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute pairwise distances between points.
        
        Args:
            X: Input data array of shape (n_samples, n_features)
            
        Returns:
            Distance matrix of shape (n_samples, n_samples)
        """
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))
        
        if self.metric == 'euclidean':
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    dist = np.sqrt(np.sum((X[i] - X[j])**2))
                    distances[i, j] = dist
                    distances[j, i] = dist
        elif self.metric == 'manhattan':
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    dist = np.sum(np.abs(X[i] - X[j]))
                    distances[i, j] = dist
                    distances[j, i] = dist
        else:
            for i in range(n_samples):
                for j in range(i+1, n_samples):
                    dist = np.sqrt(np.sum((X[i] - X[j])**2))
                    distances[i, j] = dist
                    distances[j, i] = dist
        
        return distances
    
    def _compute_fuzzy_simplicial_set(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute the fuzzy simplicial set (graph) representation of the data.
        
        Args:
            distances: Distance matrix of shape (n_samples, n_samples)
            
        Returns:
            Adjacency matrix of the fuzzy simplicial set
        """
        n_samples = distances.shape[0]
        
        knn_indices = np.zeros((n_samples, self.n_neighbors), dtype=np.int32)
        knn_distances = np.zeros((n_samples, self.n_neighbors))
        
        for i in range(n_samples):
            dist_i = distances[i].copy()
            dist_i[i] = np.inf  # Exclude self
            
            nn_indices = np.argsort(dist_i)[:self.n_neighbors]
            knn_indices[i] = nn_indices
            knn_distances[i] = dist_i[nn_indices]
        
        sigmas = np.zeros(n_samples)
        for i in range(n_samples):
            if knn_distances[i, 0] == 0:
                sigmas[i] = 1.0
            else:
                sigmas[i] = knn_distances[i, 0]
        
        adjacency = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j_idx, j in enumerate(knn_indices[i]):
                if i != j:  # Avoid self-loops
                    distance = distances[i, j]
                    weight = np.exp(-(distance / sigmas[i]))
                    
                    adjacency[i, j] = weight
                    adjacency[j, i] = weight
        
        row_sums = adjacency.sum(axis=1)
        adjacency = adjacency / row_sums[:, np.newaxis]
        
        return adjacency
    
    def _optimize_embedding(self, adjacency: np.ndarray, n_epochs: int = 200) -> np.ndarray:
        """
        Optimize the embedding using stochastic gradient descent.
        
        Args:
            adjacency: Adjacency matrix of the fuzzy simplicial set
            n_epochs: Number of optimization epochs
            
        Returns:
            Optimized embedding
        """
        n_samples = adjacency.shape[0]
        
        # Initialize embedding
        embedding = np.random.normal(scale=0.0001, size=(n_samples, self.n_components))
        
        learning_rate = 1.0
        
        negative_sample_rate = 5
        
        for epoch in range(n_epochs):
            alpha = 1.0 - (epoch / n_epochs)
            
            for i in range(n_samples):
                for j in range(n_samples):
                    if adjacency[i, j] > 0:  # Only consider connected points
                        dist_squared = np.sum((embedding[i] - embedding[j])**2)
                        
                        grad_coeff = -2.0 * alpha * adjacency[i, j] * (
                            1.0 / (1.0 + dist_squared) - 
                            (1.0 / (1.0 + self.min_dist**2))
                        )
                        
                        grad = grad_coeff * (embedding[i] - embedding[j])
                        
                        embedding[i] -= learning_rate * grad
                        embedding[j] += learning_rate * grad
                        
                        for _ in range(negative_sample_rate):
                            k = np.random.randint(n_samples)
                            if k != i and k != j:
                                dist_squared = np.sum((embedding[i] - embedding[k])**2)
                                
                                grad_coeff = 2.0 * alpha * (
                                    1.0 / (1.0 + dist_squared)
                                )
                                
                                grad = grad_coeff * (embedding[i] - embedding[k])
                                
                                embedding[i] += learning_rate * grad
        
        return embedding
    
    def fit_transform(self, X: Tensor) -> Tensor:
        """
        Fit UMAP on the data and return the embedding.
        
        Args:
            X: Input data tensor of shape (n_samples, n_features)
            
        Returns:
            Embedding of shape (n_samples, n_components)
        """
        n_samples = X.shape[0]
        
        # Set random seed if provided
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        X_np = X.numpy()
        
        distances = self._compute_distances(X_np)
        
        adjacency = self._compute_fuzzy_simplicial_set(distances)
        
        embedding = self._optimize_embedding(adjacency)
        
        self.embedding_ = Tensor(embedding)
        
        return self.embedding_
