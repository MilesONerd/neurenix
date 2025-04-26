"""
Prototypical Networks implementation for the Neurenix framework.

Prototypical Networks perform few-shot classification by computing distances
to prototype representations of each class.

Reference:
    Snell, J., Swersky, K., & Zemel, R. (2017). Prototypical Networks for Few-shot Learning.
    Advances in Neural Information Processing Systems (NeurIPS).
"""

from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor
from neurenix.device import Device
from neurenix.optim.optimizer import Optimizer
from neurenix.nn.loss import Loss, CrossEntropyLoss
from neurenix.meta.model import MetaLearningModel

class PrototypicalNetworks(MetaLearningModel):
    """
    Prototypical Networks for few-shot classification.
    
    Prototypical Networks learn an embedding space where classes can be
    represented by a single prototype (the mean of embedded support examples).
    Classification is performed by finding the nearest prototype.
    """
    
    def __init__(
        self,
        embedding_model: Module,
        distance_metric: str = 'euclidean',
        meta_lr: float = 0.001,
    ):
        """
        Initialize a Prototypical Networks model.
        
        Args:
            embedding_model: Model that maps inputs to an embedding space
            distance_metric: Distance metric to use ('euclidean' or 'cosine')
            meta_lr: Learning rate for the meta-update
        """
        super().__init__(embedding_model, inner_lr=0.0, meta_lr=meta_lr, first_order=True)
        self.distance_metric = distance_metric
    
    def compute_prototypes(self, support_x: Tensor, support_y: Tensor) -> Tensor:
        """
        Compute class prototypes from support examples.
        
        Args:
            support_x: Support set inputs [n_support, ...]
            support_y: Support set labels [n_support, n_classes] (one-hot)
            
        Returns:
            Class prototypes [n_classes, embedding_dim]
        """
        # Embed support examples
        embeddings = self.model(support_x)  # [n_support, embedding_dim]
        
        # Get number of classes
        n_classes = support_y.shape[1]
        
        # Initialize prototypes
        embedding_dim = embeddings.shape[1]
        prototypes = Tensor.zeros((n_classes, embedding_dim), device=embeddings.device)
        
        # Compute prototype for each class
        for c in range(n_classes):
            # Get mask for examples of this class
            mask = support_y[:, c]  # [n_support]
            
            # Get embeddings for this class
            class_embeddings = embeddings * mask.unsqueeze(1)  # [n_support, embedding_dim]
            
            # Compute mean embedding (prototype)
            prototype = class_embeddings.sum(dim=0) / mask.sum()  # [embedding_dim]
            prototypes[c] = prototype
        
        return prototypes
    
    def compute_distances(self, query_embeddings: Tensor, prototypes: Tensor) -> Tensor:
        """
        Compute distances between query embeddings and class prototypes.
        
        Args:
            query_embeddings: Query set embeddings [n_query, embedding_dim]
            prototypes: Class prototypes [n_classes, embedding_dim]
            
        Returns:
            Distances [n_query, n_classes]
        """
        n_query = query_embeddings.shape[0]
        n_classes = prototypes.shape[0]
        
        if self.distance_metric == 'euclidean':
            # Compute squared Euclidean distance
            # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y
            
            # Expand dimensions for broadcasting
            query_expanded = query_embeddings.unsqueeze(1)  # [n_query, 1, embedding_dim]
            prototypes_expanded = prototypes.unsqueeze(0)  # [1, n_classes, embedding_dim]
            
            # Compute squared Euclidean distance
            distances = ((query_expanded - prototypes_expanded) ** 2).sum(dim=2)  # [n_query, n_classes]
            
        elif self.distance_metric == 'cosine':
            # Normalize embeddings
            query_norm = query_embeddings / query_embeddings.norm(dim=1, keepdim=True)
            prototypes_norm = prototypes / prototypes.norm(dim=1, keepdim=True)
            
            # Compute cosine similarity
            similarity = query_norm.matmul(prototypes_norm.transpose(0, 1))  # [n_query, n_classes]
            
            # Convert to distance (1 - similarity)
            distances = 1 - similarity
            
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        return distances
    
    def forward(self, x: Tensor, support_x: Optional[Tensor] = None, support_y: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through the Prototypical Networks model.
        
        Args:
            x: Input tensor
            support_x: Support set inputs (required for inference)
            support_y: Support set labels (required for inference)
            
        Returns:
            Class logits (negative distances to prototypes)
        """
        # Embed input
        embeddings = self.model(x)
        
        # If support set is provided, compute prototypes and distances
        if support_x is not None and support_y is not None:
            prototypes = self.compute_prototypes(support_x, support_y)
            distances = self.compute_distances(embeddings, prototypes)
            
            # Return negative distances as logits
            return -distances
        
        # Otherwise, just return the embeddings
        return embeddings
    
    def meta_learn(
        self,
        tasks: List[Tuple[Tensor, Tensor, Tensor, Tensor]],
        meta_optimizer: Optimizer,
        epochs: int = 10,
        tasks_per_batch: int = 4,
    ) -> Dict[str, List[float]]:
        """
        Perform meta-learning using Prototypical Networks.
        
        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples for each task
            meta_optimizer: Optimizer for the meta-update
            epochs: Number of meta-training epochs
            tasks_per_batch: Number of tasks to use in each meta-batch
            
        Returns:
            Dictionary containing training history
        """
        # Create loss function
        loss_fn = CrossEntropyLoss()
        
        history = {
            'meta_train_loss': [],
            'meta_train_acc': [],
        }
        
        # Meta-training loop
        for epoch in range(epochs):
            meta_batch_loss = 0.0
            meta_batch_acc = 0.0
            
            # Shuffle tasks
            np.random.shuffle(tasks)
            
            # Process tasks in batches
            num_batches = (len(tasks) + tasks_per_batch - 1) // tasks_per_batch
            
            for batch_idx in range(num_batches):
                # Get batch of tasks
                start_idx = batch_idx * tasks_per_batch
                end_idx = min(start_idx + tasks_per_batch, len(tasks))
                batch_tasks = tasks[start_idx:end_idx]
                
                # Zero gradients
                meta_optimizer.zero_grad()
                
                # Process each task in the batch
                batch_loss = 0.0
                batch_acc = 0.0
                
                for support_x, support_y, query_x, query_y in batch_tasks:
                    # Embed support and query examples
                    support_embeddings = self.model(support_x)
                    query_embeddings = self.model(query_x)
                    
                    # Compute prototypes
                    prototypes = self.compute_prototypes(support_x, support_y)
                    
                    # Compute distances
                    distances = self.compute_distances(query_embeddings, prototypes)
                    
                    # Compute logits (negative distances)
                    logits = -distances
                    
                    # Compute loss
                    # Convert one-hot labels to class indices
                    query_y_indices = query_y.argmax(dim=1)
                    loss = loss_fn(logits, query_y_indices)
                    
                    # Compute accuracy
                    predictions = logits.argmax(dim=1)
                    accuracy = (predictions == query_y_indices).float().mean().item()
                    
                    # Accumulate loss and accuracy
                    batch_loss += loss.item() / len(batch_tasks)
                    batch_acc += accuracy / len(batch_tasks)
                    
                    # Backward pass
                    loss.backward()
                
                # Update model parameters
                meta_optimizer.step()
                
                # Accumulate batch metrics
                meta_batch_loss += batch_loss
                meta_batch_acc += batch_acc
            
            # Record history
            history['meta_train_loss'].append(meta_batch_loss / num_batches)
            history['meta_train_acc'].append(meta_batch_acc / num_batches)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - meta_loss: {history['meta_train_loss'][-1]:.4f} - meta_acc: {history['meta_train_acc'][-1]:.4f}")
        
        return history
