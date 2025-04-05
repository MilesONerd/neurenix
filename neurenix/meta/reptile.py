"""
Reptile meta-learning algorithm implementation for the Neurenix framework.

Reptile is a first-order meta-learning algorithm that is simpler than MAML
but often achieves comparable performance.

Reference:
    Nichol, A., Achiam, J., & Schulman, J. (2018). On First-Order Meta-Learning Algorithms.
    arXiv preprint arXiv:1803.02999.
"""

from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor
from neurenix.device import Device
from neurenix.optim.optimizer import Optimizer
from neurenix.nn.loss import Loss, MSELoss, CrossEntropyLoss
from neurenix.meta.model import MetaLearningModel

class Reptile(MetaLearningModel):
    """
    Reptile meta-learning algorithm implementation.
    
    Reptile learns an initialization for the model parameters that can be quickly
    adapted to new tasks with just a few gradient steps.
    """
    
    def __init__(
        self,
        model: Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        inner_steps: int = 5,
    ):
        """
        Initialize a Reptile model.
        
        Args:
            model: The base model to meta-train
            inner_lr: Learning rate for the inner loop (task-specific adaptation)
            meta_lr: Learning rate for the outer loop (meta-update)
            inner_steps: Number of gradient steps in the inner loop
        """
        super().__init__(model, inner_lr, meta_lr, first_order=True)
        self.inner_steps = inner_steps
    
    def adapt_to_task(self, support_x: Tensor, support_y: Tensor, steps: Optional[int] = None) -> Module:
        """
        Adapt the model to a new task using the support set.
        
        Args:
            support_x: Input tensors for the support set
            support_y: Target tensors for the support set
            steps: Number of adaptation steps (defaults to self.inner_steps)
            
        Returns:
            Adapted model for the specific task
        """
        if steps is None:
            steps = self.inner_steps
        
        # Clone the model to create a task-specific model
        adapted_model = self.clone_model()
        
        try:
            if len(support_y.shape) > 1 and support_y.shape[1] > 1:
                loss_fn = CrossEntropyLoss()
            elif support_y.dtype == Tensor.int64 or support_y.dtype == Tensor.int32:
                loss_fn = CrossEntropyLoss()
            else:
                loss_fn = MSELoss()
        except:
            loss_fn = MSELoss()
        
        try:
            try:
                from neurenix.optim import Adam
                inner_optimizer = Adam(adapted_model.parameters(), lr=self.inner_lr)
            except ImportError:
                from neurenix.optim import SGD
                inner_optimizer = SGD(adapted_model.parameters(), lr=self.inner_lr)
        except ImportError:
            inner_optimizer = None
        
        # Perform inner loop adaptation
        for _ in range(steps):
            # Forward pass
            predictions = adapted_model(support_x)
            
            # Compute loss
            loss = loss_fn(predictions, support_y)
            
            # Backward pass
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
        
        return adapted_model
    
    def meta_learn(
        self,
        tasks: List[Tuple[Tensor, Tensor, Tensor, Tensor]],
        meta_optimizer: Optional[Optimizer] = None,
        epochs: int = 10,
        tasks_per_batch: int = 4,
        loss_fn: Optional[Loss] = None,
    ) -> Dict[str, List[float]]:
        """
        Perform meta-learning using Reptile.
        
        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples for each task
            meta_optimizer: Optimizer for the meta-update (optional, uses SGD with meta_lr if None)
            epochs: Number of meta-training epochs
            tasks_per_batch: Number of tasks to use in each meta-batch
            loss_fn: Loss function to use (defaults to MSELoss)
            
        Returns:
            Dictionary containing training history
        """
        if loss_fn is None:
            loss_fn = MSELoss()
        
        if meta_optimizer is None:
            from neurenix.optim import SGD
            meta_optimizer = SGD(self.model.parameters(), lr=self.meta_lr)
        
        history = {
            'meta_train_loss': [],
            'query_loss': [],
        }
        
        # Meta-training loop
        for epoch in range(epochs):
            meta_batch_loss = 0.0
            query_loss_sum = 0.0
            
            # Shuffle tasks
            np.random.shuffle(tasks)
            
            # Process tasks in batches
            num_batches = (len(tasks) + tasks_per_batch - 1) // tasks_per_batch
            
            for batch_idx in range(num_batches):
                # Get batch of tasks
                start_idx = batch_idx * tasks_per_batch
                end_idx = min(start_idx + tasks_per_batch, len(tasks))
                batch_tasks = tasks[start_idx:end_idx]
                
                # Store original model parameters
                original_params = [param.clone() for param in self.model.parameters()]
                
                # Process each task in the batch
                for support_x, support_y, query_x, query_y in batch_tasks:
                    # Adapt model to the task
                    adapted_model = self.adapt_to_task(support_x, support_y)
                    
                    # Compute loss on query set for monitoring
                    predictions = adapted_model(query_x)
                    query_loss = loss_fn(predictions, query_y)
                    query_loss_sum += query_loss.item() / len(batch_tasks)
                    
                    # Get adapted parameters
                    adapted_params = adapted_model.parameters()
                    
                    # Update original model parameters (meta-update)
                    for i, (orig_param, adapted_param) in enumerate(zip(self.model.parameters(), adapted_params)):
                        # Reptile update: move original parameters towards adapted parameters
                        orig_param.data = orig_param.data + (self.meta_lr / len(batch_tasks)) * (adapted_param.data - orig_param.data)
                
                # Compute meta-loss (distance moved in parameter space)
                meta_loss = 0.0
                for orig_param, new_param in zip(original_params, self.model.parameters()):
                    meta_loss += ((new_param.data - orig_param.data) ** 2).sum().item()
                meta_batch_loss += meta_loss / len(batch_tasks)
            
            # Record history
            history['meta_train_loss'].append(meta_batch_loss / num_batches)
            history['query_loss'].append(query_loss_sum / num_batches)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - meta_loss: {history['meta_train_loss'][-1]:.4f} - query_loss: {history['query_loss'][-1]:.4f}")
        
        return history
