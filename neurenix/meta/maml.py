"""
Model-Agnostic Meta-Learning (MAML) implementation for the Neurenix framework.

MAML is a meta-learning algorithm that trains a model to be easily fine-tuned
for new tasks with just a few gradient steps.

Reference:
    Finn, C., Abbeel, P., & Levine, S. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks.
    International Conference on Machine Learning (ICML).
"""

from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor
from neurenix.device import Device
from neurenix.optim.optimizer import Optimizer
from neurenix.nn.loss import Loss, MSELoss, CrossEntropyLoss
from neurenix.meta.model import MetaLearningModel

class MAML(MetaLearningModel):
    """
    Model-Agnostic Meta-Learning (MAML) implementation.
    
    MAML learns an initialization for the model parameters that can be quickly
    adapted to new tasks with just a few gradient steps.
    """
    
    def __init__(
        self,
        model: Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        first_order: bool = False,
        inner_steps: int = 5,
    ):
        """
        Initialize a MAML model.
        
        Args:
            model: The base model to meta-train
            inner_lr: Learning rate for the inner loop (task-specific adaptation)
            meta_lr: Learning rate for the outer loop (meta-update)
            first_order: Whether to use first-order approximation (ignore second derivatives)
            inner_steps: Number of gradient steps in the inner loop
        """
        super().__init__(model, inner_lr, meta_lr, first_order)
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
        
        # Perform inner loop adaptation
        for _ in range(steps):
            # Forward pass
            predictions = adapted_model(support_x)
            
            # Compute loss
            loss = loss_fn(predictions, support_y)
            
            # Compute gradients
            grads = Tensor.autograd.grad(loss, adapted_model.parameters())
            
            # Update parameters
            for param, grad in zip(adapted_model.parameters(), grads):
                param.data = param.data - self.inner_lr * grad
        
        return adapted_model
    
    def meta_learn(
        self,
        tasks: List[Tuple[Tensor, Tensor, Tensor, Tensor]],
        meta_optimizer: Optimizer,
        epochs: int = 10,
        tasks_per_batch: int = 4,
        loss_fn: Optional[Loss] = None,
    ) -> Dict[str, List[float]]:
        """
        Perform meta-learning using MAML.
        
        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples for each task
            meta_optimizer: Optimizer for the meta-update
            epochs: Number of meta-training epochs
            tasks_per_batch: Number of tasks to use in each meta-batch
            loss_fn: Loss function to use (defaults to MSELoss)
            
        Returns:
            Dictionary containing training history
        """
        if loss_fn is None:
            loss_fn = MSELoss()
        
        history = {
            'meta_train_loss': [],
        }
        
        # Meta-training loop
        for epoch in range(epochs):
            meta_batch_loss = 0.0
            
            # Shuffle tasks
            np.random.shuffle(tasks)
            
            # Process tasks in batches
            num_batches = (len(tasks) + tasks_per_batch - 1) // tasks_per_batch
            
            for batch_idx in range(num_batches):
                # Get batch of tasks
                start_idx = batch_idx * tasks_per_batch
                end_idx = min(start_idx + tasks_per_batch, len(tasks))
                batch_tasks = tasks[start_idx:end_idx]
                
                # Initialize meta-gradient accumulator
                meta_grads = [Tensor.zeros_like(param) for param in self.model.parameters()]
                
                # Process each task in the batch
                for support_x, support_y, query_x, query_y in batch_tasks:
                    # Adapt model to the task
                    adapted_model = self.adapt_to_task(support_x, support_y)
                    
                    # Compute loss on query set
                    predictions = adapted_model(query_x)
                    query_loss = loss_fn(predictions, query_y)
                    
                    # Compute gradients w.r.t. original model parameters
                    task_grads = Tensor.autograd.grad(
                        query_loss,
                        self.model.parameters(),
                        create_graph=not self.first_order,
                    )
                    
                    # Accumulate gradients
                    for i, grad in enumerate(task_grads):
                        meta_grads[i] = meta_grads[i] + grad / len(batch_tasks)
                    
                    # Accumulate loss
                    meta_batch_loss += query_loss.item() / len(batch_tasks)
                
                # Update model parameters with meta-gradients
                meta_optimizer.zero_grad()
                
                # Manually set gradients
                for param, grad in zip(self.model.parameters(), meta_grads):
                    param.grad = grad
                
                # Update parameters
                meta_optimizer.step()
            
            # Record history
            history['meta_train_loss'].append(meta_batch_loss / num_batches)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs} - meta_loss: {history['meta_train_loss'][-1]:.4f}")
        
        return history
