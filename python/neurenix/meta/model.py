"""
Meta-learning model implementation for the Neurenix framework.
"""

from typing import List, Dict, Any, Optional, Union, Callable, Tuple

import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor
from neurenix.device import Device

class MetaLearningModel(Module):
    """
    Base class for meta-learning models in the Neurenix framework.
    
    Meta-learning models are designed to learn how to learn, enabling quick
    adaptation to new tasks with minimal data (few-shot learning).
    """
    
    def __init__(
        self,
        model: Module,
        inner_lr: float = 0.01,
        meta_lr: float = 0.001,
        first_order: bool = False,
    ):
        """
        Initialize a meta-learning model.
        
        Args:
            model: The base model to meta-train
            inner_lr: Learning rate for the inner loop (task-specific adaptation)
            meta_lr: Learning rate for the outer loop (meta-update)
            first_order: Whether to use first-order approximation (ignore second derivatives)
        """
        super().__init__()
        
        self.model = model
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.first_order = first_order
    
    def clone_model(self) -> Module:
        """
        Create a clone of the base model with the same architecture and parameters.
        
        Returns:
            A cloned model
        """
        try:
            import copy
            import torch
            
            if hasattr(self.model, '_parameters') and hasattr(self.model, '_modules'):
                cloned_model = copy.deepcopy(self.model)
                
                for param_name, param in self.model.named_parameters():
                    cloned_param = dict(cloned_model.named_parameters())[param_name]
                    cloned_param.data.copy_(param.data)
                
                return cloned_model
        except (ImportError, AttributeError):
            pass
        
        try:
            import copy
            
            cloned_model = copy.deepcopy(self.model)
            
            if hasattr(self.model, 'state_dict') and hasattr(cloned_model, 'load_state_dict'):
                state = self.model.state_dict()
                cloned_model.load_state_dict(state)
            
            return cloned_model
        except:
            model_class = self.model.__class__
            
            try:
                if hasattr(self.model, '_init_params'):
                    cloned_model = model_class(**self.model._init_params)
                else:
                    cloned_model = model_class()
                
                if hasattr(self.model, 'parameters') and hasattr(cloned_model, 'parameters'):
                    for p_src, p_tgt in zip(self.model.parameters(), cloned_model.parameters()):
                        if hasattr(p_tgt, 'data') and hasattr(p_src, 'data'):
                            p_tgt.data = p_src.data.clone()
                
                return cloned_model
            except:
                print("Warning: Could not properly clone model. Using original model instead.")
                return self.model
    
    def adapt_to_task(self, support_x: Tensor, support_y: Tensor, steps: int = 5) -> Module:
        """
        Adapt the model to a new task using the support set.
        
        Args:
            support_x: Input tensors for the support set
            support_y: Target tensors for the support set
            steps: Number of adaptation steps
            
        Returns:
            Adapted model for the specific task
        """
        adapted_model = self.clone_model()
        
        try:
            if len(support_y.shape) > 1 and support_y.shape[1] > 1:
                from neurenix.nn.loss import CrossEntropyLoss
                loss_fn = CrossEntropyLoss()
            else:
                from neurenix.nn.loss import MSELoss
                loss_fn = MSELoss()
        except:
            from neurenix.nn.loss import MSELoss
            loss_fn = MSELoss()
        
        try:
            from neurenix.optim import SGD
            optimizer = SGD(adapted_model.parameters(), lr=self.inner_lr)
        except:
            optimizer = None
        
        for step in range(steps):
            predictions = adapted_model(support_x)
            loss = loss_fn(predictions, support_y)
            
            if hasattr(loss, 'backward'):
                loss.backward()
            else:
                try:
                    params = list(adapted_model.parameters())
                    grads = []
                    
                    for param in params:
                        grad = Tensor.zeros_like(param)
                        
                        eps = 1e-4
                        for i in range(param.numel()):
                            original_val = param.data.flatten()[i].item()
                            
                            param.data.flatten()[i] = original_val + eps
                            pred_plus = adapted_model(support_x)
                            loss_plus = loss_fn(pred_plus, support_y).item()
                            
                            param.data.flatten()[i] = original_val - eps
                            pred_minus = adapted_model(support_x)
                            loss_minus = loss_fn(pred_minus, support_y).item()
                            
                            grad.data.flatten()[i] = (loss_plus - loss_minus) / (2 * eps)
                            
                            param.data.flatten()[i] = original_val
                        
                        grads.append(grad)
                        
                    for param, grad in zip(params, grads):
                        param.grad = grad
                except:
                    print("Warning: Could not compute gradients. Skipping adaptation step.")
                    continue
            
            if optimizer is not None:
                optimizer.step()
                optimizer.zero_grad()
            else:
                for param in adapted_model.parameters():
                    if param.grad is not None:
                        param.data -= self.inner_lr * param.grad
                        param.grad = None
        
        return adapted_model
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the meta-learning model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x)
    
    def meta_learn(
        self,
        tasks: List[Tuple[Tensor, Tensor, Tensor, Tensor]],
        epochs: int = 10,
        tasks_per_batch: int = 4,
    ) -> Dict[str, List[float]]:
        """
        Perform meta-learning on a set of tasks.
        
        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples for each task
            epochs: Number of meta-training epochs
            tasks_per_batch: Number of tasks to use in each meta-batch
            
        Returns:
            Dictionary containing training history
        """
        history = {
            'meta_train_loss': [],
            'meta_val_loss': []
        }
        
        try:
            from neurenix.optim import Adam
            meta_optimizer = Adam(self.model.parameters(), lr=self.meta_lr)
        except ImportError:
            try:
                from neurenix.optim import SGD
                meta_optimizer = SGD(self.model.parameters(), lr=self.meta_lr)
            except ImportError:
                print("Warning: Could not create optimizer. Using manual parameter updates.")
                meta_optimizer = None
        
        num_tasks = len(tasks)
        num_train_tasks = max(1, int(0.8 * num_tasks))
        
        train_tasks = tasks[:num_train_tasks]
        val_tasks = tasks[num_train_tasks:] if num_train_tasks < num_tasks else train_tasks[-1:]
        
        for epoch in range(epochs):
            import random
            random.shuffle(train_tasks)
            
            meta_train_loss = 0.0
            num_batches = max(1, len(train_tasks) // tasks_per_batch)
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * tasks_per_batch
                end_idx = min(start_idx + tasks_per_batch, len(train_tasks))
                batch_tasks = train_tasks[start_idx:end_idx]
                
                if meta_optimizer is not None:
                    meta_optimizer.zero_grad()
                else:
                    for param in self.model.parameters():
                        param.grad = Tensor.zeros_like(param)
                
                batch_loss = 0.0
                
                for task_idx, (support_x, support_y, query_x, query_y) in enumerate(batch_tasks):
                    adapted_model = self.adapt_to_task(support_x, support_y)
                    
                    predictions = adapted_model(query_x)
                    
                    try:
                        if len(query_y.shape) > 1 and query_y.shape[1] > 1:
                            from neurenix.nn.loss import CrossEntropyLoss
                            loss_fn = CrossEntropyLoss()
                        else:
                            from neurenix.nn.loss import MSELoss
                            loss_fn = MSELoss()
                    except:
                        from neurenix.nn.loss import MSELoss
                        loss_fn = MSELoss()
                    
                    task_loss = loss_fn(predictions, query_y)
                    
                    batch_loss += task_loss.item()
                    
                    if hasattr(task_loss, 'backward'):
                        task_loss.backward()
                    
                batch_loss /= len(batch_tasks)
                meta_train_loss += batch_loss
                
                if meta_optimizer is not None:
                    meta_optimizer.step()
                else:
                    for param in self.model.parameters():
                        if param.grad is not None:
                            param.data -= self.meta_lr * param.grad
                            param.grad = None
            
            meta_train_loss /= num_batches
            history['meta_train_loss'].append(meta_train_loss)
            
            meta_val_loss = 0.0
            
            for support_x, support_y, query_x, query_y in val_tasks:
                adapted_model = self.adapt_to_task(support_x, support_y)
                
                with Tensor.no_grad():
                    predictions = adapted_model(query_x)
                    
                    try:
                        if len(query_y.shape) > 1 and query_y.shape[1] > 1:
                            from neurenix.nn.loss import CrossEntropyLoss
                            loss_fn = CrossEntropyLoss()
                        else:
                            from neurenix.nn.loss import MSELoss
                            loss_fn = MSELoss()
                    except:
                        from neurenix.nn.loss import MSELoss
                        loss_fn = MSELoss()
                    
                    val_loss = loss_fn(predictions, query_y)
                    meta_val_loss += val_loss.item()
            
            meta_val_loss /= len(val_tasks)
            history['meta_val_loss'].append(meta_val_loss)
            
            print(f"Epoch {epoch+1}/{epochs}: meta_train_loss={meta_train_loss:.4f}, meta_val_loss={meta_val_loss:.4f}")
        
        return history
