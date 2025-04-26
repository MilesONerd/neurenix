"""
Fine-tuning utilities for transfer learning in the Neurenix framework.
"""

from typing import List, Dict, Any, Optional, Union, Callable

import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor
from neurenix.optim.optimizer import Optimizer
from neurenix.transfer.model import TransferModel

def freeze_layers(model: Module, layer_names: Optional[List[str]] = None):
    """
    Freeze specific layers in a model or the entire model.
    
    Args:
        model: The model to freeze layers in
        layer_names: List of layer names to freeze. If None, freeze all layers.
    """
    if layer_names is None:
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
    else:
        # Freeze specific layers
        for name, module in model._modules.items():
            if name in layer_names:
                for param in module.parameters():
                    param.requires_grad = False

def unfreeze_layers(model: Module, layer_names: Optional[List[str]] = None):
    """
    Unfreeze specific layers in a model or the entire model.
    
    Args:
        model: The model to unfreeze layers in
        layer_names: List of layer names to unfreeze. If None, unfreeze all layers.
    """
    if layer_names is None:
        # Unfreeze all parameters
        for param in model.parameters():
            param.requires_grad = True
    else:
        # Unfreeze specific layers
        for name, module in model._modules.items():
            if name in layer_names:
                for param in module.parameters():
                    param.requires_grad = True

def fine_tune(
    model: Union[TransferModel, Module],
    optimizer: Optimizer,
    train_data: List[Tensor],
    train_labels: List[Tensor],
    val_data: Optional[List[Tensor]] = None,
    val_labels: Optional[List[Tensor]] = None,
    epochs: int = 10,
    batch_size: int = 32,
    loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    callbacks: Optional[List[Callable]] = None,
    early_stopping: bool = False,
    patience: int = 3,
):
    """
    Fine-tune a model on a new dataset.
    
    Args:
        model: The model to fine-tune (either a TransferModel or a regular Module)
        optimizer: The optimizer to use for training
        train_data: List of training data tensors
        train_labels: List of training label tensors
        val_data: List of validation data tensors
        val_labels: List of validation label tensors
        epochs: Number of epochs to train for
        batch_size: Batch size for training
        loss_fn: Loss function to use. If None, uses cross-entropy for classification
                 and mean squared error for regression.
        callbacks: List of callback functions to call after each epoch
        early_stopping: Whether to use early stopping
        patience: Number of epochs to wait for improvement before stopping
        
    Returns:
        Dictionary containing training history (loss and metrics for each epoch)
    """
    from neurenix.nn.loss import MSELoss, CrossEntropyLoss
    
    # Set model to training mode
    model.train(True)
    
    # Default loss function based on output shape
    if loss_fn is None:
        # Determine if this is a classification or regression task
        # This is a simplified heuristic - in practice, we'd want a more robust approach
        if len(train_labels[0].shape) == 1 or train_labels[0].shape[-1] > 1:
            loss_fn = CrossEntropyLoss()
        else:
            loss_fn = MSELoss()
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [] if val_data is not None else None,
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(epochs):
        # Shuffle the training data
        indices = np.random.permutation(len(train_data))
        
        # Training
        epoch_loss = 0.0
        num_batches = (len(train_data) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            # Get batch indices
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(train_data))
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            batch_data = [train_data[i] for i in batch_indices]
            batch_labels = [train_labels[i] for i in batch_indices]
            
            try:
                batch_data_tensor = Tensor.stack(batch_data)
            except (ValueError, RuntimeError):
                shapes = [data.shape for data in batch_data]
                max_dims = []
                
                for i in range(max(len(shape) for shape in shapes)):
                    max_dim = max((shape[i] if i < len(shape) else 0) for shape in shapes)
                    max_dims.append(max_dim)
                
                padded_batch = []
                for data in batch_data:
                    if len(data.shape) < len(max_dims):
                        for _ in range(len(max_dims) - len(data.shape)):
                            data = data.unsqueeze(-1)
                    
                    padding = []
                    for i, dim in enumerate(data.shape):
                        pad_size = max_dims[i] - dim
                        padding.append((0, pad_size))
                    
                    padded_data = Tensor.pad(data, padding, value=0)
                    padded_batch.append(padded_data)
                
                batch_data_tensor = Tensor.stack(padded_batch)
            
            try:
                batch_labels_tensor = Tensor.stack(batch_labels)
            except (ValueError, RuntimeError):
                if all(isinstance(label.item() if label.numel() == 1 else None, int) for label in batch_labels):
                    num_classes = max(label.item() for label in batch_labels) + 1
                    one_hot_labels = []
                    
                    for label in batch_labels:
                        one_hot = Tensor.zeros(num_classes)
                        one_hot[label.item()] = 1
                        one_hot_labels.append(one_hot)
                    
                    batch_labels_tensor = Tensor.stack(one_hot_labels)
                else:
                    shapes = [label.shape for label in batch_labels]
                    max_dims = []
                    
                    for i in range(max(len(shape) for shape in shapes)):
                        max_dim = max((shape[i] if i < len(shape) else 0) for shape in shapes)
                        max_dims.append(max_dim)
                    
                    padded_batch = []
                    for label in batch_labels:
                        if len(label.shape) < len(max_dims):
                            for _ in range(len(max_dims) - len(label.shape)):
                                label = label.unsqueeze(-1)
                        
                        padding = []
                        for i, dim in enumerate(label.shape):
                            pad_size = max_dims[i] - dim
                            padding.append((0, pad_size))
                        
                        padded_label = Tensor.pad(label, padding, value=0)
                        padded_batch.append(padded_label)
                    
                    batch_labels_tensor = Tensor.stack(padded_batch)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_data_tensor)
            
            # Compute loss
            loss = loss_fn(outputs, batch_labels_tensor)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            # Accumulate loss
            epoch_loss += loss.item() * len(batch_indices)
        
        # Average loss for the epoch
        epoch_loss /= len(train_data)
        history['train_loss'].append(epoch_loss)
        
        # Validation
        if val_data is not None and val_labels is not None:
            val_loss = 0.0
            
            # Set model to evaluation mode
            model.train(False)
            
            # Disable gradient computation for validation
            with Tensor.no_grad():
                num_val_batches = (len(val_data) + batch_size - 1) // batch_size
                
                for batch_idx in range(num_val_batches):
                    # Get batch indices
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(val_data))
                    
                    # Get batch data
                    batch_data = val_data[start_idx:end_idx]
                    batch_labels = val_labels[start_idx:end_idx]
                    
                    # Stack tensors into batches
                    batch_data_tensor = Tensor.stack(batch_data)
                    batch_labels_tensor = Tensor.stack(batch_labels)
                    
                    # Forward pass
                    outputs = model(batch_data_tensor)
                    
                    # Compute loss
                    loss = loss_fn(outputs, batch_labels_tensor)
                    
                    # Accumulate loss
                    val_loss += loss.item() * len(batch_data)
            
            # Average validation loss
            val_loss /= len(val_data)
            history['val_loss'].append(val_loss)
            
            # Early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Set model back to training mode
            model.train(True)
        
        # Call callbacks
        if callbacks is not None:
            for callback in callbacks:
                callback(epoch, history)
        
        # Print progress
        if val_data is not None:
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")
    
    # Set model to evaluation mode
    model.train(False)
    
    return history
