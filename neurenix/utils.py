"""
Utility functions for the Neurenix framework.
"""

import os
import sys
import time
import random
import inspect
from typing import List, Tuple, Dict, Any, Optional, Union, Sequence

import numpy as np

from neurenix.tensor import Tensor
from neurenix.device import Device

def seed_everything(seed: int) -> None:
    """
    Set random seed for all random number generators.
    
    Args:
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Set seed for Neurenix tensors
    Tensor.manual_seed(seed)

def to_numpy(tensor: Union[Tensor, np.ndarray]) -> np.ndarray:
    """
    Convert a tensor to a numpy array.
    
    Args:
        tensor: Tensor or numpy array
        
    Returns:
        Numpy array
    """
    if isinstance(tensor, Tensor):
        return tensor.numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError(f"Cannot convert {type(tensor)} to numpy array")

def to_tensor(data: Union[Tensor, np.ndarray, List, Tuple], device: Optional[Union[str, Device]] = None) -> Tensor:
    """
    Convert data to a Neurenix tensor.
    
    Args:
        data: Data to convert
        device: Device to store the tensor on
        
    Returns:
        Neurenix tensor
    """
    if isinstance(data, Tensor):
        if device is not None:
            return data.to(device)
        return data
    elif isinstance(data, np.ndarray):
        return Tensor(data, device=device)
    else:
        return Tensor(np.array(data), device=device)

def one_hot(indices: Union[Tensor, np.ndarray, List[int]], num_classes: int) -> Tensor:
    """
    Convert indices to one-hot encoding.
    
    Args:
        indices: Class indices
        num_classes: Number of classes
        
    Returns:
        One-hot encoded tensor
    """
    indices = to_numpy(indices).astype(np.int64)
    one_hot = np.zeros((indices.size, num_classes), dtype=np.float32)
    one_hot[np.arange(indices.size), indices.flatten()] = 1.0
    one_hot = one_hot.reshape((*indices.shape, num_classes))
    return Tensor(one_hot)

def get_module_device(module) -> Optional[Device]:
    """
    Get the device of a module by checking its parameters.
    
    Args:
        module: Module to check
        
    Returns:
        Device of the module or None if the module has no parameters
    """
    for param in module.parameters():
        return param.device
    return None

def move_module_to_device(module, device: Union[str, Device]) -> None:
    """
    Move a module to a device.
    
    Args:
        module: Module to move
        device: Device to move the module to
    """
    for param in module.parameters():
        param.to(device, inplace=True)

def count_parameters(module) -> int:
    """
    Count the number of trainable parameters in a module.
    
    Args:
        module: Module to count parameters for
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def model_summary(module) -> str:
    """
    Get a summary of a module.
    
    Args:
        module: Module to summarize
        
    Returns:
        Summary string
    """
    lines = [
        f"Model Summary for {module.__class__.__name__}:",
        f"Total parameters: {count_parameters(module):,}",
        "Layers:",
    ]
    
    for name, child in module.named_children():
        lines.append(f"  {name}: {child.__class__.__name__} ({count_parameters(child):,} parameters)")
    
    return "\n".join(lines)

def timeit(func):
    """
    Decorator to measure the execution time of a function.
    
    Args:
        func: Function to measure
        
    Returns:
        Wrapped function
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute")
        return result
    return wrapper

def get_learning_rate(optimizer) -> float:
    """
    Get the learning rate of an optimizer.
    
    Args:
        optimizer: Optimizer to get learning rate from
        
    Returns:
        Learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    return 0.0

def set_learning_rate(optimizer, lr: float) -> None:
    """
    Set the learning rate of an optimizer.
    
    Args:
        optimizer: Optimizer to set learning rate for
        lr: Learning rate
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

class LRScheduler:
    """
    Base class for learning rate schedulers.
    """
    
    def __init__(self, optimizer, last_epoch: int = -1):
        """
        Initialize the learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule learning rate for
            last_epoch: The index of the last epoch
        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.step()
    
    def get_lr(self) -> List[float]:
        """
        Get the learning rate.
        
        Returns:
            List of learning rates for each parameter group
        """
        raise NotImplementedError("Subclasses must implement get_lr method")
    
    def step(self, epoch: Optional[int] = None) -> None:
        """
        Step the learning rate scheduler.
        
        Args:
            epoch: Epoch to step to
        """
        if epoch is None:
            self.last_epoch += 1
            epoch = self.last_epoch
        else:
            self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class StepLR(LRScheduler):
    """
    Step learning rate scheduler.
    
    Decays the learning rate by gamma every step_size epochs.
    
    Args:
        optimizer: Optimizer to schedule learning rate for
        step_size: Period of learning rate decay
        gamma: Multiplicative factor of learning rate decay
        last_epoch: The index of the last epoch
    """
    
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]


class ExponentialLR(LRScheduler):
    """
    Exponential learning rate scheduler.
    
    Decays the learning rate by gamma every epoch.
    
    Args:
        optimizer: Optimizer to schedule learning rate for
        gamma: Multiplicative factor of learning rate decay
        last_epoch: The index of the last epoch
    """
    
    def __init__(self, optimizer, gamma: float = 0.1, last_epoch: int = -1):
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        return [base_lr * self.gamma ** self.last_epoch for base_lr in self.base_lrs]


class ReduceLROnPlateau:
    """
    Reduce learning rate when a metric has stopped improving.
    
    Args:
        optimizer: Optimizer to schedule learning rate for
        mode: One of 'min' or 'max'. In 'min' mode, the learning rate will be reduced when the
              quantity monitored has stopped decreasing; in 'max' mode it will be reduced when
              the quantity monitored has stopped increasing
        factor: Factor by which the learning rate will be reduced
        patience: Number of epochs with no improvement after which learning rate will be reduced
        threshold: Threshold for measuring the new optimum
        threshold_mode: One of 'rel', 'abs'. In 'rel' mode, the threshold is a relative change;
                       in 'abs' mode, it is an absolute change
        cooldown: Number of epochs to wait before resuming normal operation after lr has been reduced
        min_lr: Lower bound on the learning rate
        eps: Minimal decay applied to lr
        verbose: If True, prints a message to stdout for each update
    """
    
    def __init__(
        self,
        optimizer,
        mode: str = "min",
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        threshold_mode: str = "rel",
        cooldown: int = 0,
        min_lr: float = 0.0,
        eps: float = 1e-8,
        verbose: bool = False,
    ):
        if mode not in ["min", "max"]:
            raise ValueError(f"Mode {mode} is unknown, only 'min' and 'max' are supported")
        if threshold_mode not in ["rel", "abs"]:
            raise ValueError(f"Threshold mode {threshold_mode} is unknown, only 'rel' and 'abs' are supported")
        
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.eps = eps
        self.verbose = verbose
        
        self.cooldown_counter = 0
        self.best = float("inf") if mode == "min" else float("-inf")
        self.num_bad_epochs = 0
        self.last_epoch = -1
        
        self._init_is_better(mode=mode, threshold=threshold, threshold_mode=threshold_mode)
        self._reset()
    
    def _reset(self) -> None:
        """Reset the internal state."""
        self.best = float("inf") if self.mode == "min" else float("-inf")
        self.cooldown_counter = 0
        self.num_bad_epochs = 0
    
    def _init_is_better(self, mode: str, threshold: float, threshold_mode: str) -> None:
        """Initialize the is_better function based on the mode and threshold."""
        if mode == "min" and threshold_mode == "rel":
            self.is_better = lambda a, best: a < best * (1 - threshold)
        elif mode == "min" and threshold_mode == "abs":
            self.is_better = lambda a, best: a < best - threshold
        elif mode == "max" and threshold_mode == "rel":
            self.is_better = lambda a, best: a > best * (1 + threshold)
        else:  # mode == "max" and threshold_mode == "abs":
            self.is_better = lambda a, best: a > best + threshold
    
    def step(self, metrics: float) -> None:
        """
        Step the learning rate scheduler.
        
        Args:
            metrics: Metric to monitor
        """
        self.last_epoch += 1
        
        if self.is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0
        
        if self.num_bad_epochs > self.patience:
            self._reduce_lr()
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
    
    def _reduce_lr(self) -> None:
        """Reduce the learning rate."""
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lr)
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr
                if self.verbose:
                    print(f"Reducing learning rate of group {i} to {new_lr:.4e}")
