"""
Experience Replay implementation for continual learning.

Experience Replay is a memory-based approach that stores and replays
examples from previous tasks to prevent catastrophic forgetting.
"""

import random
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from collections import deque

import neurenix
from neurenix.nn import Module
from neurenix.tensor import Tensor
from neurenix.optim import Optimizer

class ExperienceReplay:
    """
    Experience Replay for continual learning.
    
    Stores examples from previous tasks and replays them during training
    on new tasks to prevent catastrophic forgetting.
    
    Attributes:
        memory_size: Maximum number of examples to store
        memory: Storage for examples from previous tasks
        strategy: Strategy for selecting examples to replay
    """
    
    def __init__(
        self, 
        memory_size: int = 1000,
        strategy: str = "random",
        per_class: bool = True,
        sample_size: Optional[int] = None
    ):
        """
        Initialize Experience Replay.
        
        Args:
            memory_size: Maximum number of examples to store
            strategy: Strategy for selecting examples to replay
                      ("random", "reservoir", "importance")
            per_class: Whether to maintain a balanced memory per class
            sample_size: Number of examples to sample during replay
                         (defaults to 10% of memory_size if None)
        """
        self.memory_size = memory_size
        self.strategy = strategy
        self.per_class = per_class
        self.sample_size = sample_size or max(1, memory_size // 10)
        
        if per_class:
            self.memory: Dict[int, List[Tuple[Tensor, Tensor]]] = {}
        else:
            self.memory: List[Tuple[Tensor, Tensor]] = []
        
        try:
            from neurenix.binding import get_phynexus
            self._phynexus = get_phynexus()
            self._has_phynexus = True
        except ImportError:
            self._has_phynexus = False
    
    def update_memory(self, inputs: Tensor, targets: Tensor, task_id: Optional[int] = None):
        """
        Update memory with new examples.
        
        Args:
            inputs: Input tensors
            targets: Target tensors
            task_id: Task identifier (optional)
        """
        if self._has_phynexus:
            self._phynexus.continual.update_replay_memory(
                self, inputs, targets, task_id
            )
            return
        
        if self.per_class:
            for i in range(len(inputs)):
                x = inputs[i:i+1]
                y = targets[i:i+1]
                
                if len(y.shape) > 1 and y.shape[1] > 1:
                    class_idx = int(neurenix.argmax(y).item())
                else:
                    class_idx = int(y.item())
                
                if class_idx not in self.memory:
                    self.memory[class_idx] = []
                
                if self.strategy == "reservoir":
                    class_memory = self.memory[class_idx]
                    if len(class_memory) < self.memory_size // len(self.memory):
                        class_memory.append((x.clone().detach(), y.clone().detach()))
                    else:
                        t = len(class_memory) + 1
                        if random.random() < self.memory_size / t:
                            idx = random.randint(0, len(class_memory) - 1)
                            class_memory[idx] = (x.clone().detach(), y.clone().detach())
                else:
                    class_memory = self.memory[class_idx]
                    class_memory.append((x.clone().detach(), y.clone().detach()))
                    
                    max_per_class = self.memory_size // len(self.memory)
                    if len(class_memory) > max_per_class:
                        if self.strategy == "random":
                            idx = random.randint(0, len(class_memory) - 2)
                            class_memory.pop(idx)
                        else:
                            class_memory.pop(0)
        else:
            for i in range(len(inputs)):
                x = inputs[i:i+1]
                y = targets[i:i+1]
                
                if self.strategy == "reservoir":
                    if len(self.memory) < self.memory_size:
                        self.memory.append((x.clone().detach(), y.clone().detach()))
                    else:
                        t = len(self.memory) + 1
                        if random.random() < self.memory_size / t:
                            idx = random.randint(0, len(self.memory) - 1)
                            self.memory[idx] = (x.clone().detach(), y.clone().detach())
                else:
                    self.memory.append((x.clone().detach(), y.clone().detach()))
                    
                    if len(self.memory) > self.memory_size:
                        if self.strategy == "random":
                            idx = random.randint(0, len(self.memory) - 2)
                            self.memory.pop(idx)
                        else:
                            self.memory.pop(0)
    
    def sample_memory(self, batch_size: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """
        Sample examples from memory.
        
        Args:
            batch_size: Number of examples to sample (defaults to self.sample_size)
            
        Returns:
            Tuple of (inputs, targets) tensors
        """
        if self._has_phynexus:
            return self._phynexus.continual.sample_replay_memory(
                self, batch_size or self.sample_size
            )
        
        batch_size = batch_size or self.sample_size
        
        if self.per_class:
            samples_per_class = batch_size // len(self.memory)
            remainder = batch_size % len(self.memory)
            
            inputs_list = []
            targets_list = []
            
            for class_idx, class_memory in self.memory.items():
                n_samples = samples_per_class + (1 if remainder > 0 else 0)
                remainder = max(0, remainder - 1)
                
                if n_samples == 0 or not class_memory:
                    continue
                
                if n_samples > len(class_memory):
                    indices = [random.randint(0, len(class_memory) - 1) for _ in range(n_samples)]
                else:
                    indices = random.sample(range(len(class_memory)), n_samples)
                
                for idx in indices:
                    x, y = class_memory[idx]
                    inputs_list.append(x)
                    targets_list.append(y)
        else:
            if batch_size > len(self.memory):
                indices = [random.randint(0, len(self.memory) - 1) for _ in range(batch_size)]
            else:
                indices = random.sample(range(len(self.memory)), batch_size)
            
            inputs_list = [self.memory[idx][0] for idx in indices]
            targets_list = [self.memory[idx][1] for idx in indices]
        
        inputs = neurenix.cat(inputs_list, dim=0)
        targets = neurenix.cat(targets_list, dim=0)
        
        return inputs, targets
    
    def get_replay_loader(self, batch_size: int = 32) -> Callable:
        """
        Get a data loader function for replaying memory.
        
        Args:
            batch_size: Batch size for the loader
            
        Returns:
            Generator function that yields (inputs, targets) batches
        """
        def replay_loader():
            return self.sample_memory(batch_size)
        
        return replay_loader
    
    def clear_memory(self):
        """Clear all examples from memory."""
        if self.per_class:
            self.memory = {}
        else:
            self.memory = []
    
    def get_memory_size(self) -> int:
        """
        Get the current number of examples in memory.
        
        Returns:
            Number of examples in memory
        """
        if self.per_class:
            return sum(len(examples) for examples in self.memory.values())
        else:
            return len(self.memory)
