"""
Knowledge Distillation implementation for continual learning.

Knowledge Distillation transfers knowledge from a teacher model to a student
model, which can be used to preserve knowledge when learning new tasks.
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import neurenix
from neurenix.nn import Module
from neurenix.tensor import Tensor
from neurenix.optim import Optimizer

class KnowledgeDistillation:
    """
    Knowledge Distillation for continual learning.
    
    Uses a teacher model (trained on previous tasks) to guide the learning
    of a student model (being trained on new tasks) to prevent forgetting.
    
    Attributes:
        teacher_model: Model trained on previous tasks
        student_model: Model being trained on new tasks
        temperature: Temperature for softening probability distributions
        alpha: Weight for distillation loss vs. task loss
    """
    
    def __init__(
        self, 
        teacher_model: Module, 
        student_model: Module,
        temperature: float = 2.0,
        alpha: float = 0.5
    ):
        """
        Initialize Knowledge Distillation.
        
        Args:
            teacher_model: Model trained on previous tasks
            student_model: Model being trained on new tasks
            temperature: Temperature for softening probability distributions
            alpha: Weight for distillation loss vs. task loss
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        try:
            from neurenix.binding import get_phynexus
            self._phynexus = get_phynexus()
            self._has_phynexus = True
        except ImportError:
            self._has_phynexus = False
    
    def distillation_loss(
        self, 
        student_logits: Tensor, 
        teacher_logits: Tensor
    ) -> Tensor:
        """
        Compute distillation loss between teacher and student outputs.
        
        Args:
            student_logits: Output logits from student model
            teacher_logits: Output logits from teacher model
            
        Returns:
            Distillation loss tensor
        """
        if self._has_phynexus:
            return self._phynexus.continual.compute_distillation_loss(
                student_logits, teacher_logits, self.temperature
            )
        
        soft_targets = neurenix.softmax(teacher_logits / self.temperature, dim=1)
        soft_prob = neurenix.log_softmax(student_logits / self.temperature, dim=1)
        
        loss = -(soft_targets * soft_prob).sum(dim=1).mean()
        
        return loss * (self.temperature ** 2)
    
    def combined_loss(
        self, 
        student_logits: Tensor, 
        teacher_logits: Tensor, 
        targets: Tensor, 
        task_loss_fn: Callable
    ) -> Tensor:
        """
        Compute combined loss (task loss + distillation loss).
        
        Args:
            student_logits: Output logits from student model
            teacher_logits: Output logits from teacher model
            targets: Ground truth targets
            task_loss_fn: Loss function for the task
            
        Returns:
            Combined loss tensor
        """
        if self._has_phynexus:
            return self._phynexus.continual.compute_combined_distillation_loss(
                student_logits, teacher_logits, targets,
                task_loss_fn, self.temperature, self.alpha
            )
        
        task_loss = task_loss_fn(student_logits, targets)
        
        dist_loss = self.distillation_loss(student_logits, teacher_logits)
        
        return self.alpha * task_loss + (1 - self.alpha) * dist_loss
    
    def train_step(
        self, 
        inputs: Tensor, 
        targets: Tensor, 
        task_loss_fn: Callable,
        optimizer: Optimizer
    ) -> Tuple[Tensor, Tensor]:
        """
        Perform a training step with knowledge distillation.
        
        Args:
            inputs: Input tensors
            targets: Target tensors
            task_loss_fn: Loss function for the task
            optimizer: Optimizer for the student model
            
        Returns:
            Tuple of (task_loss, distillation_loss)
        """
        self.teacher_model.eval()
        self.student_model.train()
        
        with neurenix.no_grad():
            teacher_logits = self.teacher_model(inputs)
        
        student_logits = self.student_model(inputs)
        
        task_loss = task_loss_fn(student_logits, targets)
        
        dist_loss = self.distillation_loss(student_logits, teacher_logits)
        
        combined_loss = self.alpha * task_loss + (1 - self.alpha) * dist_loss
        
        optimizer.zero_grad()
        combined_loss.backward()
        optimizer.step()
        
        return task_loss, dist_loss
    
    def create_snapshot(self) -> Module:
        """
        Create a snapshot of the current student model to use as a teacher.
        
        Returns:
            Copy of the student model
        """
        snapshot = type(self.student_model)(*self.student_model._init_args, **self.student_model._init_kwargs)
        
        snapshot.load_state_dict(self.student_model.state_dict())
        
        for param in snapshot.parameters():
            param.requires_grad = False
        
        return snapshot
    
    def update_teacher(self):
        """
        Update the teacher model with the current student model.
        """
        self.teacher_model.load_state_dict(self.student_model.state_dict())
        
        for param in self.teacher_model.parameters():
            param.requires_grad = False
