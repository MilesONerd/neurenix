"""
Loss functions for the Neurenix framework.

This module provides loss functions for training neural networks.
"""

from typing import Optional, Union

import numpy as np

from neurenix.nn.module import Module
from neurenix.tensor import Tensor

class Loss(Module):
    """
    Base class for loss functions.
    """
    
    def __init__(self, reduction: str = "mean"):
        """
        Initialize the loss function.
        
        Args:
            reduction: Specifies the reduction to apply to the output:
                      'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                      'mean': the sum of the output will be divided by the number of
                      elements in the output, 'sum': the output will be summed.
        """
        super().__init__()
        
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(f"Invalid reduction: {reduction}")
        
        self.reduction = reduction
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of the loss function.
        
        Args:
            input: Predicted values
            target: Target values
            
        Returns:
            Loss value
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def __call__(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Call the loss function.
        
        Args:
            input: Predicted values
            target: Target values
            
        Returns:
            Loss value
        """
        return self.forward(input, target)


class MSELoss(Loss):
    """
    Mean Squared Error (MSE) loss.
    
    Measures the average squared difference between the predicted values and the target values.
    
    Args:
        reduction: Specifies the reduction to apply to the output:
                  'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                  'mean': the sum of the output will be divided by the number of
                  elements in the output, 'sum': the output will be summed.
    """
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of the MSE loss.
        
        Args:
            input: Predicted values
            target: Target values
            
        Returns:
            MSE loss value
        """
        # Calculate squared difference
        loss = (input - target) ** 2
        
        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        else:  # sum
            return loss.sum()
    
    def __repr__(self) -> str:
        return f"MSELoss(reduction='{self.reduction}')"


class L1Loss(Loss):
    """
    Mean Absolute Error (MAE) loss.
    
    Measures the average absolute difference between the predicted values and the target values.
    
    Args:
        reduction: Specifies the reduction to apply to the output:
                  'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                  'mean': the sum of the output will be divided by the number of
                  elements in the output, 'sum': the output will be summed.
    """
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of the L1 loss.
        
        Args:
            input: Predicted values
            target: Target values
            
        Returns:
            L1 loss value
        """
        # Calculate absolute difference
        loss = (input - target).abs()
        
        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        else:  # sum
            return loss.sum()
    
    def __repr__(self) -> str:
        return f"L1Loss(reduction='{self.reduction}')"


class CrossEntropyLoss(Loss):
    """
    Cross Entropy Loss.
    
    Combines LogSoftmax and NLLLoss in one single class.
    
    Args:
        weight: A manual rescaling weight given to each class.
               If given, has to be a Tensor of size C
        ignore_index: Specifies a target value that is ignored
                     and does not contribute to the input gradient
        reduction: Specifies the reduction to apply to the output:
                  'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                  'mean': the sum of the output will be divided by the number of
                  elements in the output, 'sum': the output will be summed.
    """
    
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__(reduction)
        
        self.weight = weight
        self.ignore_index = ignore_index
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of the cross entropy loss.
        
        Args:
            input: Predicted values of shape (N, C) where N is the batch size and C is the number of classes
            target: Target class indices of shape (N) where each value is 0 <= target[i] <= C-1
            
        Returns:
            Cross entropy loss value
        """
        # Apply log softmax to input
        log_probs = input.log_softmax(dim=1)
        
        # Calculate negative log likelihood loss
        batch_size = input.shape[0]
        loss = Tensor.zeros((batch_size,), device=input.device)
        
        for i in range(batch_size):
            if target[i].item() != self.ignore_index:
                if self.weight is not None:
                    loss[i] = -log_probs[i, target[i].item()] * self.weight[target[i].item()]
                else:
                    loss[i] = -log_probs[i, target[i].item()]
        
        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            # Count non-ignored targets
            non_ignored = (target != self.ignore_index).sum().item()
            if non_ignored == 0:
                return Tensor.zeros((), device=input.device)
            return loss.sum() / non_ignored
        else:  # sum
            return loss.sum()
    
    def __repr__(self) -> str:
        return (
            f"CrossEntropyLoss(weight={self.weight is not None}, "
            f"ignore_index={self.ignore_index}, reduction='{self.reduction}')"
        )


class BCELoss(Loss):
    """
    Binary Cross Entropy Loss.
    
    Args:
        weight: A manual rescaling weight given to the loss of each batch element.
               If given, has to be a Tensor of size N
        reduction: Specifies the reduction to apply to the output:
                  'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                  'mean': the sum of the output will be divided by the number of
                  elements in the output, 'sum': the output will be summed.
    """
    
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__(reduction)
        
        self.weight = weight
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of the binary cross entropy loss.
        
        Args:
            input: Predicted values of shape (N, *) where * means any number of additional dimensions
            target: Target values of shape (N, *), same shape as the input
            
        Returns:
            Binary cross entropy loss value
        """
        # Clamp input to avoid log(0) and log(1)
        eps = 1e-12
        input_clamped = input.clamp(min=eps, max=1 - eps)
        
        # Calculate binary cross entropy loss
        loss = -(target * input_clamped.log() + (1 - target) * (1 - input_clamped).log())
        
        # Apply weight if provided
        if self.weight is not None:
            loss = loss * self.weight
        
        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        else:  # sum
            return loss.sum()
    
    def __repr__(self) -> str:
        return f"BCELoss(weight={self.weight is not None}, reduction='{self.reduction}')"


class BCEWithLogitsLoss(Loss):
    """
    Binary Cross Entropy with Logits Loss.
    
    Combines a Sigmoid layer and the BCELoss in one single class.
    
    Args:
        weight: A manual rescaling weight given to the loss of each batch element.
               If given, has to be a Tensor of size N
        pos_weight: A weight of positive examples.
                   Must be a vector with length equal to the number of classes.
        reduction: Specifies the reduction to apply to the output:
                  'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
                  'mean': the sum of the output will be divided by the number of
                  elements in the output, 'sum': the output will be summed.
    """
    
    def __init__(
        self,
        weight: Optional[Tensor] = None,
        pos_weight: Optional[Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__(reduction)
        
        self.weight = weight
        self.pos_weight = pos_weight
    
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass of the binary cross entropy with logits loss.
        
        Args:
            input: Predicted values (logits) of shape (N, *) where * means any number of additional dimensions
            target: Target values of shape (N, *), same shape as the input
            
        Returns:
            Binary cross entropy with logits loss value
        """
        # Calculate binary cross entropy with logits loss
        if self.pos_weight is not None:
            # With positive weights: loss = -pos_weight * target * log(sigmoid(input)) - (1 - target) * log(1 - sigmoid(input))
            # This can be rewritten as:
            # loss = (1 - target) * input + (1 + (pos_weight - 1) * target) * log(1 + exp(-input))
            loss = (1 - target) * input
            
            # Use log1p(exp(-input)) for numerical stability when input > 0
            # log1p(x) = log(1 + x)
            pos_term = (1 + (self.pos_weight - 1) * target)
            neg_input = -input
            
            # For input > 0, use log1p(exp(-input))
            mask_pos = input > 0
            loss_pos = pos_term * (neg_input + neg_input.exp().log1p())
            
            # For input <= 0, use input + log1p(exp(input))
            mask_neg = ~mask_pos
            loss_neg = pos_term * (neg_input.exp().log1p())
            
            loss = loss + mask_pos * loss_pos + mask_neg * loss_neg
        else:
            # Without positive weights: loss = -target * log(sigmoid(input)) - (1 - target) * log(1 - sigmoid(input))
            # This can be rewritten as:
            # loss = input - input * target + log(1 + exp(-input))
            loss = input - input * target
            
            # Use log1p(exp(-input)) for numerical stability when input > 0
            # log1p(x) = log(1 + x)
            neg_input = -input
            
            # For input > 0, use log1p(exp(-input))
            mask_pos = input > 0
            loss_pos = neg_input.exp().log1p()
            
            # For input <= 0, use -input + log1p(exp(input))
            mask_neg = ~mask_pos
            loss_neg = neg_input + input.exp().log1p()
            
            loss = loss + mask_pos * loss_pos + mask_neg * loss_neg
        
        # Apply weight if provided
        if self.weight is not None:
            loss = loss * self.weight
        
        # Apply reduction
        if self.reduction == "none":
            return loss
        elif self.reduction == "mean":
            return loss.mean()
        else:  # sum
            return loss.sum()
    
    def __repr__(self) -> str:
        return (
            f"BCEWithLogitsLoss(weight={self.weight is not None}, "
            f"pos_weight={self.pos_weight is not None}, reduction='{self.reduction}')"
        )
