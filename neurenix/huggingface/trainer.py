"""
Hugging Face trainer integration for Neurenix.

This module provides functionality for training Hugging Face models
in Neurenix, including fine-tuning and transfer learning.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np

try:
    import torch
    import transformers
    from transformers import Trainer as HFTrainer
    from transformers import TrainingArguments as HFTrainingArguments
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

from neurenix.tensor import Tensor
from neurenix.nn.module import Module
from neurenix.huggingface.model import HuggingFaceModel


class Trainer:
    """
    Trainer for Hugging Face models in Neurenix.
    
    This class provides functionality for training Hugging Face models
    in Neurenix, including fine-tuning and transfer learning.
    """
    
    def __init__(
        self,
        model: Union[HuggingFaceModel, "transformers.PreTrainedModel"],
        args: Optional[Dict[str, Any]] = None,
        train_dataset = None,
        eval_dataset = None,
        compute_metrics: Optional[Callable] = None,
        optimizers: Optional[Tuple] = None,
        callbacks: Optional[List[Any]] = None,
        name: str = "HuggingFaceTrainer",
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            compute_metrics: Function to compute metrics
            optimizers: Tuple of (optimizer, scheduler)
            callbacks: List of callbacks
            name: Trainer name
        """
        if not HUGGINGFACE_AVAILABLE:
            raise ImportError(
                "Hugging Face is not available. Please install it with 'pip install transformers torch'."
            )
        
        self.name = name
        
        # Get Hugging Face model
        if isinstance(model, HuggingFaceModel):
            self.model = model.model
        else:
            self.model = model
        
        # Create training arguments
        self.args = self._create_training_args(args or {})
        
        # Create trainer
        self.trainer = HFTrainer(
            model=self.model,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            callbacks=callbacks,
        )
    
    def _create_training_args(self, args: Dict[str, Any]) -> "transformers.TrainingArguments":
        """
        Create Hugging Face training arguments.
        
        Args:
            args: Training arguments
            
        Returns:
            Hugging Face training arguments
        """
        # Default arguments
        default_args = {
            "output_dir": "./results",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "warmup_steps": 500,
            "weight_decay": 0.01,
            "logging_dir": "./logs",
            "logging_steps": 10,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "load_best_model_at_end": True,
        }
        
        # Update with provided arguments
        default_args.update(args)
        
        # Create training arguments
        return HFTrainingArguments(**default_args)
    
    def train(self) -> Dict[str, float]:
        """
        Train the model.
        
        Returns:
            Training metrics
        """
        # Train model
        result = self.trainer.train()
        
        # Convert to dict
        if hasattr(result, "metrics"):
            return result.metrics
        else:
            return {}
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Returns:
            Evaluation metrics
        """
        # Evaluate model
        result = self.trainer.evaluate()
        
        # Return metrics
        return result
    
    def predict(self, test_dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with the model.
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Tuple of (predictions, labels, metrics)
        """
        # Make predictions
        result = self.trainer.predict(test_dataset)
        
        # Return predictions, labels, metrics
        return result.predictions, result.label_ids, result.metrics
    
    def save_model(self, path: str):
        """
        Save model to disk.
        
        Args:
            path: Path to save model to
        """
        # Save model
        self.trainer.save_model(path)
    
    def push_to_hub(self, repo_id: str, **kwargs):
        """
        Push model to Hugging Face Hub.
        
        Args:
            repo_id: Repository ID
            **kwargs: Additional arguments
        """
        # Push to hub
        self.trainer.push_to_hub(repo_id=repo_id, **kwargs)


class FineTuningTrainer(Trainer):
    """
    Fine-tuning trainer for Hugging Face models in Neurenix.
    
    This class provides functionality for fine-tuning Hugging Face models
    in Neurenix, with additional features for transfer learning.
    """
    
    def __init__(
        self,
        model: Union[HuggingFaceModel, "transformers.PreTrainedModel"],
        args: Optional[Dict[str, Any]] = None,
        train_dataset = None,
        eval_dataset = None,
        compute_metrics: Optional[Callable] = None,
        optimizers: Optional[Tuple] = None,
        callbacks: Optional[List[Any]] = None,
        freeze_base_model: bool = False,
        freeze_layers: Optional[List[str]] = None,
        name: str = "FineTuningTrainer",
    ):
        """
        Initialize fine-tuning trainer.
        
        Args:
            model: Model to train
            args: Training arguments
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            compute_metrics: Function to compute metrics
            optimizers: Tuple of (optimizer, scheduler)
            callbacks: List of callbacks
            freeze_base_model: Whether to freeze the base model
            freeze_layers: List of layer names to freeze
            name: Trainer name
        """
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=optimizers,
            callbacks=callbacks,
            name=name,
        )
        
        # Freeze layers
        if freeze_base_model:
            self._freeze_base_model()
        elif freeze_layers:
            self._freeze_layers(freeze_layers)
    
    def _freeze_base_model(self):
        """Freeze all parameters of the base model."""
        for param in self.model.base_model.parameters():
            param.requires_grad = False
    
    def _freeze_layers(self, layer_names: List[str]):
        """
        Freeze specific layers of the model.
        
        Args:
            layer_names: List of layer names to freeze
        """
        for name, param in self.model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
    
    def unfreeze_layers(self, layer_names: Optional[List[str]] = None):
        """
        Unfreeze specific layers of the model.
        
        Args:
            layer_names: List of layer names to unfreeze
        """
        if layer_names is None:
            # Unfreeze all layers
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            # Unfreeze specific layers
            for name, param in self.model.named_parameters():
                if any(layer_name in name for layer_name in layer_names):
                    param.requires_grad = True
    
    def train_with_gradual_unfreezing(
        self,
        layer_groups: List[List[str]],
        epochs_per_group: int = 1,
    ) -> Dict[str, float]:
        """
        Train the model with gradual unfreezing.
        
        Args:
            layer_groups: List of layer name groups to unfreeze
            epochs_per_group: Number of epochs to train each group
            
        Returns:
            Training metrics
        """
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Train with gradual unfreezing
        metrics = {}
        for i, layer_group in enumerate(layer_groups):
            # Unfreeze layer group
            self.unfreeze_layers(layer_group)
            
            # Update number of epochs
            self.args.num_train_epochs = epochs_per_group
            
            # Train model
            result = self.trainer.train()
            
            # Update metrics
            if hasattr(result, "metrics"):
                metrics[f"group_{i}"] = result.metrics
        
        return metrics
