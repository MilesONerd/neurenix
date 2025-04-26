"""
Counterfactual explanation implementation for Neurenix.

This module provides tools for generating counterfactual explanations,
which show how to change the input to get a different prediction.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
import logging

from neurenix.tensor import Tensor
from neurenix.nn.module import Module

logger = logging.getLogger(__name__)

class Counterfactual:
    """
    Counterfactual explanation implementation.
    
    Generates counterfactual examples that show how to change the input
    to get a different prediction from the model.
    """
    
    def __init__(self, 
                 model: Module, 
                 feature_names: Optional[List[str]] = None,
                 categorical_features: Optional[List[int]] = None,
                 feature_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
                 random_state: int = 0):
        """
        Initialize a counterfactual explainer.
        
        Args:
            model: The model to explain
            feature_names: Names of features for the explanation
            categorical_features: Indices of categorical features
            feature_ranges: Valid ranges for features (min, max)
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.feature_names = feature_names
        self.categorical_features = categorical_features or []
        self.feature_ranges = feature_ranges or {}
        self.random_state = random_state
        
        try:
            from neurenix.binding import phynexus
            self._has_phynexus = True
            self._phynexus = phynexus
        except ImportError:
            logger.warning("Phynexus extension not found. Using pure Python implementation.")
            self._has_phynexus = False
    
    def generate(self, 
                 sample: Union[Tensor, np.ndarray], 
                 target_class: Optional[int] = None,
                 target_pred: Optional[float] = None,
                 max_iter: int = 1000,
                 **kwargs) -> Dict[str, Any]:
        """
        Generate counterfactual examples.
        
        Args:
            sample: Sample to explain
            target_class: Target class for classification (mutually exclusive with target_pred)
            target_pred: Target prediction value for regression (mutually exclusive with target_class)
            max_iter: Maximum number of iterations
            **kwargs: Additional arguments for the generator
            
        Returns:
            Dictionary containing counterfactual information
        """
        sample_tensor = self._convert_to_tensor(sample)
        
        if self._has_phynexus:
            return self._generate_native(sample_tensor, target_class, target_pred, max_iter, **kwargs)
        else:
            return self._generate_python(sample_tensor, target_class, target_pred, max_iter, **kwargs)
    
    def _convert_to_tensor(self, data: Union[Tensor, np.ndarray]) -> Tensor:
        """Convert input data to Tensor if needed."""
        if isinstance(data, np.ndarray):
            return Tensor(data)
        return data
    
    def _run_model(self, inputs: Tensor) -> Tensor:
        """Run the model on the given inputs."""
        self.model.eval()  # Set model to evaluation mode
        with Tensor.no_grad():
            return self.model(inputs)
    
    def _generate_native(self, 
                         sample: Tensor, 
                         target_class: Optional[int],
                         target_pred: Optional[float],
                         max_iter: int,
                         **kwargs) -> Dict[str, Any]:
        """Use native Phynexus implementation for counterfactual generation."""
        if not hasattr(self._phynexus, "counterfactual"):
            logger.warning("Native counterfactual not available. Using Python implementation.")
            return self._generate_python(sample, target_class, target_pred, max_iter, **kwargs)
        
        model_fn = lambda x: self._run_model(Tensor(x)).numpy()
        
        result = self._phynexus.counterfactual(
            model_fn,
            sample.numpy(),
            target_class,
            target_pred,
            self.categorical_features,
            self.feature_ranges,
            max_iter,
            self.random_state
        )
        
        feature_names = self.feature_names
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(sample.shape[0])]
        
        return {
            "counterfactual": Tensor(result["counterfactual"]),
            "original_prediction": result["original_prediction"],
            "counterfactual_prediction": result["counterfactual_prediction"],
            "feature_names": feature_names,
            "changes": result["changes"],
            "success": result["success"]
        }
    
    def _generate_python(self, 
                         sample: Tensor, 
                         target_class: Optional[int],
                         target_pred: Optional[float],
                         max_iter: int,
                         **kwargs) -> Dict[str, Any]:
        """Pure Python implementation of counterfactual generation."""
        
        np.random.seed(self.random_state)
        
        original_pred = self._run_model(sample).numpy()
        
        if target_class is not None:
            if len(original_pred.shape) > 0 and original_pred.shape[0] > 1:
                target = np.zeros_like(original_pred)
                target[target_class] = 1.0
            else:
                target = 1.0 if target_class == 1 else 0.0
        elif target_pred is not None:
            target = target_pred
        else:
            if len(original_pred.shape) > 0 and original_pred.shape[0] > 1:
                current_class = np.argmax(original_pred)
                target_class = (current_class + 1) % original_pred.shape[0]
                target = np.zeros_like(original_pred)
                target[target_class] = 1.0
            else:
                target = 1.0 - original_pred if original_pred < 0.5 else 0.0
        
        counterfactual = sample.clone().numpy()
        
        success = False
        changes = {}
        
        for i in range(max_iter):
            feature_idx = np.random.randint(0, counterfactual.shape[0])
            
            if feature_idx in self.categorical_features:
                continue
            
            if feature_idx in self.feature_ranges:
                min_val, max_val = self.feature_ranges[feature_idx]
            else:
                min_val, max_val = 0.0, 1.0
            
            if np.random.random() < 0.5:
                counterfactual[feature_idx] = min(counterfactual[feature_idx] + 0.1, max_val)
            else:
                counterfactual[feature_idx] = max(counterfactual[feature_idx] - 0.1, min_val)
            
            counterfactual_tensor = Tensor(counterfactual)
            counterfactual_pred = self._run_model(counterfactual_tensor).numpy()
            
            if target_class is not None:
                if len(counterfactual_pred.shape) > 0 and counterfactual_pred.shape[0] > 1:
                    pred_class = np.argmax(counterfactual_pred)
                    if pred_class == target_class:
                        success = True
                        break
                else:
                    if (counterfactual_pred > 0.5 and target_class == 1) or (counterfactual_pred <= 0.5 and target_class == 0):
                        success = True
                        break
            else:
                if np.abs(counterfactual_pred - target) < 0.1:
                    success = True
                    break
        
        for i in range(sample.shape[0]):
            if np.abs(counterfactual[i] - sample.numpy()[i]) > 1e-6:
                changes[i] = (sample.numpy()[i], counterfactual[i])
        
        feature_names = self.feature_names
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(sample.shape[0])]
        
        return {
            "counterfactual": Tensor(counterfactual),
            "original_prediction": original_pred,
            "counterfactual_prediction": counterfactual_pred,
            "feature_names": feature_names,
            "changes": changes,
            "success": success
        }
    
    def plot(self, 
             counterfactual_result: Dict[str, Any], 
             **kwargs):
        """
        Plot counterfactual explanation.
        
        Args:
            counterfactual_result: Dictionary containing counterfactual information from generate()
            **kwargs: Additional arguments for the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            original = counterfactual_result.get("original", None)
            counterfactual = counterfactual_result.get("counterfactual", None)
            feature_names = counterfactual_result.get("feature_names", [])
            changes = counterfactual_result.get("changes", {})
            
            if original is None or counterfactual is None:
                logger.warning("No data to plot.")
                return
            
            original_np = original.numpy() if isinstance(original, Tensor) else original
            counterfactual_np = counterfactual.numpy() if isinstance(counterfactual, Tensor) else counterfactual
            
            plt.figure(figsize=(10, 6))
            
            changed_features = list(changes.keys())
            if not changed_features:
                logger.warning("No features changed.")
                return
            
            feature_indices = np.array(changed_features)
            feature_labels = [feature_names[i] for i in feature_indices]
            
            original_values = np.array([original_np[i] for i in feature_indices])
            counterfactual_values = np.array([counterfactual_np[i] for i in feature_indices])
            
            x = np.arange(len(feature_indices))
            width = 0.35
            
            plt.bar(x - width/2, original_values, width, label='Original')
            plt.bar(x + width/2, counterfactual_values, width, label='Counterfactual')
            
            plt.xlabel('Features')
            plt.ylabel('Values')
            plt.title('Counterfactual Explanation')
            plt.xticks(x, feature_labels, rotation=45, ha='right')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not installed. Cannot plot counterfactual.")
