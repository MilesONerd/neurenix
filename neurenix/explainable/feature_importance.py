"""
Feature importance module for Neurenix.

This module provides tools for calculating and visualizing feature importance
in machine learning models.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
import logging

from neurenix.tensor import Tensor
from neurenix.nn.module import Module

logger = logging.getLogger(__name__)

class FeatureImportance:
    """Base class for feature importance methods in Neurenix."""
    
    def __init__(self, 
                 model: Module, 
                 feature_names: Optional[List[str]] = None):
        """
        Initialize a feature importance calculator.
        
        Args:
            model: The model to explain
            feature_names: Names of features for the explanation
        """
        self.model = model
        self.feature_names = feature_names
        
        try:
            from neurenix.binding import phynexus
            self._has_phynexus = True
            self._phynexus = phynexus
        except ImportError:
            logger.warning("Phynexus extension not found. Using pure Python implementation.")
            self._has_phynexus = False
    
    def calculate(self, 
                  X: Union[Tensor, np.ndarray], 
                  y: Optional[Union[Tensor, np.ndarray]] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Calculate feature importance.
        
        Args:
            X: Input data
            y: Target data (optional)
            **kwargs: Additional arguments for the calculation
            
        Returns:
            Dictionary containing feature importance information
        """
        raise NotImplementedError("Subclasses must implement calculate()")
    
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
    
    def plot(self, 
             importances: Dict[str, Any], 
             **kwargs):
        """
        Plot feature importance.
        
        Args:
            importances: Dictionary containing importance information from calculate()
            **kwargs: Additional arguments for the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            values = importances.get("importances", [])
            feature_names = importances.get("feature_names", [])
            
            if not values or not feature_names:
                logger.warning("No features to plot.")
                return
            
            indices = np.argsort(values)
            
            plt.figure(figsize=(10, 6))
            plt.barh(
                [feature_names[i] for i in indices],
                [values[i] for i in indices]
            )
            plt.xlabel("Importance")
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not installed. Cannot plot importance.")


class PermutationImportance(FeatureImportance):
    """
    Permutation importance implementation.
    
    Calculates feature importance by randomly permuting feature values
    and measuring the decrease in model performance.
    """
    
    def __init__(self, 
                 model: Module, 
                 feature_names: Optional[List[str]] = None,
                 random_state: int = 0):
        """
        Initialize permutation importance calculator.
        
        Args:
            model: The model to explain
            feature_names: Names of features for the explanation
            random_state: Random seed for reproducibility
        """
        super().__init__(model, feature_names)
        self.random_state = random_state
        
    def calculate(self, 
                  X: Union[Tensor, np.ndarray], 
                  y: Optional[Union[Tensor, np.ndarray]] = None,
                  n_repeats: int = 5,
                  scoring: Optional[Union[str, Callable]] = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Calculate permutation importance.
        
        Args:
            X: Input data
            y: Target data (optional, required for supervised models)
            n_repeats: Number of times to permute each feature
            scoring: Scoring function or metric name
            **kwargs: Additional arguments for the calculation
            
        Returns:
            Dictionary containing feature importance information
        """
        X_tensor = self._convert_to_tensor(X)
        
        if self._has_phynexus:
            return self._calculate_native(X_tensor, y, n_repeats, scoring, **kwargs)
        else:
            return self._calculate_python(X_tensor, y, n_repeats, scoring, **kwargs)
    
    def _calculate_native(self, 
                          X: Tensor, 
                          y: Optional[Union[Tensor, np.ndarray]],
                          n_repeats: int,
                          scoring: Optional[Union[str, Callable]],
                          **kwargs) -> Dict[str, Any]:
        """Use native Phynexus implementation for permutation importance."""
        if not hasattr(self._phynexus, "permutation_importance"):
            logger.warning("Native permutation_importance not available. Using Python implementation.")
            return self._calculate_python(X, y, n_repeats, scoring, **kwargs)
        
        model_fn = lambda x: self._run_model(Tensor(x)).numpy()
        
        y_np = y.numpy() if isinstance(y, Tensor) else y
        
        result = self._phynexus.permutation_importance(
            model_fn,
            X.numpy(),
            y_np,
            n_repeats,
            self.random_state
        )
        
        feature_names = self.feature_names
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        return {
            "importances": result["importances_mean"],
            "importances_std": result["importances_std"],
            "feature_names": feature_names
        }
    
    def _calculate_python(self, 
                          X: Tensor, 
                          y: Optional[Union[Tensor, np.ndarray]],
                          n_repeats: int,
                          scoring: Optional[Union[str, Callable]],
                          **kwargs) -> Dict[str, Any]:
        """Pure Python implementation of permutation importance."""
        
        np.random.seed(self.random_state)
        
        X_np = X.numpy()
        
        baseline_pred = self._run_model(X).numpy()
        
        if y is not None:
            y_np = y.numpy() if isinstance(y, Tensor) else y
            
            if scoring is None:
                baseline_score = np.mean(np.argmax(baseline_pred, axis=1) == y_np)
            else:
                baseline_score = scoring(y_np, baseline_pred)
        else:
            baseline_score = np.mean(baseline_pred)
        
        n_features = X_np.shape[1]
        importances = np.zeros((n_repeats, n_features))
        
        for i in range(n_features):
            for j in range(n_repeats):
                X_permuted = X_np.copy()
                
                perm_idx = np.random.permutation(X_np.shape[0])
                X_permuted[:, i] = X_permuted[perm_idx, i]
                
                permuted_pred = self._run_model(Tensor(X_permuted)).numpy()
                
                if y is not None:
                    if scoring is None:
                        permuted_score = np.mean(np.argmax(permuted_pred, axis=1) == y_np)
                    else:
                        permuted_score = scoring(y_np, permuted_pred)
                else:
                    permuted_score = np.mean(permuted_pred)
                
                importances[j, i] = baseline_score - permuted_score
        
        importances_mean = np.mean(importances, axis=0)
        importances_std = np.std(importances, axis=0)
        
        feature_names = self.feature_names
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(n_features)]
        
        return {
            "importances": importances_mean,
            "importances_std": importances_std,
            "feature_names": feature_names
        }
