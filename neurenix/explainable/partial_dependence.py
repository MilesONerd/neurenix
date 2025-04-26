"""
Partial Dependence Plot (PDP) implementation for Neurenix.

This module provides tools for calculating and visualizing the marginal effect
of features on the predictions of machine learning models.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
import logging

from neurenix.tensor import Tensor
from neurenix.nn.module import Module

logger = logging.getLogger(__name__)

class PartialDependence:
    """
    Partial Dependence Plot (PDP) implementation.
    
    Calculates the marginal effect of features on the predictions of a model.
    """
    
    def __init__(self, 
                 model: Module, 
                 feature_names: Optional[List[str]] = None,
                 random_state: int = 0):
        """
        Initialize a partial dependence calculator.
        
        Args:
            model: The model to explain
            feature_names: Names of features for the explanation
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.feature_names = feature_names
        self.random_state = random_state
        
        try:
            from neurenix.binding import phynexus
            self._has_phynexus = True
            self._phynexus = phynexus
        except ImportError:
            logger.warning("Phynexus extension not found. Using pure Python implementation.")
            self._has_phynexus = False
    
    def calculate(self, 
                  X: Union[Tensor, np.ndarray], 
                  features: Union[List[int], List[str]], 
                  grid_resolution: int = 20,
                  percentiles: Tuple[float, float] = (0.05, 0.95),
                  **kwargs) -> Dict[str, Any]:
        """
        Calculate partial dependence.
        
        Args:
            X: Input data
            features: Features to calculate partial dependence for
            grid_resolution: Number of points in the grid
            percentiles: Percentiles for the grid range
            **kwargs: Additional arguments for the calculation
            
        Returns:
            Dictionary containing partial dependence information
        """
        X_tensor = self._convert_to_tensor(X)
        
        if self._has_phynexus:
            return self._calculate_native(X_tensor, features, grid_resolution, percentiles, **kwargs)
        else:
            return self._calculate_python(X_tensor, features, grid_resolution, percentiles, **kwargs)
    
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
    
    def _calculate_native(self, 
                          X: Tensor, 
                          features: Union[List[int], List[str]],
                          grid_resolution: int,
                          percentiles: Tuple[float, float],
                          **kwargs) -> Dict[str, Any]:
        """Use native Phynexus implementation for partial dependence."""
        if not hasattr(self._phynexus, "partial_dependence"):
            logger.warning("Native partial_dependence not available. Using Python implementation.")
            return self._calculate_python(X, features, grid_resolution, percentiles, **kwargs)
        
        model_fn = lambda x: self._run_model(Tensor(x)).numpy()
        
        feature_indices = features
        if isinstance(features[0], str):
            if self.feature_names is None:
                raise ValueError("Feature names must be provided if features are specified as strings.")
            feature_indices = [self.feature_names.index(f) for f in features]
        
        result = self._phynexus.partial_dependence(
            model_fn,
            X.numpy(),
            feature_indices,
            grid_resolution,
            percentiles,
            self.random_state
        )
        
        feature_names = self.feature_names
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        return {
            "values": result["values"],
            "grid_points": result["grid_points"],
            "feature_indices": feature_indices,
            "feature_names": [feature_names[i] for i in feature_indices]
        }
    
    def _calculate_python(self, 
                          X: Tensor, 
                          features: Union[List[int], List[str]],
                          grid_resolution: int,
                          percentiles: Tuple[float, float],
                          **kwargs) -> Dict[str, Any]:
        """Pure Python implementation of partial dependence."""
        
        np.random.seed(self.random_state)
        
        X_np = X.numpy()
        
        feature_indices = features
        if isinstance(features[0], str):
            if self.feature_names is None:
                raise ValueError("Feature names must be provided if features are specified as strings.")
            feature_indices = [self.feature_names.index(f) for f in features]
        
        feature_names = self.feature_names
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X_np.shape[1])]
        
        grid_points = []
        for i in feature_indices:
            feature_values = X_np[:, i]
            percentile_min = np.percentile(feature_values, percentiles[0] * 100)
            percentile_max = np.percentile(feature_values, percentiles[1] * 100)
            grid = np.linspace(percentile_min, percentile_max, grid_resolution)
            grid_points.append(grid)
        
        pdp_values = []
        
        for i, feature_idx in enumerate(feature_indices):
            grid = grid_points[i]
            pdp = np.zeros(grid.shape)
            
            for j, val in enumerate(grid):
                X_modified = X_np.copy()
                
                X_modified[:, feature_idx] = val
                
                predictions = self._run_model(Tensor(X_modified)).numpy()
                
                pdp[j] = np.mean(predictions)
            
            pdp_values.append(pdp)
        
        return {
            "values": pdp_values,
            "grid_points": grid_points,
            "feature_indices": feature_indices,
            "feature_names": [feature_names[i] for i in feature_indices]
        }
    
    def plot(self, 
             pdp: Dict[str, Any], 
             **kwargs):
        """
        Plot partial dependence.
        
        Args:
            pdp: Dictionary containing partial dependence information from calculate()
            **kwargs: Additional arguments for the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            values = pdp.get("values", [])
            grid_points = pdp.get("grid_points", [])
            feature_names = pdp.get("feature_names", [])
            
            if not values or not grid_points or not feature_names:
                logger.warning("No data to plot.")
                return
            
            n_features = len(feature_names)
            
            fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4))
            if n_features == 1:
                axes = [axes]
            
            for i in range(n_features):
                ax = axes[i]
                ax.plot(grid_points[i], values[i])
                ax.set_xlabel(feature_names[i])
                if i == 0:
                    ax.set_ylabel("Partial Dependence")
                ax.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not installed. Cannot plot partial dependence.")
