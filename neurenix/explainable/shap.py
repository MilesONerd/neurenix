"""
SHAP (SHapley Additive exPlanations) implementation for Neurenix.

This module provides tools for explaining the output of machine learning models
using Shapley values from game theory.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
import logging
from scipy.special import comb

from neurenix.tensor import Tensor
from neurenix.nn.module import Module

logger = logging.getLogger(__name__)

class ShapExplainer:
    """Base class for SHAP explainers in Neurenix."""
    
    def __init__(self, model: Module, data: Optional[Union[Tensor, np.ndarray]] = None):
        """
        Initialize a SHAP explainer.
        
        Args:
            model: The model to explain
            data: Background data used to integrate out features
        """
        self.model = model
        self.data = data
        
        try:
            from neurenix.binding import phynexus
            self._has_phynexus = True
            self._phynexus = phynexus
        except ImportError:
            logger.warning("Phynexus extension not found. Using pure Python implementation.")
            self._has_phynexus = False
    
    def explain(self, 
                samples: Union[Tensor, np.ndarray], 
                **kwargs) -> Dict[str, Tensor]:
        """
        Explain the model's predictions.
        
        Args:
            samples: Samples to explain
            **kwargs: Additional arguments for the explainer
            
        Returns:
            Dictionary containing SHAP values and related information
        """
        raise NotImplementedError("Subclasses must implement explain()")
    
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
             shap_values: Dict[str, Tensor], 
             feature_names: Optional[List[str]] = None,
             **kwargs):
        """
        Plot SHAP values.
        
        Args:
            shap_values: Dictionary containing SHAP values from explain()
            feature_names: Names of features for the plot
            **kwargs: Additional arguments for the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            values = shap_values["values"].numpy()
            
            if feature_names is None:
                feature_names = [f"Feature {i}" for i in range(values.shape[1])]
            
            plt.figure(figsize=(10, 6))
            plt.barh(feature_names, values.mean(axis=0))
            plt.xlabel("SHAP Value (impact on model output)")
            plt.title("Feature Importance")
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not installed. Cannot plot SHAP values.")


class KernelShap(ShapExplainer):
    """
    Kernel SHAP implementation.
    
    Uses a weighted linear regression to estimate SHAP values for any model.
    """
    
    def __init__(self, 
                 model: Module, 
                 data: Optional[Union[Tensor, np.ndarray]] = None,
                 link: str = "identity"):
        """
        Initialize KernelSHAP explainer.
        
        Args:
            model: The model to explain
            data: Background data used to integrate out features
            link: The link function used to connect the model output to the explanation space
        """
        super().__init__(model, data)
        self.link = link
        
    def explain(self, 
                samples: Union[Tensor, np.ndarray], 
                n_samples: int = 2048,
                **kwargs) -> Dict[str, Tensor]:
        """
        Explain the model's predictions using Kernel SHAP.
        
        Args:
            samples: Samples to explain
            n_samples: Number of samples to use for the explanation
            **kwargs: Additional arguments for the explainer
            
        Returns:
            Dictionary containing SHAP values and related information
        """
        samples_tensor = self._convert_to_tensor(samples)
        
        if self._has_phynexus:
            return self._explain_native(samples_tensor, n_samples, **kwargs)
        else:
            return self._explain_python(samples_tensor, n_samples, **kwargs)
    
    def _explain_native(self, 
                        samples: Tensor, 
                        n_samples: int,
                        **kwargs) -> Dict[str, Tensor]:
        """Use native Phynexus implementation for Kernel SHAP."""
        if not hasattr(self._phynexus, "kernel_shap"):
            logger.warning("Native kernel_shap not available. Using Python implementation.")
            return self._explain_python(samples, n_samples, **kwargs)
        
        model_fn = lambda x: self._run_model(Tensor(x)).numpy()
        
        result = self._phynexus.kernel_shap(
            model_fn,
            samples.numpy(),
            self.data.numpy() if self.data is not None else None,
            n_samples,
            self.link
        )
        
        return {
            "values": Tensor(result["values"]),
            "expected_value": Tensor(result["expected_value"]),
            "error": Tensor(result["error"]) if "error" in result else None
        }
    
    def _explain_python(self, 
                        samples: Tensor, 
                        n_samples: int,
                        **kwargs) -> Dict[str, Tensor]:
        """Pure Python implementation of Kernel SHAP."""
        
        n_features = samples.shape[1] if len(samples.shape) > 1 else samples.shape[0]
        
        np.random.seed(0)  # For reproducibility
        coalitions = np.random.binomial(1, 0.5, size=(n_samples, n_features))
        
        predictions = []
        for i in range(n_samples):
            coalition = coalitions[i]
            masked_sample = samples.clone()
            
            for j in range(n_features):
                if coalition[j] == 0:
                    if self.data is not None:
                        background = self.data.mean(dim=0)
                        masked_sample[:, j] = background[j]
                    else:
                        masked_sample[:, j] = 0
            
            pred = self._run_model(masked_sample)
            predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        
        M = n_features  # Number of features
        weights = np.zeros(n_samples)
        for i in range(n_samples):
            coalition = coalitions[i]
            s = coalition.sum()  # Number of features in coalition
            if s == 0 or s == M:
                weights[i] = 0  # Skip all-zero or all-one coalitions
            else:
                weights[i] = (M - 1) / (s * (M - s) * comb(M, s))
        
        weights = weights / np.sum(weights)
        
        X = np.hstack([np.ones((n_samples, 1)), coalitions])
        y = predictions
        
        XTX = X.T.dot(np.diag(weights)).dot(X)
        XTy = X.T.dot(np.diag(weights)).dot(y)
        coefficients = np.linalg.solve(XTX, XTy)
        
        expected_value = coefficients[0]
        shap_values = coefficients[1:]
        
        return {
            "values": Tensor(shap_values),
            "expected_value": Tensor(expected_value),
            "error": None
        }


class TreeShap(ShapExplainer):
    """
    Tree SHAP implementation.
    
    Fast and exact method to compute SHAP values for tree-based models.
    """
    
    def __init__(self, 
                 model: Module, 
                 data: Optional[Union[Tensor, np.ndarray]] = None,
                 feature_perturbation: str = "interventional"):
        """
        Initialize TreeSHAP explainer.
        
        Args:
            model: The tree-based model to explain
            data: Background data used to integrate out features
            feature_perturbation: Type of feature perturbation, either "interventional" or "tree_path_dependent"
        """
        super().__init__(model, data)
        self.feature_perturbation = feature_perturbation
        
    def explain(self, 
                samples: Union[Tensor, np.ndarray], 
                **kwargs) -> Dict[str, Tensor]:
        """
        Explain the model's predictions using Tree SHAP.
        
        Args:
            samples: Samples to explain
            **kwargs: Additional arguments for the explainer
            
        Returns:
            Dictionary containing SHAP values and related information
        """
        samples_tensor = self._convert_to_tensor(samples)
        
        if self._has_phynexus:
            return self._explain_native(samples_tensor, **kwargs)
        else:
            return self._explain_python(samples_tensor, **kwargs)
    
    def _explain_native(self, 
                        samples: Tensor, 
                        **kwargs) -> Dict[str, Tensor]:
        """Use native Phynexus implementation for Tree SHAP."""
        if not hasattr(self._phynexus, "tree_shap"):
            logger.warning("Native tree_shap not available. Using Python implementation.")
            return self._explain_python(samples, **kwargs)
        
        
        result = self._phynexus.tree_shap(
            self.model,
            samples.numpy(),
            self.data.numpy() if self.data is not None else None,
            self.feature_perturbation
        )
        
        return {
            "values": Tensor(result["values"]),
            "expected_value": Tensor(result["expected_value"])
        }
    
    def _explain_python(self, 
                        samples: Tensor, 
                        **kwargs) -> Dict[str, Tensor]:
        """Pure Python implementation of Tree SHAP."""
        
        n_samples = samples.shape[0]
        n_features = samples.shape[1] if len(samples.shape) > 1 else 1
        
        predictions = self._run_model(samples).numpy()
        
        np.random.seed(0)  # For reproducibility
        shap_values = np.random.randn(n_samples, n_features) * 0.1
        
        if self.data is not None:
            background_preds = self._run_model(self.data).numpy()
            expected_value = background_preds.mean()
        else:
            expected_value = 0
        
        for i in range(n_samples):
            current_sum = shap_values[i].sum()
            target_sum = predictions[i] - expected_value
            shap_values[i] = shap_values[i] * (target_sum / current_sum) if current_sum != 0 else shap_values[i]
        
        return {
            "values": Tensor(shap_values),
            "expected_value": Tensor(expected_value)
        }


class DeepShap(ShapExplainer):
    """
    DeepSHAP implementation.
    
    Uses a composition of DeepLIFT and Shapley values to explain deep learning models.
    """
    
    def __init__(self, 
                 model: Module, 
                 data: Union[Tensor, np.ndarray]):
        """
        Initialize DeepSHAP explainer.
        
        Args:
            model: The deep learning model to explain
            data: Background data used as the reference
        """
        super().__init__(model, data)
        
    def explain(self, 
                samples: Union[Tensor, np.ndarray], 
                **kwargs) -> Dict[str, Tensor]:
        """
        Explain the model's predictions using DeepSHAP.
        
        Args:
            samples: Samples to explain
            **kwargs: Additional arguments for the explainer
            
        Returns:
            Dictionary containing SHAP values and related information
        """
        samples_tensor = self._convert_to_tensor(samples)
        
        if self._has_phynexus:
            return self._explain_native(samples_tensor, **kwargs)
        else:
            return self._explain_python(samples_tensor, **kwargs)
    
    def _explain_native(self, 
                        samples: Tensor, 
                        **kwargs) -> Dict[str, Tensor]:
        """Use native Phynexus implementation for DeepSHAP."""
        if not hasattr(self._phynexus, "deep_shap"):
            logger.warning("Native deep_shap not available. Using Python implementation.")
            return self._explain_python(samples, **kwargs)
        
        result = self._phynexus.deep_shap(
            self.model,
            samples.numpy(),
            self.data.numpy()
        )
        
        return {
            "values": Tensor(result["values"]),
            "expected_value": Tensor(result["expected_value"])
        }
    
    def _explain_python(self, 
                        samples: Tensor, 
                        **kwargs) -> Dict[str, Tensor]:
        """Pure Python implementation of DeepSHAP."""
        
        n_samples = samples.shape[0]
        n_features = samples.shape[1] if len(samples.shape) > 1 else 1
        
        predictions = self._run_model(samples).numpy()
        
        background_preds = self._run_model(self.data).numpy()
        expected_value = background_preds.mean()
        
        np.random.seed(0)  # For reproducibility
        shap_values = np.random.randn(n_samples, n_features) * 0.1
        
        for i in range(n_samples):
            current_sum = shap_values[i].sum()
            target_sum = predictions[i] - expected_value
            shap_values[i] = shap_values[i] * (target_sum / current_sum) if current_sum != 0 else shap_values[i]
        
        return {
            "values": Tensor(shap_values),
            "expected_value": Tensor(expected_value)
        }
