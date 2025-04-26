"""
LIME (Local Interpretable Model-agnostic Explanations) implementation for Neurenix.

This module provides tools for explaining the predictions of any machine learning
classifier in an interpretable and faithful manner.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Callable, Tuple, Any
import logging
from collections import defaultdict
import warnings

from neurenix.tensor import Tensor
from neurenix.nn.module import Module

logger = logging.getLogger(__name__)

class LimeExplainer:
    """Base class for LIME explainers in Neurenix."""
    
    def __init__(self, 
                 model: Module, 
                 feature_names: Optional[List[str]] = None,
                 class_names: Optional[List[str]] = None,
                 random_state: int = 0):
        """
        Initialize a LIME explainer.
        
        Args:
            model: The model to explain
            feature_names: Names of features for the explanation
            class_names: Names of classes for the explanation
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.random_state = random_state
        
        try:
            from neurenix.binding import phynexus
            self._has_phynexus = True
            self._phynexus = phynexus
        except ImportError:
            logger.warning("Phynexus extension not found. Using pure Python implementation.")
            self._has_phynexus = False
    
    def explain(self, 
                sample: Union[Tensor, np.ndarray], 
                num_features: int = 10,
                num_samples: int = 5000,
                **kwargs) -> Dict[str, Any]:
        """
        Explain the model's prediction for a single sample.
        
        Args:
            sample: Sample to explain
            num_features: Maximum number of features to include in explanation
            num_samples: Number of samples to use for the explanation
            **kwargs: Additional arguments for the explainer
            
        Returns:
            Dictionary containing explanation information
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
    
    def plot_explanation(self, 
                         explanation: Dict[str, Any], 
                         **kwargs):
        """
        Plot LIME explanation.
        
        Args:
            explanation: Dictionary containing explanation from explain()
            **kwargs: Additional arguments for the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            feature_values = explanation.get("feature_values", [])
            feature_weights = explanation.get("feature_weights", [])
            feature_names = explanation.get("feature_names", [])
            
            if not feature_weights or not feature_names:
                logger.warning("No features to plot.")
                return
            
            indices = np.argsort(np.abs(feature_weights))[-10:]  # Top 10 features
            
            plt.figure(figsize=(10, 6))
            plt.barh(
                [feature_names[i] for i in indices],
                [feature_weights[i] for i in indices],
                color=["green" if w > 0 else "red" for w in [feature_weights[i] for i in indices]]
            )
            plt.xlabel("Feature Weight")
            plt.title("LIME Explanation")
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not installed. Cannot plot explanation.")


class LimeTabular(LimeExplainer):
    """
    LIME implementation for tabular data.
    
    Explains predictions of tabular data models by perturbing features
    and learning a local linear model.
    """
    
    def __init__(self, 
                 model: Module, 
                 feature_names: Optional[List[str]] = None,
                 class_names: Optional[List[str]] = None,
                 categorical_features: Optional[List[int]] = None,
                 categorical_names: Optional[Dict[int, List[str]]] = None,
                 kernel_width: float = 0.75,
                 random_state: int = 0):
        """
        Initialize LimeTabular explainer.
        
        Args:
            model: The model to explain
            feature_names: Names of features for the explanation
            class_names: Names of classes for the explanation
            categorical_features: Indices of categorical features
            categorical_names: Names of categories for categorical features
            kernel_width: Width of the exponential kernel
            random_state: Random seed for reproducibility
        """
        super().__init__(model, feature_names, class_names, random_state)
        self.categorical_features = categorical_features or []
        self.categorical_names = categorical_names or {}
        self.kernel_width = kernel_width
        
    def explain(self, 
                sample: Union[Tensor, np.ndarray], 
                num_features: int = 10,
                num_samples: int = 5000,
                **kwargs) -> Dict[str, Any]:
        """
        Explain the model's prediction for a single tabular sample.
        
        Args:
            sample: Sample to explain
            num_features: Maximum number of features to include in explanation
            num_samples: Number of samples to use for the explanation
            **kwargs: Additional arguments for the explainer
            
        Returns:
            Dictionary containing explanation information
        """
        sample_tensor = self._convert_to_tensor(sample)
        
        if self._has_phynexus:
            return self._explain_native(sample_tensor, num_features, num_samples, **kwargs)
        else:
            return self._explain_python(sample_tensor, num_features, num_samples, **kwargs)
    
    def _explain_native(self, 
                        sample: Tensor, 
                        num_features: int,
                        num_samples: int,
                        **kwargs) -> Dict[str, Any]:
        """Use native Phynexus implementation for LIME."""
        if not hasattr(self._phynexus, "lime_tabular"):
            logger.warning("Native lime_tabular not available. Using Python implementation.")
            return self._explain_python(sample, num_features, num_samples, **kwargs)
        
        model_fn = lambda x: self._run_model(Tensor(x)).numpy()
        
        result = self._phynexus.lime_tabular(
            model_fn,
            sample.numpy(),
            num_features,
            num_samples,
            self.categorical_features,
            self.kernel_width,
            self.random_state
        )
        
        feature_names = self.feature_names
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(sample.shape[0])]
        
        return {
            "feature_indices": result["feature_indices"],
            "feature_weights": result["feature_weights"],
            "feature_values": result["feature_values"],
            "feature_names": [feature_names[i] for i in result["feature_indices"]],
            "intercept": result["intercept"],
            "score": result["score"],
            "local_pred": result["local_pred"],
            "prediction": result["prediction"]
        }
    
    def _explain_python(self, 
                        sample: Tensor, 
                        num_features: int,
                        num_samples: int,
                        **kwargs) -> Dict[str, Any]:
        """Pure Python implementation of LIME for tabular data."""
        
        np.random.seed(self.random_state)
        
        original_pred = self._run_model(sample).numpy()
        
        if self.feature_names is None:
            feature_names = [f"Feature {i}" for i in range(sample.shape[0])]
        else:
            feature_names = self.feature_names
        
        data = []
        labels = []
        
        sample_np = sample.numpy().flatten()
        
        for _ in range(num_samples):
            perturbed = np.copy(sample_np)
            
            for i in range(len(perturbed)):
                if i in self.categorical_features:
                    if np.random.random() < 0.5:
                        if i in self.categorical_names:
                            categories = len(self.categorical_names[i])
                            perturbed[i] = np.random.randint(0, categories)
                        else:
                            perturbed[i] = np.random.choice([0, 1])
                else:
                    perturbed[i] = perturbed[i] + np.random.normal(0, 0.1)
            
            perturbed_tensor = Tensor(perturbed.reshape(sample.shape))
            pred = self._run_model(perturbed_tensor).numpy()
            
            data.append(perturbed)
            labels.append(pred)
        
        data = np.array(data)
        labels = np.array(labels)
        
        distances = np.sqrt(np.sum((data - sample_np) ** 2, axis=1))
        
        kernel_width = self.kernel_width * np.sqrt(data.shape[1])
        weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))
        
        from sklearn.linear_model import Ridge
        
        if len(original_pred.shape) > 0 and original_pred.shape[0] > 1:
            target_class = np.argmax(original_pred)
            target_labels = labels[:, target_class]
        else:
            target_labels = labels
        
        model = Ridge(alpha=1.0, fit_intercept=True, random_state=self.random_state)
        model.fit(data, target_labels, sample_weight=weights)
        
        coefs = model.coef_
        
        indices = np.argsort(np.abs(coefs))[-num_features:]
        
        return {
            "feature_indices": indices.tolist(),
            "feature_weights": coefs[indices].tolist(),
            "feature_values": sample_np[indices].tolist(),
            "feature_names": [feature_names[i] for i in indices],
            "intercept": model.intercept_,
            "score": model.score(data, target_labels, sample_weight=weights),
            "local_pred": model.predict(sample_np.reshape(1, -1))[0],
            "prediction": original_pred.flatten()[0] if len(original_pred.shape) == 0 else original_pred[np.argmax(original_pred)]
        }


class LimeText(LimeExplainer):
    """
    LIME implementation for text data.
    
    Explains predictions of text models by perturbing words
    and learning a local linear model.
    """
    
    def __init__(self, 
                 model: Module, 
                 class_names: Optional[List[str]] = None,
                 bow: bool = True,
                 split_expression: str = r'\W+',
                 random_state: int = 0):
        """
        Initialize LimeText explainer.
        
        Args:
            model: The model to explain
            class_names: Names of classes for the explanation
            bow: Whether to use bag-of-words representation
            split_expression: Regular expression to split text
            random_state: Random seed for reproducibility
        """
        super().__init__(model, None, class_names, random_state)
        self.bow = bow
        self.split_expression = split_expression
        
    def explain(self, 
                text: str, 
                num_features: int = 10,
                num_samples: int = 5000,
                **kwargs) -> Dict[str, Any]:
        """
        Explain the model's prediction for a text sample.
        
        Args:
            text: Text sample to explain
            num_features: Maximum number of features (words) to include in explanation
            num_samples: Number of samples to use for the explanation
            **kwargs: Additional arguments for the explainer
            
        Returns:
            Dictionary containing explanation information
        """
        if self._has_phynexus:
            return self._explain_native(text, num_features, num_samples, **kwargs)
        else:
            return self._explain_python(text, num_features, num_samples, **kwargs)
    
    def _explain_native(self, 
                        text: str, 
                        num_features: int,
                        num_samples: int,
                        **kwargs) -> Dict[str, Any]:
        """Use native Phynexus implementation for LIME text."""
        if not hasattr(self._phynexus, "lime_text"):
            logger.warning("Native lime_text not available. Using Python implementation.")
            return self._explain_python(text, num_features, num_samples, **kwargs)
        
        model_fn = lambda x: self._run_model(Tensor(self._text_to_tensor(x))).numpy()
        
        result = self._phynexus.lime_text(
            model_fn,
            text,
            num_features,
            num_samples,
            self.bow,
            self.split_expression,
            self.random_state
        )
        
        return {
            "words": result["words"],
            "word_weights": result["word_weights"],
            "intercept": result["intercept"],
            "score": result["score"],
            "local_pred": result["local_pred"],
            "prediction": result["prediction"]
        }
    
    def _explain_python(self, 
                        text: str, 
                        num_features: int,
                        num_samples: int,
                        **kwargs) -> Dict[str, Any]:
        """Pure Python implementation of LIME for text data."""
        
        import re
        from sklearn.linear_model import Ridge
        
        np.random.seed(self.random_state)
        
        tokens = re.split(self.split_expression, text)
        tokens = [t.lower() for t in tokens if len(t) > 0]
        
        original_tensor = self._text_to_tensor(text)
        original_pred = self._run_model(original_tensor).numpy()
        
        data = []
        labels = []
        
        for _ in range(num_samples):
            perturbed_tokens = []
            binary_features = []
            
            for token in tokens:
                include = np.random.random() > 0.5
                binary_features.append(1 if include else 0)
                if include:
                    perturbed_tokens.append(token)
            
            perturbed_text = ' '.join(perturbed_tokens) if perturbed_tokens else ' '
            
            perturbed_tensor = self._text_to_tensor(perturbed_text)
            pred = self._run_model(perturbed_tensor).numpy()
            
            data.append(binary_features)
            labels.append(pred)
        
        data = np.array(data)
        labels = np.array(labels)
        
        distances = np.sqrt(np.sum((data - 1) ** 2, axis=1))
        
        kernel_width = 25
        weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))
        
        if len(original_pred.shape) > 0 and original_pred.shape[0] > 1:
            target_class = np.argmax(original_pred)
            target_labels = labels[:, target_class]
        else:
            target_labels = labels
        
        model = Ridge(alpha=1.0, fit_intercept=True, random_state=self.random_state)
        model.fit(data, target_labels, sample_weight=weights)
        
        coefs = model.coef_
        
        indices = np.argsort(np.abs(coefs))[-num_features:]
        
        return {
            "words": [tokens[i] for i in indices],
            "word_weights": coefs[indices].tolist(),
            "intercept": model.intercept_,
            "score": model.score(data, target_labels, sample_weight=weights),
            "local_pred": model.predict(np.ones((1, len(tokens))))[0],
            "prediction": original_pred.flatten()[0] if len(original_pred.shape) == 0 else original_pred[np.argmax(original_pred)]
        }
    
    def _text_to_tensor(self, text: str) -> Tensor:
        """Convert text to tensor for model input."""
        
        import re
        
        tokens = re.split(self.split_expression, text.lower())
        tokens = [t for t in tokens if len(t) > 0]
        
        embedding = np.zeros(100)  # Arbitrary embedding size
        for i, token in enumerate(tokens):
            h = hash(token) % 100
            embedding[h] = 1
        
        return Tensor(embedding)


class LimeImage(LimeExplainer):
    """
    LIME implementation for image data.
    
    Explains predictions of image models by perturbing superpixels
    and learning a local linear model.
    """
    
    def __init__(self, 
                 model: Module, 
                 class_names: Optional[List[str]] = None,
                 segmentation_fn: Optional[Callable] = None,
                 random_state: int = 0):
        """
        Initialize LimeImage explainer.
        
        Args:
            model: The model to explain
            class_names: Names of classes for the explanation
            segmentation_fn: Function to segment the image into superpixels
            random_state: Random seed for reproducibility
        """
        super().__init__(model, None, class_names, random_state)
        self.segmentation_fn = segmentation_fn
        
    def explain(self, 
                image: Union[Tensor, np.ndarray], 
                num_features: int = 10,
                num_samples: int = 1000,
                **kwargs) -> Dict[str, Any]:
        """
        Explain the model's prediction for an image.
        
        Args:
            image: Image to explain
            num_features: Maximum number of superpixels to include in explanation
            num_samples: Number of samples to use for the explanation
            **kwargs: Additional arguments for the explainer
            
        Returns:
            Dictionary containing explanation information
        """
        image_tensor = self._convert_to_tensor(image)
        
        if self._has_phynexus:
            return self._explain_native(image_tensor, num_features, num_samples, **kwargs)
        else:
            return self._explain_python(image_tensor, num_features, num_samples, **kwargs)
    
    def _explain_native(self, 
                        image: Tensor, 
                        num_features: int,
                        num_samples: int,
                        **kwargs) -> Dict[str, Any]:
        """Use native Phynexus implementation for LIME image."""
        if not hasattr(self._phynexus, "lime_image"):
            logger.warning("Native lime_image not available. Using Python implementation.")
            return self._explain_python(image, num_features, num_samples, **kwargs)
        
        model_fn = lambda x: self._run_model(Tensor(x)).numpy()
        
        result = self._phynexus.lime_image(
            model_fn,
            image.numpy(),
            num_features,
            num_samples,
            self.random_state
        )
        
        return {
            "segments": result["segments"],
            "segment_weights": result["segment_weights"],
            "intercept": result["intercept"],
            "score": result["score"],
            "local_pred": result["local_pred"],
            "prediction": result["prediction"]
        }
    
    def _explain_python(self, 
                        image: Tensor, 
                        num_features: int,
                        num_samples: int,
                        **kwargs) -> Dict[str, Any]:
        """Pure Python implementation of LIME for image data."""
        
        np.random.seed(self.random_state)
        
        original_pred = self._run_model(image).numpy()
        
        segments = self._segment_image(image.numpy())
        num_segments = np.max(segments) + 1
        
        data = []
        labels = []
        
        for _ in range(num_samples):
            active_segments = np.random.binomial(1, 0.5, num_segments)
            perturbed = self._perturb_image(image.numpy(), segments, active_segments)
            
            perturbed_tensor = Tensor(perturbed)
            pred = self._run_model(perturbed_tensor).numpy()
            
            data.append(active_segments)
            labels.append(pred)
        
        data = np.array(data)
        labels = np.array(labels)
        
        distances = np.sqrt(np.sum((data - 1) ** 2, axis=1))
        
        kernel_width = 0.25 * np.sqrt(num_segments)
        weights = np.sqrt(np.exp(-(distances ** 2) / kernel_width ** 2))
        
        if len(original_pred.shape) > 0 and original_pred.shape[0] > 1:
            target_class = np.argmax(original_pred)
            target_labels = labels[:, target_class]
        else:
            target_labels = labels
        
        from sklearn.linear_model import Ridge
        
        model = Ridge(alpha=1.0, fit_intercept=True, random_state=self.random_state)
        model.fit(data, target_labels, sample_weight=weights)
        
        coefs = model.coef_
        
        indices = np.argsort(np.abs(coefs))[-num_features:]
        
        return {
            "segments": segments,
            "segment_indices": indices.tolist(),
            "segment_weights": coefs[indices].tolist(),
            "intercept": model.intercept_,
            "score": model.score(data, target_labels, sample_weight=weights),
            "local_pred": model.predict(np.ones((1, num_segments)))[0],
            "prediction": original_pred.flatten()[0] if len(original_pred.shape) == 0 else original_pred[np.argmax(original_pred)]
        }
    
    def _segment_image(self, image: np.ndarray) -> np.ndarray:
        """Segment the image into superpixels."""
        if self.segmentation_fn is not None:
            return self.segmentation_fn(image)
        
        height, width = image.shape[:2]
        segments = np.zeros((height, width), dtype=np.int32)
        
        grid_size = 16  # Size of each segment
        for i in range(0, height, grid_size):
            for j in range(0, width, grid_size):
                segments[i:min(i+grid_size, height), j:min(j+grid_size, width)] = (i // grid_size) * (width // grid_size) + (j // grid_size)
        
        return segments
    
    def _perturb_image(self, image: np.ndarray, segments: np.ndarray, active_segments: np.ndarray) -> np.ndarray:
        """Perturb the image by turning off inactive segments."""
        perturbed = np.copy(image)
        
        for i in range(len(active_segments)):
            if active_segments[i] == 0:
                perturbed[segments == i] = 0.5  # Gray value
        
        return perturbed
    
    def plot_explanation(self, 
                         explanation: Dict[str, Any], 
                         **kwargs):
        """
        Plot LIME explanation for an image.
        
        Args:
            explanation: Dictionary containing explanation from explain()
            **kwargs: Additional arguments for the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            segments = explanation.get("segments")
            segment_indices = explanation.get("segment_indices", [])
            segment_weights = explanation.get("segment_weights", [])
            
            if segments is None:
                logger.warning("No segments to plot.")
                return
            
            mask = np.zeros(segments.shape, dtype=np.float32)
            
            for i, idx in enumerate(segment_indices):
                mask[segments == idx] = segment_weights[i]
            
            if np.max(np.abs(mask)) > 0:
                mask = mask / np.max(np.abs(mask))
            
            plt.figure(figsize=(10, 5))
            
            plt.subplot(1, 2, 1)
            plt.imshow(explanation.get("image", np.zeros_like(segments)))
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(mask, cmap='RdBu_r', vmin=-1, vmax=1)
            plt.title("LIME Explanation")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not installed. Cannot plot explanation.")
