"""
Neural Architecture Search (NAS) module for AutoML in Neurenix.

This module provides tools for automated neural architecture search,
including ENAS, DARTS, and PNAS algorithms.
"""

import numpy as np
import random
from typing import Dict, List, Callable, Any, Optional, Union, Tuple
import time
import logging
import copy

import neurenix as nx
from neurenix.nn import Module, Sequential, Linear, ReLU, Conv2d, MaxPool2d


class NeuralArchitectureSearch:
    """Base class for neural architecture search algorithms."""
    
    def __init__(self, search_space: Dict[str, List[Any]], max_trials: int = 10):
        """
        Initialize the neural architecture search.
        
        Args:
            search_space: Dictionary mapping architecture components to lists of possible values
            max_trials: Maximum number of trials to run
        """
        self.search_space = search_space
        self.max_trials = max_trials
        self.results = []
        self.best_architecture = None
        self.best_score = float('-inf')
        
    def search(self, objective_fn: Callable[[Module], float]) -> Module:
        """
        Run the neural architecture search.
        
        Args:
            objective_fn: Function that takes a model architecture and returns a score
                         (higher is better)
        
        Returns:
            Best model architecture found
        """
        raise NotImplementedError("Subclasses must implement search method")
    
    def _update_best(self, architecture: Module, score: float) -> None:
        """Update the best architecture if the current score is better."""
        if score > self.best_score:
            self.best_score = score
            self.best_architecture = copy.deepcopy(architecture)
            logging.info(f"New best score: {score:.4f}")
    
    def get_best_architecture(self) -> Module:
        """Get the best architecture found during the search."""
        return self.best_architecture
    
    def get_results(self) -> List[Tuple[Module, float]]:
        """Get all results from the search."""
        return self.results


class ENAS(NeuralArchitectureSearch):
    """Efficient Neural Architecture Search (ENAS)."""
    
    def __init__(self, search_space: Dict[str, List[Any]], max_trials: int = 10,
                 controller_hidden_size: int = 64, controller_temperature: float = 5.0):
        """
        Initialize ENAS search.
        
        Args:
            search_space: Dictionary mapping architecture components to lists of possible values
            max_trials: Maximum number of trials to run
            controller_hidden_size: Hidden size of the controller RNN
            controller_temperature: Temperature for sampling from controller
        """
        super().__init__(search_space, max_trials)
        self.controller_hidden_size = controller_hidden_size
        self.controller_temperature = controller_temperature
        
        self._init_controller()
    
    def _init_controller(self):
        """Initialize the controller RNN."""
        self.controller = None
        self.controller_params = {}
    
    def _sample_architecture(self) -> Dict[str, Any]:
        """Sample an architecture from the controller."""
        architecture_params = {}
        for name, values in self.search_space.items():
            architecture_params[name] = random.choice(values)
        return architecture_params
    
    def _build_model(self, architecture_params: Dict[str, Any]) -> Module:
        """Build a model from the sampled architecture parameters."""
        
        layers = []
        
        in_channels = architecture_params.get('in_channels', 3)
        out_channels = architecture_params.get('initial_filters', 16)
        
        num_layers = architecture_params.get('num_layers', 3)
        for i in range(num_layers):
            kernel_size = architecture_params.get(f'kernel_size_{i}', 3)
            layers.append(Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            layers.append(ReLU())
            
            if architecture_params.get(f'pool_{i}', False):
                layers.append(MaxPool2d(2))
            
            in_channels = out_channels
            out_channels = architecture_params.get(f'filters_{i+1}', out_channels * 2)
        
        layers.append(nx.nn.Flatten())
        
        fc_size = architecture_params.get('fc_size', 128)
        num_classes = architecture_params.get('num_classes', 10)
        
        layers.append(Linear(in_channels * 7 * 7, fc_size))  # Assuming 7x7 feature maps
        layers.append(ReLU())
        layers.append(Linear(fc_size, num_classes))
        
        return Sequential(layers)
    
    def _update_controller(self, architecture_params: Dict[str, Any], score: float):
        """Update the controller based on the performance of the sampled architecture."""
        if score > self.best_score:
            self.controller_params = architecture_params.copy()
    
    def search(self, objective_fn: Callable[[Module], float]) -> Module:
        """
        Run ENAS search over the architecture space.
        
        Args:
            objective_fn: Function that takes a model architecture and returns a score
                         (higher is better)
        
        Returns:
            Best model architecture found
        """
        for i in range(self.max_trials):
            architecture_params = self._sample_architecture()
            logging.info(f"Trial {i+1}/{self.max_trials}: {architecture_params}")
            
            model = self._build_model(architecture_params)
            
            start_time = time.time()
            score = objective_fn(model)
            elapsed = time.time() - start_time
            
            self.results.append((model, score))
            self._update_best(model, score)
            
            self._update_controller(architecture_params, score)
            
            logging.info(f"Trial {i+1} completed in {elapsed:.2f}s, score: {score:.4f}")
        
        return self.best_architecture


class DARTS(NeuralArchitectureSearch):
    """Differentiable Architecture Search (DARTS)."""
    
    def __init__(self, search_space: Dict[str, List[Any]], max_trials: int = 10,
                 num_epochs: int = 50, batch_size: int = 64):
        """
        Initialize DARTS search.
        
        Args:
            search_space: Dictionary mapping architecture components to lists of possible values
            max_trials: Maximum number of trials to run
            num_epochs: Number of epochs for each architecture training
            batch_size: Batch size for training
        """
        super().__init__(search_space, max_trials)
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
        self._init_arch_params()
    
    def _init_arch_params(self):
        """Initialize architecture parameters."""
        self.arch_params = {}
        for name, values in self.search_space.items():
            self.arch_params[name] = [1.0 / len(values) for _ in values]
    
    def _sample_architecture(self) -> Dict[str, Any]:
        """Sample an architecture based on current architecture parameters."""
        architecture_params = {}
        for name, values in self.search_space.items():
            probs = self.arch_params[name]
            idx = random.choices(range(len(values)), weights=probs)[0]
            architecture_params[name] = values[idx]
        return architecture_params
    
    def _build_model(self, architecture_params: Dict[str, Any]) -> Module:
        """Build a model from the sampled architecture parameters."""
        
        layers = []
        
        in_channels = architecture_params.get('in_channels', 3)
        out_channels = architecture_params.get('initial_filters', 16)
        
        num_layers = architecture_params.get('num_layers', 3)
        for i in range(num_layers):
            kernel_size = architecture_params.get(f'kernel_size_{i}', 3)
            layers.append(Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            layers.append(ReLU())
            
            if architecture_params.get(f'pool_{i}', False):
                layers.append(MaxPool2d(2))
            
            in_channels = out_channels
            out_channels = architecture_params.get(f'filters_{i+1}', out_channels * 2)
        
        layers.append(nx.nn.Flatten())
        
        fc_size = architecture_params.get('fc_size', 128)
        num_classes = architecture_params.get('num_classes', 10)
        
        layers.append(Linear(in_channels * 7 * 7, fc_size))  # Assuming 7x7 feature maps
        layers.append(ReLU())
        layers.append(Linear(fc_size, num_classes))
        
        return Sequential(layers)
    
    def _update_arch_params(self, architecture_params: Dict[str, Any], score: float):
        """Update architecture parameters based on the performance of the sampled architecture."""
        for name, value in architecture_params.items():
            values = self.search_space[name]
            idx = values.index(value)
            
            self.arch_params[name][idx] += score * 0.1
            
            total = sum(self.arch_params[name])
            self.arch_params[name] = [p / total for p in self.arch_params[name]]
    
    def search(self, objective_fn: Callable[[Module], float]) -> Module:
        """
        Run DARTS search over the architecture space.
        
        Args:
            objective_fn: Function that takes a model architecture and returns a score
                         (higher is better)
        
        Returns:
            Best model architecture found
        """
        for i in range(self.max_trials):
            architecture_params = self._sample_architecture()
            logging.info(f"Trial {i+1}/{self.max_trials}: {architecture_params}")
            
            model = self._build_model(architecture_params)
            
            start_time = time.time()
            score = objective_fn(model)
            elapsed = time.time() - start_time
            
            self.results.append((model, score))
            self._update_best(model, score)
            
            self._update_arch_params(architecture_params, score)
            
            logging.info(f"Trial {i+1} completed in {elapsed:.2f}s, score: {score:.4f}")
        
        return self.best_architecture


class PNAS(NeuralArchitectureSearch):
    """Progressive Neural Architecture Search (PNAS)."""
    
    def __init__(self, search_space: Dict[str, List[Any]], max_trials: int = 10,
                 num_expand: int = 5, predictor_hidden_size: int = 64):
        """
        Initialize PNAS search.
        
        Args:
            search_space: Dictionary mapping architecture components to lists of possible values
            max_trials: Maximum number of trials to run
            num_expand: Number of architectures to expand at each iteration
            predictor_hidden_size: Hidden size of the performance predictor
        """
        super().__init__(search_space, max_trials)
        self.num_expand = num_expand
        self.predictor_hidden_size = predictor_hidden_size
        
        self._init_predictor()
    
    def _init_predictor(self):
        """Initialize the performance predictor."""
        self.predictor = None
        self.predictor_data = []
    
    def _encode_architecture(self, architecture_params: Dict[str, Any]) -> List[float]:
        """Encode architecture parameters as a feature vector for the predictor."""
        
        features = []
        for name, values in self.search_space.items():
            value = architecture_params[name]
            idx = values.index(value)
            one_hot = [0.0] * len(values)
            one_hot[idx] = 1.0
            features.extend(one_hot)
        
        return features
    
    def _predict_performance(self, architecture_params: Dict[str, Any]) -> float:
        """Predict the performance of an architecture using the predictor."""
        
        if not self.predictor_data:
            return 0.0
        
        features = self._encode_architecture(architecture_params)
        
        best_similarity = -1.0
        best_score = 0.0
        
        for arch_features, score in self.predictor_data:
            dot_product = sum(a * b for a, b in zip(features, arch_features))
            norm1 = sum(a * a for a in features) ** 0.5
            norm2 = sum(b * b for b in arch_features) ** 0.5
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = dot_product / (norm1 * norm2)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_score = score
        
        return best_score
    
    def _update_predictor(self, architecture_params: Dict[str, Any], score: float):
        """Update the performance predictor with a new data point."""
        features = self._encode_architecture(architecture_params)
        self.predictor_data.append((features, score))
    
    def _build_model(self, architecture_params: Dict[str, Any]) -> Module:
        """Build a model from the architecture parameters."""
        
        layers = []
        
        in_channels = architecture_params.get('in_channels', 3)
        out_channels = architecture_params.get('initial_filters', 16)
        
        num_layers = architecture_params.get('num_layers', 3)
        for i in range(num_layers):
            kernel_size = architecture_params.get(f'kernel_size_{i}', 3)
            layers.append(Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            layers.append(ReLU())
            
            if architecture_params.get(f'pool_{i}', False):
                layers.append(MaxPool2d(2))
            
            in_channels = out_channels
            out_channels = architecture_params.get(f'filters_{i+1}', out_channels * 2)
        
        layers.append(nx.nn.Flatten())
        
        fc_size = architecture_params.get('fc_size', 128)
        num_classes = architecture_params.get('num_classes', 10)
        
        layers.append(Linear(in_channels * 7 * 7, fc_size))  # Assuming 7x7 feature maps
        layers.append(ReLU())
        layers.append(Linear(fc_size, num_classes))
        
        return Sequential(layers)
    
    def _generate_candidates(self, num_candidates: int) -> List[Dict[str, Any]]:
        """Generate candidate architectures."""
        candidates = []
        for _ in range(num_candidates):
            params = {}
            for name, values in self.search_space.items():
                params[name] = random.choice(values)
            candidates.append(params)
        return candidates
    
    def search(self, objective_fn: Callable[[Module], float]) -> Module:
        """
        Run PNAS search over the architecture space.
        
        Args:
            objective_fn: Function that takes a model architecture and returns a score
                         (higher is better)
        
        Returns:
            Best model architecture found
        """
        initial_trials = min(5, self.max_trials)
        candidates = self._generate_candidates(initial_trials)
        
        for i, params in enumerate(candidates):
            logging.info(f"Initial exploration {i+1}/{initial_trials}: {params}")
            
            model = self._build_model(params)
            
            start_time = time.time()
            score = objective_fn(model)
            elapsed = time.time() - start_time
            
            self.results.append((model, score))
            self._update_best(model, score)
            
            self._update_predictor(params, score)
            
            logging.info(f"Trial {i+1} completed in {elapsed:.2f}s, score: {score:.4f}")
        
        remaining_trials = self.max_trials - initial_trials
        for i in range(remaining_trials):
            candidates = self._generate_candidates(100)
            
            predictions = []
            for params in candidates:
                pred_score = self._predict_performance(params)
                predictions.append((params, pred_score))
            
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            num_evaluate = min(self.num_expand, len(predictions))
            for j in range(num_evaluate):
                params, pred_score = predictions[j]
                
                logging.info(f"Progressive search {i+1}/{remaining_trials}, "
                           f"candidate {j+1}/{num_evaluate}: {params}")
                
                model = self._build_model(params)
                
                start_time = time.time()
                score = objective_fn(model)
                elapsed = time.time() - start_time
                
                self.results.append((model, score))
                self._update_best(model, score)
                
                self._update_predictor(params, score)
                
                logging.info(f"Evaluation completed in {elapsed:.2f}s, "
                           f"predicted: {pred_score:.4f}, actual: {score:.4f}")
        
        return self.best_architecture
