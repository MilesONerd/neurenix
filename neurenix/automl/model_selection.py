"""
Model selection module for AutoML in Neurenix.

This module provides tools for automated model selection,
including cross-validation and nested cross-validation.
"""

import numpy as np
import random
from typing import Dict, List, Callable, Any, Optional, Union, Tuple, Type
import time
import logging
import copy
import itertools

import neurenix as nx
from neurenix.nn import Module


class AutoModelSelection:
    """Base class for automated model selection."""
    
    def __init__(self, model_classes: List[Type[Module]], 
                 hyperparams: Dict[Type[Module], Dict[str, List[Any]]],
                 max_trials: int = 10):
        """
        Initialize the automated model selection.
        
        Args:
            model_classes: List of model classes to consider
            hyperparams: Dictionary mapping model classes to their hyperparameter spaces
            max_trials: Maximum number of trials to run
        """
        self.model_classes = model_classes
        self.hyperparams = hyperparams
        self.max_trials = max_trials
        self.results = []
        self.best_model = None
        self.best_score = float('-inf')
        
    def select(self, X_train, y_train, X_val, y_val) -> Module:
        """
        Run the model selection process.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        
        Returns:
            Best model found
        """
        raise NotImplementedError("Subclasses must implement select method")
    
    def _update_best(self, model: Module, score: float) -> None:
        """Update the best model if the current score is better."""
        if score > self.best_score:
            self.best_score = score
            self.best_model = copy.deepcopy(model)
            logging.info(f"New best score: {score:.4f}")
    
    def get_best_model(self) -> Module:
        """Get the best model found during the selection process."""
        return self.best_model
    
    def get_results(self) -> List[Tuple[Module, float]]:
        """Get all results from the selection process."""
        return self.results


class CrossValidation:
    """Cross-validation for model evaluation."""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_seed: Optional[int] = None):
        """
        Initialize cross-validation.
        
        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle the data before splitting
            random_seed: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed
        
    def split(self, X, y):
        """
        Generate indices to split data into training and validation sets.
        
        Args:
            X: Features
            y: Labels
        
        Yields:
            train_indices, val_indices for each fold
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            if self.random_seed is not None:
                np.random.seed(self.random_seed)
            np.random.shuffle(indices)
        
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            
            val_indices = indices[start:end]
            train_indices = np.concatenate([indices[:start], indices[end:]])
            
            yield train_indices, val_indices
    
    def evaluate(self, model_fn: Callable[[], Module], X, y, 
                 fit_params: Dict[str, Any] = None) -> float:
        """
        Evaluate a model using cross-validation.
        
        Args:
            model_fn: Function that returns a new model instance
            X: Features
            y: Labels
            fit_params: Additional parameters for model fitting
        
        Returns:
            Mean score across all folds
        """
        if fit_params is None:
            fit_params = {}
        
        scores = []
        
        for fold, (train_indices, val_indices) in enumerate(self.split(X, y)):
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_val_fold = X[val_indices]
            y_val_fold = y[val_indices]
            
            model = model_fn()
            
            X_train_tensor = nx.Tensor(X_train_fold)
            y_train_tensor = nx.Tensor(y_train_fold)
            X_val_tensor = nx.Tensor(X_val_fold)
            y_val_tensor = nx.Tensor(y_val_fold)
            
            logging.info(f"Training fold {fold+1}/{self.n_splits}")
            model.fit(X_train_tensor, y_train_tensor, **fit_params)
            
            score = model.evaluate(X_val_tensor, y_val_tensor)
            scores.append(score)
            
            logging.info(f"Fold {fold+1} score: {score:.4f}")
        
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        logging.info(f"Cross-validation results: {mean_score:.4f} ± {std_score:.4f}")
        
        return mean_score


class NestedCrossValidation:
    """Nested cross-validation for model selection and evaluation."""
    
    def __init__(self, outer_splits: int = 5, inner_splits: int = 3, 
                 shuffle: bool = True, random_seed: Optional[int] = None):
        """
        Initialize nested cross-validation.
        
        Args:
            outer_splits: Number of folds for outer loop
            inner_splits: Number of folds for inner loop
            shuffle: Whether to shuffle the data before splitting
            random_seed: Random seed for reproducibility
        """
        self.outer_cv = CrossValidation(n_splits=outer_splits, shuffle=shuffle, 
                                       random_seed=random_seed)
        self.inner_cv = CrossValidation(n_splits=inner_splits, shuffle=shuffle, 
                                       random_seed=random_seed)
        
    def evaluate(self, model_fn: Callable[[Dict[str, Any]], Module], 
                 param_grid: Dict[str, List[Any]], X, y, 
                 fit_params: Dict[str, Any] = None) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Evaluate a model using nested cross-validation.
        
        Args:
            model_fn: Function that takes hyperparameters and returns a new model instance
            param_grid: Dictionary mapping parameter names to lists of possible values
            X: Features
            y: Labels
            fit_params: Additional parameters for model fitting
        
        Returns:
            Mean score across all outer folds and list of best parameters for each fold
        """
        if fit_params is None:
            fit_params = {}
        
        outer_scores = []
        best_params_list = []
        
        for fold, (train_indices, val_indices) in enumerate(self.outer_cv.split(X, y)):
            X_train_fold = X[train_indices]
            y_train_fold = y[train_indices]
            X_val_fold = X[val_indices]
            y_val_fold = y[val_indices]
            
            best_score = float('-inf')
            best_params = None
            
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())
            param_combinations = list(itertools.product(*param_values))
            
            for values in param_combinations:
                params = {name: value for name, value in zip(param_names, values)}
                
                def create_model():
                    return model_fn(params)
                
                score = self.inner_cv.evaluate(create_model, X_train_fold, y_train_fold, fit_params)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
            
            model = model_fn(best_params)
            
            X_train_tensor = nx.Tensor(X_train_fold)
            y_train_tensor = nx.Tensor(y_train_fold)
            X_val_tensor = nx.Tensor(X_val_fold)
            y_val_tensor = nx.Tensor(y_val_fold)
            
            logging.info(f"Training outer fold {fold+1}/{self.outer_cv.n_splits} "
                        f"with best parameters: {best_params}")
            model.fit(X_train_tensor, y_train_tensor, **fit_params)
            
            score = model.evaluate(X_val_tensor, y_val_tensor)
            outer_scores.append(score)
            best_params_list.append(best_params)
            
            logging.info(f"Outer fold {fold+1} score: {score:.4f}")
        
        mean_score = np.mean(outer_scores)
        std_score = np.std(outer_scores)
        
        logging.info(f"Nested cross-validation results: {mean_score:.4f} ± {std_score:.4f}")
        
        return mean_score, best_params_list
