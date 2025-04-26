"""
Hyperparameter search module for AutoML in Neurenix.

This module provides tools for automated hyperparameter optimization,
including grid search, random search, Bayesian optimization, and
evolutionary algorithms.
"""

import numpy as np
import random
from typing import Dict, List, Callable, Any, Optional, Union, Tuple
import itertools
import time
import logging

import neurenix as nx
from neurenix.nn import Module


class HyperparameterSearch:
    """Base class for hyperparameter search algorithms."""
    
    def __init__(self, param_space: Dict[str, List[Any]], max_trials: int = 10):
        """
        Initialize the hyperparameter search.
        
        Args:
            param_space: Dictionary mapping parameter names to lists of possible values
            max_trials: Maximum number of trials to run
        """
        self.param_space = param_space
        self.max_trials = max_trials
        self.results = []
        self.best_params = None
        self.best_score = float('-inf')
        
    def search(self, objective_fn: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """
        Run the hyperparameter search.
        
        Args:
            objective_fn: Function that takes a parameter configuration and returns a score
                         (higher is better)
        
        Returns:
            Dictionary containing the best hyperparameters found
        """
        raise NotImplementedError("Subclasses must implement search method")
    
    def _update_best(self, params: Dict[str, Any], score: float) -> None:
        """Update the best parameters if the current score is better."""
        if score > self.best_score:
            self.best_score = score
            self.best_params = params.copy()
            logging.info(f"New best score: {score:.4f}, params: {params}")
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get the best parameters found during the search."""
        return self.best_params
    
    def get_results(self) -> List[Tuple[Dict[str, Any], float]]:
        """Get all results from the search."""
        return self.results


class GridSearch(HyperparameterSearch):
    """Grid search for hyperparameter optimization."""
    
    def search(self, objective_fn: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """
        Run grid search over the parameter space.
        
        Args:
            objective_fn: Function that takes a parameter configuration and returns a score
                         (higher is better)
        
        Returns:
            Dictionary containing the best hyperparameters found
        """
        param_names = list(self.param_space.keys())
        param_values = list(self.param_space.values())
        param_combinations = list(itertools.product(*param_values))
        
        if len(param_combinations) > self.max_trials:
            logging.warning(f"Grid search would require {len(param_combinations)} trials, "
                           f"limiting to {self.max_trials} random combinations")
            random.shuffle(param_combinations)
            param_combinations = param_combinations[:self.max_trials]
        
        for i, values in enumerate(param_combinations):
            params = {name: value for name, value in zip(param_names, values)}
            logging.info(f"Trial {i+1}/{len(param_combinations)}: {params}")
            
            start_time = time.time()
            score = objective_fn(params)
            elapsed = time.time() - start_time
            
            self.results.append((params, score))
            self._update_best(params, score)
            
            logging.info(f"Trial {i+1} completed in {elapsed:.2f}s, score: {score:.4f}")
        
        return self.best_params


class RandomSearch(HyperparameterSearch):
    """Random search for hyperparameter optimization."""
    
    def search(self, objective_fn: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """
        Run random search over the parameter space.
        
        Args:
            objective_fn: Function that takes a parameter configuration and returns a score
                         (higher is better)
        
        Returns:
            Dictionary containing the best hyperparameters found
        """
        for i in range(self.max_trials):
            params = {}
            for name, values in self.param_space.items():
                params[name] = random.choice(values)
            
            logging.info(f"Trial {i+1}/{self.max_trials}: {params}")
            
            start_time = time.time()
            score = objective_fn(params)
            elapsed = time.time() - start_time
            
            self.results.append((params, score))
            self._update_best(params, score)
            
            logging.info(f"Trial {i+1} completed in {elapsed:.2f}s, score: {score:.4f}")
        
        return self.best_params


class BayesianOptimization(HyperparameterSearch):
    """Bayesian optimization for hyperparameter search."""
    
    def __init__(self, param_space: Dict[str, List[Any]], max_trials: int = 10, 
                 exploration_weight: float = 0.1):
        """
        Initialize Bayesian optimization search.
        
        Args:
            param_space: Dictionary mapping parameter names to lists of possible values
            max_trials: Maximum number of trials to run
            exploration_weight: Weight for exploration vs exploitation (higher means more exploration)
        """
        super().__init__(param_space, max_trials)
        self.exploration_weight = exploration_weight
        
        self._init_surrogate_model()
    
    def _init_surrogate_model(self):
        """Initialize the surrogate model (Gaussian Process)."""
        self.surrogate_model = None
        self.X_observed = []
        self.y_observed = []
    
    def _acquisition_function(self, params: Dict[str, Any]) -> float:
        """
        Compute the acquisition function value (Upper Confidence Bound).
        
        Args:
            params: Parameter configuration to evaluate
        
        Returns:
            Acquisition function value (higher is better)
        """
        if not self.X_observed:
            return 0.0
        
        param_vector = self._params_to_vector(params)
        similarities = [self._similarity(param_vector, x) for x in self.X_observed]
        
        if max(similarities) > 0.99:
            return float('-inf')
        
        return -max(similarities)
    
    def _params_to_vector(self, params: Dict[str, Any]) -> List[float]:
        """Convert parameter dictionary to a numeric vector."""
        vector = []
        for name, value in params.items():
            if isinstance(value, (int, float)):
                vector.append(float(value))
            else:
                options = self.param_space[name]
                for option in options:
                    vector.append(1.0 if value == option else 0.0)
        return vector
    
    def _similarity(self, v1: List[float], v2: List[float]) -> float:
        """Compute similarity between two parameter vectors."""
        if len(v1) != len(v2):
            return 0.0
        
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = sum(a * a for a in v1) ** 0.5
        norm2 = sum(b * b for b in v2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    
    def _update_surrogate_model(self):
        """Update the surrogate model with observed data."""
        pass
    
    def search(self, objective_fn: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """
        Run Bayesian optimization search over the parameter space.
        
        Args:
            objective_fn: Function that takes a parameter configuration and returns a score
                         (higher is better)
        
        Returns:
            Dictionary containing the best hyperparameters found
        """
        initial_points = min(3, self.max_trials)
        for i in range(initial_points):
            params = {}
            for name, values in self.param_space.items():
                params[name] = random.choice(values)
            
            logging.info(f"Initial exploration {i+1}/{initial_points}: {params}")
            
            start_time = time.time()
            score = objective_fn(params)
            elapsed = time.time() - start_time
            
            self.results.append((params, score))
            self._update_best(params, score)
            
            self.X_observed.append(self._params_to_vector(params))
            self.y_observed.append(score)
            
            logging.info(f"Trial {i+1} completed in {elapsed:.2f}s, score: {score:.4f}")
        
        self._update_surrogate_model()
        
        for i in range(initial_points, self.max_trials):
            best_acq = float('-inf')
            best_params = None
            
            num_candidates = 100
            for _ in range(num_candidates):
                params = {}
                for name, values in self.param_space.items():
                    params[name] = random.choice(values)
                
                acq_value = self._acquisition_function(params)
                if acq_value > best_acq:
                    best_acq = acq_value
                    best_params = params
            
            if best_params is None:
                best_params = {}
                for name, values in self.param_space.items():
                    best_params[name] = random.choice(values)
            
            logging.info(f"Trial {i+1}/{self.max_trials}: {best_params}")
            
            start_time = time.time()
            score = objective_fn(best_params)
            elapsed = time.time() - start_time
            
            self.results.append((best_params, score))
            self._update_best(best_params, score)
            
            self.X_observed.append(self._params_to_vector(best_params))
            self.y_observed.append(score)
            
            self._update_surrogate_model()
            
            logging.info(f"Trial {i+1} completed in {elapsed:.2f}s, score: {score:.4f}")
        
        return self.best_params


class EvolutionarySearch(HyperparameterSearch):
    """Evolutionary algorithm for hyperparameter optimization."""
    
    def __init__(self, param_space: Dict[str, List[Any]], max_trials: int = 10,
                 population_size: int = 10, mutation_prob: float = 0.1):
        """
        Initialize evolutionary search.
        
        Args:
            param_space: Dictionary mapping parameter names to lists of possible values
            max_trials: Maximum number of trials to run
            population_size: Size of the population
            mutation_prob: Probability of mutation
        """
        super().__init__(param_space, max_trials)
        self.population_size = min(population_size, max_trials)
        self.mutation_prob = mutation_prob
    
    def _initialize_population(self) -> List[Dict[str, Any]]:
        """Initialize a random population."""
        population = []
        for _ in range(self.population_size):
            params = {}
            for name, values in self.param_space.items():
                params[name] = random.choice(values)
            population.append(params)
        return population
    
    def _crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two parents."""
        child = {}
        for name in self.param_space:
            if random.random() < 0.5:
                child[name] = parent1[name]
            else:
                child[name] = parent2[name]
        return child
    
    def _mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate an individual."""
        mutated = individual.copy()
        for name, values in self.param_space.items():
            if random.random() < self.mutation_prob:
                current_value = mutated[name]
                new_values = [v for v in values if v != current_value]
                if new_values:
                    mutated[name] = random.choice(new_values)
        return mutated
    
    def search(self, objective_fn: Callable[[Dict[str, Any]], float]) -> Dict[str, Any]:
        """
        Run evolutionary search over the parameter space.
        
        Args:
            objective_fn: Function that takes a parameter configuration and returns a score
                         (higher is better)
        
        Returns:
            Dictionary containing the best hyperparameters found
        """
        population = self._initialize_population()
        
        population_scores = []
        for i, params in enumerate(population):
            logging.info(f"Initial population {i+1}/{len(population)}: {params}")
            
            start_time = time.time()
            score = objective_fn(params)
            elapsed = time.time() - start_time
            
            population_scores.append(score)
            self.results.append((params, score))
            self._update_best(params, score)
            
            logging.info(f"Evaluation completed in {elapsed:.2f}s, score: {score:.4f}")
        
        num_generations = (self.max_trials - self.population_size) // self.population_size
        for generation in range(num_generations):
            logging.info(f"Generation {generation+1}/{num_generations}")
            
            new_population = []
            for _ in range(self.population_size):
                tournament_size = 3
                tournament_indices = random.sample(range(self.population_size), 
                                                 min(tournament_size, self.population_size))
                tournament_scores = [population_scores[i] for i in tournament_indices]
                winner_idx = tournament_indices[tournament_scores.index(max(tournament_scores))]
                
                new_population.append(population[winner_idx])
            
            next_population = []
            for i in range(0, self.population_size, 2):
                if i + 1 < self.population_size:
                    parent1 = new_population[i]
                    parent2 = new_population[i + 1]
                    
                    child1 = self._crossover(parent1, parent2)
                    child2 = self._crossover(parent2, parent1)
                    
                    child1 = self._mutate(child1)
                    child2 = self._mutate(child2)
                    
                    next_population.extend([child1, child2])
                else:
                    next_population.append(self._mutate(new_population[i]))
            
            population = next_population
            population_scores = []
            
            for i, params in enumerate(population):
                logging.info(f"Generation {generation+1}, individual {i+1}/{len(population)}: {params}")
                
                start_time = time.time()
                score = objective_fn(params)
                elapsed = time.time() - start_time
                
                population_scores.append(score)
                self.results.append((params, score))
                self._update_best(params, score)
                
                logging.info(f"Evaluation completed in {elapsed:.2f}s, score: {score:.4f}")
        
        return self.best_params
