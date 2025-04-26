"""
Evolution Strategies (ES) implementation.

This module implements various Evolution Strategies algorithms, which are
powerful black-box optimization techniques that use stochastic population-based
approaches to find optimal solutions.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from collections import defaultdict

from neurenix.tensor import Tensor
from neurenix.nn.module import Module


class ESConfig:
    """Configuration parameters for Evolution Strategies."""
    
    def __init__(self, 
                 population_size: int = 100,
                 sigma: float = 0.1,
                 learning_rate: float = 0.01,
                 decay: float = 0.999,
                 noise_std: float = 0.01,
                 weight_decay: float = 0.0,
                 antithetic: bool = True,
                 rank_based: bool = True,
                 normalize_observations: bool = True,
                 normalize_updates: bool = True,
                 adam: bool = True,
                 adam_beta1: float = 0.9,
                 adam_beta2: float = 0.999,
                 adam_epsilon: float = 1e-8):
        """Initialize ES configuration.
        
        Args:
            population_size: Number of individuals in the population
            sigma: Standard deviation of the noise
            learning_rate: Learning rate for parameter updates
            decay: Decay rate for learning rate and sigma
            noise_std: Standard deviation of the noise
            weight_decay: Weight decay coefficient
            antithetic: Whether to use antithetic sampling
            rank_based: Whether to use rank-based fitness shaping
            normalize_observations: Whether to normalize observations
            normalize_updates: Whether to normalize updates
            adam: Whether to use Adam optimizer
            adam_beta1: Beta1 parameter for Adam
            adam_beta2: Beta2 parameter for Adam
            adam_epsilon: Epsilon parameter for Adam
        """
        self.population_size = population_size
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.noise_std = noise_std
        self.weight_decay = weight_decay
        self.antithetic = antithetic
        self.rank_based = rank_based
        self.normalize_observations = normalize_observations
        self.normalize_updates = normalize_updates
        self.adam = adam
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon


class ESPopulation:
    """Population of individuals for Evolution Strategies."""
    
    def __init__(self, config: ESConfig, dimension: int):
        """Initialize a population for Evolution Strategies.
        
        Args:
            config: Configuration parameters
            dimension: Dimension of the search space
        """
        self.config = config
        self.dimension = dimension
        self.population_size = config.population_size
        
        if config.antithetic:
            assert self.population_size % 2 == 0, "Population size must be even for antithetic sampling"
            self.half_popsize = self.population_size // 2
        else:
            self.half_popsize = self.population_size
            
        self.solutions = []
        self.fitnesses = []
        self.best_fitness = float('-inf')
        self.best_solution = None
        
    def sample(self, mean: np.ndarray) -> List[np.ndarray]:
        """Sample new solutions from the population.
        
        Args:
            mean: Mean vector
            
        Returns:
            List of sampled solutions
        """
        self.solutions = []
        
        if self.config.antithetic:
            noise = np.random.randn(self.half_popsize, self.dimension)
            for i in range(self.half_popsize):
                self.solutions.append(mean + self.config.sigma * noise[i])
                self.solutions.append(mean - self.config.sigma * noise[i])
        else:
            for i in range(self.population_size):
                noise = np.random.randn(self.dimension)
                self.solutions.append(mean + self.config.sigma * noise)
                
        return self.solutions
    
    def update(self, fitnesses: List[float]) -> Tuple[np.ndarray, float]:
        """Update the population based on fitness values.
        
        Args:
            fitnesses: List of fitness values for the solutions
            
        Returns:
            Tuple of (best solution, best fitness)
        """
        self.fitnesses = fitnesses
        
        best_idx = np.argmax(fitnesses)
        if fitnesses[best_idx] > self.best_fitness:
            self.best_fitness = fitnesses[best_idx]
            self.best_solution = self.solutions[best_idx].copy()
            
        return self.best_solution, self.best_fitness
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        """Get the best solution found so far.
        
        Returns:
            Tuple of (best solution, best fitness)
        """
        return self.best_solution, self.best_fitness


class EvolutionStrategy:
    """Implementation of Evolution Strategies."""
    
    def __init__(self, dimension: int, mean: Optional[np.ndarray] = None, 
                 config: Optional[ESConfig] = None):
        """Initialize the Evolution Strategy.
        
        Args:
            dimension: Dimension of the search space
            mean: Initial mean vector. If None, defaults to zeros
            config: Configuration parameters
        """
        self.dimension = dimension
        self.mean = mean if mean is not None else np.zeros(dimension)
        self.config = config if config is not None else ESConfig()
        
        self.population = ESPopulation(self.config, dimension)
        
        if self.config.adam:
            self.m = np.zeros(dimension)
            self.v = np.zeros(dimension)
            self.t = 0
            
        if self.config.normalize_observations:
            self.obs_mean = np.zeros(dimension)
            self.obs_std = np.ones(dimension)
            self.obs_count = 0
            
    def ask(self) -> List[np.ndarray]:
        """Sample new candidate solutions.
        
        Returns:
            List of candidate solutions
        """
        return self.population.sample(self.mean)
    
    def tell(self, fitnesses: List[float]) -> None:
        """Update the strategy parameters using the evaluated solutions.
        
        Args:
            fitnesses: List of fitness values for the solutions
        """
        self.population.update(fitnesses)
        
        if self.config.rank_based:
            ranks = np.argsort(np.argsort(-np.array(fitnesses)))
            ranks = ranks / (len(fitnesses) - 1) - 0.5
            fitnesses = ranks
            
        if self.config.antithetic:
            half_popsize = self.population.half_popsize
            noise = []
            for i in range(half_popsize):
                noise_i = (self.population.solutions[i] - self.mean) / self.config.sigma
                noise.append(noise_i)
                
            weighted_noise = np.zeros(self.dimension)
            for i in range(half_popsize):
                weighted_noise += fitnesses[i] * noise[i] - fitnesses[i + half_popsize] * noise[i]
                
            weighted_noise /= half_popsize
        else:
            weighted_noise = np.zeros(self.dimension)
            for i in range(self.population.population_size):
                noise_i = (self.population.solutions[i] - self.mean) / self.config.sigma
                weighted_noise += fitnesses[i] * noise_i
                
            weighted_noise /= self.population.population_size
            
        if self.config.normalize_updates:
            weighted_noise = weighted_noise / (np.std(weighted_noise) + 1e-8)
            
        if self.config.weight_decay > 0:
            weighted_noise -= self.config.weight_decay * self.mean
            
        if self.config.adam:
            self.t += 1
            self.m = self.config.adam_beta1 * self.m + (1 - self.config.adam_beta1) * weighted_noise
            self.v = self.config.adam_beta2 * self.v + (1 - self.config.adam_beta2) * (weighted_noise ** 2)
            
            m_hat = self.m / (1 - self.config.adam_beta1 ** self.t)
            v_hat = self.v / (1 - self.config.adam_beta2 ** self.t)
            
            update = self.config.learning_rate * m_hat / (np.sqrt(v_hat) + self.config.adam_epsilon)
        else:
            update = self.config.learning_rate * weighted_noise
            
        self.mean += update
        
        self.config.learning_rate *= self.config.decay
        self.config.sigma *= self.config.decay
        
    def optimize(self, objective_function: Callable[[np.ndarray], float], 
                iterations: int = 100, verbose: bool = False) -> Tuple[np.ndarray, float]:
        """Optimize the objective function.
        
        Args:
            objective_function: Function to maximize
            iterations: Number of iterations
            verbose: Whether to print progress
            
        Returns:
            Tuple of (best solution, best fitness)
        """
        for i in range(iterations):
            solutions = self.ask()
            
            fitnesses = [objective_function(x) for x in solutions]
            
            self.tell(fitnesses)
            
            if verbose and (i % 10 == 0 or i == iterations - 1):
                best_solution, best_fitness = self.population.get_best()
                print(f"Iteration {i}: Best fitness = {best_fitness}")
                
        return self.population.get_best()
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        """Get the best solution found so far.
        
        Returns:
            Tuple of (best solution, best fitness)
        """
        return self.population.get_best()


class ESModel(Module):
    """Neural network model optimized using Evolution Strategies."""
    
    def __init__(self, model: Module, config: Optional[ESConfig] = None):
        """Initialize a model to be optimized with Evolution Strategies.
        
        Args:
            model: Neural network model to optimize
            config: Evolution Strategies configuration
        """
        super().__init__()
        self.model = model
        self.config = config if config is not None else ESConfig()
        self.es = None
        self.best_params = None
        
    def _count_parameters(self) -> int:
        """Count the number of parameters in the model.
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.model.parameters())
        
    def _model_params_to_vector(self) -> np.ndarray:
        """Convert model parameters to a flat vector.
        
        Returns:
            Flat vector of parameters
        """
        params = []
        for p in self.model.parameters():
            params.append(p.detach().numpy().flatten())
        return np.concatenate(params)
        
    def _vector_to_model_params(self, vector: np.ndarray) -> None:
        """Update model parameters from a flat vector.
        
        Args:
            vector: Flat vector of parameters
        """
        start = 0
        for p in self.model.parameters():
            size = p.numel()
            p.data = Tensor(vector[start:start+size].reshape(p.shape))
            start += size
            
    def _evaluate(self, params: np.ndarray, X: Tensor, y: Tensor, 
                 loss_fn: Callable[[Tensor, Tensor], float]) -> float:
        """Evaluate the model with the given parameters.
        
        Args:
            params: Flat vector of parameters
            X: Input data
            y: Target data
            loss_fn: Loss function
            
        Returns:
            Negative loss value (for maximization)
        """
        self._vector_to_model_params(params)
        y_pred = self.model(X)
        loss = loss_fn(y_pred, y)
        return -loss  # Negate loss for maximization
        
    def fit(self, X: Tensor, y: Tensor, loss_fn: Callable[[Tensor, Tensor], float], 
           iterations: int = 100, verbose: bool = False) -> float:
        """Fit the model to the data using Evolution Strategies.
        
        Args:
            X: Input data
            y: Target data
            loss_fn: Loss function
            iterations: Number of iterations
            verbose: Whether to print progress
            
        Returns:
            Best loss value
        """
        dimension = self._count_parameters()
        initial_params = self._model_params_to_vector()
        
        self.es = EvolutionStrategy(dimension, initial_params, self.config)
        
        def objective(params):
            return self._evaluate(params, X, y, loss_fn)
            
        best_params, best_fitness = self.es.optimize(objective, iterations, verbose)
        
        self._vector_to_model_params(best_params)
        self.best_params = best_params
        
        if verbose:
            print(f"ES optimization completed. Best loss: {-best_fitness}")
            
        return -best_fitness  # Return loss (not negative)
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x)
