"""
Covariance Matrix Adaptation Evolution Strategy (CMA-ES) implementation.

This module implements the CMA-ES algorithm, a state-of-the-art evolutionary
algorithm for difficult non-linear non-convex optimization problems in continuous
domains.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
from collections import defaultdict

from neurenix.tensor import Tensor
from neurenix.nn.module import Module


class CMAESConfig:
    """Configuration parameters for CMA-ES algorithm."""
    
    def __init__(self, 
                 sigma: float = 0.5,
                 population_size: Optional[int] = None,
                 parent_number: Optional[int] = None,
                 weights_option: str = "default",
                 cs: Optional[float] = None,
                 ds: Optional[float] = None,
                 cc: Optional[float] = None,
                 c1: Optional[float] = None,
                 cmu: Optional[float] = None,
                 active: bool = True,
                 diagonal_iterations: int = 0,
                 tolx: float = 1e-12,
                 tolfun: float = 1e-12,
                 tolstagnation: int = 100,
                 max_iterations: int = 1000):
        """Initialize CMA-ES configuration.
        
        Args:
            sigma: Initial step size
            population_size: Population size (lambda). If None, defaults to 4 + floor(3 * log(N))
            parent_number: Number of parents (mu). If None, defaults to population_size // 2
            weights_option: How to compute recombination weights ('default', 'equal', or 'linear')
            cs: Step size cumulation parameter. If None, defaults to (mueff + 2) / (N + mueff + 5)
            ds: Step size damping parameter. If None, defaults to 1 + 2 * max(0, sqrt((mueff - 1) / (N + 1)) - 1) + cs
            cc: Cumulation parameter for C. If None, defaults to 4 / (N + 4)
            c1: Learning rate for rank-one update. If None, defaults to 2 / ((N + 1.3)^2 + mueff)
            cmu: Learning rate for rank-mu update. If None, defaults to min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((N + 2)^2 + mueff))
            active: Whether to use active CMA (negative weights for bad solutions)
            diagonal_iterations: Number of iterations with diagonal covariance matrix
            tolx: Tolerance for changes in solution
            tolfun: Tolerance for changes in objective function
            tolstagnation: Maximum number of iterations without improvement
            max_iterations: Maximum number of iterations
        """
        self.sigma = sigma
        self.population_size = population_size
        self.parent_number = parent_number
        self.weights_option = weights_option
        self.cs = cs
        self.ds = ds
        self.cc = cc
        self.c1 = c1
        self.cmu = cmu
        self.active = active
        self.diagonal_iterations = diagonal_iterations
        self.tolx = tolx
        self.tolfun = tolfun
        self.tolstagnation = tolstagnation
        self.max_iterations = max_iterations


class CMAES:
    """Implementation of the CMA-ES algorithm."""
    
    def __init__(self, dimension: int, mean: Optional[np.ndarray] = None, 
                 config: Optional[CMAESConfig] = None):
        """Initialize the CMA-ES algorithm.
        
        Args:
            dimension: Dimension of the search space
            mean: Initial mean vector. If None, defaults to zeros
            config: Configuration parameters
        """
        self.dimension = dimension
        self.mean = mean if mean is not None else np.zeros(dimension)
        self.config = config if config is not None else CMAESConfig()
        
        if self.config.population_size is None:
            self.population_size = 4 + int(3 * np.log(dimension))
        else:
            self.population_size = self.config.population_size
            
        if self.config.parent_number is None:
            self.parent_number = self.population_size // 2
        else:
            self.parent_number = self.config.parent_number
            
        if self.config.weights_option == "equal":
            self.weights = np.ones(self.parent_number) / self.parent_number
        elif self.config.weights_option == "linear":
            self.weights = np.array([self.parent_number - i for i in range(self.parent_number)])
            self.weights = self.weights / np.sum(self.weights)
        else:  # default
            self.weights = np.array([np.log(self.parent_number + 0.5) - np.log(i + 1) for i in range(self.parent_number)])
            self.weights = self.weights / np.sum(self.weights)
            
        self.mueff = 1.0 / np.sum(np.square(self.weights))
        
        if self.config.cs is None:
            self.cs = (self.mueff + 2) / (self.dimension + self.mueff + 5)
        else:
            self.cs = self.config.cs
            
        if self.config.ds is None:
            self.ds = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dimension + 1)) - 1) + self.cs
        else:
            self.ds = self.config.ds
            
        if self.config.cc is None:
            self.cc = 4 / (self.dimension + 4)
        else:
            self.cc = self.config.cc
            
        if self.config.c1 is None:
            self.c1 = 2 / ((self.dimension + 1.3) ** 2 + self.mueff)
        else:
            self.c1 = self.config.c1
            
        if self.config.cmu is None:
            self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / 
                          ((self.dimension + 2) ** 2 + self.mueff))
        else:
            self.cmu = self.config.cmu
            
        self.sigma = self.config.sigma
        self.pc = np.zeros(self.dimension)
        self.ps = np.zeros(self.dimension)
        self.B = np.eye(self.dimension)
        self.D = np.ones(self.dimension)
        self.C = np.eye(self.dimension)
        self.invsqrtC = np.eye(self.dimension)
        
        self.chiN = np.sqrt(self.dimension) * (1 - 1/(4*self.dimension) + 1/(21*self.dimension**2))
        
        self.iterations = 0
        self.evaluations = 0
        self.best_fitness = float('inf')
        self.best_solution = None
        self.stop_dict = {}
        self.diag_mode = self.config.diagonal_iterations > 0
        self.eigen_decomposition_count = 0
        self.eigen_decomposition_interval = 1
        
    def ask(self) -> List[np.ndarray]:
        """Sample new candidate solutions.
        
        Returns:
            List of candidate solutions
        """
        if self.diag_mode:
            samples = []
            for _ in range(self.population_size):
                z = np.random.normal(0, 1, self.dimension)
                y = z * self.D
                x = self.mean + self.sigma * y
                samples.append(x)
            return samples
        else:
            if (self.iterations - self.eigen_decomposition_count) >= self.eigen_decomposition_interval:
                self._update_eigen_decomposition()
                
            samples = []
            for _ in range(self.population_size):
                z = np.random.normal(0, 1, self.dimension)
                y = np.dot(self.B, self.D * z)
                x = self.mean + self.sigma * y
                samples.append(x)
            return samples
        
    def tell(self, solutions: List[np.ndarray], fitnesses: List[float]) -> None:
        """Update the strategy parameters using the evaluated solutions.
        
        Args:
            solutions: List of candidate solutions
            fitnesses: List of fitness values for the solutions
        """
        self.iterations += 1
        self.evaluations += len(solutions)
        
        sorted_indices = np.argsort(fitnesses)
        sorted_solutions = [solutions[i] for i in sorted_indices]
        sorted_fitnesses = [fitnesses[i] for i in sorted_indices]
        
        if sorted_fitnesses[0] < self.best_fitness:
            self.best_fitness = sorted_fitnesses[0]
            self.best_solution = sorted_solutions[0].copy()
            
        selected_solutions = sorted_solutions[:self.parent_number]
        
        old_mean = self.mean.copy()
        self.mean = np.zeros_like(self.mean)
        for i, solution in enumerate(selected_solutions):
            self.mean += self.weights[i] * solution
            
        y = (self.mean - old_mean) / self.sigma
        
        if self.diag_mode:
            z = y / self.D
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * z
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.iterations)) < self.chiN * (1.4 + 2 / (self.dimension + 1))
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
            
            c1a = self.c1 * (1 - (1 - hsig**2) * self.cc * (2 - self.cc))
            self.C = (1 - c1a - self.cmu) * self.C + c1a * np.outer(self.pc, self.pc)
            
            for i, solution in enumerate(selected_solutions):
                w = self.weights[i]
                z = (solution - old_mean) / self.sigma
                if self.config.active and w < 0:
                    self.C += w * self.cmu * np.outer(z, z) / self.C
                else:
                    self.C += w * self.cmu * np.outer(z, z)
                    
            self.D = np.sqrt(np.diag(self.C))
            
        else:
            z = np.dot(self.invsqrtC, y)
            self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * z
            hsig = np.linalg.norm(self.ps) / np.sqrt(1 - (1 - self.cs) ** (2 * self.iterations)) < self.chiN * (1.4 + 2 / (self.dimension + 1))
            self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y
            
            c1a = self.c1 * (1 - (1 - hsig**2) * self.cc * (2 - self.cc))
            self.C = (1 - c1a - self.cmu) * self.C + c1a * np.outer(self.pc, self.pc)
            
            for i, solution in enumerate(selected_solutions):
                w = self.weights[i]
                z = (solution - old_mean) / self.sigma
                if self.config.active and w < 0:
                    artmp = np.dot(self.invsqrtC, z)
                    self.C += w * self.cmu * np.outer(artmp, artmp)
                else:
                    self.C += w * self.cmu * np.outer(z, z)
                    
        self.sigma *= np.exp((np.linalg.norm(self.ps) / self.chiN - 1) * self.cs / self.ds)
        
        if self.diag_mode and self.iterations >= self.config.diagonal_iterations:
            self.diag_mode = False
            self._update_eigen_decomposition()
            
        self._check_stop()
        
    def _update_eigen_decomposition(self) -> None:
        """Update the eigendecomposition of the covariance matrix."""
        self.eigen_decomposition_count = self.iterations
        
        self.C = np.triu(self.C) + np.triu(self.C, 1).T
        
        D2, B = np.linalg.eigh(self.C)
        
        D2 = np.maximum(D2, 1e-14)
        
        idx = np.argsort(D2)[::-1]
        self.D = np.sqrt(D2[idx])
        self.B = B[:, idx]
        
        self.invsqrtC = np.dot(self.B, np.diag(1/self.D)).dot(self.B.T)
        
        self.eigen_decomposition_interval = int(0.5 * self.iterations)
        
    def _check_stop(self) -> None:
        """Check stopping criteria."""
        if self.iterations >= self.config.max_iterations:
            self.stop_dict['maxiter'] = True
            
        if len(self.stop_dict) > 0:
            return
            
        
    def optimize(self, objective_function: Callable[[np.ndarray], float], 
                iterations: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """Optimize the objective function.
        
        Args:
            objective_function: Function to minimize
            iterations: Maximum number of iterations. If None, uses config.max_iterations
            
        Returns:
            Tuple of (best solution, best fitness)
        """
        max_iter = iterations if iterations is not None else self.config.max_iterations
        
        for _ in range(max_iter):
            if len(self.stop_dict) > 0:
                break
                
            solutions = self.ask()
            
            fitnesses = [objective_function(x) for x in solutions]
            
            self.tell(solutions, fitnesses)
            
        return self.best_solution, self.best_fitness
    
    def get_best(self) -> Tuple[np.ndarray, float]:
        """Get the best solution found so far.
        
        Returns:
            Tuple of (best solution, best fitness)
        """
        return self.best_solution, self.best_fitness


class CMAESModel(Module):
    """Neural network model optimized using CMA-ES."""
    
    def __init__(self, model: Module, config: Optional[CMAESConfig] = None):
        """Initialize a model to be optimized with CMA-ES.
        
        Args:
            model: Neural network model to optimize
            config: CMA-ES configuration
        """
        super().__init__()
        self.model = model
        self.config = config if config is not None else CMAESConfig()
        self.cmaes = None
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
            Loss value
        """
        self._vector_to_model_params(params)
        y_pred = self.model(X)
        loss = loss_fn(y_pred, y)
        return loss
        
    def fit(self, X: Tensor, y: Tensor, loss_fn: Callable[[Tensor, Tensor], float], 
           iterations: int = 100, verbose: bool = False) -> float:
        """Fit the model to the data using CMA-ES.
        
        Args:
            X: Input data
            y: Target data
            loss_fn: Loss function
            iterations: Maximum number of iterations
            verbose: Whether to print progress
            
        Returns:
            Best loss value
        """
        dimension = self._count_parameters()
        initial_params = self._model_params_to_vector()
        
        self.cmaes = CMAES(dimension, initial_params, self.config)
        
        def objective(params):
            return self._evaluate(params, X, y, loss_fn)
            
        best_params, best_loss = self.cmaes.optimize(objective, iterations)
        
        self._vector_to_model_params(best_params)
        self.best_params = best_params
        
        if verbose:
            print(f"CMA-ES optimization completed. Best loss: {best_loss}")
            
        return best_loss
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x)
