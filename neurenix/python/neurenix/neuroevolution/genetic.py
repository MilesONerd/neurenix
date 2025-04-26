"""
Genetic Algorithms for Neural Network Optimization.

This module provides implementations of genetic algorithms for optimizing neural networks,
including population management, selection, crossover, and mutation operations.
"""

import random
import numpy as np
from typing import List, Tuple, Callable, Dict, Any, Optional, Union

from neurenix.tensor import Tensor
from neurenix.nn.module import Module
from neurenix.device import get_device

class Individual:
    """Represents an individual in a genetic algorithm population."""
    
    def __init__(self, genotype: Union[Dict[str, Tensor], Module], fitness: float = 0.0):
        """Initialize an individual with a genotype and fitness value.
        
        Args:
            genotype: Either a dictionary of tensors representing network weights or a Module
            fitness: The fitness value of this individual
        """
        self.genotype = genotype
        self.fitness = fitness
        self.age = 0
        self.metadata = {}
    
    def clone(self) -> 'Individual':
        """Create a copy of this individual."""
        if isinstance(self.genotype, Module):
            new_genotype = self.genotype.__class__()
            new_genotype.load_state_dict(self.genotype.state_dict())
        else:
            new_genotype = {k: v.clone() for k, v in self.genotype.items()}
        
        new_individual = Individual(new_genotype, self.fitness)
        new_individual.age = self.age
        new_individual.metadata = self.metadata.copy()
        
        return new_individual
    
    def to_model(self, model_class) -> Module:
        """Convert this individual to a neural network model.
        
        Args:
            model_class: The class of the model to create
            
        Returns:
            A neural network model with weights from this individual
        """
        if isinstance(self.genotype, Module):
            return self.genotype
        
        model = model_class()
        model.load_state_dict(self.genotype)
        return model
    
    @classmethod
    def from_model(cls, model: Module) -> 'Individual':
        """Create an individual from a neural network model.
        
        Args:
            model: The neural network model
            
        Returns:
            An individual with genotype from the model
        """
        return cls(model)


class Mutation:
    """Base class for mutation operators."""
    
    def __call__(self, individual: Individual) -> Individual:
        """Apply mutation to an individual.
        
        Args:
            individual: The individual to mutate
            
        Returns:
            The mutated individual
        """
        raise NotImplementedError("Subclasses must implement this method")


class GaussianMutation(Mutation):
    """Applies Gaussian noise to weights."""
    
    def __init__(self, std: float = 0.1, mutation_rate: float = 0.1):
        """Initialize the mutation operator.
        
        Args:
            std: Standard deviation of the Gaussian noise
            mutation_rate: Probability of mutating each weight
        """
        self.std = std
        self.mutation_rate = mutation_rate
    
    def __call__(self, individual: Individual) -> Individual:
        """Apply Gaussian mutation to an individual.
        
        Args:
            individual: The individual to mutate
            
        Returns:
            The mutated individual
        """
        new_individual = individual.clone()
        
        if isinstance(new_individual.genotype, Module):
            for param in new_individual.genotype.parameters():
                if random.random() < self.mutation_rate:
                    noise = Tensor(np.random.normal(0, self.std, param.shape), 
                                  device=param.device)
                    param.data += noise
        else:
            for key, tensor in new_individual.genotype.items():
                if random.random() < self.mutation_rate:
                    noise = Tensor(np.random.normal(0, self.std, tensor.shape), 
                                  device=tensor.device)
                    new_individual.genotype[key] = tensor + noise
        
        return new_individual


class Crossover:
    """Base class for crossover operators."""
    
    def __call__(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Apply crossover to two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring individuals
        """
        raise NotImplementedError("Subclasses must implement this method")


class UniformCrossover(Crossover):
    """Performs uniform crossover between two individuals."""
    
    def __init__(self, crossover_rate: float = 0.5):
        """Initialize the crossover operator.
        
        Args:
            crossover_rate: Probability of swapping each parameter
        """
        self.crossover_rate = crossover_rate
    
    def __call__(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Apply uniform crossover to two parents.
        
        Args:
            parent1: First parent
            parent2: Second parent
            
        Returns:
            Two offspring individuals
        """
        child1 = parent1.clone()
        child2 = parent2.clone()
        
        if isinstance(parent1.genotype, Module) and isinstance(parent2.genotype, Module):
            for (name1, param1), (name2, param2) in zip(
                child1.genotype.named_parameters(), child2.genotype.named_parameters()
            ):
                if name1 == name2 and random.random() < self.crossover_rate:
                    temp = param1.data.clone()
                    param1.data = param2.data.clone()
                    param2.data = temp
        else:
            for key in child1.genotype:
                if key in child2.genotype and random.random() < self.crossover_rate:
                    temp = child1.genotype[key].clone()
                    child1.genotype[key] = child2.genotype[key].clone()
                    child2.genotype[key] = temp
        
        return child1, child2


class Selection:
    """Base class for selection operators."""
    
    def __call__(self, population: List[Individual], num_selected: int) -> List[Individual]:
        """Select individuals from the population.
        
        Args:
            population: The population to select from
            num_selected: Number of individuals to select
            
        Returns:
            Selected individuals
        """
        raise NotImplementedError("Subclasses must implement this method")


class TournamentSelection(Selection):
    """Performs tournament selection."""
    
    def __init__(self, tournament_size: int = 3):
        """Initialize the selection operator.
        
        Args:
            tournament_size: Number of individuals in each tournament
        """
        self.tournament_size = tournament_size
    
    def __call__(self, population: List[Individual], num_selected: int) -> List[Individual]:
        """Apply tournament selection.
        
        Args:
            population: The population to select from
            num_selected: Number of individuals to select
            
        Returns:
            Selected individuals
        """
        selected = []
        
        for _ in range(num_selected):
            tournament = random.sample(population, min(self.tournament_size, len(population)))
            
            winner = max(tournament, key=lambda ind: ind.fitness)
            selected.append(winner)
        
        return selected


class Population:
    """Manages a population of individuals for genetic algorithms."""
    
    def __init__(self, individuals: List[Individual] = None, size: int = 0):
        """Initialize the population.
        
        Args:
            individuals: Initial list of individuals
            size: Size of the population if individuals is None
        """
        self.individuals = individuals if individuals is not None else []
        self.size = size if individuals is None else len(individuals)
        self.generation = 0
    
    def initialize(self, model_class, init_params: Dict[str, Any] = None):
        """Initialize the population with random individuals.
        
        Args:
            model_class: The class of the model to create
            init_params: Parameters for model initialization
        """
        init_params = init_params or {}
        
        for _ in range(self.size):
            model = model_class(**init_params)
            self.individuals.append(Individual.from_model(model))
    
    def evaluate(self, fitness_function: Callable[[Individual], float]):
        """Evaluate the fitness of all individuals.
        
        Args:
            fitness_function: Function to evaluate fitness
        """
        for individual in self.individuals:
            individual.fitness = fitness_function(individual)
    
    def sort(self):
        """Sort the population by fitness in descending order."""
        self.individuals.sort(key=lambda ind: ind.fitness, reverse=True)
    
    def get_best(self) -> Individual:
        """Get the best individual in the population.
        
        Returns:
            The individual with the highest fitness
        """
        return max(self.individuals, key=lambda ind: ind.fitness)
    
    def get_worst(self) -> Individual:
        """Get the worst individual in the population.
        
        Returns:
            The individual with the lowest fitness
        """
        return min(self.individuals, key=lambda ind: ind.fitness)
    
    def get_average_fitness(self) -> float:
        """Get the average fitness of the population.
        
        Returns:
            Average fitness
        """
        return sum(ind.fitness for ind in self.individuals) / len(self.individuals)
    
    def next_generation(self):
        """Increment the generation counter."""
        self.generation += 1
        for individual in self.individuals:
            individual.age += 1


class GeneticAlgorithm:
    """Genetic algorithm for neural network optimization."""
    
    def __init__(self, 
                population_size: int = 100,
                selection: Selection = None,
                crossover: Crossover = None,
                mutation: Mutation = None,
                elitism: int = 1):
        """Initialize the genetic algorithm.
        
        Args:
            population_size: Size of the population
            selection: Selection operator
            crossover: Crossover operator
            mutation: Mutation operator
            elitism: Number of best individuals to preserve
        """
        self.population_size = population_size
        self.selection = selection or TournamentSelection()
        self.crossover = crossover or UniformCrossover()
        self.mutation = mutation or GaussianMutation()
        self.elitism = elitism
        self.population = Population(size=population_size)
        self.best_individual = None
        self.history = {
            'best_fitness': [],
            'avg_fitness': [],
            'worst_fitness': []
        }
    
    def initialize(self, model_class, init_params: Dict[str, Any] = None):
        """Initialize the population.
        
        Args:
            model_class: The class of the model to create
            init_params: Parameters for model initialization
        """
        self.population.initialize(model_class, init_params)
    
    def evolve(self, fitness_function: Callable[[Individual], float], generations: int = 100,
              callback: Callable[[int, Population], None] = None):
        """Run the genetic algorithm.
        
        Args:
            fitness_function: Function to evaluate fitness
            generations: Number of generations to run
            callback: Function called after each generation
        """
        self.population.evaluate(fitness_function)
        self.population.sort()
        self._update_history()
        
        if callback:
            callback(0, self.population)
        
        for generation in range(1, generations + 1):
            elite = self.population.individuals[:self.elitism]
            
            parents = self.selection(self.population.individuals, 
                                    self.population_size - self.elitism)
            
            new_individuals = [ind.clone() for ind in elite]
            
            while len(new_individuals) < self.population_size:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = self.crossover(parent1, parent2)
                
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                new_individuals.append(child1)
                new_individuals.append(child2)
            
            new_individuals = new_individuals[:self.population_size]
            
            self.population.individuals = new_individuals
            self.population.next_generation()
            
            self.population.evaluate(fitness_function)
            self.population.sort()
            self._update_history()
            
            if callback:
                callback(generation, self.population)
    
    def _update_history(self):
        """Update history with current population statistics."""
        best = self.population.get_best()
        worst = self.population.get_worst()
        avg = self.population.get_average_fitness()
        
        self.history['best_fitness'].append(best.fitness)
        self.history['worst_fitness'].append(worst.fitness)
        self.history['avg_fitness'].append(avg)
        
        if self.best_individual is None or best.fitness > self.best_individual.fitness:
            self.best_individual = best.clone()
    
    def get_best_model(self, model_class):
        """Get the best model found by the algorithm.
        
        Args:
            model_class: The class of the model to create
            
        Returns:
            The best model
        """
        if self.best_individual is None:
            return None
        
        return self.best_individual.to_model(model_class)
