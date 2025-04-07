"""
Neuroevolution and Evolutionary Algorithms for Neurenix.

This module provides implementations of neuroevolution and evolutionary algorithms
for neural network optimization, including NEAT, HyperNEAT, CMA-ES, and genetic algorithms.
"""

from .genetic import (
    GeneticAlgorithm,
    Population,
    Individual,
    Mutation,
    Crossover,
    Selection
)

from .neat import (
    NEAT,
    NEATConfig,
    NEATGenome,
    NEATSpecies,
    NEATPopulation
)

from .hyperneat import (
    HyperNEAT,
    CPPN,
    Substrate
)

from .cmaes import (
    CMAES,
    CMAESConfig
)

from .evolution_strategy import (
    EvolutionStrategy,
    ESConfig,
    ESPopulation
)

__all__ = [
    'GeneticAlgorithm',
    'Population',
    'Individual',
    'Mutation',
    'Crossover',
    'Selection',
    
    'NEAT',
    'NEATConfig',
    'NEATGenome',
    'NEATSpecies',
    'NEATPopulation',
    
    'HyperNEAT',
    'CPPN',
    'Substrate',
    
    'CMAES',
    'CMAESConfig',
    
    'EvolutionStrategy',
    'ESConfig',
    'ESPopulation'
]
