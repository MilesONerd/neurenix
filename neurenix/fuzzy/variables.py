"""
Fuzzy variables module for Neurenix.

This module provides implementations of fuzzy variables and
linguistic variables for fuzzy logic systems.
"""

from typing import Dict, List, Optional, Union, Tuple

import neurenix as nx
from neurenix.fuzzy.sets import FuzzySet


class FuzzyVariable:
    """Base class for fuzzy variables."""
    
    def __init__(self, name: str, universe: nx.Tensor = None, 
                 universe_range: Tuple[float, float] = None, resolution: int = 100):
        """
        Initialize a fuzzy variable.
        
        Args:
            name: Name of the variable
            universe: Universe of discourse
            universe_range: Range of the universe of discourse
            resolution: Resolution of the universe of discourse
        """
        self.name = name
        
        if universe is None and universe_range is not None:
            start, end = universe_range
            self.universe = nx.linspace(start, end, resolution)
        else:
            self.universe = universe
        
        self.sets = {}
    
    def add_set(self, name: str, fuzzy_set: FuzzySet):
        """
        Add a fuzzy set to the variable.
        
        Args:
            name: Name of the fuzzy set
            fuzzy_set: Fuzzy set
        """
        fuzzy_set.universe = self.universe
        self.sets[name] = fuzzy_set
    
    def get_set(self, name: str) -> FuzzySet:
        """
        Get a fuzzy set by name.
        
        Args:
            name: Name of the fuzzy set
            
        Returns:
            Fuzzy set
        """
        if name not in self.sets:
            raise ValueError(f"Fuzzy set '{name}' not found in variable '{self.name}'")
        
        return self.sets[name]
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        """
        Fuzzify a crisp value.
        
        Args:
            value: Crisp value
            
        Returns:
            Dictionary mapping set names to membership degrees
        """
        result = {}
        
        for name, fuzzy_set in self.sets.items():
            result[name] = fuzzy_set(value)
        
        return result


class LinguisticVariable(FuzzyVariable):
    """Linguistic variable for fuzzy logic systems."""
    
    def __init__(self, name: str, universe: nx.Tensor = None, 
                 universe_range: Tuple[float, float] = None, resolution: int = 100):
        """
        Initialize a linguistic variable.
        
        Args:
            name: Name of the variable
            universe: Universe of discourse
            universe_range: Range of the universe of discourse
            resolution: Resolution of the universe of discourse
        """
        super().__init__(name, universe, universe_range, resolution)
        
        self.terms = {}
    
    def add_term(self, name: str, fuzzy_set: FuzzySet):
        """
        Add a linguistic term to the variable.
        
        Args:
            name: Name of the linguistic term
            fuzzy_set: Fuzzy set representing the term
        """
        self.add_set(name, fuzzy_set)
        self.terms[name] = fuzzy_set
    
    def get_term(self, name: str) -> FuzzySet:
        """
        Get a linguistic term by name.
        
        Args:
            name: Name of the linguistic term
            
        Returns:
            Fuzzy set representing the term
        """
        return self.get_set(name)
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        """
        Fuzzify a crisp value.
        
        Args:
            value: Crisp value
            
        Returns:
            Dictionary mapping term names to membership degrees
        """
        return super().fuzzify(value)
