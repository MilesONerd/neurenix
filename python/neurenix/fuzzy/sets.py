"""
Fuzzy sets module for Neurenix.

This module provides implementations of various fuzzy sets and
membership functions for fuzzy logic systems.
"""

from typing import Callable, Optional, Union, List, Tuple

import neurenix as nx


class FuzzySet:
    """Base class for fuzzy sets."""
    
    def __init__(self, universe: nx.Tensor = None):
        """
        Initialize a fuzzy set.
        
        Args:
            universe: Universe of discourse
        """
        self.universe = universe
    
    def membership(self, x: Union[float, nx.Tensor]) -> Union[float, nx.Tensor]:
        """
        Calculate the membership degree of x in the fuzzy set.
        
        Args:
            x: Input value(s)
            
        Returns:
            Membership degree(s)
        """
        raise NotImplementedError("Subclasses must implement membership method")
    
    def __call__(self, x: Union[float, nx.Tensor]) -> Union[float, nx.Tensor]:
        """
        Calculate the membership degree of x in the fuzzy set.
        
        Args:
            x: Input value(s)
            
        Returns:
            Membership degree(s)
        """
        return self.membership(x)
    
    def __and__(self, other: 'FuzzySet') -> 'FuzzySet':
        """
        Intersection of two fuzzy sets.
        
        Args:
            other: Another fuzzy set
            
        Returns:
            Intersection fuzzy set
        """
        return IntersectionSet(self, other)
    
    def __or__(self, other: 'FuzzySet') -> 'FuzzySet':
        """
        Union of two fuzzy sets.
        
        Args:
            other: Another fuzzy set
            
        Returns:
            Union fuzzy set
        """
        return UnionSet(self, other)
    
    def __invert__(self) -> 'FuzzySet':
        """
        Complement of the fuzzy set.
        
        Returns:
            Complement fuzzy set
        """
        return ComplementSet(self)


class TriangularSet(FuzzySet):
    """Triangular fuzzy set."""
    
    def __init__(self, a: float, b: float, c: float, universe: nx.Tensor = None):
        """
        Initialize a triangular fuzzy set.
        
        Args:
            a: Left boundary
            b: Peak
            c: Right boundary
            universe: Universe of discourse
        """
        super().__init__(universe)
        
        if not (a <= b <= c):
            raise ValueError("Parameters must satisfy a <= b <= c")
        
        self.a = a
        self.b = b
        self.c = c
    
    def membership(self, x: Union[float, nx.Tensor]) -> Union[float, nx.Tensor]:
        """
        Calculate the membership degree of x in the fuzzy set.
        
        Args:
            x: Input value(s)
            
        Returns:
            Membership degree(s)
        """
        if isinstance(x, (int, float)):
            if x <= self.a or x >= self.c:
                return 0.0
            elif x <= self.b:
                return (x - self.a) / (self.b - self.a)
            else:
                return (self.c - x) / (self.c - self.b)
        else:
            result = nx.zeros_like(x, dtype=nx.float32)
            
            mask_left = (x > self.a) & (x <= self.b)
            result[mask_left] = (x[mask_left] - self.a) / (self.b - self.a)
            
            mask_right = (x > self.b) & (x < self.c)
            result[mask_right] = (self.c - x[mask_right]) / (self.c - self.b)
            
            return result


class TrapezoidalSet(FuzzySet):
    """Trapezoidal fuzzy set."""
    
    def __init__(self, a: float, b: float, c: float, d: float, universe: nx.Tensor = None):
        """
        Initialize a trapezoidal fuzzy set.
        
        Args:
            a: Left boundary
            b: Left peak
            c: Right peak
            d: Right boundary
            universe: Universe of discourse
        """
        super().__init__(universe)
        
        if not (a <= b <= c <= d):
            raise ValueError("Parameters must satisfy a <= b <= c <= d")
        
        self.a = a
        self.b = b
        self.c = c
        self.d = d
    
    def membership(self, x: Union[float, nx.Tensor]) -> Union[float, nx.Tensor]:
        """
        Calculate the membership degree of x in the fuzzy set.
        
        Args:
            x: Input value(s)
            
        Returns:
            Membership degree(s)
        """
        if isinstance(x, (int, float)):
            if x <= self.a or x >= self.d:
                return 0.0
            elif x >= self.b and x <= self.c:
                return 1.0
            elif x < self.b:
                return (x - self.a) / (self.b - self.a)
            else:
                return (self.d - x) / (self.d - self.c)
        else:
            result = nx.zeros_like(x, dtype=nx.float32)
            
            mask_left = (x > self.a) & (x < self.b)
            result[mask_left] = (x[mask_left] - self.a) / (self.b - self.a)
            
            mask_plateau = (x >= self.b) & (x <= self.c)
            result[mask_plateau] = 1.0
            
            mask_right = (x > self.c) & (x < self.d)
            result[mask_right] = (self.d - x[mask_right]) / (self.d - self.c)
            
            return result


class GaussianSet(FuzzySet):
    """Gaussian fuzzy set."""
    
    def __init__(self, mean: float, sigma: float, universe: nx.Tensor = None):
        """
        Initialize a Gaussian fuzzy set.
        
        Args:
            mean: Mean of the Gaussian function
            sigma: Standard deviation of the Gaussian function
            universe: Universe of discourse
        """
        super().__init__(universe)
        
        if sigma <= 0:
            raise ValueError("Sigma must be positive")
        
        self.mean = mean
        self.sigma = sigma
    
    def membership(self, x: Union[float, nx.Tensor]) -> Union[float, nx.Tensor]:
        """
        Calculate the membership degree of x in the fuzzy set.
        
        Args:
            x: Input value(s)
            
        Returns:
            Membership degree(s)
        """
        if isinstance(x, (int, float)):
            return nx.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)
        else:
            return nx.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)


class BellSet(FuzzySet):
    """Generalized bell-shaped fuzzy set."""
    
    def __init__(self, a: float, b: float, c: float, universe: nx.Tensor = None):
        """
        Initialize a bell-shaped fuzzy set.
        
        Args:
            a: Width of the bell
            b: Slope of the bell
            c: Center of the bell
            universe: Universe of discourse
        """
        super().__init__(universe)
        
        if a <= 0:
            raise ValueError("Parameter 'a' must be positive")
        
        if b <= 0:
            raise ValueError("Parameter 'b' must be positive")
        
        self.a = a
        self.b = b
        self.c = c
    
    def membership(self, x: Union[float, nx.Tensor]) -> Union[float, nx.Tensor]:
        """
        Calculate the membership degree of x in the fuzzy set.
        
        Args:
            x: Input value(s)
            
        Returns:
            Membership degree(s)
        """
        if isinstance(x, (int, float)):
            return 1.0 / (1.0 + abs((x - self.c) / self.a) ** (2 * self.b))
        else:
            return 1.0 / (1.0 + nx.abs((x - self.c) / self.a) ** (2 * self.b))


class SigmoidSet(FuzzySet):
    """Sigmoid fuzzy set."""
    
    def __init__(self, a: float, c: float, universe: nx.Tensor = None):
        """
        Initialize a sigmoid fuzzy set.
        
        Args:
            a: Slope of the sigmoid
            c: Center of the sigmoid
            universe: Universe of discourse
        """
        super().__init__(universe)
        
        self.a = a
        self.c = c
    
    def membership(self, x: Union[float, nx.Tensor]) -> Union[float, nx.Tensor]:
        """
        Calculate the membership degree of x in the fuzzy set.
        
        Args:
            x: Input value(s)
            
        Returns:
            Membership degree(s)
        """
        if isinstance(x, (int, float)):
            return 1.0 / (1.0 + nx.exp(-self.a * (x - self.c)))
        else:
            return 1.0 / (1.0 + nx.exp(-self.a * (x - self.c)))


class IntersectionSet(FuzzySet):
    """Intersection of two fuzzy sets."""
    
    def __init__(self, set1: FuzzySet, set2: FuzzySet, universe: nx.Tensor = None):
        """
        Initialize an intersection fuzzy set.
        
        Args:
            set1: First fuzzy set
            set2: Second fuzzy set
            universe: Universe of discourse
        """
        super().__init__(universe)
        
        self.set1 = set1
        self.set2 = set2
    
    def membership(self, x: Union[float, nx.Tensor]) -> Union[float, nx.Tensor]:
        """
        Calculate the membership degree of x in the fuzzy set.
        
        Args:
            x: Input value(s)
            
        Returns:
            Membership degree(s)
        """
        return nx.minimum(self.set1(x), self.set2(x))


class UnionSet(FuzzySet):
    """Union of two fuzzy sets."""
    
    def __init__(self, set1: FuzzySet, set2: FuzzySet, universe: nx.Tensor = None):
        """
        Initialize a union fuzzy set.
        
        Args:
            set1: First fuzzy set
            set2: Second fuzzy set
            universe: Universe of discourse
        """
        super().__init__(universe)
        
        self.set1 = set1
        self.set2 = set2
    
    def membership(self, x: Union[float, nx.Tensor]) -> Union[float, nx.Tensor]:
        """
        Calculate the membership degree of x in the fuzzy set.
        
        Args:
            x: Input value(s)
            
        Returns:
            Membership degree(s)
        """
        return nx.maximum(self.set1(x), self.set2(x))


class ComplementSet(FuzzySet):
    """Complement of a fuzzy set."""
    
    def __init__(self, set1: FuzzySet, universe: nx.Tensor = None):
        """
        Initialize a complement fuzzy set.
        
        Args:
            set1: Fuzzy set to complement
            universe: Universe of discourse
        """
        super().__init__(universe)
        
        self.set1 = set1
    
    def membership(self, x: Union[float, nx.Tensor]) -> Union[float, nx.Tensor]:
        """
        Calculate the membership degree of x in the fuzzy set.
        
        Args:
            x: Input value(s)
            
        Returns:
            Membership degree(s)
        """
        return 1.0 - self.set1(x)
