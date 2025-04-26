"""
Fuzzy rules module for Neurenix.

This module provides implementations of fuzzy rules and
rule sets for fuzzy logic systems.
"""

from typing import Dict, List, Optional, Union, Tuple, Callable

import neurenix as nx
from neurenix.fuzzy.sets import FuzzySet
from neurenix.fuzzy.variables import FuzzyVariable, LinguisticVariable


class FuzzyRule:
    """Base class for fuzzy rules."""
    
    def __init__(self, antecedent: Dict[str, Tuple[FuzzyVariable, str]], 
                 consequent: Dict[str, Tuple[FuzzyVariable, str]],
                 operator: str = 'and'):
        """
        Initialize a fuzzy rule.
        
        Args:
            antecedent: Dictionary mapping variable names to (variable, term) tuples
            consequent: Dictionary mapping variable names to (variable, term) tuples
            operator: Logical operator for combining antecedents ('and' or 'or')
        """
        self.antecedent = antecedent
        self.consequent = consequent
        
        if operator not in ['and', 'or']:
            raise ValueError(f"Invalid operator: {operator}. Must be 'and' or 'or'")
        
        self.operator = operator
    
    def evaluate_antecedent(self, inputs: Dict[str, float]) -> float:
        """
        Evaluate the antecedent of the rule.
        
        Args:
            inputs: Dictionary mapping variable names to crisp values
            
        Returns:
            Firing strength of the rule
        """
        memberships = []
        
        for var_name, (var, term) in self.antecedent.items():
            if var_name not in inputs:
                raise ValueError(f"Input value for variable '{var_name}' not provided")
            
            value = inputs[var_name]
            membership = var.get_set(term)(value)
            memberships.append(membership)
        
        if not memberships:
            return 0.0
        
        if self.operator == 'and':
            return min(memberships)
        else:  # 'or'
            return max(memberships)
    
    def evaluate_consequent(self, firing_strength: float) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the consequent of the rule.
        
        Args:
            firing_strength: Firing strength of the rule
            
        Returns:
            Dictionary mapping variable names to dictionaries mapping term names to firing strengths
        """
        result = {}
        
        for var_name, (var, term) in self.consequent.items():
            if var_name not in result:
                result[var_name] = {}
            
            result[var_name][term] = firing_strength
        
        return result


class FuzzyRuleSet:
    """A set of fuzzy rules."""
    
    def __init__(self, rules: List[FuzzyRule] = None):
        """
        Initialize a fuzzy rule set.
        
        Args:
            rules: List of fuzzy rules
        """
        self.rules = rules or []
    
    def add_rule(self, rule: FuzzyRule):
        """
        Add a rule to the rule set.
        
        Args:
            rule: Fuzzy rule
        """
        self.rules.append(rule)
    
    def evaluate(self, inputs: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all rules in the rule set.
        
        Args:
            inputs: Dictionary mapping variable names to crisp values
            
        Returns:
            Dictionary mapping variable names to dictionaries mapping term names to firing strengths
        """
        result = {}
        
        for rule in self.rules:
            firing_strength = rule.evaluate_antecedent(inputs)
            
            if firing_strength > 0:
                consequent_result = rule.evaluate_consequent(firing_strength)
                
                for var_name, terms in consequent_result.items():
                    if var_name not in result:
                        result[var_name] = {}
                    
                    for term, strength in terms.items():
                        if term not in result[var_name]:
                            result[var_name][term] = strength
                        else:
                            result[var_name][term] = max(result[var_name][term], strength)
        
        return result


class FuzzyRuleBuilder:
    """Builder for fuzzy rules."""
    
    def __init__(self):
        """Initialize a fuzzy rule builder."""
        self.antecedent = {}
        self.consequent = {}
        self.operator = 'and'
    
    def if_var(self, var: FuzzyVariable, term: str):
        """
        Add a variable to the antecedent.
        
        Args:
            var: Fuzzy variable
            term: Term name
            
        Returns:
            Self for method chaining
        """
        self.antecedent[var.name] = (var, term)
        return self
    
    def and_var(self, var: FuzzyVariable, term: str):
        """
        Add a variable to the antecedent with AND operator.
        
        Args:
            var: Fuzzy variable
            term: Term name
            
        Returns:
            Self for method chaining
        """
        self.operator = 'and'
        self.antecedent[var.name] = (var, term)
        return self
    
    def or_var(self, var: FuzzyVariable, term: str):
        """
        Add a variable to the antecedent with OR operator.
        
        Args:
            var: Fuzzy variable
            term: Term name
            
        Returns:
            Self for method chaining
        """
        self.operator = 'or'
        self.antecedent[var.name] = (var, term)
        return self
    
    def then_var(self, var: FuzzyVariable, term: str):
        """
        Add a variable to the consequent.
        
        Args:
            var: Fuzzy variable
            term: Term name
            
        Returns:
            Self for method chaining
        """
        self.consequent[var.name] = (var, term)
        return self
    
    def build(self) -> FuzzyRule:
        """
        Build a fuzzy rule.
        
        Returns:
            Fuzzy rule
        """
        return FuzzyRule(self.antecedent, self.consequent, self.operator)
