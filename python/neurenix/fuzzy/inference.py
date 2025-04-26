"""
Fuzzy inference systems module for Neurenix.

This module provides implementations of various fuzzy inference systems,
including Mamdani, Sugeno, and Tsukamoto systems.
"""

from typing import Dict, List, Optional, Union, Tuple, Callable

import neurenix as nx
from neurenix.fuzzy.sets import FuzzySet
from neurenix.fuzzy.variables import FuzzyVariable, LinguisticVariable
from neurenix.fuzzy.rules import FuzzyRule, FuzzyRuleSet
from neurenix.fuzzy.defuzzification import centroid, bisector, mean_of_maximum


class FuzzyInferenceSystem:
    """Base class for fuzzy inference systems."""
    
    def __init__(self, name: str = "FIS"):
        """
        Initialize a fuzzy inference system.
        
        Args:
            name: Name of the fuzzy inference system
        """
        self.name = name
        self.input_variables = {}
        self.output_variables = {}
        self.rule_set = FuzzyRuleSet()
    
    def add_input_variable(self, variable: FuzzyVariable):
        """
        Add an input variable to the system.
        
        Args:
            variable: Fuzzy variable
        """
        self.input_variables[variable.name] = variable
    
    def add_output_variable(self, variable: FuzzyVariable):
        """
        Add an output variable to the system.
        
        Args:
            variable: Fuzzy variable
        """
        self.output_variables[variable.name] = variable
    
    def add_rule(self, rule: FuzzyRule):
        """
        Add a rule to the system.
        
        Args:
            rule: Fuzzy rule
        """
        self.rule_set.add_rule(rule)
    
    def evaluate(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate the fuzzy inference system.
        
        Args:
            inputs: Dictionary mapping input variable names to crisp values
            
        Returns:
            Dictionary mapping output variable names to crisp values
        """
        raise NotImplementedError("Subclasses must implement evaluate method")


class MamdaniSystem(FuzzyInferenceSystem):
    """Mamdani fuzzy inference system."""
    
    def __init__(self, name: str = "Mamdani FIS", 
                 defuzzification_method: str = "centroid"):
        """
        Initialize a Mamdani fuzzy inference system.
        
        Args:
            name: Name of the fuzzy inference system
            defuzzification_method: Method for defuzzification ('centroid', 'bisector', 'mom', 'som', 'lom')
        """
        super().__init__(name)
        
        self.defuzzification_method = defuzzification_method
    
    def evaluate(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate the Mamdani fuzzy inference system.
        
        Args:
            inputs: Dictionary mapping input variable names to crisp values
            
        Returns:
            Dictionary mapping output variable names to crisp values
        """
        fuzzified_inputs = {}
        for var_name, value in inputs.items():
            if var_name not in self.input_variables:
                raise ValueError(f"Input variable '{var_name}' not found")
            
            fuzzified_inputs[var_name] = self.input_variables[var_name].fuzzify(value)
        
        rule_outputs = self.rule_set.evaluate(inputs)
        
        crisp_outputs = {}
        
        for var_name, var in self.output_variables.items():
            if var_name not in rule_outputs:
                crisp_outputs[var_name] = 0.0
                continue
            
            aggregated_set = self._aggregate_output_sets(var, rule_outputs[var_name])
            
            if self.defuzzification_method == 'centroid':
                crisp_outputs[var_name] = centroid(var.universe, aggregated_set)
            elif self.defuzzification_method == 'bisector':
                crisp_outputs[var_name] = bisector(var.universe, aggregated_set)
            elif self.defuzzification_method == 'mom':
                crisp_outputs[var_name] = mean_of_maximum(var.universe, aggregated_set)
            else:
                raise ValueError(f"Unknown defuzzification method: {self.defuzzification_method}")
        
        return crisp_outputs
    
    def _aggregate_output_sets(self, variable: FuzzyVariable, 
                               term_strengths: Dict[str, float]) -> nx.Tensor:
        """
        Aggregate output fuzzy sets.
        
        Args:
            variable: Output variable
            term_strengths: Dictionary mapping term names to firing strengths
            
        Returns:
            Aggregated fuzzy set
        """
        if not variable.universe.numel():
            raise ValueError(f"Variable '{variable.name}' has no universe of discourse")
        
        result = nx.zeros_like(variable.universe)
        
        for term, strength in term_strengths.items():
            if term not in variable.sets:
                continue
            
            fuzzy_set = variable.sets[term]
            
            term_result = nx.minimum(fuzzy_set(variable.universe), 
                                    nx.ones_like(variable.universe) * strength)
            
            result = nx.maximum(result, term_result)
        
        return result


class SugenoSystem(FuzzyInferenceSystem):
    """Sugeno fuzzy inference system."""
    
    def __init__(self, name: str = "Sugeno FIS"):
        """
        Initialize a Sugeno fuzzy inference system.
        
        Args:
            name: Name of the fuzzy inference system
        """
        super().__init__(name)
        
        self.output_functions = {}
    
    def add_output_function(self, var_name: str, term: str, 
                           function: Callable[[Dict[str, float]], float]):
        """
        Add an output function to the system.
        
        Args:
            var_name: Name of the output variable
            term: Name of the term
            function: Function that takes input values and returns a crisp output
        """
        if var_name not in self.output_functions:
            self.output_functions[var_name] = {}
        
        self.output_functions[var_name][term] = function
    
    def evaluate(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate the Sugeno fuzzy inference system.
        
        Args:
            inputs: Dictionary mapping input variable names to crisp values
            
        Returns:
            Dictionary mapping output variable names to crisp values
        """
        rule_outputs = self.rule_set.evaluate(inputs)
        
        crisp_outputs = {}
        
        for var_name in self.output_variables:
            if var_name not in rule_outputs:
                crisp_outputs[var_name] = 0.0
                continue
            
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for term, strength in rule_outputs[var_name].items():
                if var_name not in self.output_functions or term not in self.output_functions[var_name]:
                    continue
                
                output_value = self.output_functions[var_name][term](inputs)
                
                weighted_sum += strength * output_value
                weight_sum += strength
            
            if weight_sum > 0:
                crisp_outputs[var_name] = weighted_sum / weight_sum
            else:
                crisp_outputs[var_name] = 0.0
        
        return crisp_outputs


class TsukamotoSystem(FuzzyInferenceSystem):
    """Tsukamoto fuzzy inference system."""
    
    def __init__(self, name: str = "Tsukamoto FIS"):
        """
        Initialize a Tsukamoto fuzzy inference system.
        
        Args:
            name: Name of the fuzzy inference system
        """
        super().__init__(name)
    
    def evaluate(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate the Tsukamoto fuzzy inference system.
        
        Args:
            inputs: Dictionary mapping input variable names to crisp values
            
        Returns:
            Dictionary mapping output variable names to crisp values
        """
        rule_outputs = self.rule_set.evaluate(inputs)
        
        crisp_outputs = {}
        
        for var_name, var in self.output_variables.items():
            if var_name not in rule_outputs:
                crisp_outputs[var_name] = 0.0
                continue
            
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for term, strength in rule_outputs[var_name].items():
                if term not in var.sets:
                    continue
                
                fuzzy_set = var.sets[term]
                output_value = self._invert_membership_function(fuzzy_set, strength, var.universe)
                
                weighted_sum += strength * output_value
                weight_sum += strength
            
            if weight_sum > 0:
                crisp_outputs[var_name] = weighted_sum / weight_sum
            else:
                crisp_outputs[var_name] = 0.0
        
        return crisp_outputs
    
    def _invert_membership_function(self, fuzzy_set: FuzzySet, 
                                   strength: float, universe: nx.Tensor) -> float:
        """
        Invert a membership function to find the value with the given membership degree.
        
        Args:
            fuzzy_set: Fuzzy set
            strength: Membership degree
            universe: Universe of discourse
            
        Returns:
            Value with the given membership degree
        """
        memberships = fuzzy_set(universe)
        
        diff = nx.abs(memberships - strength)
        idx = nx.argmin(diff)
        
        return universe[idx].item()
