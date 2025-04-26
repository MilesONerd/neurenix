"""
Symbolic reasoning components for hybrid neuro-symbolic models.

This module provides implementations of symbolic reasoning systems that can be
integrated with neural networks to create hybrid neuro-symbolic models.
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Union, Any, Callable
from collections import defaultdict

from neurenix.tensor import Tensor
from neurenix.nn.module import Module


class SymbolicKnowledgeBase:
    """A knowledge base for storing symbolic facts and rules."""
    
    def __init__(self):
        """Initialize an empty knowledge base."""
        self.facts = set()
        self.rules = []
        self.predicates = set()
        
    def add_fact(self, fact: str) -> None:
        """Add a fact to the knowledge base.
        
        Args:
            fact: A string representing a fact (e.g., "parent(john, mary)")
        """
        self.facts.add(fact)
        predicate = fact.split('(')[0]
        self.predicates.add(predicate)
        
    def add_rule(self, head: str, body: List[str]) -> None:
        """Add a rule to the knowledge base.
        
        Args:
            head: The head of the rule (e.g., "grandparent(X, Z)")
            body: The body of the rule as a list of literals (e.g., ["parent(X, Y)", "parent(Y, Z)"])
        """
        self.rules.append((head, body))
        head_predicate = head.split('(')[0]
        self.predicates.add(head_predicate)
        
        for literal in body:
            predicate = literal.split('(')[0]
            self.predicates.add(predicate)
            
    def get_facts(self) -> Set[str]:
        """Get all facts in the knowledge base.
        
        Returns:
            A set of all facts in the knowledge base
        """
        return self.facts
    
    def get_rules(self) -> List[Tuple[str, List[str]]]:
        """Get all rules in the knowledge base.
        
        Returns:
            A list of (head, body) tuples representing rules
        """
        return self.rules
    
    def get_predicates(self) -> Set[str]:
        """Get all predicates in the knowledge base.
        
        Returns:
            A set of all predicates in the knowledge base
        """
        return self.predicates


class RuleSet:
    """A set of logical rules for symbolic reasoning."""
    
    def __init__(self):
        """Initialize an empty rule set."""
        self.rules = []
        self.rule_weights = []
        
    def add_rule(self, rule: Tuple[str, List[str]], weight: float = 1.0) -> None:
        """Add a rule to the rule set.
        
        Args:
            rule: A (head, body) tuple representing a rule
            weight: The weight of the rule (for weighted reasoning)
        """
        self.rules.append(rule)
        self.rule_weights.append(weight)
        
    def get_rules(self) -> List[Tuple[str, List[str]]]:
        """Get all rules in the rule set.
        
        Returns:
            A list of (head, body) tuples representing rules
        """
        return self.rules
    
    def get_rule_weights(self) -> List[float]:
        """Get the weights of all rules in the rule set.
        
        Returns:
            A list of weights for the rules
        """
        return self.rule_weights
    
    def __len__(self) -> int:
        """Get the number of rules in the rule set.
        
        Returns:
            The number of rules
        """
        return len(self.rules)


class LogicProgram:
    """A logic program for symbolic reasoning."""
    
    def __init__(self, kb: Optional[SymbolicKnowledgeBase] = None):
        """Initialize a logic program.
        
        Args:
            kb: An optional knowledge base to initialize the program with
        """
        self.kb = kb or SymbolicKnowledgeBase()
        self.inferred_facts = set()
        
    def add_knowledge_base(self, kb: SymbolicKnowledgeBase) -> None:
        """Add a knowledge base to the logic program.
        
        Args:
            kb: The knowledge base to add
        """
        self.kb = kb
        
    def query(self, query: str) -> bool:
        """Query the logic program.
        
        Args:
            query: The query to evaluate
            
        Returns:
            True if the query is entailed by the knowledge base, False otherwise
        """
        return self._backward_chain(query, set())
    
    def _backward_chain(self, query: str, visited: Set[str]) -> bool:
        """Perform backward chaining to evaluate a query.
        
        Args:
            query: The query to evaluate
            visited: A set of already visited queries (to avoid infinite recursion)
            
        Returns:
            True if the query is entailed by the knowledge base, False otherwise
        """
        if query in visited:
            return False
        
        visited.add(query)
        
        if query in self.kb.facts or query in self.inferred_facts:
            return True
        
        query_predicate = query.split('(')[0]
        query_args = query.split('(')[1].split(')')[0].split(',')
        
        for head, body in self.kb.rules:
            head_predicate = head.split('(')[0]
            
            if head_predicate == query_predicate:
                head_args = head.split('(')[1].split(')')[0].split(',')
                
                substitution = self._unify(query_args, head_args)
                
                if substitution is not None:
                    all_satisfied = True
                    
                    for literal in body:
                        literal_predicate = literal.split('(')[0]
                        literal_args = literal.split('(')[1].split(')')[0].split(',')
                        
                        substituted_args = [substitution.get(arg.strip(), arg.strip()) for arg in literal_args]
                        substituted_literal = f"{literal_predicate}({','.join(substituted_args)})"
                        
                        if not self._backward_chain(substituted_literal, visited.copy()):
                            all_satisfied = False
                            break
                    
                    if all_satisfied:
                        self.inferred_facts.add(query)
                        return True
        
        return False
    
    def _unify(self, query_args: List[str], head_args: List[str]) -> Optional[Dict[str, str]]:
        """Unify the arguments of a query with the arguments of a rule head.
        
        Args:
            query_args: The arguments of the query
            head_args: The arguments of the rule head
            
        Returns:
            A substitution (as a dictionary) if unification is possible, None otherwise
        """
        if len(query_args) != len(head_args):
            return None
        
        substitution = {}
        
        for i in range(len(query_args)):
            query_arg = query_args[i].strip()
            head_arg = head_args[i].strip()
            
            if head_arg.isupper():  # Variable in the rule head
                if head_arg in substitution and substitution[head_arg] != query_arg:
                    return None
                substitution[head_arg] = query_arg
            elif query_arg != head_arg:  # Constants must match
                return None
        
        return substitution


class SymbolicReasoner:
    """A symbolic reasoner for hybrid neuro-symbolic models."""
    
    def __init__(self, logic_program: Optional[LogicProgram] = None):
        """Initialize a symbolic reasoner.
        
        Args:
            logic_program: An optional logic program to initialize the reasoner with
        """
        self.logic_program = logic_program or LogicProgram()
        
    def set_logic_program(self, logic_program: LogicProgram) -> None:
        """Set the logic program for the reasoner.
        
        Args:
            logic_program: The logic program to use
        """
        self.logic_program = logic_program
        
    def reason(self, query: str) -> bool:
        """Perform symbolic reasoning to evaluate a query.
        
        Args:
            query: The query to evaluate
            
        Returns:
            True if the query is entailed by the knowledge base, False otherwise
        """
        return self.logic_program.query(query)
    
    def batch_reason(self, queries: List[str]) -> List[bool]:
        """Perform symbolic reasoning on a batch of queries.
        
        Args:
            queries: The queries to evaluate
            
        Returns:
            A list of boolean values indicating whether each query is entailed
        """
        return [self.reason(query) for query in queries]
    
    def to_tensor(self, queries: List[str]) -> Tensor:
        """Convert the results of reasoning on a batch of queries to a tensor.
        
        Args:
            queries: The queries to evaluate
            
        Returns:
            A tensor of boolean values indicating whether each query is entailed
        """
        results = self.batch_reason(queries)
        return Tensor([float(result) for result in results])
