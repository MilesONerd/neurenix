"""
Multi-Agent Systems (MAS) module for Neurenix.

This module provides implementations of multi-agent systems for distributed
AI applications where multiple intelligent agents interact, communicate,
and coordinate to solve complex problems.
"""

from .agent import Agent, ReactiveAgent, DeliberativeAgent, HybridAgent
from .communication import Message, Channel, Protocol
from .coordination import TaskAllocation, Auction, ContractNet, Voting
from .learning import IndependentLearner, JointActionLearner, TeamLearner

__all__ = [
    'Agent', 'ReactiveAgent', 'DeliberativeAgent', 'HybridAgent',
    'Message', 'Channel', 'Protocol',
    'TaskAllocation', 'Auction', 'ContractNet', 'Voting',
    'IndependentLearner', 'JointActionLearner', 'TeamLearner',
]
