"""
Multi-Agent Systems (MAS) for Neurenix.

This module provides implementations of multi-agent systems for distributed
artificial intelligence, agent-based modeling, and collaborative learning.
"""

from .agent import (
    Agent,
    ReactiveAgent,
    DeliberativeAgent,
    HybridAgent,
    AgentState
)

from .environment import (
    Environment,
    GridEnvironment,
    ContinuousEnvironment,
    NetworkEnvironment,
    StateSpace,
    ActionSpace
)

from .communication import (
    Message,
    Channel,
    Protocol,
    Mailbox,
    CommunicationNetwork
)

from .coordination import (
    Coordinator,
    Auction,
    ContractNet,
    VotingMechanism,
    CoalitionFormation
)

from .learning import (
    MultiAgentLearning,
    IndependentLearners,
    JointActionLearners,
    TeamLearning,
    OpponentModeling
)

__all__ = [
    'Agent',
    'ReactiveAgent',
    'DeliberativeAgent',
    'HybridAgent',
    'AgentState',
    
    'Environment',
    'GridEnvironment',
    'ContinuousEnvironment',
    'NetworkEnvironment',
    'StateSpace',
    'ActionSpace',
    
    'Message',
    'Channel',
    'Protocol',
    'Mailbox',
    'CommunicationNetwork',
    
    'Coordinator',
    'Auction',
    'ContractNet',
    'VotingMechanism',
    'CoalitionFormation',
    
    'MultiAgentLearning',
    'IndependentLearners',
    'JointActionLearners',
    'TeamLearning',
    'OpponentModeling'
]
