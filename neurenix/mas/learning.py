"""
Learning module for Multi-Agent Systems in Neurenix.

This module provides implementations of learning algorithms for
multi-agent systems, including independent learning, joint action learning,
and team learning.
"""

from typing import Dict, List, Any, Optional, Callable, Tuple
import numpy as np

from ..core import PhynexusExtension
from ..optim import Optimizer

class IndependentLearner:
    """
    Independent learner that learns without considering other agents.
    """
    
    def __init__(self, agent_id: str, state_size: int, action_size: int,
                 learning_rate: float = 0.01, discount_factor: float = 0.99):
        """
        Initialize an independent learner.
        
        Args:
            agent_id: Unique identifier for the agent
            state_size: Dimensionality of the state space
            action_size: Dimensionality of the action space
            learning_rate: Learning rate for updates
            discount_factor: Discount factor for future rewards
        """
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.q_table = {}
        
    def get_state_key(self, state: np.ndarray) -> str:
        """
        Convert a state array to a hashable key.
        
        Args:
            state: State array
            
        Returns:
            Hashable key for the state
        """
        return str(state.tolist())
        
    def get_q_value(self, state: np.ndarray, action: int) -> float:
        """
        Get the Q-value for a state-action pair.
        
        Args:
            state: State array
            action: Action index
            
        Returns:
            Q-value for the state-action pair
        """
        state_key = self.get_state_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
            
        return self.q_table[state_key][action]
        
    def update_q_value(self, state: np.ndarray, action: int, 
                       reward: float, next_state: np.ndarray) -> None:
        """
        Update the Q-value for a state-action pair.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
            
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_size)
            
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key])
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
        
    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration probability
            
        Returns:
            Selected action index
        """
        state_key = self.get_state_key(state)
        
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_size)
            
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
            
        return np.argmax(self.q_table[state_key])


class JointActionLearner:
    """
    Joint action learner that considers the actions of other agents.
    """
    
    def __init__(self, agent_id: str, state_size: int, action_size: int,
                 other_agents: List[str], learning_rate: float = 0.01, 
                 discount_factor: float = 0.99):
        """
        Initialize a joint action learner.
        
        Args:
            agent_id: Unique identifier for the agent
            state_size: Dimensionality of the state space
            action_size: Dimensionality of the action space
            other_agents: List of other agent IDs
            learning_rate: Learning rate for updates
            discount_factor: Discount factor for future rewards
        """
        self.agent_id = agent_id
        self.state_size = state_size
        self.action_size = action_size
        self.other_agents = other_agents
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.joint_q_table = {}
        
        self.other_agent_models = {
            agent_id: np.ones((state_size, action_size)) / action_size
            for agent_id in other_agents
        }
        
    def get_joint_state_key(self, state: np.ndarray, 
                           other_actions: Dict[str, int]) -> str:
        """
        Convert a state and other agents' actions to a hashable key.
        
        Args:
            state: State array
            other_actions: Dictionary mapping agent IDs to actions
            
        Returns:
            Hashable key for the joint state
        """
        action_str = "_".join([
            f"{agent_id}:{action}" 
            for agent_id, action in sorted(other_actions.items())
        ])
        
        return f"{str(state.tolist())}|{action_str}"
        
    def get_q_value(self, state: np.ndarray, action: int, 
                   other_actions: Dict[str, int]) -> float:
        """
        Get the Q-value for a joint state-action pair.
        
        Args:
            state: State array
            action: Action index
            other_actions: Dictionary mapping agent IDs to actions
            
        Returns:
            Q-value for the joint state-action pair
        """
        joint_state_key = self.get_joint_state_key(state, other_actions)
        
        if joint_state_key not in self.joint_q_table:
            self.joint_q_table[joint_state_key] = np.zeros(self.action_size)
            
        return self.joint_q_table[joint_state_key][action]
        
    def update_q_value(self, state: np.ndarray, action: int, 
                       other_actions: Dict[str, int], reward: float, 
                       next_state: np.ndarray) -> None:
        """
        Update the Q-value for a joint state-action pair.
        
        Args:
            state: Current state
            action: Action taken
            other_actions: Dictionary mapping agent IDs to actions
            reward: Reward received
            next_state: Next state
        """
        joint_state_key = self.get_joint_state_key(state, other_actions)
        
        if joint_state_key not in self.joint_q_table:
            self.joint_q_table[joint_state_key] = np.zeros(self.action_size)
            
        for agent_id, action in other_actions.items():
            state_idx = hash(str(state.tolist())) % self.state_size
            self.other_agent_models[agent_id][state_idx, :] *= 0.9  # Decay
            self.other_agent_models[agent_id][state_idx, action] += 0.1  # Increment
            
            self.other_agent_models[agent_id][state_idx, :] /= \
                np.sum(self.other_agent_models[agent_id][state_idx, :])
            
        expected_future_value = 0.0
        
        
        current_q = self.joint_q_table[joint_state_key][action]
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * expected_future_value - current_q
        )
        
        self.joint_q_table[joint_state_key][action] = new_q
        
    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration probability
            
        Returns:
            Selected action index
        """
        predicted_actions = {}
        for agent_id in self.other_agents:
            state_idx = hash(str(state.tolist())) % self.state_size
            probs = self.other_agent_models[agent_id][state_idx, :]
            predicted_actions[agent_id] = np.argmax(probs)
            
        joint_state_key = self.get_joint_state_key(state, predicted_actions)
        
        if joint_state_key not in self.joint_q_table:
            self.joint_q_table[joint_state_key] = np.zeros(self.action_size)
            
        if np.random.random() < epsilon:
            return np.random.randint(self.action_size)
            
        return np.argmax(self.joint_q_table[joint_state_key])


class TeamLearner:
    """
    Team learner that optimizes for team performance.
    """
    
    def __init__(self, team_id: str, agent_ids: List[str], 
                 state_size: int, action_sizes: Dict[str, int],
                 learning_rate: float = 0.01, discount_factor: float = 0.99):
        """
        Initialize a team learner.
        
        Args:
            team_id: Unique identifier for the team
            agent_ids: List of agent IDs in the team
            state_size: Dimensionality of the state space
            action_sizes: Dictionary mapping agent IDs to action space sizes
            learning_rate: Learning rate for updates
            discount_factor: Discount factor for future rewards
        """
        self.team_id = team_id
        self.agent_ids = agent_ids
        self.state_size = state_size
        self.action_sizes = action_sizes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.team_q_table = {}
        
    def get_joint_action_key(self, actions: Dict[str, int]) -> str:
        """
        Convert a joint action to a hashable key.
        
        Args:
            actions: Dictionary mapping agent IDs to actions
            
        Returns:
            Hashable key for the joint action
        """
        return "_".join([
            f"{agent_id}:{action}" 
            for agent_id, action in sorted(actions.items())
        ])
        
    def get_state_key(self, state: np.ndarray) -> str:
        """
        Convert a state array to a hashable key.
        
        Args:
            state: State array
            
        Returns:
            Hashable key for the state
        """
        return str(state.tolist())
        
    def get_q_value(self, state: np.ndarray, 
                   joint_action: Dict[str, int]) -> float:
        """
        Get the Q-value for a state and joint action.
        
        Args:
            state: State array
            joint_action: Dictionary mapping agent IDs to actions
            
        Returns:
            Q-value for the state and joint action
        """
        state_key = self.get_state_key(state)
        joint_action_key = self.get_joint_action_key(joint_action)
        
        if state_key not in self.team_q_table:
            self.team_q_table[state_key] = {}
            
        if joint_action_key not in self.team_q_table[state_key]:
            self.team_q_table[state_key][joint_action_key] = 0.0
            
        return self.team_q_table[state_key][joint_action_key]
        
    def update_q_value(self, state: np.ndarray, joint_action: Dict[str, int],
                       reward: float, next_state: np.ndarray) -> None:
        """
        Update the Q-value for a state and joint action.
        
        Args:
            state: Current state
            joint_action: Dictionary mapping agent IDs to actions
            reward: Team reward received
            next_state: Next state
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        joint_action_key = self.get_joint_action_key(joint_action)
        
        if state_key not in self.team_q_table:
            self.team_q_table[state_key] = {}
            
        if joint_action_key not in self.team_q_table[state_key]:
            self.team_q_table[state_key][joint_action_key] = 0.0
            
        if next_state_key not in self.team_q_table:
            self.team_q_table[next_state_key] = {}
            
        max_next_q = 0.0
        if self.team_q_table[next_state_key]:
            max_next_q = max(self.team_q_table[next_state_key].values())
            
        current_q = self.team_q_table[state_key][joint_action_key]
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.team_q_table[state_key][joint_action_key] = new_q
        
    def select_joint_action(self, state: np.ndarray, 
                           epsilon: float = 0.1) -> Dict[str, int]:
        """
        Select a joint action for the team using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration probability
            
        Returns:
            Dictionary mapping agent IDs to selected actions
        """
        state_key = self.get_state_key(state)
        
        if state_key not in self.team_q_table:
            self.team_q_table[state_key] = {}
            
        if np.random.random() < epsilon:
            return {
                agent_id: np.random.randint(action_size)
                for agent_id, action_size in self.action_sizes.items()
            }
            
        if not self.team_q_table[state_key]:
            return {
                agent_id: np.random.randint(action_size)
                for agent_id, action_size in self.action_sizes.items()
            }
            
        best_joint_action_key = max(
            self.team_q_table[state_key].items(),
            key=lambda x: x[1]
        )[0]
        
        joint_action = {}
        for agent_action in best_joint_action_key.split("_"):
            agent_id, action = agent_action.split(":")
            joint_action[agent_id] = int(action)
            
        return joint_action
