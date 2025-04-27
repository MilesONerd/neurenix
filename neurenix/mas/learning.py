"""
Learning algorithms for Multi-Agent Systems in Neurenix.

This module provides implementations of learning algorithms for multi-agent systems,
including independent learners, joint action learners, and team learning.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from collections import defaultdict

from neurenix.tensor import Tensor
from neurenix.nn.module import Module


class MultiAgentLearning(Module):
    """Base class for multi-agent learning algorithms."""
    
    def __init__(self, agent_ids: List[str], state_dim: int, action_dim: int,
                learning_rate: float = 0.01):
        """Initialize a multi-agent learning algorithm."""
        super().__init__()
        self.agent_ids = agent_ids
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.policies = {}
        
    def forward(self, states: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass through the learning algorithm."""
        raise NotImplementedError("MultiAgentLearning must implement forward method")
    
    def update(self, states: Dict[str, Tensor], actions: Dict[str, Tensor],
             rewards: Dict[str, float], next_states: Dict[str, Tensor],
             dones: Dict[str, bool]) -> Dict[str, float]:
        """Update the learning algorithm based on experience."""
        raise NotImplementedError("MultiAgentLearning must implement update method")
    
    def reset(self) -> None:
        """Reset the learning algorithm."""
        self.policies = {}


class IndependentLearners(MultiAgentLearning):
    """Independent learners for multi-agent systems."""
    
    def __init__(self, agent_ids: List[str], state_dim: int, action_dim: int,
                learning_rate: float = 0.01, discount_factor: float = 0.99):
        """Initialize independent learners."""
        super().__init__(agent_ids, state_dim, action_dim, learning_rate)
        self.discount_factor = discount_factor
        self.q_tables = {agent_id: {} for agent_id in agent_ids}
    
    def _get_state_key(self, state: Tensor) -> str:
        """Convert a state tensor to a hashable key."""
        return str(state.numpy().tolist())
    
    def forward(self, states: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass through the independent learners."""
        action_probs = {}
        
        for agent_id, state in states.items():
            state_key = self._get_state_key(state)
            
            if state_key not in self.q_tables[agent_id]:
                self.q_tables[agent_id][state_key] = np.zeros(self.action_dim)
                
            q_values = self.q_tables[agent_id][state_key]
            action_probs[agent_id] = Tensor(q_values).softmax()
            
        return action_probs
    
    def update(self, states: Dict[str, Tensor], actions: Dict[str, Tensor],
             rewards: Dict[str, float], next_states: Dict[str, Tensor],
             dones: Dict[str, bool]) -> Dict[str, float]:
        """Update the independent learners based on experience."""
        losses = {}
        
        for agent_id in self.agent_ids:
            if agent_id not in states:
                continue
                
            state = states[agent_id]
            action = int(actions[agent_id].item())
            reward = rewards[agent_id]
            next_state = next_states[agent_id]
            done = dones[agent_id]
            
            state_key = self._get_state_key(state)
            if state_key not in self.q_tables[agent_id]:
                self.q_tables[agent_id][state_key] = np.zeros(self.action_dim)
            current_q = self.q_tables[agent_id][state_key][action]
            
            next_state_key = self._get_state_key(next_state)
            if next_state_key not in self.q_tables[agent_id]:
                self.q_tables[agent_id][next_state_key] = np.zeros(self.action_dim)
            next_q_max = np.max(self.q_tables[agent_id][next_state_key])
            
            target_q = reward if done else reward + self.discount_factor * next_q_max
                
            self.q_tables[agent_id][state_key][action] += self.learning_rate * (target_q - current_q)
            
            losses[agent_id] = (target_q - current_q) ** 2
            
        return losses


class JointActionLearners(MultiAgentLearning):
    """Joint action learners for multi-agent systems."""
    
    def __init__(self, agent_ids: List[str], state_dim: int, action_dim: int,
                learning_rate: float = 0.01, discount_factor: float = 0.99):
        """Initialize joint action learners."""
        super().__init__(agent_ids, state_dim, action_dim, learning_rate)
        self.discount_factor = discount_factor
        self.joint_q_tables = {agent_id: {} for agent_id in agent_ids}
        self.agent_models = {agent_id: {other_id: {} for other_id in agent_ids if other_id != agent_id} 
                           for agent_id in agent_ids}
    
    def _get_state_key(self, state: Tensor) -> str:
        """Convert a state tensor to a hashable key."""
        return str(state.numpy().tolist())
    
    def _get_joint_action_key(self, actions: Dict[str, int]) -> str:
        """Convert a joint action to a hashable key."""
        return str(sorted([(agent_id, action) for agent_id, action in actions.items()]))
    
    def forward(self, states: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass through the joint action learners."""
        action_probs = {}
        
        for agent_id, state in states.items():
            expected_q_values = np.zeros(self.action_dim)
            
            for action in range(self.action_dim):
                state_key = self._get_state_key(state)
                
                joint_action = {}
                for other_id in self.agent_ids:
                    if other_id != agent_id:
                        if other_id in states:
                            other_state_key = self._get_state_key(states[other_id])
                            if other_state_key in self.agent_models[agent_id][other_id]:
                                action_probs = self.agent_models[agent_id][other_id][other_state_key]
                                other_action = np.argmax(action_probs) if np.sum(action_probs) > 0 else 0
                            else:
                                other_action = 0
                            joint_action[other_id] = other_action
                
                joint_action[agent_id] = action
                
                joint_action_key = self._get_joint_action_key(joint_action)
                
                if state_key in self.joint_q_tables[agent_id] and joint_action_key in self.joint_q_tables[agent_id][state_key]:
                    expected_q_values[action] = self.joint_q_tables[agent_id][state_key][joint_action_key]
                else:
                    expected_q_values[action] = np.random.uniform(0, 0.1)
                
            action_probs[agent_id] = Tensor(expected_q_values).softmax()
            
        return action_probs
    
    def update(self, states: Dict[str, Tensor], actions: Dict[str, Tensor],
             rewards: Dict[str, float], next_states: Dict[str, Tensor],
             dones: Dict[str, bool]) -> Dict[str, float]:
        """Update the joint action learners based on experience."""
        return {agent_id: 0.0 for agent_id in self.agent_ids if agent_id in states}


class TeamLearning(MultiAgentLearning):
    """Team learning for multi-agent systems."""
    
    def __init__(self, agent_ids: List[str], state_dim: int, action_dim: int,
                learning_rate: float = 0.01, discount_factor: float = 0.99):
        """Initialize team learning."""
        super().__init__(agent_ids, state_dim, action_dim, learning_rate)
        self.discount_factor = discount_factor
        self.team_q_table = {}
    
    def _get_state_key(self, states: Dict[str, Tensor]) -> str:
        """Convert a joint state to a hashable key."""
        state_list = []
        for agent_id in sorted(states.keys()):
            state_list.append((agent_id, str(states[agent_id].numpy().tolist())))
        return str(state_list)
    
    def forward(self, states: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass through the team learning algorithm."""
        state_key = self._get_state_key(states)
        
        if state_key not in self.team_q_table:
            self.team_q_table[state_key] = np.zeros((self.action_dim,) * len(states))
        
        action_probs = {}
        for agent_id in states:
            probs = np.ones(self.action_dim) / self.action_dim
            action_probs[agent_id] = Tensor(probs)
            
        return action_probs
    
    def update(self, states: Dict[str, Tensor], actions: Dict[str, Tensor],
             rewards: Dict[str, float], next_states: Dict[str, Tensor],
             dones: Dict[str, bool]) -> Dict[str, float]:
        """Update the team learning algorithm based on experience."""
        team_reward = sum(rewards.values()) / len(rewards) if rewards else 0.0
        
        return {agent_id: 0.0 for agent_id in self.agent_ids if agent_id in states}


class OpponentModeling(MultiAgentLearning):
    """Learning with opponent modeling for multi-agent systems."""
    
    def __init__(self, agent_ids: List[str], state_dim: int, action_dim: int,
                learning_rate: float = 0.01, discount_factor: float = 0.99):
        """Initialize opponent modeling."""
        super().__init__(agent_ids, state_dim, action_dim, learning_rate)
        self.discount_factor = discount_factor
        self.q_tables = {agent_id: {} for agent_id in agent_ids}
        self.opponent_models = {agent_id: {other_id: {} for other_id in agent_ids if other_id != agent_id} 
                              for agent_id in agent_ids}
    
    def _get_state_key(self, state: Tensor) -> str:
        """Convert a state tensor to a hashable key."""
        return str(state.numpy().tolist())
    
    def forward(self, states: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass through the opponent modeling algorithm."""
        action_probs = {}
        
        for agent_id, state in states.items():
            state_key = self._get_state_key(state)
            
            if state_key not in self.q_tables[agent_id]:
                self.q_tables[agent_id][state_key] = np.zeros(self.action_dim)
                
            q_values = self.q_tables[agent_id][state_key]
            action_probs[agent_id] = Tensor(q_values).softmax()
            
        return action_probs
    
    def update(self, states: Dict[str, Tensor], actions: Dict[str, Tensor],
             rewards: Dict[str, float], next_states: Dict[str, Tensor],
             dones: Dict[str, bool]) -> Dict[str, float]:
        """Update the opponent modeling algorithm based on experience."""
        for agent_id in self.agent_ids:
            if agent_id not in states:
                continue
                
            state = states[agent_id]
            state_key = self._get_state_key(state)
            
            for other_id, action in actions.items():
                if other_id != agent_id:
                    action_idx = int(action.item())
                    
                    if state_key not in self.opponent_models[agent_id][other_id]:
                        self.opponent_models[agent_id][other_id][state_key] = np.zeros(self.action_dim)
                        
                    self.opponent_models[agent_id][other_id][state_key][action_idx] += 1
        
        losses = {}
        for agent_id in self.agent_ids:
            if agent_id not in states:
                continue
                
            prediction_errors = []
            
            for other_id in self.agent_ids:
                if other_id != agent_id and other_id in actions and other_id in states:
                    other_state_key = self._get_state_key(states[other_id])
                    other_action = int(actions[other_id].item())
                    
                    if other_state_key in self.opponent_models[agent_id][other_id]:
                        action_counts = self.opponent_models[agent_id][other_id][other_state_key]
                        total_count = np.sum(action_counts)
                        
                        if total_count > 0:
                            action_probs = action_counts / total_count
                            prediction_error = -np.log(action_probs[other_action] + 1e-10)  # Add small epsilon to avoid log(0)
                            prediction_errors.append(prediction_error)
            
            if prediction_errors:
                losses[agent_id] = np.mean(prediction_errors)
            else:
                losses[agent_id] = 0.0
            
        return losses
