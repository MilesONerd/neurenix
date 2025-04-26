"""
Policy module for reinforcement learning in Neurenix.

This module provides the base Policy class and implementations of various
reinforcement learning policies.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from neurenix.tensor import Tensor
from neurenix.nn.module import Module


class Policy:
    """
    Base class for reinforcement learning policies.
    
    This class provides the basic functionality for reinforcement learning policies,
    which map states to actions.
    """
    
    def __init__(self, name: str = "Policy"):
        """
        Initialize policy.
        
        Args:
            name: Policy name
        """
        self.name = name
    
    def __call__(self, state: Union[np.ndarray, Tensor]) -> Any:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        return self.select_action(state)
    
    def select_action(self, state: Union[np.ndarray, Tensor]) -> Any:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        raise NotImplementedError("Subclasses must implement select_action")
    
    def step(self):
        """Update policy parameters (e.g., exploration rate)."""
        pass
    
    def reset(self):
        """Reset policy parameters."""
        pass
    
    def save(self, path: str):
        """
        Save policy to disk.
        
        Args:
            path: Path to save policy to
        """
        pass
    
    def load(self, path: str):
        """
        Load policy from disk.
        
        Args:
            path: Path to load policy from
        """
        pass


class RandomPolicy(Policy):
    """
    Random policy for reinforcement learning.
    
    This policy selects actions uniformly at random from the action space.
    """
    
    def __init__(self, action_space: Dict[str, Any], name: str = "RandomPolicy"):
        """
        Initialize random policy.
        
        Args:
            action_space: Action space specification
            name: Policy name
        """
        super().__init__(name=name)
        self.action_space = action_space
    
    def select_action(self, state: Union[np.ndarray, Tensor]) -> Any:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        if self.action_space["type"] == "discrete":
            return np.random.randint(self.action_space["n"])
        elif self.action_space["type"] == "box":
            return np.random.uniform(
                self.action_space["low"],
                self.action_space["high"],
                self.action_space["shape"],
            )
        else:
            raise ValueError(f"Unsupported action space type: {self.action_space['type']}")


class GreedyPolicy(Policy):
    """
    Greedy policy for reinforcement learning.
    
    This policy selects the action with the highest value according to a value function.
    """
    
    def __init__(
        self,
        value_function: Module,
        action_space: Dict[str, Any],
        name: str = "GreedyPolicy",
    ):
        """
        Initialize greedy policy.
        
        Args:
            value_function: Value function for action evaluation
            action_space: Action space specification
            name: Policy name
        """
        super().__init__(name=name)
        self.value_function = value_function
        self.action_space = action_space
    
    def select_action(self, state: Union[np.ndarray, Tensor]) -> Any:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Convert to tensor if necessary
        if isinstance(state, np.ndarray):
            state = Tensor(state)
        
        # Get action values
        with Tensor.no_grad():
            action_values = self.value_function(state)
        
        # Select action with highest value
        if self.action_space["type"] == "discrete":
            return action_values.argmax().item()
        else:
            raise ValueError(f"Unsupported action space type: {self.action_space['type']}")


class EpsilonGreedyPolicy(Policy):
    """
    Epsilon-greedy policy for reinforcement learning.
    
    This policy selects the action with the highest value with probability 1-epsilon,
    and a random action with probability epsilon.
    """
    
    def __init__(
        self,
        value_function: Module,
        action_space: Dict[str, Any],
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        name: str = "EpsilonGreedyPolicy",
    ):
        """
        Initialize epsilon-greedy policy.
        
        Args:
            value_function: Value function for action evaluation
            action_space: Action space specification
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration rate decay
            name: Policy name
        """
        super().__init__(name=name)
        self.value_function = value_function
        self.action_space = action_space
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        
        # Create greedy and random policies
        self.greedy_policy = GreedyPolicy(value_function, action_space)
        self.random_policy = RandomPolicy(action_space)
    
    def select_action(self, state: Union[np.ndarray, Tensor]) -> Any:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Explore with probability epsilon
        if np.random.random() < self.epsilon:
            return self.random_policy(state)
        
        # Exploit with probability 1-epsilon
        return self.greedy_policy(state)
    
    def step(self):
        """Update epsilon."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def reset(self):
        """Reset epsilon."""
        self.epsilon = self.epsilon_start


class SoftmaxPolicy(Policy):
    """
    Softmax policy for reinforcement learning.
    
    This policy selects actions according to a softmax distribution over action values.
    """
    
    def __init__(
        self,
        value_function: Module,
        action_space: Dict[str, Any],
        temperature: float = 1.0,
        name: str = "SoftmaxPolicy",
    ):
        """
        Initialize softmax policy.
        
        Args:
            value_function: Value function for action evaluation
            action_space: Action space specification
            temperature: Temperature parameter for softmax
            name: Policy name
        """
        super().__init__(name=name)
        self.value_function = value_function
        self.action_space = action_space
        self.temperature = temperature
    
    def select_action(self, state: Union[np.ndarray, Tensor]) -> Any:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Convert to tensor if necessary
        if isinstance(state, np.ndarray):
            state = Tensor(state)
        
        # Get action values
        with Tensor.no_grad():
            action_values = self.value_function(state)
        
        # Apply softmax
        if self.action_space["type"] == "discrete":
            # Compute probabilities
            probs = Tensor.softmax(action_values / self.temperature, dim=-1)
            
            # Convert to numpy
            probs = probs.numpy()
            
            # Sample action
            return np.random.choice(self.action_space["n"], p=probs)
        else:
            raise ValueError(f"Unsupported action space type: {self.action_space['type']}")


class GaussianPolicy(Policy):
    """
    Gaussian policy for reinforcement learning.
    
    This policy selects actions according to a Gaussian distribution with
    mean given by a policy network and fixed standard deviation.
    """
    
    def __init__(
        self,
        policy_network: Module,
        action_space: Dict[str, Any],
        std: float = 0.1,
        name: str = "GaussianPolicy",
    ):
        """
        Initialize Gaussian policy.
        
        Args:
            policy_network: Policy network for action mean
            action_space: Action space specification
            std: Standard deviation of Gaussian
            name: Policy name
        """
        super().__init__(name=name)
        self.policy_network = policy_network
        self.action_space = action_space
        self.std = std
    
    def select_action(self, state: Union[np.ndarray, Tensor]) -> Any:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Convert to tensor if necessary
        if isinstance(state, np.ndarray):
            state = Tensor(state)
        
        # Get action mean
        with Tensor.no_grad():
            action_mean = self.policy_network(state)
        
        # Sample action
        if self.action_space["type"] == "box":
            # Convert to numpy
            action_mean = action_mean.numpy()
            
            # Sample action
            action = np.random.normal(action_mean, self.std)
            
            # Clip action to valid range
            action = np.clip(
                action,
                self.action_space["low"],
                self.action_space["high"],
            )
            
            return action
        else:
            raise ValueError(f"Unsupported action space type: {self.action_space['type']}")
