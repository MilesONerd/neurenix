"""
Value function module for reinforcement learning in Neurenix.

This module provides the base ValueFunction class and implementations of various
reinforcement learning value functions.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from neurenix.tensor import Tensor
from neurenix.nn.module import Module


class ValueFunction:
    """
    Base class for reinforcement learning value functions.
    
    This class provides the basic functionality for reinforcement learning value functions,
    which estimate the value of states or state-action pairs.
    """
    
    def __init__(self, name: str = "ValueFunction"):
        """
        Initialize value function.
        
        Args:
            name: Value function name
        """
        self.name = name
    
    def __call__(self, state: Union[np.ndarray, Tensor]) -> Tensor:
        """
        Estimate the value of a state.
        
        Args:
            state: State to estimate value for
            
        Returns:
            Estimated value
        """
        return self.estimate_value(state)
    
    def estimate_value(self, state: Union[np.ndarray, Tensor]) -> Tensor:
        """
        Estimate the value of a state.
        
        Args:
            state: State to estimate value for
            
        Returns:
            Estimated value
        """
        raise NotImplementedError("Subclasses must implement estimate_value")
    
    def update(
        self,
        states: Tensor,
        actions: Any,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        gamma: float,
    ) -> Dict[str, float]:
        """
        Update value function based on experience.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            gamma: Discount factor
            
        Returns:
            Dictionary of update metrics
        """
        raise NotImplementedError("Subclasses must implement update")
    
    def save(self, path: str):
        """
        Save value function to disk.
        
        Args:
            path: Path to save value function to
        """
        pass
    
    def load(self, path: str):
        """
        Load value function from disk.
        
        Args:
            path: Path to load value function from
        """
        pass


class QFunction(ValueFunction):
    """
    Q-function for reinforcement learning.
    
    This class implements a Q-function, which estimates the value of
    state-action pairs.
    """
    
    def __init__(
        self,
        q_network: Module,
        target_network: Optional[Module] = None,
        optimizer = None,
        observation_space: Optional[Dict[str, Any]] = None,
        action_space: Optional[Dict[str, Any]] = None,
        name: str = "QFunction",
    ):
        """
        Initialize Q-function.
        
        Args:
            q_network: Q-network for value estimation
            target_network: Target network for stable learning
            optimizer: Optimizer for Q-network
            observation_space: Observation space specification
            action_space: Action space specification
            name: Value function name
        """
        super().__init__(name=name)
        self.q_network = q_network
        self.target_network = target_network
        self.optimizer = optimizer
        self.observation_space = observation_space
        self.action_space = action_space
    
    def estimate_value(self, state: Union[np.ndarray, Tensor]) -> Tensor:
        """
        Estimate the value of a state.
        
        Args:
            state: State to estimate value for
            
        Returns:
            Estimated Q-values for all actions
        """
        # Convert to tensor if necessary
        if isinstance(state, np.ndarray):
            state = Tensor(state)
        
        # Add batch dimension if necessary
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Get Q-values
        return self.q_network(state)
    
    def update(
        self,
        states: Tensor,
        actions: Any,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        gamma: float,
    ) -> Dict[str, float]:
        """
        Update Q-function based on experience.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            gamma: Discount factor
            
        Returns:
            Dictionary of update metrics
        """
        # Check if optimizer is available
        if self.optimizer is None:
            raise ValueError("Optimizer is required for Q-function update")
        
        # Convert actions to tensor if necessary
        if not isinstance(actions, Tensor):
            actions = Tensor(actions)
        
        # Get current Q-values
        q_values = self.q_network(states)
        
        # Get Q-values for taken actions
        if self.action_space["type"] == "discrete":
            # Discrete action space
            q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        else:
            raise ValueError(f"Unsupported action space type: {self.action_space['type']}")
        
        # Compute target Q-values
        with Tensor.no_grad():
            if self.target_network is not None:
                # Use target network for stable learning
                next_q_values = self.target_network(next_states)
            else:
                # Use Q-network
                next_q_values = self.q_network(next_states)
            
            # Get maximum Q-values
            next_q_values = next_q_values.max(1)[0]
            
            # Compute target
            target_q_values = rewards + gamma * next_q_values * (1 - dones)
        
        # Compute loss
        loss = ((q_values - target_q_values) ** 2).mean()
        
        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"value_loss": loss.item()}
    
    def update_target(self):
        """Update target network with Q-network weights."""
        if self.target_network is None:
            return
        
        # Copy Q-network weights to target network
        for target_param, param in zip(
            self.target_network.parameters(), self.q_network.parameters()
        ):
            target_param.data.copy_(param.data)
    
    def save(self, path: str):
        """
        Save Q-function to disk.
        
        Args:
            path: Path to save Q-function to
        """
        # Save Q-network
        self.q_network.save(f"{path}_q_network")
        
        # Save target network
        if self.target_network is not None:
            self.target_network.save(f"{path}_target_network")
    
    def load(self, path: str):
        """
        Load Q-function from disk.
        
        Args:
            path: Path to load Q-function from
        """
        # Load Q-network
        self.q_network.load(f"{path}_q_network")
        
        # Load target network
        if self.target_network is not None:
            self.target_network.load(f"{path}_target_network")


class ValueNetworkFunction(ValueFunction):
    """
    Value network function for reinforcement learning.
    
    This class implements a value network function, which estimates the value of
    states.
    """
    
    def __init__(
        self,
        value_network: Module,
        optimizer = None,
        observation_space: Optional[Dict[str, Any]] = None,
        name: str = "ValueNetworkFunction",
    ):
        """
        Initialize value network function.
        
        Args:
            value_network: Value network for value estimation
            optimizer: Optimizer for value network
            observation_space: Observation space specification
            name: Value function name
        """
        super().__init__(name=name)
        self.value_network = value_network
        self.optimizer = optimizer
        self.observation_space = observation_space
    
    def estimate_value(self, state: Union[np.ndarray, Tensor]) -> Tensor:
        """
        Estimate the value of a state.
        
        Args:
            state: State to estimate value for
            
        Returns:
            Estimated value
        """
        # Convert to tensor if necessary
        if isinstance(state, np.ndarray):
            state = Tensor(state)
        
        # Add batch dimension if necessary
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        # Get value
        return self.value_network(state)
    
    def update(
        self,
        states: Tensor,
        actions: Any,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        gamma: float,
    ) -> Dict[str, float]:
        """
        Update value network function based on experience.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            gamma: Discount factor
            
        Returns:
            Dictionary of update metrics
        """
        # Check if optimizer is available
        if self.optimizer is None:
            raise ValueError("Optimizer is required for value network function update")
        
        # Get current values
        values = self.value_network(states).squeeze(1)
        
        # Compute target values
        with Tensor.no_grad():
            next_values = self.value_network(next_states).squeeze(1)
            target_values = rewards + gamma * next_values * (1 - dones)
        
        # Compute loss
        loss = ((values - target_values) ** 2).mean()
        
        # Update value network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {"value_loss": loss.item()}
    
    def save(self, path: str):
        """
        Save value network function to disk.
        
        Args:
            path: Path to save value network function to
        """
        # Save value network
        self.value_network.save(f"{path}_value_network")
    
    def load(self, path: str):
        """
        Load value network function from disk.
        
        Args:
            path: Path to load value network function from
        """
        # Load value network
        self.value_network.load(f"{path}_value_network")


class AdvantageFunction(ValueFunction):
    """
    Advantage function for reinforcement learning.
    
    This class implements an advantage function, which estimates the advantage of
    actions over the baseline value of states.
    """
    
    def __init__(
        self,
        value_function: ValueFunction,
        q_function: QFunction,
        name: str = "AdvantageFunction",
    ):
        """
        Initialize advantage function.
        
        Args:
            value_function: Value function for baseline estimation
            q_function: Q-function for state-action value estimation
            name: Value function name
        """
        super().__init__(name=name)
        self.value_function = value_function
        self.q_function = q_function
    
    def estimate_value(self, state: Union[np.ndarray, Tensor]) -> Tensor:
        """
        Estimate the value of a state.
        
        Args:
            state: State to estimate value for
            
        Returns:
            Estimated baseline value
        """
        return self.value_function(state)
    
    def estimate_advantage(
        self,
        state: Union[np.ndarray, Tensor],
        action: Optional[Any] = None,
    ) -> Tensor:
        """
        Estimate the advantage of an action.
        
        Args:
            state: State to estimate advantage for
            action: Action to estimate advantage for
            
        Returns:
            Estimated advantage
        """
        # Get baseline value
        baseline = self.value_function(state)
        
        # Get Q-value
        if action is None:
            # Return advantage for all actions
            q_values = self.q_function(state)
            return q_values - baseline.unsqueeze(1)
        else:
            # Return advantage for specific action
            q_value = self.q_function(state, action)
            return q_value - baseline
    
    def update(
        self,
        states: Tensor,
        actions: Any,
        rewards: Tensor,
        next_states: Tensor,
        dones: Tensor,
        gamma: float,
    ) -> Dict[str, float]:
        """
        Update advantage function based on experience.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
            gamma: Discount factor
            
        Returns:
            Dictionary of update metrics
        """
        # Update value function
        value_metrics = self.value_function.update(
            states, actions, rewards, next_states, dones, gamma
        )
        
        # Update Q-function
        q_metrics = self.q_function.update(
            states, actions, rewards, next_states, dones, gamma
        )
        
        # Combine metrics
        metrics = {
            "value_loss": value_metrics["value_loss"],
            "q_loss": q_metrics["value_loss"],
        }
        
        return metrics
