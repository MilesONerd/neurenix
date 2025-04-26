"""
Agent module for reinforcement learning in Neurenix.

This module provides the base Agent class and implementations of various
reinforcement learning agents.
"""

from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import numpy as np

from neurenix.tensor import Tensor
from neurenix.nn.module import Module
from neurenix.rl.policy import Policy
from neurenix.rl.value import ValueFunction


class Agent:
    """
    Base class for reinforcement learning agents.
    
    This class provides the basic functionality for reinforcement learning agents,
    including interaction with environments and learning from experience.
    """
    
    def __init__(
        self,
        policy: Policy,
        value_function: Optional[ValueFunction] = None,
        gamma: float = 0.99,
        name: str = "Agent",
    ):
        """
        Initialize agent.
        
        Args:
            policy: Policy for action selection
            value_function: Value function for state evaluation
            gamma: Discount factor
            name: Agent name
        """
        self.policy = policy
        self.value_function = value_function
        self.gamma = gamma
        self.name = name
        
        # Experience buffer
        self.experiences = []
        
        # Training metrics
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "value_losses": [],
            "policy_losses": [],
        }
    
    def act(self, state: Union[np.ndarray, Tensor]) -> Any:
        """
        Select an action based on the current state.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        return self.policy(state)
    
    def update(
        self,
        state: Union[np.ndarray, Tensor],
        action: Any,
        reward: float,
        next_state: Union[np.ndarray, Tensor],
        done: bool,
    ) -> Dict[str, float]:
        """
        Update agent based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            
        Returns:
            Dictionary of update metrics
        """
        # Store experience
        self.experiences.append((state, action, reward, next_state, done))
        
        # Update metrics
        metrics = {}
        
        return metrics
    
    def train(
        self,
        env,
        episodes: int = 1000,
        max_steps: int = 1000,
        render: bool = False,
        verbose: bool = True,
        callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the agent on an environment.
        
        Args:
            env: Environment to train on
            episodes: Number of episodes to train for
            max_steps: Maximum number of steps per episode
            render: Whether to render the environment
            verbose: Whether to print training progress
            callback: Callback function called after each episode
            
        Returns:
            Dictionary of training metrics
        """
        # Reset metrics
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "value_losses": [],
            "policy_losses": [],
        }
        
        # Training loop
        for episode in range(episodes):
            # Reset environment
            state = env.reset()
            
            # Episode variables
            episode_reward = 0
            episode_length = 0
            done = False
            
            # Episode loop
            while not done and episode_length < max_steps:
                # Select action
                action = self.act(state)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                
                # Update agent
                update_metrics = self.update(state, action, reward, next_state, done)
                
                # Update episode variables
                episode_reward += reward
                episode_length += 1
                state = next_state
                
                # Render environment
                if render:
                    env.render()
            
            # Update metrics
            self.metrics["episode_rewards"].append(episode_reward)
            self.metrics["episode_lengths"].append(episode_length)
            
            # Update value and policy losses
            if "value_loss" in update_metrics:
                self.metrics["value_losses"].append(update_metrics["value_loss"])
            if "policy_loss" in update_metrics:
                self.metrics["policy_losses"].append(update_metrics["policy_loss"])
            
            # Print progress
            if verbose and (episode + 1) % (episodes // 10 or 1) == 0:
                mean_reward = np.mean(self.metrics["episode_rewards"][-100:])
                print(f"Episode {episode + 1}/{episodes} | "
                      f"Mean Reward (last 100): {mean_reward:.2f}")
            
            # Call callback
            if callback is not None:
                callback_metrics = {
                    "episode": episode,
                    "reward": episode_reward,
                    "length": episode_length,
                    **update_metrics,
                }
                if callback(callback_metrics):
                    break
        
        return self.metrics
    
    def save(self, path: str):
        """
        Save agent to disk.
        
        Args:
            path: Path to save agent to
        """
        # Save policy
        if hasattr(self.policy, "save"):
            self.policy.save(f"{path}_policy")
        
        # Save value function
        if self.value_function is not None and hasattr(self.value_function, "save"):
            self.value_function.save(f"{path}_value")
    
    def load(self, path: str):
        """
        Load agent from disk.
        
        Args:
            path: Path to load agent from
        """
        # Load policy
        if hasattr(self.policy, "load"):
            self.policy.load(f"{path}_policy")
        
        # Load value function
        if self.value_function is not None and hasattr(self.value_function, "load"):
            self.value_function.load(f"{path}_value")


class DQN(Agent):
    """
    Deep Q-Network agent.
    
    This class implements the Deep Q-Network (DQN) algorithm for
    reinforcement learning.
    """
    
    def __init__(
        self,
        observation_space: Dict[str, Any],
        action_space: Dict[str, Any],
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10000,
        batch_size: int = 64,
        update_target_every: int = 100,
        learning_rate: float = 0.001,
        name: str = "DQN",
    ):
        """
        Initialize DQN agent.
        
        Args:
            observation_space: Observation space specification
            action_space: Action space specification
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Exploration rate decay
            buffer_size: Experience replay buffer size
            batch_size: Batch size for training
            update_target_every: Number of steps between target network updates
            learning_rate: Learning rate for optimizer
            name: Agent name
        """
        # Create Q-network
        from neurenix.nn import Sequential, Linear, ReLU
        
        # Get input and output dimensions
        if observation_space["type"] == "box":
            input_dim = np.prod(observation_space["shape"])
        else:
            raise ValueError(f"Unsupported observation space type: {observation_space['type']}")
        
        if action_space["type"] == "discrete":
            output_dim = action_space["n"]
        else:
            raise ValueError(f"Unsupported action space type: {action_space['type']}")
        
        # Create Q-network
        q_network = Sequential(
            Linear(input_dim, 64),
            ReLU(),
            Linear(64, 64),
            ReLU(),
            Linear(64, output_dim),
        )
        
        # Create target network
        target_network = q_network.clone()
        
        # Create optimizer
        from neurenix.optim import Adam
        optimizer = Adam(q_network.parameters(), lr=learning_rate)
        
        # Create policy
        from neurenix.rl.policy import EpsilonGreedyPolicy
        policy = EpsilonGreedyPolicy(
            q_network,
            action_space,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
        )
        
        # Create value function
        from neurenix.rl.value import QFunction
        value_function = QFunction(
            q_network,
            target_network,
            optimizer,
            observation_space,
            action_space,
        )
        
        # Initialize agent
        super().__init__(
            policy=policy,
            value_function=value_function,
            gamma=gamma,
            name=name,
        )
        
        # DQN-specific attributes
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.steps = 0
        
        # Experience replay buffer
        self.buffer = []
    
    def update(
        self,
        state: Union[np.ndarray, Tensor],
        action: Any,
        reward: float,
        next_state: Union[np.ndarray, Tensor],
        done: bool,
    ) -> Dict[str, float]:
        """
        Update DQN agent based on experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            
        Returns:
            Dictionary of update metrics
        """
        # Convert to tensors if necessary
        if isinstance(state, np.ndarray):
            state = Tensor(state)
        if isinstance(next_state, np.ndarray):
            next_state = Tensor(next_state)
        
        # Add experience to buffer
        self.buffer.append((state, action, reward, next_state, done))
        
        # Limit buffer size
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        
        # Increment steps
        self.steps += 1
        
        # Update target network
        if self.steps % self.update_target_every == 0:
            self.value_function.update_target()
        
        # Update policy (epsilon decay)
        self.policy.step()
        
        # Train on batch
        metrics = {}
        if len(self.buffer) >= self.batch_size:
            # Sample batch
            indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
            batch = [self.buffer[i] for i in indices]
            
            # Unpack batch
            states, actions, rewards, next_states, dones = zip(*batch)
            
            # Convert to tensors
            states = Tensor.stack(states)
            rewards = Tensor(rewards)
            next_states = Tensor.stack(next_states)
            dones = Tensor(dones)
            
            # Train value function
            metrics = self.value_function.update(
                states, actions, rewards, next_states, dones, self.gamma
            )
        
        return metrics


class MultiAgentSystem:
    """
    Multi-agent system for reinforcement learning.
    
    This class manages multiple agents interacting in a shared environment,
    enabling multi-agent reinforcement learning.
    """
    
    def __init__(
        self,
        agents: List[Agent],
        env,
        name: str = "MultiAgentSystem",
    ):
        """
        Initialize multi-agent system.
        
        Args:
            agents: List of agents
            env: Environment
            name: System name
        """
        self.agents = agents
        self.env = env
        self.name = name
        
        # System metrics
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "agent_rewards": [[] for _ in agents],
        }
    
    def train(
        self,
        episodes: int = 1000,
        max_steps: int = 1000,
        render: bool = False,
        verbose: bool = True,
        callback: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the multi-agent system.
        
        Args:
            episodes: Number of episodes to train for
            max_steps: Maximum number of steps per episode
            render: Whether to render the environment
            verbose: Whether to print training progress
            callback: Callback function called after each episode
            
        Returns:
            Dictionary of training metrics
        """
        # Reset metrics
        self.metrics = {
            "episode_rewards": [],
            "episode_lengths": [],
            "agent_rewards": [[] for _ in self.agents],
        }
        
        # Training loop
        for episode in range(episodes):
            # Reset environment
            states = self.env.reset()
            
            # Episode variables
            episode_reward = 0
            episode_length = 0
            agent_rewards = [0 for _ in self.agents]
            done = False
            
            # Episode loop
            while not done and episode_length < max_steps:
                # Select actions
                actions = [agent.act(state) for agent, state in zip(self.agents, states)]
                
                # Take actions
                next_states, rewards, done, info = self.env.step(actions)
                
                # Update agents
                for i, agent in enumerate(self.agents):
                    agent.update(
                        states[i], actions[i], rewards[i], next_states[i], done
                    )
                
                # Update episode variables
                episode_reward += sum(rewards)
                episode_length += 1
                for i, reward in enumerate(rewards):
                    agent_rewards[i] += reward
                states = next_states
                
                # Render environment
                if render:
                    self.env.render()
            
            # Update metrics
            self.metrics["episode_rewards"].append(episode_reward)
            self.metrics["episode_lengths"].append(episode_length)
            for i, reward in enumerate(agent_rewards):
                self.metrics["agent_rewards"][i].append(reward)
            
            # Print progress
            if verbose and (episode + 1) % (episodes // 10 or 1) == 0:
                mean_reward = np.mean(self.metrics["episode_rewards"][-100:])
                print(f"Episode {episode + 1}/{episodes} | "
                      f"Mean Reward (last 100): {mean_reward:.2f}")
            
            # Call callback
            if callback is not None:
                callback_metrics = {
                    "episode": episode,
                    "reward": episode_reward,
                    "length": episode_length,
                    "agent_rewards": agent_rewards,
                }
                if callback(callback_metrics):
                    break
        
        return self.metrics
    
    def save(self, path: str):
        """
        Save multi-agent system to disk.
        
        Args:
            path: Path to save system to
        """
        # Save agents
        for i, agent in enumerate(self.agents):
            agent.save(f"{path}_agent{i}")
    
    def load(self, path: str):
        """
        Load multi-agent system from disk.
        
        Args:
            path: Path to load system from
        """
        # Load agents
        for i, agent in enumerate(self.agents):
            agent.load(f"{path}_agent{i}")
