# Reinforcement Learning Module

## Overview

The Reinforcement Learning (RL) module in Neurenix provides a comprehensive suite of algorithms and tools for developing, training, and deploying reinforcement learning agents. Built on Neurenix's high-performance multi-language architecture, this module delivers efficient implementations of classical and state-of-the-art RL algorithms.

The module features a unified agent interface that works seamlessly with Neurenix's tensor and neural network modules, enabling users to easily switch between different RL algorithms without changing their environment code. It supports various RL paradigms, including value-based methods, policy-based methods, actor-critic methods, and model-based methods.

Implemented with a combination of Rust and C++ for performance-critical components and Python for the user-friendly interface, the RL module ensures both computational efficiency and ease of use. It provides native support for various hardware accelerators, including GPUs, TPUs, and specialized AI hardware, with automatic optimization for the available hardware.

## Key Concepts

### Agent Interface

The RL module provides a unified interface for all agents, making it easy to switch between different algorithms:

- **Policy Management**: Representation and management of agent policies
- **State Processing**: Preprocessing and feature extraction from environment states
- **Action Selection**: Mechanisms for selecting actions based on policies
- **Learning Algorithms**: Methods for updating policies based on experience
- **Exploration Strategies**: Techniques for balancing exploration and exploitation

### Value-Based Methods

The module includes various value-based RL methods that learn value functions to guide action selection:

- **Q-Learning**: Classic algorithm for learning action-value functions
- **Deep Q-Network (DQN)**: Neural network-based Q-learning with experience replay and target networks
- **Double DQN**: Extension of DQN that addresses overestimation bias
- **Dueling DQN**: Architecture that separates state value and advantage functions
- **Prioritized Experience Replay**: Sampling strategy that prioritizes important transitions

### Policy-Based Methods

The module includes policy-based RL methods that directly learn policy functions:

- **REINFORCE**: Classic policy gradient algorithm
- **Trust Region Policy Optimization (TRPO)**: Policy optimization with trust region constraint
- **Proximal Policy Optimization (PPO)**: Simplified version of TRPO with clipped objective
- **Soft Actor-Critic (SAC)**: Off-policy actor-critic algorithm with entropy regularization
- **Deterministic Policy Gradient (DPG)**: Learning deterministic policies with policy gradients

### Actor-Critic Methods

The module includes actor-critic RL methods that combine value-based and policy-based approaches:

- **Advantage Actor-Critic (A2C)**: Synchronous version of A3C
- **Asynchronous Advantage Actor-Critic (A3C)**: Parallel actor-critic with asynchronous updates
- **Generalized Advantage Estimation (GAE)**: Technique for reducing variance in policy gradients
- **Actor-Critic with Experience Replay (ACER)**: Off-policy actor-critic with experience replay

### Model-Based Methods

The module includes model-based RL methods that learn and use environment models:

- **Dyna-Q**: Integrating planning, acting, and learning with a learned model
- **Model-Based Policy Optimization (MBPO)**: Model-based policy optimization with uncertainty
- **Probabilistic Ensembles with Trajectory Sampling (PETS)**: Ensemble of probabilistic dynamics models
- **Model-Predictive Control (MPC)**: Planning with a learned model over a finite horizon
- **World Models**: Learning environment dynamics for planning and policy learning

## API Reference

### Agent Interface

```python
import neurenix
from neurenix.rl import Agent

# Base agent class (abstract)
class Agent:
    def __init__(self, observation_space, action_space, config=None):
        """
        Initialize the agent.
        
        Args:
            observation_space: Space object defining the observation space
            action_space: Space object defining the action space
            config: Dictionary containing agent configuration
        """
        pass
    
    def act(self, observation, explore=True):
        """
        Select an action based on the current observation.
        
        Args:
            observation: Current observation from the environment
            explore: Whether to use exploration or act greedily
        
        Returns:
            Selected action
        """
        pass
    
    def learn(self, experiences):
        """
        Update the agent's policy based on experiences.
        
        Args:
            experiences: Batch of experiences (observations, actions, rewards, next_observations, dones)
        
        Returns:
            Dictionary of learning metrics
        """
        pass
    
    def save(self, path):
        """
        Save the agent's state to disk.
        
        Args:
            path: Path to save the agent
        """
        pass
    
    def load(self, path):
        """
        Load the agent's state from disk.
        
        Args:
            path: Path to load the agent from
        """
        pass
```

### Value-Based Agents

```python
from neurenix.rl.agents import DQNAgent, DoubleDQNAgent, DuelingDQNAgent

# DQN agent
agent = DQNAgent(
    observation_space=env.observation_space,
    action_space=env.action_space,
    config={
        "learning_rate": 0.001,        # Learning rate for the optimizer
        "gamma": 0.99,                 # Discount factor
        "epsilon_start": 1.0,          # Initial exploration rate
        "epsilon_end": 0.01,           # Final exploration rate
        "epsilon_decay": 0.995,        # Decay rate for exploration
        "buffer_size": 100000,         # Replay buffer size
        "batch_size": 64,              # Batch size for learning
        "update_frequency": 4,         # Frequency of learning updates
        "target_update_frequency": 1000,  # Frequency of target network updates
        "hidden_layers": [128, 128],   # Hidden layer sizes
        "activation": "relu",          # Activation function
        "optimizer": "adam"            # Optimizer type
    }
)
```

### Policy-Based Agents

```python
from neurenix.rl.agents import REINFORCEAgent, PPOAgent, SACAgent

# PPO agent
agent = PPOAgent(
    observation_space=env.observation_space,
    action_space=env.action_space,
    config={
        "learning_rate": 0.0003,       # Learning rate for the optimizer
        "gamma": 0.99,                 # Discount factor
        "lambda_gae": 0.95,            # Lambda for GAE
        "epsilon_clip": 0.2,           # Clipping parameter for PPO
        "value_coef": 0.5,             # Value function coefficient
        "entropy_coef": 0.01,          # Entropy coefficient
        "hidden_layers": [64, 64],     # Hidden layer sizes
        "activation": "tanh",          # Activation function
        "optimizer": "adam",           # Optimizer type
        "epochs": 10,                  # Number of epochs per update
        "batch_size": 64               # Batch size for learning
    }
)
```

## Framework Comparison

### Neurenix RL vs. TensorFlow RL

| Feature | Neurenix RL | TensorFlow RL |
|---------|-------------|---------------|
| Performance | Multi-language implementation with Rust/C++ backends | Python implementation with TensorFlow backend |
| Hardware Support | Comprehensive support for various hardware | Primarily focused on TPUs and GPUs |
| Algorithm Variety | Extensive collection of classical and modern RL algorithms | Good selection of common RL algorithms |
| Distributed Training | Multiple strategies with efficient communication | Limited to specific distributed strategies |
| Model-Based RL | Comprehensive support for model-based methods | Limited support for model-based methods |
| Exploration Strategies | Wide variety of exploration strategies | Basic exploration strategies |
| Environment Compatibility | Seamless integration with various environment types | Good compatibility with standard environments |
| Edge Device Support | Native support for edge devices | Limited through TensorFlow Lite |

Neurenix's RL module provides better performance through its multi-language implementation and offers more comprehensive hardware support, especially for edge devices. It also provides a wider variety of RL algorithms, more advanced exploration strategies, and better support for model-based methods.

### Neurenix RL vs. PyTorch RL

| Feature | Neurenix RL | PyTorch RL |
|---------|-------------|------------|
| Performance | Multi-language implementation with Rust/C++ backends | Python implementation with PyTorch backend |
| Hardware Support | Comprehensive support for various hardware | Primarily focused on CUDA devices |
| Algorithm Variety | Extensive collection of classical and modern RL algorithms | Good selection of common RL algorithms |
| Distributed Training | Multiple strategies with efficient communication | Good support through PyTorch Distributed |
| Model-Based RL | Comprehensive support for model-based methods | Limited support for model-based methods |
| Exploration Strategies | Wide variety of exploration strategies | Basic exploration strategies |
| Environment Compatibility | Seamless integration with various environment types | Good compatibility with standard environments |
| Edge Device Support | Native support for edge devices | Limited through separate tools |

While PyTorch has good RL libraries, Neurenix's RL module offers better performance through its multi-language implementation and provides more comprehensive hardware support, especially for edge devices. It also offers more advanced exploration strategies and better support for model-based methods.

### Neurenix RL vs. Scikit-Learn RL

| Feature | Neurenix RL | Scikit-Learn RL |
|---------|-------------|-----------------|
| Deep RL Support | Full support for deep reinforcement learning | Limited to classical RL algorithms |
| Hardware Acceleration | Native support for various hardware accelerators | Limited hardware acceleration |
| Algorithm Variety | Extensive collection of classical and modern RL algorithms | Focus on classical RL algorithms |
| Model-Based RL | Comprehensive support for model-based methods | Limited support for model-based methods |
| Exploration Strategies | Wide variety of exploration strategies | Basic exploration strategies |
| Environment Compatibility | Seamless integration with various environment types | Limited environment compatibility |
| Edge Device Support | Native support for edge devices | Limited edge support |

Scikit-Learn's RL capabilities are primarily focused on classical RL algorithms, while Neurenix's RL module is designed for both classical and deep RL. Neurenix provides better hardware acceleration, more comprehensive support for various RL algorithms, and better integration with the deep learning ecosystem.

## Best Practices

### Choosing the Right Algorithm

1. **Start with Simple Algorithms**: For discrete action spaces, start with DQN; for continuous action spaces, start with DDPG or SAC.

```python
# For discrete action spaces
if isinstance(env.action_space, neurenix.rl.spaces.Discrete):
    agent = neurenix.rl.agents.DQNAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        config={"learning_rate": 0.001, "gamma": 0.99}
    )
# For continuous action spaces
elif isinstance(env.action_space, neurenix.rl.spaces.Box):
    agent = neurenix.rl.agents.SACAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        config={"learning_rate": 0.0003, "gamma": 0.99, "tau": 0.005}
    )
```

2. **Consider Sample Efficiency**: For tasks with limited samples, use off-policy algorithms like DQN, DDPG, or SAC.

3. **Use Model-Based Methods for Complex Tasks**: For complex tasks with sparse rewards, consider model-based methods like MBPO or Dreamer.

### Hyperparameter Tuning

1. **Start with Default Hyperparameters**: Begin with the default hyperparameters provided in the documentation.

2. **Tune Learning Rate and Batch Size**: These are often the most important hyperparameters to tune.

3. **Adjust Exploration Parameters**: Fine-tune exploration parameters based on the task complexity.

### Environment Preprocessing

1. **Normalize Observations**: Normalize observations to improve learning stability.

2. **Stack Frames for Partial Observability**: Stack consecutive frames for environments with partial observability.

3. **Scale Rewards**: Scale rewards to improve learning stability.

## Tutorials

### Training a DQN Agent on CartPole

```python
import neurenix
import gym
from neurenix.rl.agents import DQNAgent
from neurenix.rl.wrappers import Monitor

# Create environment
env = gym.make("CartPole-v1")
env = Monitor(env, directory="./logs")

# Create agent
agent = DQNAgent(
    observation_space=env.observation_space,
    action_space=env.action_space,
    config={
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "buffer_size": 10000,
        "batch_size": 64,
        "update_frequency": 4,
        "target_update_frequency": 1000,
        "hidden_layers": [64, 64],
        "activation": "relu",
        "optimizer": "adam"
    }
)

# Training loop
num_episodes = 500
max_steps = 500
rewards = []

for episode in range(num_episodes):
    observation, _ = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        # Select action
        action = agent.act(observation, explore=True)
        
        # Take action
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store experience
        agent.memory.add(observation, action, reward, next_observation, done)
        
        # Learn
        if len(agent.memory) > agent.batch_size:
            experiences = agent.memory.sample(agent.batch_size)
            agent.learn(experiences)
        
        # Update observation
        observation = next_observation
        episode_reward += reward
        
        # Check if episode is done
        if done:
            break
    
    # Update target network
    if episode % agent.target_update_frequency == 0:
        agent.update_target_network()
    
    # Decay epsilon
    agent.epsilon = max(
        agent.epsilon_end,
        agent.epsilon * agent.epsilon_decay
    )
    
    # Track rewards
    rewards.append(episode_reward)
    
    # Print progress
    if episode % 10 == 0:
        mean_reward = sum(rewards[-10:]) / 10
        print(f"Episode {episode}, Mean Reward: {mean_reward:.2f}, Epsilon: {agent.epsilon:.2f}")
```
