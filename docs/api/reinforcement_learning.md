# Reinforcement Learning Documentation

## Overview

The Reinforcement Learning module in Neurenix provides tools and utilities for developing and training reinforcement learning agents. Reinforcement learning is a paradigm where agents learn to make decisions by taking actions in an environment to maximize cumulative rewards.

Neurenix's reinforcement learning capabilities are implemented using a multi-language architecture, where the high-performance operations are implemented in the Rust/C++ Phynexus engine, while the Python API provides a user-friendly interface. This architecture enables Neurenix to deliver optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

## Key Concepts

### Agents

Agents are the decision-makers in reinforcement learning. They observe the environment, take actions, and learn from the resulting rewards. Neurenix provides a flexible Agent framework that can be extended to implement various reinforcement learning algorithms.

### Environments

Environments represent the world in which agents operate. They define the state space, action space, and reward function. Neurenix provides a standard Environment interface that can be used to implement custom environments or wrap existing ones.

### Policies

Policies define how agents select actions based on their current state. Neurenix provides several policy implementations:

- **RandomPolicy**: Selects actions uniformly at random
- **GreedyPolicy**: Selects the action with the highest estimated value
- **EpsilonGreedyPolicy**: Balances exploration and exploitation using an epsilon parameter
- **SoftmaxPolicy**: Selects actions according to a softmax distribution over action values
- **GaussianPolicy**: Selects actions according to a Gaussian distribution for continuous action spaces

### Value Functions

Value functions estimate the expected return (cumulative reward) of states or state-action pairs. Neurenix provides several value function implementations:

- **QFunction**: Estimates the value of state-action pairs
- **ValueNetworkFunction**: Estimates the value of states
- **AdvantageFunction**: Estimates the advantage of actions over the baseline value of states

### Algorithms

Neurenix implements several state-of-the-art reinforcement learning algorithms:

- **DQN (Deep Q-Network)**: Learns a Q-function to estimate the value of state-action pairs
- **A2C (Advantage Actor-Critic)**: Learns both a policy and a value function
- **PPO (Proximal Policy Optimization)**: Learns a policy and a value function with a clipped surrogate objective
- **DDPG (Deep Deterministic Policy Gradient)**: Learns a deterministic policy and a Q-function for continuous action spaces
- **SAC (Soft Actor-Critic)**: Learns a stochastic policy and a Q-function with entropy regularization

### Multi-Agent Systems

Neurenix supports multi-agent reinforcement learning through the MultiAgentSystem class, which manages multiple agents interacting in a shared environment.

## API Reference

### Agent

```python
neurenix.rl.Agent(policy, value_function=None, gamma=0.99, name="Agent")
```

Base class for reinforcement learning agents.

**Parameters:**
- `policy`: Policy for action selection
- `value_function`: Value function for state evaluation (optional)
- `gamma`: Discount factor
- `name`: Agent name

**Methods:**
- `act(state)`: Select an action based on the current state
- `update(state, action, reward, next_state, done)`: Update agent based on experience
- `train(env, episodes=1000, max_steps=1000, render=False, verbose=True, callback=None)`: Train the agent on an environment
- `save(path)`: Save agent to disk
- `load(path)`: Load agent from disk

### Environment

```python
neurenix.rl.Environment(name="Environment", max_steps=1000)
```

Base class for reinforcement learning environments.

**Parameters:**
- `name`: Environment name
- `max_steps`: Maximum number of steps per episode

**Methods:**
- `reset()`: Reset the environment and return the initial state
- `step(action)`: Take a step in the environment and return (next_state, reward, done, info)
- `render(mode="human")`: Render the environment
- `close()`: Close the environment
- `seed(seed=None)`: Set the random seed
- `get_observation_space()`: Get the observation space specification
- `get_action_space()`: Get the action space specification

### Policy Classes

```python
neurenix.rl.Policy(name="Policy")
neurenix.rl.RandomPolicy(action_space, name="RandomPolicy")
neurenix.rl.GreedyPolicy(value_function, action_space, name="GreedyPolicy")
neurenix.rl.EpsilonGreedyPolicy(value_function, action_space, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, name="EpsilonGreedyPolicy")
neurenix.rl.SoftmaxPolicy(value_function, action_space, temperature=1.0, name="SoftmaxPolicy")
neurenix.rl.GaussianPolicy(policy_network, action_space, std=0.1, name="GaussianPolicy")
```

Policy classes for action selection in reinforcement learning.

**Common Methods:**
- `select_action(state)`: Select an action based on the current state
- `step()`: Update policy parameters (e.g., exploration rate)
- `reset()`: Reset policy parameters

### Value Function Classes

```python
neurenix.rl.ValueFunction(name="ValueFunction")
neurenix.rl.QFunction(q_network, target_network=None, optimizer=None, observation_space=None, action_space=None, name="QFunction")
neurenix.rl.ValueNetworkFunction(value_network, optimizer=None, observation_space=None, name="ValueNetworkFunction")
neurenix.rl.AdvantageFunction(value_function, q_function, name="AdvantageFunction")
```

Value function classes for state and state-action value estimation in reinforcement learning.

**Common Methods:**
- `estimate_value(state)`: Estimate the value of a state
- `update(states, actions, rewards, next_states, dones, gamma)`: Update value function based on experience

### Algorithm Classes

```python
neurenix.rl.DQN(observation_space, action_space, hidden_dims=[64, 64], learning_rate=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, buffer_size=10000, batch_size=64, update_target_every=100, double_q=False, dueling=False, name="DQN")
neurenix.rl.A2C(observation_space, action_space, actor_hidden_dims=[64, 64], critic_hidden_dims=[64, 64], actor_learning_rate=0.0003, critic_learning_rate=0.001, gamma=0.99, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5, name="A2C")
neurenix.rl.PPO(observation_space, action_space, actor_hidden_dims=[64, 64], critic_hidden_dims=[64, 64], actor_learning_rate=0.0003, critic_learning_rate=0.001, gamma=0.99, gae_lambda=0.95, clip_ratio=0.2, target_kl=0.01, value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5, name="PPO")
neurenix.rl.DDPG(observation_space, action_space, actor_hidden_dims=[64, 64], critic_hidden_dims=[64, 64], actor_learning_rate=0.001, critic_learning_rate=0.001, gamma=0.99, tau=0.005, buffer_size=10000, batch_size=64, exploration_noise=0.1, name="DDPG")
neurenix.rl.SAC(observation_space, action_space, actor_hidden_dims=[64, 64], critic_hidden_dims=[64, 64], actor_learning_rate=0.0003, critic_learning_rate=0.0003, alpha_learning_rate=0.0003, gamma=0.99, tau=0.005, alpha=0.2, auto_alpha=True, buffer_size=10000, batch_size=64, name="SAC")
```

Algorithm classes for reinforcement learning.

**Common Methods:**
- `train(env, episodes=1000, max_steps=1000, render=False, verbose=True, callback=None)`: Train the agent on an environment
- `save(path)`: Save agent to disk
- `load(path)`: Load agent from disk

### MultiAgentSystem

```python
neurenix.rl.MultiAgentSystem(agents, env, name="MultiAgentSystem")
```

Multi-agent system for reinforcement learning.

**Parameters:**
- `agents`: List of agents
- `env`: Environment
- `name`: System name

**Methods:**
- `train(episodes=1000, max_steps=1000, render=False, verbose=True, callback=None)`: Train the multi-agent system
- `save(path)`: Save multi-agent system to disk
- `load(path)`: Load multi-agent system from disk

## Framework Comparison

### Neurenix vs. TensorFlow

| Feature | Neurenix | TensorFlow |
|---------|----------|------------|
| **Agent Framework** | Unified agent framework with built-in support for various algorithms | Requires custom implementation or third-party libraries (e.g., TF-Agents) |
| **Algorithm Implementations** | Built-in implementations of DQN, A2C, PPO, DDPG, SAC | Limited built-in support, requires TF-Agents |
| **Multi-Agent Support** | Native support through MultiAgentSystem | Limited support through third-party libraries |
| **Edge Device Support** | Native optimization for edge devices | TensorFlow Lite for edge devices |
| **Hardware Acceleration** | Multi-device support (CPU, CUDA, ROCm, WebGPU) | Primarily optimized for CPU and CUDA |
| **API Design** | Consistent API across all algorithms | Varies between different libraries and implementations |

Neurenix's reinforcement learning capabilities offer a more unified and integrated approach compared to TensorFlow, which often requires third-party libraries like TF-Agents for comprehensive reinforcement learning support. The native implementation of multiple reinforcement learning algorithms in Neurenix provides a consistent API and seamless integration with other framework components. Additionally, Neurenix's multi-language architecture and edge device optimization make it particularly well-suited for deploying reinforcement learning models in resource-constrained environments.

### Neurenix vs. PyTorch

| Feature | Neurenix | PyTorch |
|---------|----------|---------|
| **Agent Framework** | Unified agent framework with built-in support for various algorithms | Requires custom implementation or third-party libraries (e.g., Stable Baselines3) |
| **Algorithm Implementations** | Built-in implementations of DQN, A2C, PPO, DDPG, SAC | Limited built-in support, requires third-party libraries |
| **Multi-Agent Support** | Native support through MultiAgentSystem | Limited support through third-party libraries |
| **Hardware Support** | CPU, CUDA, ROCm, WebGPU | CPU, CUDA |
| **Edge Device Optimization** | Native optimization for edge devices | PyTorch Mobile for edge devices |
| **API Design** | Consistent API across all algorithms | Varies between different libraries and implementations |

Neurenix provides a more comprehensive and integrated reinforcement learning solution compared to PyTorch, which requires third-party libraries like Stable Baselines3 for most reinforcement learning algorithms. While PyTorch's dynamic computation graph makes it flexible for implementing custom reinforcement learning algorithms, Neurenix's built-in implementations offer a more streamlined experience with less boilerplate code. Neurenix also extends hardware support to include ROCm and WebGPU, making it more versatile across different hardware platforms.

### Neurenix vs. Stable Baselines3

| Feature | Neurenix | Stable Baselines3 |
|---------|----------|-------------------|
| **Framework Integration** | Fully integrated with Neurenix's tensor operations and neural networks | Built on top of PyTorch |
| **Algorithm Implementations** | DQN, A2C, PPO, DDPG, SAC with consistent API | DQN, A2C, PPO, DDPG, SAC, TD3, and more |
| **Multi-Agent Support** | Native support through MultiAgentSystem | Limited support |
| **Hardware Support** | CPU, CUDA, ROCm, WebGPU | CPU, CUDA (via PyTorch) |
| **Edge Device Optimization** | Native optimization for edge devices | Limited support |
| **Customizability** | Modular design with extensible components | Modular design with extensible components |

Neurenix and Stable Baselines3 both offer comprehensive reinforcement learning capabilities, but Neurenix's integration with its own tensor operations and neural networks provides a more cohesive experience. While Stable Baselines3 offers a wider range of algorithm implementations, Neurenix's consistent API and native multi-agent support make it more suitable for complex reinforcement learning tasks. Neurenix also provides better support for edge devices and a wider range of hardware platforms.

## Best Practices

### Choosing the Right Algorithm

Different reinforcement learning algorithms have different strengths and weaknesses:

1. **DQN**:
   - Best for discrete action spaces
   - Good for problems with complex state spaces
   - Requires tuning of exploration parameters

2. **A2C**:
   - Works with both discrete and continuous action spaces
   - More sample-efficient than DQN
   - Can be unstable during training

3. **PPO**:
   - Works with both discrete and continuous action spaces
   - More stable than A2C
   - Good default choice for many problems

4. **DDPG**:
   - Best for continuous action spaces
   - Can be sample-efficient
   - Sensitive to hyperparameters

5. **SAC**:
   - Best for continuous action spaces
   - More stable than DDPG
   - Automatically balances exploration and exploitation

### Optimizing for Edge Devices

When deploying reinforcement learning models to edge devices, consider these optimizations:

1. **Model Size**: Use smaller networks with fewer parameters
2. **Quantization**: Quantize model weights to reduce memory usage
3. **Pruning**: Remove unnecessary connections in neural networks
4. **Efficient Architectures**: Use architectures specifically designed for edge devices
5. **Action Space**: Consider discretizing continuous action spaces for better performance

### Multi-Agent Reinforcement Learning

When working with multi-agent systems, consider these best practices:

1. **Centralized Training, Decentralized Execution**: Train agents with access to global information, but execute with only local information
2. **Communication Protocols**: Implement efficient communication between agents
3. **Reward Shaping**: Design rewards to encourage cooperation or competition as needed
4. **Curriculum Learning**: Start with simpler tasks and gradually increase complexity

## Tutorials

### Training a DQN Agent on GridWorld

```python
import neurenix
from neurenix.rl import DQN
from neurenix.rl import GridWorld

# Create environment
env = GridWorld(width=10, height=10, obstacle_density=0.2)

# Get observation and action space
observation_space = env.get_observation_space()
action_space = env.get_action_space()

# Create DQN agent
dqn = DQN(
    observation_space=observation_space,
    action_space=action_space,
    hidden_dims=[64, 64],
    learning_rate=0.001,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
    buffer_size=10000,
    batch_size=64,
    update_target_every=100,
)

# Train agent
metrics = dqn.train(
    env=env,
    episodes=1000,
    max_steps=100,
    render=False,
    verbose=True,
)

# Plot training metrics
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(metrics["episode_rewards"])
plt.title("Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")

plt.subplot(1, 2, 2)
plt.plot(metrics["episode_lengths"])
plt.title("Episode Lengths")
plt.xlabel("Episode")
plt.ylabel("Length")

plt.tight_layout()
plt.show()

# Save agent
dqn.save("dqn_gridworld")

# Test agent
state = env.reset()
done = False
total_reward = 0

while not done:
    # Select action
    action = dqn.agent.act(state)
    
    # Take action
    next_state, reward, done, info = env.step(action)
    
    # Update state and reward
    state = next_state
    total_reward += reward
    
    # Render environment
    env.render()

print(f"Total reward: {total_reward}")
```

### Implementing a Custom Environment

```python
import neurenix
import numpy as np
from neurenix.rl import Environment

class CartPole(Environment):
    """
    CartPole environment for reinforcement learning.
    
    This environment simulates a pole balancing on a cart, where the
    agent must apply forces to the cart to keep the pole upright.
    """
    
    def __init__(
        self,
        gravity=9.8,
        cart_mass=1.0,
        pole_mass=0.1,
        pole_length=0.5,
        max_steps=200,
        name="CartPole",
    ):
        """
        Initialize CartPole environment.
        
        Args:
            gravity: Acceleration due to gravity
            cart_mass: Mass of the cart
            pole_mass: Mass of the pole
            pole_length: Length of the pole
            max_steps: Maximum number of steps per episode
            name: Environment name
        """
        super().__init__(name=name, max_steps=max_steps)
        
        self.gravity = gravity
        self.cart_mass = cart_mass
        self.pole_mass = pole_mass
        self.pole_length = pole_length
        
        # State variables
        self.x = 0.0  # Cart position
        self.x_dot = 0.0  # Cart velocity
        self.theta = 0.0  # Pole angle
        self.theta_dot = 0.0  # Pole angular velocity
        
        # Actions
        self.LEFT = 0
        self.RIGHT = 1
        
        # Action forces
        self.force_mag = 10.0
        
        # Simulation parameters
        self.tau = 0.02  # Time step
        
        # Threshold for termination
        self.x_threshold = 2.4
        self.theta_threshold = 12.0 * np.pi / 180.0
    
    def _reset_state(self):
        """
        Reset the state.
        
        Returns:
            Initial state
        """
        self.x = 0.0
        self.x_dot = 0.0
        self.theta = np.random.uniform(-0.05, 0.05)
        self.theta_dot = np.random.uniform(-0.05, 0.05)
        
        return self._get_state()
    
    def _get_state(self):
        """
        Get the current state.
        
        Returns:
            Current state
        """
        return np.array([self.x, self.x_dot, self.theta, self.theta_dot])
    
    def _step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Check if action is valid
        if action not in [self.LEFT, self.RIGHT]:
            raise ValueError(f"Invalid action: {action}")
        
        # Apply force
        force = -self.force_mag if action == self.LEFT else self.force_mag
        
        # Physics simulation
        costheta = np.cos(self.theta)
        sintheta = np.sin(self.theta)
        
        # Calculate acceleration
        temp = (force + self.pole_mass * self.pole_length * self.theta_dot**2 * sintheta) / (self.cart_mass + self.pole_mass)
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.pole_length * (4.0/3.0 - self.pole_mass * costheta**2 / (self.cart_mass + self.pole_mass)))
        xacc = temp - self.pole_mass * self.pole_length * thetaacc * costheta / (self.cart_mass + self.pole_mass)
        
        # Update state
        self.x = self.x + self.tau * self.x_dot
        self.x_dot = self.x_dot + self.tau * xacc
        self.theta = self.theta + self.tau * self.theta_dot
        self.theta_dot = self.theta_dot + self.tau * thetaacc
        
        # Check if done
        done = (
            self.x < -self.x_threshold
            or self.x > self.x_threshold
            or self.theta < -self.theta_threshold
            or self.theta > self.theta_threshold
        )
        
        # Calculate reward
        reward = 1.0 if not done else 0.0
        
        # Return state, reward, done, info
        return self._get_state(), reward, done, {}
    
    def get_observation_space(self):
        """
        Get the observation space.
        
        Returns:
            Observation space specification
        """
        return {
            "type": "box",
            "shape": (4,),
            "low": np.array([-self.x_threshold * 2, -np.inf, -self.theta_threshold * 2, -np.inf]),
            "high": np.array([self.x_threshold * 2, np.inf, self.theta_threshold * 2, np.inf]),
            "dtype": np.float32,
        }
    
    def get_action_space(self):
        """
        Get the action space.
        
        Returns:
            Action space specification
        """
        return {
            "type": "discrete",
            "n": 2,
        }
```

## Conclusion

The Reinforcement Learning module of Neurenix provides a comprehensive set of tools for developing and training reinforcement learning agents. Its multi-language architecture with a high-performance Rust/C++ core enables optimal performance across a wide range of devices, from edge devices to multi-GPU clusters.

Compared to other frameworks like TensorFlow, PyTorch, and Stable Baselines3, Neurenix's Reinforcement Learning module offers advantages in terms of API design, hardware support, and edge device optimization. The unified agent framework and implementations of multiple reinforcement learning algorithms provide a consistent and integrated experience, making Neurenix particularly well-suited for reinforcement learning tasks and AI agent development.
